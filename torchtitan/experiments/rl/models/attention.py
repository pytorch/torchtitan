# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from dataclasses import dataclass

import torch
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from torchtitan.models.common.attention import (
    AttentionMasksType,
    GQAttention,
    LocalMapAttention,
)
from torchtitan.models.common.rope import (
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)
from torchtitan.tools.utils import has_cuda_capability

from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


@register_backend(AttentionBackendEnum.CUSTOM)
class PyTorchFlashAttentionBackend(FlashAttentionBackend):
    """Custom vLLM attention backend using PyTorch's native FlashAttention kernel.

    This class is not directly referenced in user code. It is registered into
    vLLM's attention backend registry via the ``@register_backend`` decorator
    and selected at runtime when the vLLM engine is configured to use a CUSTOM
    attention backend.

    Inheriting from ``FlashAttentionBackend`` is not strictly required for all
    backends, but it is convenient here to reuse metadata construction logic.
    """

    @staticmethod
    def get_name():
        # vLLM requires any custom attention backend to return "CUSTOM" as its
        # name so the backend registry can look it up correctly.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PyTorchFlashAttentionImpl


class PyTorchFlashAttentionImpl(FlashAttentionImpl):
    """
    Custom vLLM attention backend impl using PyTorch's native FlashAttention varlen API.
    Instead of using vLLM's FlashAttention kernel, this implementation takes the kernel
    dependency from torch directly while supporting the same interface.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # FA3 requires SM 9.0+ (e.g. H100); check capability explicitly because
        # activate_flash_attention_impl("FA3") succeeds even on SM80.
        # Fall back to FA2 which requires page_size to be a multiple of 256.
        if has_cuda_capability(9, 0):
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")
            self._use_fa3 = True
        else:
            logger.warning(
                "FA3 not available (requires SM 9.0+), falling back to FA2. "
                "vLLM block_size must be set to 256 for FA2 paged attention."
            )
            self._use_fa3 = False

    # Based on vLLM's FlashAttentionImpl.forward():
    # https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        assert (
            self.vllm_flash_attn_version is not None
        ), "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        assert attn_type not in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ), "Encoder-only attention not supported yet."

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        assert not self.kv_cache_dtype.startswith(
            "fp8"
        ), "FP8 KV cache not supported yet."

        assert not attn_metadata.use_cascade, "Cascade not supported yet."

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        assert self.dcp_world_size == 1, "DCP not supported yet."

        if attn_metadata.causal:
            sliding_window_size = (-1, 0)
        else:
            raise RuntimeError("Non-causal attention not supported yet.")

        assert self.alibi_slopes is None, "Alibi slopes not supported yet."

        # FA3 can infer cu_seqlens_k from block_table + seqused_k.
        # FA2 requires cu_seqlens_k to be explicitly set.
        if self._use_fa3:
            cu_seqlens_k = None
        else:
            num_seqs = seqused_k.shape[0]
            cu_seqlens_k = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k[1:] = torch.cumsum(seqused_k, dim=0)

        return torch.nn.attention.varlen.varlen_attn_out(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=sliding_window_size,
            block_table=block_table,
            seqused_k=seqused_k,
        )


logger = logging.getLogger(__name__)


class VLLMInnerAttention(LocalMapAttention):
    """Adapter from TorchTitan tensor layout to ``vllm.Attention``.

    vLLM's ``Attention`` layer manages KV-cache and paged attention internally,
    but expects flattened ``(num_tokens, num_heads, head_dim)`` inputs.

    Called by :class:`VLLMGQAttention` with ``(bs, seq, heads, dim)`` layout
    (heads on dim 2). DTensor with ``Shard(2)`` placements is handled by
    the base class ``LocalMapAttention.__call__``.

    Used by the **generator** (via :class:`VLLMGQAttention`).
    """

    # vLLM requires a unique prefix per Attention layer for
    # static_forward_context registration.
    # TODO: Pass layer_id through the build chain instead of using a
    # global counter. The counter breaks with pipeline parallelism
    # where layers are built on different ranks.
    _layer_counter: itertools.count = itertools.count()

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if scale is None:
            self.scale = head_dim**-0.5
        else:
            self.scale = scale

        cache_config = (
            vllm_config.cache_config if hasattr(vllm_config, "cache_config") else None
        )

        layer_id = next(VLLMInnerAttention._layer_counter)
        self.vllm_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=self.scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"model.layers.{layer_id}.attention.inner_attention",
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Run vLLM paged attention on local (non-DTensor) tensors.

        Called by :class:`VLLMGQAttention` which passes tensors in
        ``(batch, seq_len, num_heads, head_dim)`` layout — the natural
        contiguous layout after QKV projection + RoPE. This allows a
        zero-copy reshape to ``(batch*seq_len, num_heads, head_dim)``
        for vLLM.

        Args:
            q: ``(batch, seq_len, num_heads, head_dim)``
            k: ``(batch, seq_len, num_kv_heads, head_dim)``
            v: ``(batch, seq_len, num_kv_heads, head_dim)``

        Returns:
            ``(batch, seq_len, num_heads * head_dim)`` — ready for ``wo``
        """
        batch_size, seq_len, _, head_dim = q.shape

        # (bs, seq, heads, dim) is contiguous, so reshape is zero-copy
        q = q.reshape(batch_size * seq_len, -1, head_dim)
        k = k.reshape(batch_size * seq_len, -1, head_dim)
        v = v.reshape(batch_size * seq_len, -1, head_dim)

        output_flat = self.vllm_attn(q, k, v)

        # vLLM's flash attention backend may pad the token count (e.g.
        # round up to an even number), which introduces a new symbolic
        # shape under torch.compile.  Narrow to trim this padding.
        output_flat = output_flat.narrow(0, 0, batch_size * seq_len)

        # (bs*seq, heads*dim) -> (bs, seq, heads*dim)
        return output_flat.view(batch_size, seq_len, -1)


class VLLMGQAttention(GQAttention):
    """GQAttention subclass with zero-copy path for vLLM attention.

    Eliminates redundant transposes between GQAttention and VLLMInnerAttention.

    Standard GQAttention + VLLMInnerAttention flow:
      (bs,seq,heads,dim) -> transpose -> (bs,heads,seq,dim)
      -> VLLMInnerAttention transpose back -> clone -> (bs*seq,heads,dim)
      -> vllm_attn -> reshape -> transpose -> contiguous  (4 layout ops)

    This subclass:
      (bs,seq,heads,dim) -> reshape -> (bs*seq,heads,dim)  (1 zero-copy op)
      -> vllm_attn -> view -> (bs,seq,hidden)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        attn_backend: str = "varlen"

    def __init__(self, config: Config, *, dim: int):
        super().__init__(config, dim=dim)

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        tp_degree = vllm_config.parallel_config.tensor_parallel_size

        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads or config.n_heads
        head_dim = config.head_dim if config.head_dim is not None else dim // n_heads

        if n_kv_heads < tp_degree:
            raise ValueError(
                f"n_kv_heads ({n_kv_heads}) must be >= "
                f"tensor_parallel_size ({tp_degree})"
            )
        assert n_kv_heads % tp_degree == 0
        assert n_heads % tp_degree == 0

        self.inner_attention = VLLMInnerAttention(
            hidden_size=dim,
            num_heads=n_heads // tp_degree,
            num_kv_heads=n_kv_heads // tp_degree,
            head_dim=head_dim,
            scale=head_dim**-0.5,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        if self.use_rope:
            if self.rope_backend == "cos_sin":
                xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        output = self.inner_attention(xq, xk, xv)

        return self.wo(output)
