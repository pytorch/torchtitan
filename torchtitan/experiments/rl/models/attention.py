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
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import LocalMapInnerAttention
from torchtitan.tools.logging import warn_once
from torchtitan.tools.utils import has_cuda_capability
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

logger = logging.getLogger(__name__)


@register_backend(AttentionBackendEnum.CUSTOM)
class PyTorchVarlenAttentionBackend(FlashAttentionBackend):
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
        return PyTorchVarlenAttentionImpl


class PyTorchVarlenAttentionImpl(FlashAttentionImpl):
    """
    Custom vLLM attention backend impl using PyTorch's native FlashAttention varlen API.
    Instead of using vLLM's FlashAttention kernel, this implementation takes the kernel
    dependency from torch directly while supporting the same interface.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Hopper (SM 9.0) uses FA3
        if has_cuda_capability(9, 0):
            # activate_flash_attention_impl() will restore internal global state
            # and re-run register function, so we want to only call it once.
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")
        else:
            warn_once(
                logger, "FA3 not available (requires SM 9.0+), falling back to FA2. "
            )

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
        if current_flash_attention_impl() == "FA3":
            cu_seqlens_k = None
        else:
            num_seqs = seqused_k.shape[0]
            cu_seqlens_k = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k[1:] = torch.cumsum(seqused_k, dim=0)
        # FA3 + batch-invariant: fix num_splits=1 to prevent non-deterministic
        # split-k reductions. FA2 is automatically batch-invariant and does
        # not accept num_splits.
        extra_kwargs = {}

        # Disable split_kv in Flash Attention to ensure bitwise identical output.
        # see https://github.com/pytorch/pytorch/pull/176905
        if is_in_batch_invariant_mode() and current_flash_attention_impl() == "FA3":
            extra_kwargs["num_splits"] = 1

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
            **extra_kwargs,
        )


class VLLMAttentionWrapper(LocalMapInnerAttention):
    """Adapter from TorchTitan tensor layout to ``vllm.Attention``.

    vLLM's ``Attention`` layer manages KV-cache and paged attention internally,
    but expects flattened ``(num_tokens, num_heads, head_dim)`` inputs.

    Receives ``(bs, seq, heads, dim)`` layout from GQAttention. DTensor with
    ``Shard(2)`` placements is handled by the base class
    ``LocalMapInnerAttention.__call__``.

    Used as ``inner_attention`` in GQAttention via Config-based construction.
    """

    # vLLM requires a unique prefix per Attention layer for
    # static_forward_context registration.
    # TODO: Pass layer_id through the build chain instead of using a
    # global counter. The counter breaks with pipeline parallelism
    # where layers are built on different ranks.
    _layer_counter: itertools.count = itertools.count()

    @dataclass(kw_only=True, slots=True)
    class Config(LocalMapInnerAttention.Config):
        hidden_size: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        scale: float | None = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        tp_degree = vllm_config.parallel_config.tensor_parallel_size

        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads

        if num_kv_heads < tp_degree:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be >= "
                f"tensor_parallel_size ({tp_degree})"
            )
        if num_kv_heads % tp_degree != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )
        if num_heads % tp_degree != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )

        num_heads = num_heads // tp_degree
        num_kv_heads = num_kv_heads // tp_degree
        head_dim = config.head_dim
        scale = config.scale if config.scale is not None else head_dim**-0.5

        self.hidden_size = config.hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale

        cache_config = (
            vllm_config.cache_config if hasattr(vllm_config, "cache_config") else None
        )

        # TODO: This need to be compatible with Pipeline Parallelism
        layer_id = next(VLLMAttentionWrapper._layer_counter)
        self.vllm_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=scale,
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
        **kwargs,
    ) -> torch.Tensor:
        """Run vLLM paged attention on local (non-DTensor) tensors.

        Args:
            q: ``(batch, seq_len, num_heads, head_dim)``
            k: ``(batch, seq_len, num_kv_heads, head_dim)``
            v: ``(batch, seq_len, num_kv_heads, head_dim)``

        Returns:
            ``(batch, seq_len, num_heads * head_dim)`` — ready for
            ``output.view(bs, seqlen, -1)`` in GQAttention.forward
        """
        batch_size, seq_len, _, head_dim = q.shape

        # vllm attention expects (bs*seqlen, n_heads, head_dim)
        # (bs, seq, heads, dim) is contiguous, so reshape is zero-copy
        q = q.reshape(batch_size * seq_len, -1, head_dim)
        k = k.reshape(batch_size * seq_len, -1, head_dim)
        v = v.reshape(batch_size * seq_len, -1, head_dim)

        output_flat = self.vllm_attn(q, k, v)

        # vLLM's flash attention backend may pad the token count (e.g.
        # round up to an even number), which introduces a new symbolic
        # shape under torch.compile.  Narrow to trim this padding.
        output_flat = output_flat.narrow(0, 0, batch_size * seq_len)

        # Reshape back to the format expected by GQAttention.forward()
        output = output_flat.view(batch_size, seq_len, -1, head_dim)

        return output
