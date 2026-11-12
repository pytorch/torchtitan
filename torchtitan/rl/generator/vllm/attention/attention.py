# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import spmd_types as spmd

import torch
from torch.nn.attention import current_flash_attention_impl
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.rl.generator.attention.attention import paged_varlen_attention_out
from torchtitan.tools.utils import (
    activate_cuda_flash_attention_impl,
    get_cuda_flash_attention_impl,
    set_default_dtype,
)


class _FA4PageSize:
    """Pick the KV kernel page size for FA4 on SM100/SM103: 128 tokens, or 16 if batch-invariant.

    There, FA4 loads prefill K/V with one bulk copy (TMA) only when a page holds
    exactly one 128-key tile, and decodes one token per sequence on its SwapAB
    kernel. 16-token pages get a slower gather. vLLM already picks 128 for head
    dim 256. Other GPUs that run FA4 (e.g. SM120, whose FA4 forward is the SM80
    kernel) keep vLLM's default.
    Costs: prefix-cache reuse in whole 128-token pages, up to 127 KV slots wasted
    per sequence, and an explicit ``block_size`` that is not a multiple of 128 now
    fails.
    Batch-invariant runs get exactly 16-token pages. FA4 runs its SwapAB decode
    kernel on any page that is a multiple of 128 (``SwapABPolicy.page_multiple`` in
    ``flash_attn/cute/config.py``). SwapAB reads 256 keys per tile and prefill reads
    128, so decode would not match prefill bitwise. ``MultipleOf(16)`` is not enough:
    it keeps a hybrid model's 1152-token KV block as one page.
    TODO: ask FA4 for its standard decode kernel per call once torch's varlen
    attention can pass that choice; batch-invariant runs can then keep 128-token pages.
    """

    @classmethod
    def get_supported_kernel_block_sizes(
        cls, kv_cache_spec: KVCacheSpec | None = None
    ) -> list[int | MultipleOf]:
        if (
            get_cuda_flash_attention_impl() == "FA4"
            and torch.cuda.get_device_capability()[0] == 10
        ):
            return [16] if is_in_batch_invariant_mode() else [128]
        # vLLM main passes kv_cache_spec; the vLLM nightly on PyTorch's index
        # (9f03b51) takes no argument.
        # TODO: always forward it once that nightly includes vllm-project/vllm#53175.
        if kv_cache_spec is None:
            return super().get_supported_kernel_block_sizes()
        return super().get_supported_kernel_block_sizes(kv_cache_spec)


@register_backend(AttentionBackendEnum.CUSTOM)
class TorchTitanVarlenInnerAttentionBackend(_FA4PageSize, FlashAttentionBackend):
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
        return TorchTitanVarlenInnerAttentionImpl

    @staticmethod
    def get_builder_cls():
        class TorchTitanVarlenInnerAttentionMetadataBuilder(
            FlashAttentionMetadataBuilder
        ):
            _cudagraph_support = AttentionCGSupport.ALWAYS
            # Single-token rows first: TorchTitanVarlenInnerAttentionImpl.forward runs them
            # as their own FA4 call.
            reorder_batch_threshold = 1

        return TorchTitanVarlenInnerAttentionMetadataBuilder


class TorchTitanVarlenInnerAttentionDiffKVBackend(
    _FA4PageSize, FlashAttentionDiffKVBackend
):
    """PyTorch varlen backend using vLLM's unequal K/V cache support."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # vLLM sizes a hybrid model's attention page from a spec without head_size_v,
        # so V gets K's head size and the page comes out larger than this layer's real
        # page. vLLM then cannot split a block into smaller kernel blocks (FA4 on SM100).
        # vLLM's DiffKV backends keep V's head size on the class (set_head_size_v), so
        # one V head size per process; Kimi K3 has one.
        # TODO: drop once vLLM's DiffKV backends set head_size_v in customize_spec.
        return super().customize_spec(replace(spec, head_size_v=cls.head_size_v))

    @staticmethod
    def get_impl_cls():
        return TorchTitanVarlenInnerAttentionDiffKVImpl

    @staticmethod
    def get_builder_cls():
        return TorchTitanVarlenInnerAttentionBackend.get_builder_cls()


class TorchTitanVarlenInnerAttentionImpl(FlashAttentionImpl):
    """
    Custom vLLM attention backend impl using PyTorch's native FlashAttention varlen API.
    Instead of using vLLM's FlashAttention kernel, this implementation takes the kernel
    dependency from torch directly while supporting the same interface.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # optional post-attention epilogue transform (out, lse) -> out
        # set by VLLMModelWrapper in vllm_wrapper.py
        self.out_transform = None

        self.enable_gqa = self.num_heads > self.num_kv_heads

        activate_cuda_flash_attention_impl()

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
            kv_cache: shape is ``[num_blocks, num_kv_heads, block_size,
                2 * head_size]``, with K and V packed in the last dimension.
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

        # Breakable cuda_graph: under VLLM_USE_BREAKABLE_CUDAGRAPH the recorded
        # forward closure pins capture-time args (attn_metadata None, varlen metadata
        # absent). Re-read live per-layer metadata + kv_cache from the forward context
        # (vLLM refreshes them before each replay) BEFORE the None check, else replay
        # short-circuits to output.fill_(0) (zeroed attention). No-op outside capture.
        attn_metadata, _, kv_cache, _ = get_attention_context(layer.layer_name)

        if attn_metadata is None:
            # Profiling / CUDA graph dummy-capture run (no real metadata yet).
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

        # vLLM #44455 packs K and V into the content dimension for full-attention
        # layers, including those in hybrid models. Restore the layout expected by
        # varlen attention before splitting them.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        assert not self.kv_cache_dtype.startswith(
            "fp8"
        ), "FP8 KV cache not supported yet."

        assert not attn_metadata.use_cascade, "Cascade not supported yet."

        assert self.dcp_world_size == 1, "DCP not supported yet."

        if not attn_metadata.causal:
            raise RuntimeError("Non-causal attention not supported yet.")

        # vLLM assigns sliding_window_size = None to (-1, -1) w/ optional causal flag
        # but varlen only encode with sliding window. so we need to convert vllm (-1, -1) to (-1, 0)
        # for proper full causal attention instead of bidirectional
        if self.sliding_window == (-1, -1):
            sliding_window_size = (-1, 0)
        else:
            # by default vLLM sets attention type = DECODER, which will set (W-1, 0)
            sliding_window_size = self.sliding_window

        assert self.alibi_slopes is None, "Alibi slopes not supported yet."

        # TODO(https://github.com/Dao-AILab/flash-attention/issues/3146): FA4 picks one
        # kernel configuration per call from max_seqlen_q, so the single-token decode rows
        # of a mixed prefill/decode step lose the configuration they get in a decode-only
        # step and run 3-8x slower (e.g. at head dim 256, when the GQA ratio does not
        # divide 128, FA4 turns PackGQA off above 128 query tokens). Until FA4 decides per
        # sequence, run those rows as their own call. The builder puts them first
        # (reorder_batch_threshold = 1), so clamping cu_seqlens_q at their token count
        # splits the batch on device, inside CUDA graphs too. Each call gets the other
        # call's rows with zero KV length, since FA4 can still process the KV of a row
        # that has no query tokens:
        #   cu_seqlens_q            [0, 1, 2, 3, 653]   (3 decode rows, one 650-token row)
        #   decode call   min(., 3) [0, 1, 2, 3, 3]     max_seqlen_q 1    seqused_k [k0, k1, k2, 0]
        #   prefill call  max(., 3) [3, 3, 3, 3, 653]   max_seqlen_q 650  seqused_k [0, 0, 0, k3]
        # A 1-token prompt sorts after the prefills and runs in the prefill call.
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        seqused_k = attn_metadata.seq_lens
        if (
            current_flash_attention_impl() == "FA4"
            and max_seqlen_q > 1
            and self.out_transform is None
        ):
            is_single_token = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) == 1
            is_leading_single_token = is_single_token.int().cumprod(0).bool()
            num_leading_single_tokens = is_leading_single_token.sum(dtype=torch.int32)
            calls = [
                (
                    torch.minimum(cu_seqlens_q, num_leading_single_tokens),
                    1,
                    torch.where(is_leading_single_token, seqused_k, 0),
                ),
                (
                    torch.maximum(cu_seqlens_q, num_leading_single_tokens),
                    max_seqlen_q,
                    torch.where(is_leading_single_token, 0, seqused_k),
                ),
            ]
        else:
            calls = [(cu_seqlens_q, max_seqlen_q, seqused_k)]
        for call_cu_seqlens_q, call_max_seqlen_q, call_seqused_k in calls:
            result = paged_varlen_attention_out(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                cu_seqlens_q=call_cu_seqlens_q,
                max_seqlen_q=call_max_seqlen_q,
                max_seqlen_k=attn_metadata.max_seq_len,
                seqused_k=call_seqused_k,
                block_table=attn_metadata.block_table,
                scale=self.scale,
                window_size=sliding_window_size,
                enable_gqa=self.enable_gqa,
                return_lse=self.out_transform is not None,
            )
        if self.out_transform is None:
            return result

        out, lse = result
        out = self.out_transform(out, lse.transpose(0, 1))
        output[:num_actual_tokens].copy_(out)
        return output[:num_actual_tokens]


class TorchTitanVarlenInnerAttentionDiffKVImpl(TorchTitanVarlenInnerAttentionImpl):
    """PyTorch varlen implementation using vLLM's unequal K/V cache writer."""

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        return FlashAttentionDiffKVImpl.do_kv_cache_update(
            self, layer, key, value, kv_cache, slot_mapping
        )


class _VLLMAttention(Attention, Module):
    """vLLM attention that participates in the TorchTitan Module protocol."""


def _exclude_vllm_attn_scales_from_state_dict(layer: torch.nn.Module) -> None:
    """Re-register a vLLM attention layer's quantization scales as non-persistent.

    vLLM registers them as persistent buffers, but they are not model weights, so
    keeping them in the state dict breaks HF state dict adapters that reject
    unmapped keys.
    """
    for name in VLLMInnerAttention.vllm_attn_scale_buffer_names:
        layer.register_buffer(name, layer._buffers.pop(name), persistent=False)


class VLLMInnerAttention(InnerAttention):
    """Adapter from TorchTitan tensor layout to ``vllm.Attention``.

    vLLM's ``Attention`` layer manages KV-cache and paged attention internally,
    and expects ``(num_tokens, num_heads, head_dim)`` inputs.

    Used as ``inner_attention`` in GQAttention via Config-based construction.
    """

    # vLLM requires a unique prefix per Attention layer for
    # static_forward_context registration.
    # TODO: Pass layer_id through the build chain instead of using a
    # global counter. The counter breaks with pipeline parallelism
    # where layers are built on different ranks.
    _layer_counter: itertools.count = itertools.count()
    vllm_attn_scale_buffer_names: ClassVar[tuple[str, ...]] = (
        "_k_scale",
        "_prob_scale",
        "_q_scale",
        "_v_scale",
    )

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        hidden_size: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        value_head_dim: int | None = None
        scale: float | None = None
        sliding_window_size: int | None = None
        """Causal sliding-window size (``None`` => full attention)."""

        @property
        def is_full_attention(self) -> bool:
            return self.sliding_window_size is None

    def __init__(self, config: Config) -> None:
        super().__init__()

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
        value_head_dim = (
            config.value_head_dim if config.value_head_dim is not None else head_dim
        )
        scale = config.scale if config.scale is not None else head_dim**-0.5

        self.hidden_size = config.hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.scale = scale

        cache_config = (
            vllm_config.cache_config if hasattr(vllm_config, "cache_config") else None
        )

        # TODO: This need to be compatible with Pipeline Parallelism
        layer_id = next(VLLMInnerAttention._layer_counter)
        diff_kv_kwargs: dict[str, Any] = {}
        if value_head_dim != head_dim:
            FlashAttentionDiffKVBackend.set_head_size_v(value_head_dim)
            diff_kv_kwargs = {
                "head_size_v": value_head_dim,
                "attn_backend": TorchTitanVarlenInnerAttentionDiffKVBackend,
            }
        # vLLM selects the attention backend from the default dtype. Build the layer
        # in vLLM's model dtype, as vLLM builds its own models; the wrapper builds
        # the TorchTitan model in the trainer's master dtype (``precision.dtype``).
        with set_default_dtype(vllm_config.model_config.dtype):
            self.vllm_attn = _VLLMAttention(
                num_heads=num_heads,
                head_size=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                cache_config=cache_config,
                quant_config=None,
                per_layer_sliding_window=config.sliding_window_size,
                prefix=f"model.layers.{layer_id}.attention.inner_attention",
                **diff_kv_kwargs,
            )
        _exclude_vllm_attn_scales_from_state_dict(self.vllm_attn)
        # Set sharding config for the inner vLLM attention module, which has scale
        # buffers that need to be replicated on all ranks in the dense mesh
        replicated_dense = dense_param_placement(tp=spmd.R)
        self.vllm_attn._sharding_config = ShardingConfig(
            state_shardings={
                name: replicated_dense for name in self.vllm_attn_scale_buffer_names
            },
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: None = None,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run vLLM paged attention on rank-local tensors.

        Args:
            q_THK: ``(num_tokens, num_heads, query/key_head_dim)``
            k_THK: ``(num_tokens, num_kv_heads, query/key_head_dim)``
            v_THV: ``(num_tokens, num_kv_heads, value_head_dim)``
            scale: Optional per-call scale; must match the scale set at construction.

        Returns:
            ``(num_tokens, num_heads, value_head_dim)``.
        """
        if attention_metadata is not None:
            raise ValueError(
                "VLLMInnerAttention does not support attention_metadata; vLLM "
                "manages causal masking and the KV-cache internally."
            )

        if scale is not None and scale != self.scale:
            raise ValueError(
                f"Model attention scale {scale} != vLLM scale {self.scale}"
            )

        if self.value_head_dim != self.head_dim:
            # V may be a strided view (e.g. MLA splits it from a fused per-head
            # KV projection), but vLLM's cache writer assumes V heads are packed
            # with stride value_head_dim, so repack before updating the cache.
            v_THV = v_THV.contiguous()
        out_TD = self.vllm_attn(q_THK, k_THK, v_THV)

        # vLLM's flash attention backend may pad the token count (e.g.
        # round up to an even number), which introduces a new symbolic
        # shape under torch.compile.  Narrow to trim this padding.
        num_tokens = q_THK.shape[0]
        out_TD = out_TD.narrow(0, 0, num_tokens)
        return out_TD.view(num_tokens, -1, self.value_head_dim)
