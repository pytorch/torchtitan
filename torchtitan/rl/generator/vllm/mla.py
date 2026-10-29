# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM native MLA adapter for TorchTitan's multi-head latent attention.

vLLM's MLA layer caches the compressed latent ``[kv_c_normed, k_pe]`` and
decodes in latent space by folding ``wkv_b`` into the query and output.
Batch-invariant mode does not use this module; it keeps materialized K/V
through ``VLLMInnerAttention`` for bitwise parity with the trainer.
"""

from dataclasses import dataclass

import spmd_types as spmd

import torch
from vllm.model_executor.layers.attention import MLAAttention as VLLMNativeMLAAttention

from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import InnerAttention, MLAAttention
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.rl.generator.vllm.attention import (
    _exclude_vllm_attn_scales_from_state_dict,
    VLLMInnerAttention,
)

# Shape suffixes:
# T = packed token count (num_tokens)
# H = local attention head count (num_heads)
# K = full per-head query dimension (qk_nope_head_dim + qk_rope_head_dim)
# L = latent KV dimension (kv_lora_rank)
# R = head-shared key dimension (qk_rope_head_dim)

# TODO: replace vLLM's MLA kernel with attn-gym latent-space decode kernel


class _VLLMKVBProj:
    """Expose a TorchTitan ``Linear`` through the ``kv_b_proj`` interface vLLM MLA reads.

    Supports a bias-free bf16 ``Linear`` only: no LoRA, no MXFP8/NVFP4, no bias.
    vLLM absorbs ``weight`` into its decode matmuls, so it must be the matrix the
    linear applies.
    """

    quant_method = None

    def __init__(self, linear: Linear) -> None:
        self.linear = linear

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.linear._linear(x, self.linear.weight, None), None


class _VLLMNativeMLAAttention(VLLMNativeMLAAttention, Module):
    """vLLM MLA attention that participates in the TorchTitan Module protocol.

    vLLM derives ``W_UK_T``/``W_UV`` from ``kv_b_proj.weight`` in
    ``process_weights_after_loading`` and registers them as parameters. They are
    not model weights (the trainer has no such keys), so they are moved to
    non-persistent buffers to keep the generator's state dict identical to the
    trainer's. For a bf16 ``Linear`` they are views of ``wkv_b.weight``, so weight
    sync's in-place copy updates them without re-deriving.
    """

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Derive the absorbed decode weights as non-persistent views of ``wkv_b``."""
        super().process_weights_after_loading(act_dtype)
        wkv_b_storage = self.kv_b_proj.weight.untyped_storage().data_ptr()
        for name in ("W_UK_T", "W_UV"):
            weight = self._parameters.pop(name)
            assert weight.untyped_storage().data_ptr() == wkv_b_storage, (
                f"vLLM's {name} must be a view of wkv_b.weight so weight sync "
                "updates it, but it has its own storage; decode would keep using "
                "the initial weights."
            )
            self.register_buffer(name, weight.data, persistent=False)


class VLLMMLAInnerAttention(InnerAttention):
    """Adapter from compact TorchTitan MLA inputs to vLLM's native MLA layer."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        num_heads: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        scale: float

        @property
        def is_full_attention(self) -> bool:
            return True

    def __init__(self, config: Config, *, kv_b_proj: Linear) -> None:
        super().__init__()

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        tp_degree = vllm_config.parallel_config.tensor_parallel_size
        if config.num_heads % tp_degree != 0:
            raise ValueError(
                f"num_heads ({config.num_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )
        self.num_heads = config.num_heads // tp_degree
        self.v_head_dim = config.v_head_dim

        self.vllm_mla = _VLLMNativeMLAAttention(
            num_heads=self.num_heads,
            scale=config.scale,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=None,
            kv_lora_rank=config.kv_lora_rank,
            kv_b_proj=_VLLMKVBProj(kv_b_proj),
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=(
                f"model.layers.{next(VLLMInnerAttention._layer_counter)}"
                ".attention.inner_attention"
            ),
        )
        _exclude_vllm_attn_scales_from_state_dict(self.vllm_mla)
        # Non-persistent buffers still need device placement.
        self.vllm_mla._sharding_config = ShardingConfig(
            state_shardings={
                name: dense_param_placement(tp=spmd.R)
                for name in VLLMInnerAttention.vllm_attn_scale_buffer_names
            },
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        *,
        attention_metadata: None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run vLLM MLA attention on rank-local tensors.

        Args:
            q_THK: ``(num_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim)``
            kv_c_normed_TL: ``(num_tokens, kv_lora_rank)``
            k_shared_TR: ``(num_tokens, qk_rope_head_dim)``, shared by all heads

        Returns:
            ``(num_tokens, num_heads, v_head_dim)``.
        """
        if attention_metadata is not None:
            raise ValueError(
                "VLLMMLAInnerAttention does not support attention_metadata; vLLM "
                "manages causal masking and the KV-cache internally."
            )
        num_tokens = q_THK.shape[0]
        return self.vllm_mla(
            q_THK,
            kv_c_normed_TL,
            k_shared_TR.unsqueeze(1),
            output_shape=(num_tokens, self.num_heads * self.v_head_dim),
        ).view(num_tokens, -1, self.v_head_dim)


class VLLMMLAAttention(MLAAttention):
    """MLA adapter whose vLLM paged cache stores ``[kv_c_normed, k_pe]``."""

    inner_attention: VLLMMLAInnerAttention

    @dataclass(kw_only=True, slots=True)
    class Config(MLAAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        InnerAttention.__init__(self)
        # TODO: support low-precision wkv_b with a latent-space MLA kernel that
        # consumes the quantized operands directly.
        if (
            config.wkv_b._owner is not Linear
            or config.wkv_b.num_linears != 1
            or config.wkv_b.bias
        ):
            raise ValueError(
                "vLLM's compressed MLA path needs wkv_b to be a bias-free Linear, "
                f"but wkv_b is {config.wkv_b._owner.__qualname__} "
                f"(num_linears={config.wkv_b.num_linears}, bias={config.wkv_b.bias}). "
                "MXFP8/NVFP4 quantization and LoRA on wkv_b are not supported "
                "on this path yet; exclude wkv_b from the quantization "
                "converter's FQNs."
            )
        self.wkv_b = config.wkv_b.build()
        self.inner_attention = config.inner_attention.build(kv_b_proj=self.wkv_b)

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.inner_attention(q_THK, kv_c_normed_TL, k_shared_TR, **kwargs)
