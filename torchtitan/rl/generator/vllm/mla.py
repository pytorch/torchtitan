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
from typing import ClassVar

import spmd_types as spmd

import torch
from vllm.model_executor.layers.attention import MLAAttention as VLLMNativeMLAAttention

from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import InnerAttention, MLAAttention
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.rl.generator.vllm.attention import VLLMInnerAttention


class _VLLMKVBProj:
    """Expose a TorchTitan ``Linear`` through the ``kv_b_proj`` interface vLLM MLA reads.

    vLLM folds ``weight`` into its decode matmuls, so ``weight`` is the matrix
    the module computes. A plain linear shares its own weight; other linears
    are probed by applying their local projection to the identity, which gives
    ``W + s * B @ A`` for LoRA and the rounded weight for MXFP8/NVFP4. Prefill
    calls the local projection itself.
    """

    quant_method = None

    def __init__(self, linear: Linear) -> None:
        self.linear = linear
        self.weight: torch.Tensor | None = None

    @property
    def params_dtype(self) -> torch.dtype:
        assert self.weight is not None
        return self.weight.dtype

    def _local_linear(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear._linear(x, self.linear.weight, self.linear.bias)

    def refresh_weight(self, dtype: torch.dtype) -> None:
        """Point ``weight`` at a plain linear's weight, or re-probe it in place."""
        if type(self.linear) is Linear:
            self.weight = self.linear.weight
            return
        eye = torch.eye(
            self.linear.in_features, dtype=dtype, device=self.linear.weight.device
        )
        with torch.no_grad():
            weight = self._local_linear(eye).T
        if self.weight is None:
            self.weight = weight.contiguous()
        else:
            self.weight.copy_(weight)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self._local_linear(x), None


class _VLLMNativeMLAAttention(VLLMNativeMLAAttention, Module):
    """vLLM MLA attention that participates in the TorchTitan Module protocol."""

    absorbed_weight_names: ClassVar[tuple[str, ...]] = ("W_UK_T", "W_UV")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        for name in VLLMInnerAttention.vllm_attn_scale_buffer_names:
            self.register_buffer(name, self._buffers.pop(name), persistent=False)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Derive the absorbed decode weights as non-persistent buffers."""
        self._act_dtype = act_dtype
        self.kv_b_proj.refresh_weight(act_dtype)
        super().process_weights_after_loading(act_dtype)
        for name in self.absorbed_weight_names:
            weight = self._parameters.pop(name, None)
            if weight is not None:
                self.register_buffer(name, weight.data, persistent=False)

    def refresh_absorbed_weights(self) -> None:
        """Re-derive the absorbed weights after ``wkv_b`` or its adapters changed."""
        with torch.inference_mode():
            self.process_weights_after_loading(self._act_dtype)


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

        layer_id = next(VLLMInnerAttention._layer_counter)
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
            prefix=f"model.layers.{layer_id}.attention.inner_attention",
        )
        replicated_dense = dense_param_placement(tp=spmd.R)
        self.vllm_mla._sharding_config = ShardingConfig(
            state_shardings={
                name: replicated_dense
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
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run vLLM MLA attention on local (non-DTensor) tensors.

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
        if scale is not None and scale != self.vllm_mla.scale:
            raise ValueError(
                f"Attention scale {scale} differs from the vLLM MLA scale "
                f"{self.vllm_mla.scale} fixed at construction."
            )
        num_tokens = q_THK.shape[0]
        out_TD = self.vllm_mla(
            q_THK,
            kv_c_normed_TL,
            k_shared_TR.unsqueeze(1),
            output_shape=(num_tokens, self.num_heads * self.v_head_dim),
        )
        return out_TD.view(num_tokens, -1, self.v_head_dim)


class VLLMMLAAttention(MLAAttention):
    """MLA adapter whose vLLM paged cache stores ``[kv_c_normed, k_pe]``."""

    inner_attention: VLLMMLAInnerAttention

    @dataclass(kw_only=True, slots=True)
    class Config(MLAAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        InnerAttention.__init__(self)
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
