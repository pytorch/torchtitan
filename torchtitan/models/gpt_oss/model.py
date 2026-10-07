# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import torch
import torch_remat as remat
from torch import nn

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttentionMetadata,
    QKVLinear,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.cp_attention import UlyssesCPInnerAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

from .state_dict_adapter import GptOssStateDictAdapter


def apply_attention_sink_rescale(
    out: torch.Tensor, lse: torch.Tensor, sinks: torch.Tensor
) -> torch.Tensor:
    """Rescale attention output by the learned per-head sink term."""
    sinks = sinks.view(*([1] * (lse.ndim - 1)), -1)
    sink_scale = torch.sigmoid(lse - sinks).unsqueeze(-1)
    return out * sink_scale.to(out.dtype)


class Attention(BaseAttention):
    """
    Multi-head attention (MLA) module with sink attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int = 64
        n_kv_heads: int = 8
        head_dim: int = 64
        dim: int
        qkv_linear: QKVLinear.Config
        wo: Linear.Config  # output projection
        inner_attention: Module.Config = dataclasses.field(
            default_factory=VarlenInnerAttention.Config
        )
        sliding_window_size: int | None = None
        """Per-layer causal sliding-window size"""
        rope: RoPE.Config

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        # Standard attention softmax scale (1/sqrt(head_dim))
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        if config.rope.scaling == "yarn" and config.rope.rope_factor > 1.0:
            mscale = 0.1 * math.log(config.rope.rope_factor) + 1.0
            # Merge YaRN attention mscale into softmax_scale, with
            # m**2 being equivalent to scaling q / k each by mscale.
            self.softmax_scale *= mscale * mscale

        self.qkv_linear = config.qkv_linear.build()
        self.wo = config.wo.build()
        self.sinks = nn.Parameter(torch.empty(config.n_heads))
        self.inner_attention = config.inner_attention.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x: Input tensor with shape ``[T, D]``.
            attention_metadata: ``FlexAttentionMetadata`` or
                ``VarlenAttentionMetadata`` (varlen).
            positions: Optional position indices (unused, for API compatibility).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        q, k, v = self.qkv_linear(x)

        q, k = self.rope(q, k, positions)

        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q,
            k,
            v,
            attention_metadata=attention_metadata,
            scale=self.softmax_scale,
            enable_gqa=self.enable_gqa,
            out_transform=self._apply_sinks,
        )

        # The reshape below copies the inner_attention output with bare ops.
        remat.recompute_needs_tensor(output)
        # Reshape and project output
        output = output.reshape(output.shape[0], -1).contiguous()
        return self.wo(output)

    def _apply_sinks(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        """out_transform hook: rescale attention output by this layer's sinks."""
        return apply_attention_sink_rescale(out, lse, self.sinks)


class GptOssTransformerBlock(TransformerBlock):
    """
    GptOss Transformer block with sliding window attention support.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        assert isinstance(config.attention, Attention.Config)
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        assert config.moe is not None
        self.moe = config.moe.build()
        self.moe_enabled = True  # for composability with load balancing

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (num_tokens, dim).
            attention_metadata: Flex metadata selected for this layer's inner
                attention, or the shared ``VarlenAttentionMetadata``.
            positions: Optional position indices.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """

        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        x = x + attn_out
        moe_out = self.moe(
            self.ffn_norm(x),
            padding_mask_T=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x, moe_out)


class GptOssModel(Decoder):
    state_dict_adapter_cls = GptOssStateDictAdapter

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2880
        vocab_size: int = 201088
        local_compile_regions: list[str] = field(
            default_factory=lambda: [
                "loss",
                "fused_binary_activation",
                "cos_sin_rope",
                "fp32_to_bf16_split",
            ]
        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=attention.head_dim,
                    v_head_dim=attention.head_dim,
                    seq_len=seq_len,
                    sliding_window_size=attention.sliding_window_size,
                )
            return nparams, 6 * active_nparams + attention_op_flops

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import set_gpt_oss_sharding_config

            set_gpt_oss_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

    def __init__(self, config: Config):
        super().__init__(config)

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile_regions: list[str],
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
        skip_dp: bool = False,
    ) -> GptOssModel:
        if parallelism_context.cp_enabled and any(
            isinstance(backend, UlyssesCPInnerAttention.Config)
            for backend in self.config.base_attention_backends
        ):
            raise NotImplementedError(
                "GPT-OSS does not support Ulysses CP because its per-head "
                "sinks are not sharded over CP."
            )

        return super().parallelize(
            parallelism_context=parallelism_context,
            training=training,
            parallelism=parallelism,
            local_compile_regions=local_compile_regions,
            ac_config=ac_config,
            dump_folder=dump_folder,
            skip_dp=skip_dp,
        )
