# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch_remat as remat
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.deepseek_v3.mtp import MTPDecoder
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from .state_dict_adapter import DeepSeekV3StateDictAdapter


class DeepSeekV3TransformerBlock(TransformerBlock):
    """
    DeepSeek V3 Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        x = x + attn_out
        if self.moe_enabled:
            ffn_out = self.moe(
                self.ffn_norm(x),
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_out = self.feed_forward(self.ffn_norm(x))
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x, ffn_out)


def get_deepseek_v3_nparams_and_flops(
    model_config: MTPDecoder.Config,
    model: nn.Module,
    seq_len: int,
    *,
    modules_excluded_from_active_params: Iterable[nn.Module | None] = (),
) -> tuple[int, int]:
    """Estimate DeepSeek-style decoder FLOPs from the final model config."""
    nparams, active_nparams = get_nparams_and_active_nparams(
        model,
        modules_excluded_from_active_params=modules_excluded_from_active_params,
    )

    attention_op_flops = 0
    for layers in (model_config.layers, model_config.mtp_layers):
        for layer in layers:
            attention = layer.attention
            attention_op_flops += quadratic_attention_flops_per_token(
                num_heads=attention.n_heads,
                qk_head_dim=(attention.qk_nope_head_dim + attention.qk_rope_head_dim),
                v_head_dim=attention.v_head_dim,
                seq_len=seq_len,
            )

    # The base parameter term counts one lm_head use. MTP applies that same
    # output projection once more for every prediction depth.
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Module):
        active_nparams += len(model_config.mtp_layers) * sum(
            param.numel() for param in lm_head.parameters()
        )

    return nparams, 6 * active_nparams + attention_op_flops


class DeepSeekV3Model(MTPDecoder):
    state_dict_adapter_cls = DeepSeekV3StateDictAdapter

    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MTPDecoder.Config):
        dim: int = 2048
        vocab_size: int = 102400
        local_compile_regions: list[str] = field(
            default_factory=lambda: [
                "loss",
                "fused_binary_activation",
                "fp32_to_bf16_split",
            ]
        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_deepseek_v3_nparams_and_flops(self, model, seq_len)

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import set_deepseek_v3_sharding_config

            set_deepseek_v3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

    def __init__(self, config: Config):
        super().__init__(config)

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
