# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass, field

import torch
import torch_remat as remat
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from .state_dict_adapter import Llama3StateDictAdapter


class Llama3TransformerBlock(TransformerBlock):
    """
    Llama3 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Llama3TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        del padding_mask
        attn_out = self.attention(self.attention_norm(x), attention_masks, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h))
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h, ffn_out)


class Llama3Model(Decoder):
    state_dict_adapter_cls = Llama3StateDictAdapter

    """
    Llama3Model Module

    Args:
        config (Llama3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 128256
        local_compile_regions: list[str] = field(
            default_factory=lambda: ["loss", "swiglu"]
        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                head_dim = (
                    attention.head_dim
                    if attention.head_dim is not None
                    else attention.dim // attention.n_heads
                )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * active_nparams + attention_op_flops

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import set_llama3_sharding_config

            set_llama3_sharding_config(
                self, enable_sp=parallelism.enable_sequence_parallel
            )

    def __init__(self, config: Config):
        super().__init__(config)
