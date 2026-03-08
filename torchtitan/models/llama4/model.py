# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_fixed_block_mask_mod,
    GQAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


def compute_moe_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 256,
    ffn_dim_multiplier: float | None = None,
    auto_scale_hidden_dim: bool = True,
    top_k: int = 1,
    num_shared_experts: int = 1,
) -> int:
    """Compute the MoE expert hidden dimension for Llama4-style models.

    This replicates the original Llama4 computation order:
    1. int(2 * 4 * dim / 3)
    2. Apply ffn_dim_multiplier
    3. Auto-scale (divide by top_k + num_shared_experts)
    4. Round up to multiple_of

    Note: This differs from compute_ffn_hidden_dim which applies multiple_of
    rounding BEFORE any auto-scaling.
    """
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)

    if auto_scale_hidden_dim:
        hidden_dim_denom = top_k + num_shared_experts
        hidden_dim = int(hidden_dim / hidden_dim_denom)

    hidden_dim += -hidden_dim % multiple_of
    return hidden_dim


class Llama4TransformerBlock(TransformerBlock):
    """
    Llama4 TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        dim (int): Model dimension.
        n_layers (int): Total number of layers.
        config (Llama4TransformerBlock.Config): Block configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True
        every_n_layers_nope: int | None = None
        interleave_moe_layer_step: int = 2
        fixed_attn_block_size: int = 8192

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()

        # iRoPE: determine per-layer use_rope and fixed_attn_block_size
        attn_use_rope = True
        if config.every_n_layers_nope is not None:
            if config.every_n_layers_nope <= 1:
                raise ValueError("every_n_layers_nope must be greater than 1")
            if layer_id % config.every_n_layers_nope == 0:
                attn_use_rope = False

        # Create per-layer attention config with potentially overridden use_rope
        if not attn_use_rope:
            assert isinstance(config.attention, GQAttention.Config)
            layer_attention = dataclasses.replace(config.attention, use_rope=False)
        else:
            layer_attention = config.attention

        self.attention = layer_attention.build(dim=dim)

        # use MoE layer for every interleave_moe_layer_step FFN layers
        self.moe_enabled = (layer_id + 1) % config.interleave_moe_layer_step == 0
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build(dim=dim)
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build(dim=dim)

        self.attention_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            out = h + self.moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, **kwargs):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(
                init_std=self.weight_init_std, buffer_device=buffer_device
            )
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Llama4Model(Decoder):
    """
    Llama4Model Module

    Args:
        config (Llama4Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        n_layers: int = 32
        vocab_size: int = 202048
        layer: TransformerBlock.Config

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            assert self.layer.moe is not None
            if self.layer.moe.use_grouped_mm and not has_cuda_capability(9, 0):
                logger.warning(
                    "Failed to use grouped mm, which is only supported on SM90 or later",
                )
                self.layer.moe.use_grouped_mm = False

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "Context Parallel is not supported for Llama4 "
                    "(Llama4 requires FlexAttention, which is not supported with CP)."
                )

            self.layer.moe._debug_force_load_balance = debug.moe_force_load_balance

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * (self.dim // self.layer.attention.n_heads),
                seq_len,
            )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.config.layer.attention.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
                B = input_batch.shape[0]
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.config.layer.attention.attn_mask_type}"
                )

        assert isinstance(self.config.layer, Llama4TransformerBlock.Config)
        rope_mask_mod = and_masks(
            *mask_mods,
            get_fixed_block_mask_mod(self.config.layer.fixed_attn_block_size),
        )
        nope_mask_mod = and_masks(*mask_mods)

        seqlen = input_batch.shape[1]
        return {
            "rope": create_attention_mask(rope_mask_mod, B, None, seqlen, seqlen),
            "nope": create_attention_mask(nope_mask_mod, B, None, seqlen, seqlen),
        }
