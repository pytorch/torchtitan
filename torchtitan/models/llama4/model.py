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

from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_fixed_block_mask_mod,
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
        fixed_attn_block_size: int = 8192

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

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


class Llama4Model(Decoder):
    """
    Llama4Model Module

    Args:
        config (Llama4Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        vocab_size: int = 202048

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

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    if (
                        layer_cfg.moe.experts.use_grouped_mm
                        and not has_cuda_capability(9, 0)
                    ):
                        logger.warning(
                            "Failed to use grouped mm, which is only supported on SM90 or later",
                        )
                        layer_cfg.moe.experts.use_grouped_mm = False
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )
                    comm_backend = getattr(
                        layer_cfg.moe.experts.token_dispatcher,
                        "comm_backend",
                        "standard",
                    )
                    if (
                        comm_backend in ("deepep", "hybridep")
                        and parallelism.expert_parallel_degree == 1
                    ):
                        raise ValueError(
                            f"{comm_backend.upper()} requires expert parallelism "
                            "(expert_parallel_degree > 1)."
                        )

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "Context Parallel is not supported for Llama4 "
                    "(Llama4 requires FlexAttention, which is not supported with CP)."
                )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )

            from torchtitan.models.llama4.sharding import set_llama4_sharding_config

            set_llama4_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
                tp_enabled=parallelism.tensor_parallel_degree > 1,
                ep_enabled=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:

            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len,
            )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        attn_config = self.config.layers[0].attention
        match attn_config.mask_type:
            case "causal":
                B = 1
            case "block_causal":
                mask_mods.append(get_document_mask_mod(positions))
                B = positions.shape[0]
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {attn_config.mask_type}"
                )

        layer0 = self.config.layers[0]
        assert isinstance(layer0, Llama4TransformerBlock.Config)
        rope_mask_mod = and_masks(
            *mask_mods,
            get_fixed_block_mask_mod(layer0.fixed_attn_block_size),
        )
        nope_mask_mod = and_masks(*mask_mods)

        seqlen = positions.shape[1]
        return {
            "rope": create_attention_mask(rope_mask_mod, B, None, seqlen, seqlen),
            "nope": create_attention_mask(nope_mask_mod, B, None, seqlen, seqlen),
        }
