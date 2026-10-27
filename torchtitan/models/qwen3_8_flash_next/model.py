# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.8-Flash-Next's hybrid multimodal decoder."""

from dataclasses import dataclass, field
from typing import Any, cast

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_local_context,
)
from torchtitan.models.common.attention import AttentionMetadata, AttentionMetadataMap
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.decoder_sharding import token_id_placement
from torchtitan.models.common.flops import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5.model import Qwen35Model

from .attention import QSAAttention
from .embedding import PLELayer
from .residual import GatedResidual, inject_residual
from .state_dict_adapter import Qwen38FlashNextStateDictAdapter

# T = packed tokens, D = model dimension, S = residual streams, C = S * D.


class Qwen38FlashNextBlock(TransformerBlock):
    """Lexical injection followed by gated attention and MoE residual writes."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attn_hyper_connection: GatedResidual.Config
        mlp_hyper_connection: GatedResidual.Config
        ple: PLELayer.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attn_hyper_connection = config.attn_hyper_connection.build()
        self.mlp_hyper_connection = config.mlp_hyper_connection.build()
        self.ple = config.ple.build() if config.ple is not None else None

    def forward(
        self,
        x_TC: torch.Tensor,
        attention_metadata: AttentionMetadata | None,
        positions: torch.Tensor | None,
        *,
        ple_input_ids_T: torch.Tensor | None,
        document_positions_T: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ple is not None:
            if ple_input_ids_T is None:
                raise ValueError(
                    "PLE requires original input token IDs on its pipeline stage"
                )
            x_TC = x_TC + self.ple(
                x_TC,
                input_ids_T=ple_input_ids_T,
                document_positions_T=document_positions_T,
                padding_mask_T=padding_mask_T,
            )
        mixed_TD, residual_TC, write_TS = self.attn_hyper_connection(x_TC)
        mixed_TD = self.attention_norm(mixed_TD)
        if isinstance(self.attention, QSAAttention):
            update_TD = self.attention(
                mixed_TD,
                attention_metadata,
                positions,
                document_positions_T=document_positions_T,
                padding_mask_T=padding_mask_T,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            update_TD = self.attention(mixed_TD, attention_metadata)
        remat.recompute_needs_tensor(update_TD, write_TS)
        x_TC = inject_residual(update_TD, residual_TC, write_TS)
        mixed_TD, residual_TC, write_TS = self.mlp_hyper_connection(x_TC)
        mixed_TD = self.ffn_norm(mixed_TD)
        if isinstance(self.feed_forward, MoE):
            update_TD = self.feed_forward(
                mixed_TD,
                padding_mask=padding_mask_T,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            update_TD = self.feed_forward(mixed_TD)
        remat.recompute_needs_tensor(update_TD, write_TS)
        return inject_residual(update_TD, residual_TC, write_TS)


class Qwen38FlashNextModel(Qwen35Model):
    """Four-stream Flash-Next decoder sharing Qwen3.5's vision and training lifecycle."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        num_streams: int
        hyper_connection_mixer: GatedResidual.Config
        local_compile_regions: list[str] = field(default_factory=list)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(
                    getattr(model, "vision_encoder", None),
                ),
            )
            attention_flops = 0
            for layer in self.layers:
                attention = layer.attention
                if isinstance(attention, QSAAttention.Config):
                    indexer = attention.indexer
                    attended = min(
                        seq_len, indexer.token_budget + indexer.compress_ratio - 1
                    )
                    attention_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.head_dim,
                        v_head_dim=attention.head_dim,
                        seq_len=attended,
                    )
                    attention_flops += (
                        6
                        * indexer.num_heads
                        * indexer.head_dim
                        * (seq_len // indexer.compress_ratio)
                    )
                elif isinstance(attention, GDN.Config):
                    attention_flops += delta_rule_flops_per_token(
                        num_heads=attention.n_heads,
                        key_head_dim=attention.key_head_dim,
                        v_head_dim=attention.value_head_dim,
                    )
            return nparams, 6 * active + attention_flops

    state_dict_adapter_cls = Qwen38FlashNextStateDictAdapter
    pipeline_last_stage_module_fqns = ("hyper_connection_mixer",)

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_streams = config.num_streams
        self.hyper_connection_mixer = config.hyper_connection_mixer.build()

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        # These full token histories must reach PLE even when it belongs to a
        # later PP stage. Rotary coordinates alone cannot recover document IDs.
        lexical_inputs = {
            "ple_input_ids": input_dict["input"],
            "document_positions": input_dict["positions"],
        }
        inputs_T, labels_T, model_kwargs, loss_kwargs = super().preprocess_inputs(
            input_dict,
            parallelism_context=parallelism_context,
            parallelism=parallelism,
            **kwargs,
        )
        lexical_inputs = annotate_input_spmd_types(
            parallelism_context,
            lexical_inputs,
            {name: token_id_placement() for name in lexical_inputs},
        )
        model_kwargs.update(lexical_inputs)
        return inputs_T, labels_T, model_kwargs, loss_kwargs

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        document_positions: torch.Tensor | None = None,
        ple_input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        padding_mask: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        image_vision_bank_indices_T: torch.Tensor | None = None,
        video_vision_bank_indices_T: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if document_positions is None:
            if positions is None or positions.ndim != 1:
                raise ValueError(
                    "Flash-Next requires document positions in addition to multimodal rotary positions"
                )
            document_positions = positions
        with spmd_local_context("dp"):
            if self.tok_embeddings is not None:
                if ple_input_ids is None:
                    ple_input_ids = tokens
                x_TD = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw=grid_thw,
                    grid_thw_videos=grid_thw_videos,
                    special_tokens=special_tokens,
                    image_vision_bank_indices_T=image_vision_bank_indices_T,
                    video_vision_bank_indices_T=video_vision_bank_indices_T,
                )
                x_TC = x_TD.repeat(1, self.num_streams)
            else:
                x_TC = tokens
        with spmd.no_typecheck():
            denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for module in self.layers.values():
            layer = cast(Qwen38FlashNextBlock, module)
            x_TC = layer(
                x_TC,
                attention_metadata.get(type(layer.attention.inner_attention))
                if attention_metadata
                else None,
                positions,
                ple_input_ids_T=ple_input_ids,
                document_positions_T=document_positions,
                padding_mask_T=padding_mask,
                aux_loss_denominator=denominator,
            )
        x_TD = (
            self.hyper_connection_mixer(x_TC)
            if self.hyper_connection_mixer is not None
            else x_TC
        )
        x_TD = self.norm(x_TD) if self.norm is not None else x_TD
        if self._skip_lm_head:
            return x_TD
        return self.lm_head(x_TD) if self.lm_head is not None else x_TD
