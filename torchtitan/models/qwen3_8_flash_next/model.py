# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.8-Flash-Next's hybrid multimodal decoder."""

from dataclasses import dataclass
from typing import Any, cast

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    RowShardedEmbeddingMetadata,
)
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    set_current_spmd_mesh,
    spmd_local_context,
)
from torchtitan.models.common.attention import AttentionMetadata, AttentionMetadataMap
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.decoder_sharding import token_id_placement
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.qwen3_5.model import Qwen35Model

from .attention import QSAAttention, QSAIndexMetadata
from .embedding import NGramTable, PLELayer
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
        ple_input_ids_T: torch.Tensor,
        document_positions_T: torch.Tensor,
        ple_routing: RowShardedEmbeddingMetadata | None = None,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ple is not None:
            x_TC = x_TC + self.ple(
                x_TC,
                input_ids_T=ple_input_ids_T,
                document_positions_T=document_positions_T,
                routing=ple_routing,
                padding_mask_T=padding_mask_T,
            )
        mixed_TD, residual_TC, write_TS = self.attn_hyper_connection(x_TC)
        mixed_TD = self.attention_norm(mixed_TD)
        if isinstance(self.attention, QSAAttention):
            update_TD = self.attention(
                mixed_TD,
                cast(QSAIndexMetadata | None, attention_metadata),
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
                padding_mask_T=padding_mask_T,
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
        norm: Identity.Config  # pyrefly: ignore [bad-override]

    state_dict_adapter_cls = Qwen38FlashNextStateDictAdapter

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_streams = config.num_streams
        self.hyper_connection_mixer = config.hyper_connection_mixer.build()

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        precision: PrecisionConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torch.distributed.fsdp import fully_shard

        for table in self.modules():
            if isinstance(table, NGramTable) and table.row_mesh is not None:
                if precision.reduce_dtype != precision.dtype:
                    raise ValueError(
                        "Row-owned PLE requires reduce_dtype to match the master parameter dtype"
                    )
                # An ignored-only FSDP child keeps ancestor FSDP units from
                # gathering this parameter, while retaining their lifecycle.
                fully_shard(
                    table,
                    mesh=table.row_mesh,
                    ignored_params={table.get_parameter("weight")},
                )
        super()._apply_fsdp(
            parallelism_context=parallelism_context,
            precision=precision,
            parallelism=parallelism,
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        # PLE needs original token history; multimodal rotary coordinates cannot
        # recover token IDs or one-dimensional document positions.
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
        for module in self.layers.values():
            layer = cast(Qwen38FlashNextBlock, module)
            if layer.ple is None:
                continue
            embedding = layer.ple.ple_embedding
            table = embedding.ngram_embedding
            if table.row_mesh is not None:
                with spmd.no_typecheck():
                    hashes_TN = embedding.hash_ids(
                        lexical_inputs["ple_input_ids"],
                        document_positions_T=lexical_inputs["document_positions"],
                    )
                    routing = prepare_row_sharded_embedding(
                        hashes_TN,
                        num_embeddings=table.num_embeddings,
                        row_mesh=table.row_mesh,
                    )
                with set_current_spmd_mesh(parallelism_context.spmd_dense_mesh()):
                    spmd.assert_type(
                        routing.input_ids,
                        spmd.SpmdType(
                            {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                            spmd.PartitionSpec(("dp", "cp"), None),
                        ),
                    )
                    for tensor in (
                        routing.local_row_ids,
                        routing.restore_indices,
                        routing.inverse_indices,
                    ):
                        spmd.assert_type(
                            tensor, {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R}
                        )
                model_kwargs["ple_routing"] = routing
        for metadata in model_kwargs.get("attention_metadata", {}).values():
            if isinstance(metadata, QSAIndexMetadata):
                metadata.annotate_spmd_types()
        model_kwargs.update(lexical_inputs)
        return inputs_T, labels_T, model_kwargs, loss_kwargs

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        document_positions: torch.Tensor,
        ple_input_ids: torch.Tensor,
        ple_routing: RowShardedEmbeddingMetadata | None = None,
        positions: torch.Tensor | None = None,
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
        with spmd_local_context("dp"):
            if self.tok_embeddings is not None:
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
        if spmd.is_type_checking():
            spmd.assert_type(
                x_TC,
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                spmd.PartitionSpec(("dp", "cp"), None),
            )
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
                ple_routing=ple_routing,
                padding_mask_T=padding_mask,
                aux_loss_denominator=denominator,
            )
        x_TD = self.norm(self.hyper_connection_mixer(x_TC))
        if self._skip_lm_head:
            return x_TD
        return self.lm_head(x_TD) if self.lm_head is not None else x_TD
