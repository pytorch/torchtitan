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
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import local_compile, LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    set_current_spmd_mesh,
    spmd_dense_sp_enabled,
    spmd_local_context,
)
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import AttentionMetadata, AttentionMetadataMap
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.decoder_sharding import (
    residual_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.multimodal import MultimodalModel, replicate_to_residual
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.qwen3_5.model import Qwen35Model, Qwen35MultimodalModel
from torchtitan.models.qwen3_5.vision_encoder import Qwen35VisionEncoder

from .attention import QSAAttention, QSAIndexMetadata
from .embedding import NGramTable, PLELayer
from .norm import GroupedOffsetRMSNorm
from .row_sharded_embedding import (
    prepare_row_sharded_embedding,
    RowShardedEmbeddingMetadata,
)
from .state_dict_adapter import Qwen38FlashNextStateDictAdapter

# T = packed tokens, D = model dimension, S = residual streams, C = S * D.
# E = experts, K = selected experts, R = gated residual mixing rank.


class GatedResidual(Module):
    """Read residual streams and optionally produce per-stream write gates."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_streams: int
        hc_norm: GroupedOffsetRMSNorm.Config
        input_mix_weight_down: Linear.Config
        input_mix_weight_up: Linear.Config
        block_inject_weight: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.num_streams = config.num_streams
        self.hc_norm = config.hc_norm.build()
        self.input_mix_weight_down = config.input_mix_weight_down.build()
        self.input_mix_weight_up = config.input_mix_weight_up.build()
        self.block_inject_weight = (
            config.block_inject_weight.build()
            if config.block_inject_weight is not None
            else None
        )

    @staticmethod
    @local_compile("gated_residual", batch_invariant=True)
    def read_streams(
        read_logits_TC: torch.Tensor, x_norm_TC: torch.Tensor, num_streams: int
    ) -> torch.Tensor:
        """Average the streams weighted by their sigmoid read gates."""
        read_TSD = read_logits_TC.sigmoid().unflatten(-1, (num_streams, -1))
        return (read_TSD * x_norm_TC.unflatten(-1, (num_streams, -1))).mean(-2)

    def forward(
        self, x_TC: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm_TC = self.hc_norm(x_TC)
        remat.recompute_needs_tensor(x_norm_TC)
        # Bare ops read the projections below, so selective remat must keep
        # their outputs.
        down_TR = self.input_mix_weight_down(x_norm_TC)
        remat.recompute_needs_tensor(down_TR)
        mix_TR = F.silu(down_TR / self.num_streams)
        remat.recompute_needs_tensor(mix_TR)
        up_TC = self.input_mix_weight_up(mix_TR)
        remat.recompute_needs_tensor(up_TC)
        mixed_TD = self.read_streams(up_TC, x_norm_TC, self.num_streams)
        if self.block_inject_weight is None:
            return mixed_TD
        inject_TS = self.block_inject_weight(x_norm_TC)
        remat.recompute_needs_tensor(inject_TS)
        write_TS = 2 * (inject_TS / self.num_streams).sigmoid()
        return mixed_TD, x_TC, write_TS


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

    @staticmethod
    @local_compile("gated_residual", batch_invariant=True)
    def add_residuals(
        update_TD: torch.Tensor, residual_TC: torch.Tensor, write_TS: torch.Tensor
    ) -> torch.Tensor:
        """Write the update to every stream, scaled by its write gate."""
        update_TSD = update_TD.unsqueeze(-2) * write_TS.unsqueeze(-1)
        return residual_TC + update_TSD.flatten(-2)

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
            # PLE runs TP-replicated: gather the residual stream in and restore
            # the residual layout on the update.
            update_TC = self.ple(
                maybe_gather_tp_input(self.ple, x_TC),
                input_ids_T=ple_input_ids_T,
                document_positions_T=document_positions_T,
                routing=ple_routing,
                padding_mask_T=padding_mask_T,
            )
            x_TC = x_TC + replicate_to_residual(update_TC)
        mixed_TD, residual_TC, write_TS = self.attn_hyper_connection(x_TC)
        mixed_TD = self.attention_norm(mixed_TD)
        if isinstance(self.attention, QSAAttention):
            update_TD = self.attention(
                mixed_TD,
                cast(QSAIndexMetadata, attention_metadata),
                positions,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            update_TD = self.attention(mixed_TD, attention_metadata)
        remat.recompute_needs_tensor(update_TD, write_TS)
        x_TC = self.add_residuals(update_TD, residual_TC, write_TS)
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
        return self.add_residuals(update_TD, residual_TC, write_TS)


class Qwen38FlashNextModel(Qwen35Model):
    """Four-stream Flash-Next text decoder sharing Qwen3.5's training lifecycle.

    ``Qwen38FlashNextMultimodalModel`` adds the vision encoder.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        layers: list[Qwen38FlashNextBlock.Config]  # pyrefly: ignore [bad-override]
        num_streams: int
        hyper_connection_mixer: GatedResidual.Config
        # The reference has no final norm (the checkpoint has no norm.weight):
        # hyper_connection_mixer normalizes the streams with its own hc_norm and
        # feeds the LM head. Decoder.Config requires a norm slot, so it holds
        # Identity.
        norm: Identity.Config  # pyrefly: ignore [bad-override]
        # Qwen3.5's regions plus the gated residual stream read and write.
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "gated_rmsnorm",
                    "gdn_split_in_proj",
                    "offset_rmsnorm",
                    "partial_rope",
                    "shared_expert_gate",
                    "fp32_to_bf16_split",
                    "attn_output_gate",
                    "gdn_gate",
                    "gated_residual",
                    "indexer_scores",
                ]
            )
        )

        def __post_init__(self) -> None:
            super(Qwen38FlashNextModel.Config, self).__post_init__()
            if sum(layer.ple is not None for layer in self.layers) != 1:
                raise ValueError("Flash-Next requires exactly one PLE layer")

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

        # Config validation allows exactly one PLE layer, so at most one table
        # (none on a pipeline stage without it).
        table = next((m for m in self.modules() if isinstance(m, NGramTable)), None)
        if table is not None and table.row_mesh is not None:
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
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
        dict[str, Any],
    ]:
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
                        # Packing pads every microbatch to the configured token
                        # count, so T and the bound agree on all row owners.
                        max_ids_per_owner=embedding.max_ids_per_owner(
                            hashes_TN.shape[0], num_owners=table.row_mesh.size()
                        ),
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
                        routing.inverse_indices,
                    ):
                        spmd.assert_type(
                            tensor, {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R}
                        )
                model_kwargs["ple_routing"] = routing
            # Config validation allows exactly one PLE layer.
            break
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
        attention_metadata: AttentionMetadataMap | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.tok_embeddings is None:
            x_TC = tokens
        else:
            # Embed in the global context so the output records its DP shard.
            x_TC = self.tok_embeddings(tokens).repeat(1, self.num_streams)
        return self._forward_streams(
            x_TC,
            positions,
            attention_metadata,
            document_positions=document_positions,
            ple_input_ids=ple_input_ids,
            ple_routing=ple_routing,
            padding_mask=padding_mask,
            aux_loss_denominators=aux_loss_denominators,
        )

    def _forward_streams(
        self,
        x_TC: torch.Tensor,
        positions: torch.Tensor | None,
        attention_metadata: AttentionMetadataMap | None,
        *,
        document_positions: torch.Tensor,
        ple_input_ids: torch.Tensor,
        ple_routing: RowShardedEmbeddingMetadata | None,
        padding_mask: torch.Tensor | None,
        aux_loss_denominators: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the layers, the final mixer and the LM head on the residual streams."""
        if spmd.is_type_checking():
            spmd.assert_type(
                x_TC,
                residual_activation_placement(enable_sp=spmd_dense_sp_enabled()),
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
        # The vocab-sharded lm_head, or the chunked loss, needs TP-Replicate input.
        x_TD = maybe_gather_tp_input(
            self.norm, self.norm(self.hyper_connection_mixer(x_TC))
        )
        if self._skip_lm_head:
            return x_TD
        return self.lm_head(x_TD) if self.lm_head is not None else x_TD


class Qwen38FlashNextMultimodalModel(Qwen38FlashNextModel, Qwen35MultimodalModel):
    """Flash-Next decoder with Qwen3.5's vision encoder."""

    # Two slotted bases that both add fields cannot be combined, so this repeats
    # Qwen35MultimodalModel.Config's only field instead of inheriting it. The
    # config is therefore not a Qwen35MultimodalModel.Config; check
    # MultimodalModel.Config instead.
    @dataclass(kw_only=True, slots=True)
    class Config(  # pyrefly: ignore [bad-override]
        Qwen38FlashNextModel.Config, MultimodalModel.Config
    ):
        vision_encoder: Qwen35VisionEncoder.Config

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
        if self.tok_embeddings is None:
            x_TC = tokens
        else:
            with spmd_local_context("dp"):
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
            if spmd.is_type_checking():
                spmd.assert_type(
                    x_TC,
                    {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                    spmd.PartitionSpec(("dp", "cp"), None),
                )
            # The vision scatter leaves the embeddings TP-Replicate.
            x_TC = replicate_to_residual(x_TC)
        return self._forward_streams(
            x_TC,
            positions,
            attention_metadata,
            document_positions=document_positions,
            ple_input_ids=ple_input_ids,
            ple_routing=ple_routing,
            padding_mask=padding_mask,
            aux_loss_denominators=aux_loss_denominators,
        )
