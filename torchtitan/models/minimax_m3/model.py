# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes:
#   T = packed tokens, D = model dimension.

from __future__ import annotations

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
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.flops import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.multimodal import (
    add_zero_vision_dependency,
    build_dummy_vision_inputs,
    get_vision_positions,
    MultimodalModel,
    scatter_vision_embeds,
)
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding

from .attention import MiniMaxSparseAttention, MSAInnerAttention
from .state_dict_adapter import MiniMaxM3StateDictAdapter
from .vision_encoder import MiniMaxM3VisionEncoder

__all__ = ["MiniMaxM3Model", "MiniMaxM3TransformerBlock"]


class MiniMaxM3TransformerBlock(TransformerBlock):
    """Pre-norm block with GQA or MiniMax Sparse Attention and a dense FFN or MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.sparse_attention = isinstance(
            config.attention, MiniMaxSparseAttention.Config
        )
        self.attention = config.attention.build()
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: AttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_TD = self.attention_norm(x_TD)
        if self.sparse_attention:
            h_TD = self.attention(
                h_TD,
                attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            h_TD = self.attention(h_TD, attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(h_TD)
        x_TD = x_TD + h_TD

        h_TD = self.ffn_norm(x_TD)
        if isinstance(self.feed_forward, MoE):
            h_TD = self.feed_forward(
                h_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            h_TD = self.feed_forward(h_TD)
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x_TD, h_TD)


class MiniMaxM3Model(MultimodalModel):
    """MiniMax M3: a natively multimodal MoE decoder with MiniMax Sparse Attention.

    Images and videos are encoded by a CLIP-style ViT and scattered into their
    placeholder tokens; the decoder uses plain 1D positions.
    """

    state_dict_adapter_cls = MiniMaxM3StateDictAdapter
    multimodal_encoder_fqns = ("vision_encoder",)
    pipeline_first_stage_module_fqns = ("vision_encoder",)

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: MiniMaxM3VisionEncoder.Config | None = None
        local_compile_regions: list[str] = field(
            default_factory=lambda: [
                "loss",
                "fused_binary_activation",
                "offset_rmsnorm",
                "partial_rope",
                "fp32_to_bf16_split",
            ]
        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # The vision encoder cost scales with patches rather than text
            # sequence length, so this remains a decoder-only MFU estimate.
            m3_model = cast("MiniMaxM3Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(m3_model.vision_encoder,),
            )
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                assert attention.head_dim is not None
                inner_attention = attention.inner_attention
                sliding_window_size = None
                if isinstance(attention, MiniMaxSparseAttention.Config):
                    assert isinstance(inner_attention, MSAInnerAttention.Config)
                    if inner_attention.sparse:
                        sliding_window_size = (
                            inner_attention.topk_blocks
                            * inner_attention.index_block_size
                        )
                    indexer = attention.indexer
                    num_index_heads = indexer.wq.out_features // indexer.head_dim
                    attention_op_flops += (
                        6 * num_index_heads * indexer.head_dim * seq_len
                    )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=attention.head_dim,
                    v_head_dim=attention.head_dim,
                    seq_len=seq_len,
                    sliding_window_size=sliding_window_size,
                )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        if spmd_mesh_size("cp") > 1:
            raise NotImplementedError(
                "Context Parallel is not yet supported for MiniMax Sparse Attention."
            )
        super().__init__(config)
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Build attention metadata and annotate the multimodal batch."""
        del kwargs
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        positions = input_dict.get("positions")
        if positions is not None:
            input_dict["attention_metadata"] = self._get_attention_metadata(
                positions=positions,
                padding_mask=input_dict.get("padding_mask"),
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

        input_shardings = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(),
        }
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )

        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        input_dict["aux_loss_denominators"] = global_routing_token_counts
        return inputs, labels, input_dict, loss_kwargs

    def _scatter_modality(
        self,
        inputs_embeds_TD: torch.Tensor,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        placeholder_id: int | None,
    ) -> torch.Tensor:
        """Encode one modality and scatter it into its placeholder tokens."""
        assert self.vision_encoder is not None
        is_dummy = pixel_values is None or grid_thw is None
        if is_dummy:
            # Every rank runs the encoder so the FSDP collectives match.
            merge = self.vision_encoder.spatial_merge_size
            pixel_values, grid_thw = build_dummy_vision_inputs(
                patch_dim=self.vision_encoder.patch_embed.in_features,
                grid_thw=(1, merge, merge),
                device=inputs_embeds_TD.device,
            )
        assert pixel_values is not None and grid_thw is not None
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        if is_dummy:
            return add_zero_vision_dependency(inputs_embeds_TD, vision_embeds)
        if placeholder_id is None:
            raise ValueError("special_tokens are required for vision inputs.")
        num_tokens_per_item = (
            grid_thw.prod(-1) // self.vision_encoder.spatial_merge_unit
        )
        vision_positions = get_vision_positions(
            tokens, num_tokens_per_item, placeholder_id
        )
        if not vision_positions:
            return inputs_embeds_TD
        return scatter_vision_embeds(
            inputs_embeds_TD,
            vision_embeds=vision_embeds,
            vision_positions=vision_positions,
        )

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        """Embed tokens, then encode images and videos into their placeholders."""
        assert self.tok_embeddings is not None
        inputs_embeds_TD = self.tok_embeddings(tokens)
        if self.vision_encoder is None:
            if pixel_values is not None or pixel_values_videos is not None:
                raise ValueError(
                    "Vision inputs were provided without a vision encoder."
                )
            return inputs_embeds_TD
        special_tokens = special_tokens or {}
        # TODO: Configure the job-wide modality set from the dataset so
        # single-modality jobs can avoid the second encoder call.
        inputs_embeds_TD = self._scatter_modality(
            inputs_embeds_TD,
            tokens,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            placeholder_id=special_tokens.get("image_id"),
        )
        return self._scatter_modality(
            inputs_embeds_TD,
            tokens,
            pixel_values=pixel_values_videos,
            grid_thw=grid_thw_videos,
            placeholder_id=special_tokens.get("video_id"),
        )

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        if self.tok_embeddings is not None:
            with spmd_local_context("dp"):
                x_TD = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw=grid_thw,
                    grid_thw_videos=grid_thw_videos,
                    special_tokens=special_tokens,
                )
        else:
            x_TD = tokens

        if spmd.is_type_checking():
            spmd.assert_type(
                x_TD,
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                spmd.PartitionSpec(("dp", "cp"), None),
            )

        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            x_TD = layer(
                x_TD,
                (
                    attention_metadata.get(type(layer.attention.inner_attention))
                    if attention_metadata is not None
                    else None
                ),
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )

        x_TD = self.norm(x_TD) if self.norm is not None else x_TD
        if self._skip_lm_head:
            return x_TD
        return self.lm_head(x_TD) if self.lm_head is not None else x_TD
