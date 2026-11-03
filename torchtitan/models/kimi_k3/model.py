# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn

from torchtitan.config import TrainingConfig
from torchtitan.config.function import Function
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.distributed.local_compile import local_compile, LocalCompileConfig
from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    annotate_replicated_parameters,
    spmd_local_context,
)
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    AttentionMetadata,
    AttentionMetadataMap,
    LinearAttentionMetadata,
)
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import (
    decoder_input_sharding,
    token_id_placement,
)
from torchtitan.models.common.flops import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.models.common.multimodal import (
    add_zero_vision_dependency,
    build_dummy_vision_inputs,
    build_vision_bank_indices,
    gather_vision_embeds,
    get_vision_positions,
    MultimodalModel,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding

from .attention import KimiMLAAttention
from .moe import KimiLatentMoE
from .state_dict_adapter import KimiK3StateDictAdapter
from .vision_encoder import KimiK3VisionEncoder

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels, H = heads,
# K = query/key head dimension, V = value head dimension,
# N = attention-residual entries.


class AttentionResidual(Function[torch.Tensor]):
    """Configurable callable for Kimi K3 attention residual aggregation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Function.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    # N grows across attention-residual blocks, so keep one shape-polymorphic graph.
    @local_compile("attention_residual", batch_invariant=True, dynamic=True)
    def __call__(
        self,
        partial_block_TD: torch.Tensor | None,
        block_residual_TND: torch.Tensor,
        projection: Linear,
        norm: RMSNorm,
    ) -> torch.Tensor:
        """Apply Kimi's block-level attention residual in FP32."""
        assert norm.eps is not None

        values_TND = (
            block_residual_TND
            if partial_block_TD is None
            else torch.cat((block_residual_TND, partial_block_TD.unsqueeze(1)), dim=1)
        )
        values_float = values_TND.float()
        variance = values_float.pow(2).mean(dim=-1, keepdim=True)
        keys_TND = values_float * torch.rsqrt(variance + norm.eps)
        score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
        scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
        probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
        output_TD = torch.matmul(probs_T1N, values_float).squeeze(1)
        return output_TD.to(values_TND.dtype)


class KimiK3TransformerBlock(TransformerBlock):
    """Hybrid KDA/MLA decoder block with Kimi attention residuals."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        layer_id: int
        attn_res_block_size: int
        attention_res_norm: RMSNorm.Config | None
        attention_res_proj: Linear.Config | None
        ffn_res_norm: RMSNorm.Config
        ffn_res_proj: Linear.Config
        attention_residual: AttentionResidual.Config

    def __init__(self, config: Config):
        super().__init__()
        # A block's first layer closes the previous block and joins the stack.
        self.first_layer_in_block = config.layer_id % config.attn_res_block_size == 0
        self.attention = config.attention.build()
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attention_res_norm = (
            config.attention_res_norm.build()
            if config.attention_res_norm is not None
            else None
        )
        self.attention_res_proj = (
            config.attention_res_proj.build()
            if config.attention_res_proj is not None
            else None
        )
        self.ffn_res_norm = config.ffn_res_norm.build()
        self.ffn_res_proj = config.ffn_res_proj.build()
        self.attention_residual = config.attention_residual.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_metadata: AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.first_layer_in_block:
            block_residual_TND = torch.cat(
                (block_residual_TND, x_TD.unsqueeze(1)), dim=1
            )
            partial_block_TD = None
        else:
            partial_block_TD = x_TD

        if self.attention_res_proj is None:
            h_TD = x_TD
        else:
            assert self.attention_res_norm is not None
            h_TD = self.attention_residual(
                partial_block_TD,
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            )
        h_TD = self.attention_norm(h_TD)
        h_TD = self.attention(h_TD, attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(h_TD)
        prefix_sum_TD = h_TD if self.first_layer_in_block else x_TD + h_TD

        h_TD = self.attention_residual(
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if isinstance(self.feed_forward, KimiLatentMoE):
            h_TD = self.feed_forward(
                h_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            h_TD = self.feed_forward(h_TD)
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        out_TD = remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(prefix_sum_TD, h_TD)
        return out_TD, block_residual_TND


class KimiK3Model(MultimodalModel):
    state_dict_adapter_cls = KimiK3StateDictAdapter
    multimodal_encoder_fqns = ("vision_encoder",)

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_quantile_balancing_hook

        register_moe_quantile_balancing_hook(
            optimizers, model_parts, parallelism_context
        )

    pipeline_first_stage_module_fqns = ("vision_encoder",)
    pipeline_last_stage_module_fqns = ("output_res_proj", "output_res_norm")

    def pipeline(self, **kwargs):
        """Partition the model, then route the attention-residual blocks along the split."""
        from .pipeline_parallel import pipeline_kimi_k3

        return pipeline_kimi_k3(self, **kwargs)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: RMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None
        attention_residual: AttentionResidual.Config = field(
            default_factory=AttentionResidual.Config
        )
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "gated_rmsnorm",
                    "fused_binary_activation",
                    "fp32_to_bf16_split",
                    "attention_residual",
                ]
            )
        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            kimi_model = cast("KimiK3Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(kimi_model.vision_encoder,),
            )
            attention_op_flops = 0
            for layer in self.layers:
                if isinstance(layer.attention, KimiMLAAttention.Config):
                    attention = layer.attention
                    attention_op_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=(
                            attention.qk_nope_head_dim + attention.qk_rope_head_dim
                        ),
                        v_head_dim=attention.v_head_dim,
                        seq_len=seq_len,
                    )
                elif isinstance(layer.attention, KDA.Config):
                    delta_attention = layer.attention
                    attention_op_flops += delta_rule_flops_per_token(
                        num_heads=delta_attention.n_heads,
                        key_head_dim=delta_attention.head_dim,
                        v_head_dim=delta_attention.head_dim,
                    )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        tp = spmd_mesh_size("tp")
        vision_heads = (
            config.vision_encoder.block.attn.num_heads
            if config.vision_encoder is not None
            else None
        )
        if tp > 1 and vision_heads is not None and vision_heads % tp != 0:
            raise ValueError(
                f"tensor parallel degree ({tp}) must divide "
                f"vision num_heads ({vision_heads})."
            )
        super().__init__(config)
        self.output_res_norm = config.output_res_norm.build()
        self.output_res_proj = config.output_res_proj.build()
        self.attention_residual = config.attention_residual.build()
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile: LocalCompileConfig,
        activation_remat: ActivationRematConfig | None,
        dump_folder: str,
    ) -> KimiK3Model:
        # Bind local implementations early; torch.compile traces on first use.
        local_compile.apply()
        with parallelism_context.activate_spmd():
            annotate_replicated_parameters(self, parallelism_context)
            self._parallelize(parallelism_context)
            if activation_remat is not None:
                activation_remat.apply(self)
                if self.vision_encoder is not None:
                    activation_remat.apply(self.vision_encoder)
            self._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
        return self

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
        """Build metadata, CP-shard inputs, and annotate K3 layouts."""
        del kwargs
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask", None)
        if positions is not None:
            input_dict["attention_metadata"] = self._get_attention_metadata(
                positions=positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

        pixel_values = input_dict.get("pixel_values")
        if (
            parallelism_context.cp_enabled
            and pixel_values is not None
            and self.tok_embeddings is not None
        ):
            special_tokens = input_dict.get("special_tokens")
            if special_tokens is None or "image_id" not in special_tokens:
                raise ValueError(
                    "pixel_values require special_tokens with an 'image_id' entry."
                )
            input_dict["vision_bank_indices_T"] = build_vision_bank_indices(
                input_dict["input"],
                placeholder_id=special_tokens["image_id"],
            )

        input_shardings = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(),
        }
        if "vision_bank_indices_T" in input_dict:
            input_shardings["vision_bank_indices_T"] = token_id_placement()
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
        if parallelism_context.cp_enabled:
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )
        attention_metadata = input_dict.get("attention_metadata")
        if attention_metadata is not None:
            for metadata in attention_metadata.values():
                if isinstance(metadata, LinearAttentionMetadata):
                    metadata.annotate_spmd_types()

        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        input_dict["aux_loss_denominators"] = global_routing_token_counts
        return inputs, labels, input_dict, loss_kwargs

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        vision_bank_indices_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings_TD = self.tok_embeddings(tokens)
        if (pixel_values is None) != (grid_thw is None):
            raise ValueError(
                "pixel_values and grid_thw must either both be provided or "
                "both be omitted."
            )
        is_dummy = pixel_values is None
        if is_dummy:
            if self.vision_encoder is None:
                return embeddings_TD
            kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
            pixel_values, grid_thw = build_dummy_vision_inputs(
                patch_dim=self.vision_encoder.patch_embed.in_features,
                grid_thw=(1, kernel_h, kernel_w),
                device=embeddings_TD.device,
            )
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")

        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        if is_dummy:
            return add_zero_vision_dependency(embeddings_TD, vision_embeds)

        if vision_bank_indices_T is not None:
            return gather_vision_embeds(
                embeddings_TD,
                vision_bank_VD=vision_embeds,
                vision_bank_indices_T=vision_bank_indices_T,
            )

        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")
        # MoonViT collapses time and merges spatially, so the text-side token
        # count per item is (h/kh)*(w/kw), independent of t.
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kernel_h) * (
            grid_thw[:, 2] // kernel_w
        )
        vision_positions = get_vision_positions(
            tokens,
            num_tokens_per_item,
            special_tokens["image_id"],
        )
        return scatter_vision_embeds(
            embeddings_TD,
            vision_embeds=vision_embeds,
            vision_positions=vision_positions,
        )

    def forward(  # pyrefly: ignore[bad-param-name-override, bad-override]
        self,
        tokens: torch.Tensor,
        block_residual_TND: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        padding_mask: torch.Tensor | None = None,
        vision_bank_indices_T: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")

        if self.tok_embeddings is not None:
            with spmd_local_context("dp"):
                h_TD = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    special_tokens=special_tokens,
                    vision_bank_indices_T=vision_bank_indices_T,
                )
        else:
            h_TD = tokens

        if spmd.is_type_checking():
            spmd.assert_type(
                h_TD,
                spmd.SpmdType(
                    {
                        MeshAxisName.DP: spmd.V,
                        MeshAxisName.CP: spmd.V,
                        MeshAxisName.TP: spmd.R,
                    },
                    partition_spec=spmd.PartitionSpec(
                        (MeshAxisName.DP, MeshAxisName.CP), None
                    ),
                ),
            )

        if block_residual_TND is None:
            block_residual_TND = h_TD.unsqueeze(1)[:, :0]
        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for block in self.layers.values():
            h_TD, block_residual_TND = block(
                h_TD,
                block_residual_TND,
                (
                    attention_metadata[type(block.attention.inner_attention)]
                    if attention_metadata is not None
                    else None
                ),
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )

        if self.output_res_proj is None:
            return h_TD, block_residual_TND
        h_TD = self.attention_residual(
            h_TD,
            block_residual_TND,
            self.output_res_proj,
            self.output_res_norm,
        )
        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD
