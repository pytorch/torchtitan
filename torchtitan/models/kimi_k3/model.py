# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, TypedDict

import spmd_types as spmd
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import CompileConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    annotate_replicated_parameters,
    spmd_local_context,
)
from torchtitan.models.common import FeedForward, Linear
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    FlexInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import (
    decoder_input_sharding,
    dense_activation_placement,
    token_id_placement,
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
from torchtitan.models.kimi_k3.sharding import set_kimi_k3_sharding_config
from torchtitan.models.utils import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

from .attention import KimiMLAAttention
from .kda import KDA, KDAAttentionMetadata
from .moe import KimiLatentMoE
from .state_dict_adapter import KimiK3StateDictAdapter
from .vision_encoder import KimiK3VisionEncoder


class KimiK3AttentionMetadata(TypedDict):
    """Per-batch metadata for Kimi K3 attention backends."""

    quadratic_attention: BlockMask | VarlenMetadata | None
    kda: KDAAttentionMetadata


# Shape suffixes:
# T = packed token count (num_tokens)
# D = model dimension (dim)
# A = residual candidate count per token


def _apply_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TAD: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32."""
    assert norm.eps is not None

    values_TAD = torch.cat((block_residual_TAD, prefix_sum_TD.unsqueeze(1)), dim=1)
    values_float = values_TAD.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TAD = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TA = (keys_TAD * score_weight_D).sum(dim=-1)
    probs_T1A = torch.softmax(scores_TA, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1A, values_float).squeeze(1)
    return output_TD.to(values_TAD.dtype)


class KimiK3TransformerBlock(Module):
    """Hybrid KDA/MLA decoder block with Kimi attention residuals."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        attn_res_block_size: int
        attention: KimiMLAAttention.Config | None
        delta_attention: KDA.Config | None
        feed_forward: FeedForward.Config | None
        moe: KimiLatentMoE.Config | None
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        attention_res_norm: RMSNorm.Config | None
        attention_res_proj: Linear.Config | None
        ffn_res_norm: RMSNorm.Config
        ffn_res_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        if (config.attention is None) == (config.delta_attention is None):
            raise ValueError(
                "Exactly one of attention or delta_attention must be configured."
            )
        if (config.feed_forward is None) == (config.moe is None):
            raise ValueError("Exactly one of feed_forward or moe must be configured.")
        self.layer_id = config.layer_id
        self.attn_res_block_size = config.attn_res_block_size
        self.attention = (
            config.attention.build() if config.attention is not None else None
        )
        self.delta_attention = (
            config.delta_attention.build()
            if config.delta_attention is not None
            else None
        )
        self.feed_forward = (
            config.feed_forward.build() if config.feed_forward is not None else None
        )
        self.moe = config.moe.build() if config.moe is not None else None
        self.moe_enabled = self.moe is not None
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

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TAD: torch.Tensor,
        attention_metadata: KimiK3AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum_TD = x_TD

        if block_residual_TAD.shape[1] > 0:
            assert self.attention_res_proj is not None
            assert self.attention_res_norm is not None
            x_TD = _apply_attention_residual(
                prefix_sum_TD,
                block_residual_TAD,
                self.attention_res_proj,
                self.attention_res_norm,
            )

        opens_block = self.layer_id % self.attn_res_block_size == 0
        if opens_block:
            block_residual_TAD = torch.cat(
                (
                    block_residual_TAD,
                    prefix_sum_TD.unsqueeze(1),
                ),
                dim=1,
            )

        h_TD = self.attention_norm(x_TD)
        if self.attention is not None:
            layer_mask = (
                attention_metadata["quadratic_attention"]
                if attention_metadata is not None
                else None
            )
            h_TD = self.attention(h_TD, layer_mask, positions)
        else:
            assert self.delta_attention is not None
            kda_metadata = attention_metadata["kda"] if attention_metadata else None
            h_TD = self.delta_attention(h_TD, kda_metadata, positions)
        prefix_sum_TD = h_TD if opens_block else prefix_sum_TD + h_TD

        h_TD = _apply_attention_residual(
            prefix_sum_TD,
            block_residual_TAD,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if self.moe is not None:
            h_TD = self.moe(h_TD, padding_mask_T=padding_mask)
        else:
            assert self.feed_forward is not None
            h_TD = self.feed_forward(h_TD)
        return prefix_sum_TD + h_TD, block_residual_TAD


class KimiK3Model(MultimodalModel):
    state_dict_adapter_cls = KimiK3StateDictAdapter
    multimodal_encoder_fqns = ("vision_encoder",)

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallel_dims) -> None:
        from torchtitan.components.optimizer import register_moe_quantile_balancing_hook

        register_moe_quantile_balancing_hook(optimizers, model_parts, parallel_dims)

    supports_pipeline_parallel = False

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: RMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            if parallelism.context_parallel_degree > 1:
                load_balancer_config = parallelism.context_parallel_load_balancer
                if load_balancer_config is not None and not isinstance(
                    load_balancer_config, HeadTailCPLoadBalancer.Config
                ):
                    raise ValueError(
                        "Kimi K3 KDA context parallelism supports only contiguous "
                        "or head-tail token partitions."
                    )
                conv_kernel_sizes = {
                    layer.delta_attention.conv_kernel_size
                    for layer in self.layers
                    if layer.delta_attention is not None
                }
                if len(conv_kernel_sizes) != 1:
                    raise ValueError(
                        "Kimi K3 context parallelism requires every KDA layer "
                        "to use the same convolution kernel size."
                    )

            # Vision attention is also head-sharded; validate its head count.
            tp = parallelism.tensor_parallel_degree
            vision_heads = (
                self.vision_encoder.block.attn.num_heads
                if self.vision_encoder is not None
                else None
            )
            if tp > 1 and vision_heads is not None and vision_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"vision num_heads ({vision_heads})."
                )

            set_kimi_k3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
                cp_enabled=parallelism.context_parallel_degree > 1,
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
                elif isinstance(layer.delta_attention, KDA.Config):
                    delta_attention = layer.delta_attention
                    attention_op_flops += delta_rule_flops_per_token(
                        num_heads=delta_attention.num_heads,
                        key_head_dim=delta_attention.head_dim,
                        v_head_dim=delta_attention.head_dim,
                    )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_res_norm = config.output_res_norm.build()
        self.output_res_proj = config.output_res_proj.build()
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def parallelize(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: CompileConfig | None,
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
        skip_dp: bool = False,
    ) -> KimiK3Model:
        unsupported = [
            name
            for name, enabled in (("pipeline parallel", parallel_dims.pp_enabled),)
            if enabled
        ]
        if unsupported:
            raise NotImplementedError(
                f"Kimi K3 does not support {', '.join(unsupported)}."
            )
        if compile_config is not None and "model" in compile_config.components:
            raise NotImplementedError("Kimi K3 does not support model compilation yet.")

        from torchtitan.distributed.utils import get_spmd_context

        with get_spmd_context(parallel_dims=parallel_dims):
            annotate_replicated_parameters(self, parallel_dims)
            self._parallelize(parallel_dims)
            if ac_config is not None:
                policy = ac_config.build(dump_folder=dump_folder)
                policy.apply(self)
                if self.vision_encoder is not None:
                    policy.apply(self.vision_encoder)
            if not skip_dp:
                self._apply_fsdp(
                    parallel_dims=parallel_dims,
                    training=training,
                    parallelism=parallelism,
                )
        return self

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build masks and CP metadata, shard inputs, and annotate layouts."""
        del kwargs
        batch: dict[str, Any] = dict(input_dict)
        positions = batch.get("positions")
        padding_mask = batch.get("padding_mask", None)
        if positions is not None:
            inner = self.config.first_full_attention_backend
            if isinstance(
                inner, (FlexInnerAttention.Config, VarlenInnerAttention.Config)
            ):
                batch["attention_masks"] = self.get_attention_masks(
                    positions=positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )

        pixel_values = batch.get("pixel_values")
        grid_thw = batch.get("grid_thw")
        special_tokens = batch.get("special_tokens")
        if parallel_dims.cp_enabled and pixel_values is not None:
            if grid_thw is None:
                raise ValueError(
                    "pixel_values were provided but grid_thw was not provided."
                )
            if self.vision_encoder is None:
                raise ValueError("pixel_values were provided without a vision encoder.")
            if special_tokens is None or "image_id" not in special_tokens:
                raise ValueError(
                    "pixel_values require special_tokens with an 'image_id' entry."
                )
            if self.tok_embeddings is not None:
                batch["vision_bank_indices_T"] = build_vision_bank_indices(
                    batch["input"],
                    placeholder_id=special_tokens["image_id"],
                )
            batch.pop("special_tokens")

        input_shardings = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=parallel_dims.cp_enabled),
        }
        if "vision_bank_indices_T" in batch:
            input_shardings["vision_bank_indices_T"] = token_id_placement()
        if parallel_dims.cp_enabled:
            batch = self._prepare_cp_batch(
                batch,
                input_shardings=input_shardings,
                parallel_dims=parallel_dims,
                parallelism=parallelism,
            )
        batch = annotate_input_spmd_types(parallel_dims, batch, input_shardings)

        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> KimiK3AttentionMetadata:
        attn_config = self.config.first_attention

        kda_metadata = create_varlen_metadata_for_document(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )

        if attn_config is None:
            quadratic_attention = None
        elif isinstance(attn_config.inner_attention, VarlenInnerAttention.Config):
            # Under varlen both consumers read the same document offsets.
            quadratic_attention = kda_metadata
        else:
            quadratic_attention = super().get_attention_masks(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
        # pyrefly: ignore [bad-return]
        return {
            "quadratic_attention": quadratic_attention,  # pyrefly: ignore [bad-assignment]
            "kda": KDAAttentionMetadata(varlen=kda_metadata),
        }

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        vision_bank_indices_T: torch.Tensor | None,
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

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: KimiK3AttentionMetadata | None = None,
        padding_mask: torch.Tensor | None = None,
        vision_bank_indices_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
                dense_activation_placement(tp=spmd.I, cp=spmd.S(0)),
            )

        block_residual_TAD = h_TD.unsqueeze(1)[:, :0]
        for layer in self.layers.values():
            h_TD, block_residual_TAD = layer(
                h_TD,
                block_residual_TAD,
                attention_metadata=attention_masks,
                positions=positions,
                padding_mask=padding_mask,
            )

        h_TD = _apply_attention_residual(
            h_TD,
            block_residual_TAD,
            self.output_res_proj,
            self.output_res_norm,
        )
        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD
