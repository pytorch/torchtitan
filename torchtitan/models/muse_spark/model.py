# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast, Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_local_context,
)
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    FlexAttentionMetadata,
    GQAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.flops import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.multimodal import (
    add_zero_vision_dependency as add_zero_perception_dependency,
    build_dummy_vision_inputs,
    build_vision_bank_indices,
    gather_vision_embeds,
    MultimodalModel,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding

from torchtitan.models.muse_spark.audio_encoder import MuseSparkAudioEncoder
from torchtitan.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter
from torchtitan.models.muse_spark.vision_encoder import (
    MuseSparkNaViTAdapter,
    NaViTEncoder,
)

# Shape suffix legend:
# T = token or patch rows, D = model dimension, N = query heads, K = KV heads,
# H = head dimension, O = flattened attention output dimension,
# V = perception bank rows, P = flattened patch features, I = visual items,
# S = waveform samples, C = audio channels.


def build_dummy_audio_inputs(
    *,
    patch_size: int,
    input_channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one patch-aligned zero waveform for the audio encoder."""
    waveforms_SC = torch.zeros(patch_size, input_channels, device=device)
    waveform_padded_lengths_N = torch.tensor([patch_size], device=device)
    if spmd.is_type_checking():
        for tensor in (waveforms_SC, waveform_padded_lengths_N):
            spmd.mutate_type(tensor, "dp", src=spmd.R, dst=spmd.V)
            spmd.mutate_type(tensor, "tp", src=spmd.R, dst=spmd.I)
            spmd.mutate_type(tensor, "cp", src=spmd.R, dst=spmd.I)
    return waveforms_SC, waveform_padded_lengths_N


class MuseSparkAttention(GQAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        use_rope: bool = True
        scale_query_by: float
        window_size: int | None = None
        position_temperature_floor_scale: float | None = None
        position_temperature_query_scale: float | None = None
        output_norm: RMSNorm.Config | None = None
        output_gate: Linear.Config | None = None

        @property
        def sliding_window_size(self) -> int | None:
            return self.window_size

    def __init__(self, config: Config):
        super().__init__(config)
        if not config.use_rope:
            self.rope = None
        self.use_rope = self.rope is not None
        self.window_size = config.window_size
        self.scale_query_by = config.scale_query_by
        self.position_temperature_floor_scale = config.position_temperature_floor_scale
        self.position_temperature_query_scale = config.position_temperature_query_scale
        self.output_norm = (
            config.output_norm.build() if config.output_norm is not None else None
        )
        self.output_gate = (
            config.output_gate.build() if config.output_gate is not None else None
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_TNH, key_TKH, value_TKH = self.qkv_linear(x_TD)
        output_gate_TO = None
        if self.output_gate is not None:
            output_gate_TO = self.output_gate(x_TD)

        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            remat.recompute_needs_tensor(query_TNH)
            query_TNH = self.q_norm(query_TNH) * self.scale_query_by
            remat.recompute_needs_tensor(key_TKH)
            key_TKH = self.k_norm(key_TKH)

        floor_scale = self.position_temperature_floor_scale
        query_scale = self.position_temperature_query_scale
        if floor_scale is not None and query_scale is not None:
            if positions is None:
                raise ValueError(
                    "positions are required for attention temperature tuning"
                )
            temperature_T = (
                torch.log((positions.float() + 1.0) / floor_scale) * query_scale + 1.0
            ).clamp_min(1.0)
            remat.recompute_needs_tensor(query_TNH)
            query_TNH = (query_TNH * temperature_T.reshape(-1, 1, 1)).to(
                query_TNH.dtype
            )

        if self.rope is not None:
            remat.recompute_needs_tensor(query_TNH, key_TKH)
            query_TNH, key_TKH = self.rope(
                query_TNH,
                key_TKH,
                positions,
            )

        output_TNH = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            query_TNH,
            key_TKH,
            value_TKH,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        remat.recompute_needs_tensor(output_TNH)
        output_TNH = output_TNH.contiguous()
        if self.output_norm is not None:
            output_TNH = self.output_norm(output_TNH)
        output_TO = output_TNH.reshape(output_TNH.shape[0], -1)
        if output_gate_TO is not None:
            remat.recompute_needs_tensor(output_gate_TO)
            output_TO = output_TO * torch.sigmoid(output_gate_TO)
        output_TD = self.wo(output_TO)
        return output_TD


class MuseSparkRMSNorm(RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        gain_center: float = 1.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.gain_center = config.gain_center

    def _get_effective_weight(self) -> torch.Tensor | None:
        if self.weight is None:
            return None
        return self.weight + self.gain_center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        output_TD = F.rms_norm(
            input,
            self.normalized_shape,
            self._get_effective_weight(),
            self.eps,
        )
        return output_TD.to(input_dtype)


class MuseSparkRMSNormWithInputScale(MuseSparkRMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(MuseSparkRMSNorm.Config):
        pass

    @local_compile("muse_spark_input_scaled_rmsnorm", batch_invariant=False)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        effective_weight_D = self._get_effective_weight()
        if effective_weight_D is not None:
            input = effective_weight_D * input
        output_TD = F.rms_norm(input, self.normalized_shape, None, self.eps)
        return output_TD.to(input_dtype)

    def to_dist_moe_postprocess(
        self,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the learned RMSNorm as an eager Dist-MoE callback."""
        return self


class MuseSparkNormPreservingResidualConnection(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        temperature: float = 0.3
        eps: float = 1e-3

    def __init__(self, config: Config):
        super().__init__()
        self.temperature = config.temperature
        self.eps = config.eps
        self.gate = nn.Parameter(torch.empty(config.dim))

    def _coefficients(
        self,
        scaled_gate_1D: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta_1D = torch.sigmoid(scaled_gate_1D)
        alpha_squared_1D = torch.sigmoid(-scaled_gate_1D) * (1.0 + beta_1D)
        alpha_1D = torch.sqrt(torch.clamp(alpha_squared_1D, min=self.eps))
        return alpha_1D, beta_1D

    @local_compile("muse_spark_residual", batch_invariant=True)
    def forward(
        self,
        stream_TD: torch.Tensor,
        branch_TD: torch.Tensor,
    ) -> torch.Tensor:
        scaled_gate_1D = self.gate.reshape(1, -1).float() / self.temperature
        alpha_1D, beta_1D = self._coefficients(scaled_gate_1D)
        output_TD = alpha_1D * stream_TD + beta_1D * branch_TD.to(stream_TD.dtype)
        return output_TD.to(stream_TD.dtype)


class MuseSparkEmbeddingWithNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        embedding: Embedding.Config
        norm: MuseSparkRMSNorm.Config | None = None
        multiplier: float = 1.0

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embedding = config.embedding.build()
        self.norm = config.norm.build() if config.norm is not None else None
        self.multiplier = config.multiplier

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output_TD = self.embedding(tokens)
        if self.norm is not None:
            output_TD = self.norm(output_TD)
        return output_TD * self.multiplier


class MuseSparkLMHead(HiMidLoLinear):
    @dataclass(kw_only=True, slots=True)
    class Config(HiMidLoLinear.Config):
        output_multiplier: float = 1.0
        output_soft_cap_temp: float | None = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.output_multiplier = config.output_multiplier
        self.output_soft_cap_temp = config.output_soft_cap_temp

    @local_compile("muse_spark_logit_postprocess", batch_invariant=True)
    def postprocess_logits(self, logits_TV: torch.Tensor) -> torch.Tensor:
        """Apply Muse Spark Open's output scaling and smooth logit cap."""
        scaled_logits_TV = logits_TV * self.output_multiplier
        if self.output_soft_cap_temp is None:
            return scaled_logits_TV
        return self.output_soft_cap_temp * torch.tanh(
            scaled_logits_TV / self.output_soft_cap_temp
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raw_logits_TV = super().forward(input)
        return self.postprocess_logits(raw_logits_TV)


class MuseSparkTransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # pyrefly: ignore [bad-override]
        attention: MuseSparkAttention.Config
        post_attention_norm: MuseSparkRMSNorm.Config
        attention_residual: MuseSparkNormPreservingResidualConnection.Config
        post_ffn_norm: MuseSparkRMSNormWithInputScale.Config
        ffn_residual: MuseSparkNormPreservingResidualConnection.Config
        branch_dtype: torch.dtype = torch.bfloat16

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.post_attention_norm = config.post_attention_norm.build()
        self.attention_residual = config.attention_residual.build()
        self.ffn_norm = config.ffn_norm.build()
        self.post_ffn_norm = config.post_ffn_norm.build()
        self.ffn_residual = config.ffn_residual.build()
        self.branch_dtype = config.branch_dtype

        self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input_TD = self.attention_norm(x).to(self.branch_dtype)
        attention_output_TD = self.attention(
            attention_input_TD,
            attention_metadata,
            positions,
        )
        remat.recompute_needs_tensor(attention_output_TD)
        attention_branch_TD = self.post_attention_norm(attention_output_TD)
        stream_TD = self.attention_residual(x, attention_branch_TD)

        ffn_input_TD = self.ffn_norm(stream_TD).to(self.branch_dtype)
        if isinstance(self.feed_forward, MoE):
            ffn_output_TD = self.feed_forward(
                ffn_input_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_output_TD = self.feed_forward(ffn_input_TD)
        remat.recompute_needs_tensor(ffn_output_TD)
        ffn_branch_TD = self.post_ffn_norm(ffn_output_TD)
        return self.ffn_residual(stream_TD, ffn_branch_TD)


class MuseSparkModel(MultimodalModel):
    state_dict_adapter_cls = MuseSparkStateDictAdapter
    multimodal_encoder_fqns = ("vision_encoder", "audio_encoder")
    pipeline_first_stage_module_fqns = (
        "vision_encoder",
        "vision_adapter",
        "vision_projection",
        "audio_encoder",
        "perception_emb_norm",
    )

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    @dataclass(kw_only=True, slots=True)
    class Config(MultimodalModel.Config):
        """Configure stage modalities consistently across ranks.

        ``dataset_modalities`` is the union of modalities the current dataset can
        supply, shared across ranks. Listed encoders always run, using dummy
        inputs when locally absent, even on steps with no media on any rank.
        Static participation and zero-valued dependencies keep FSDP all-gathers
        and reduce-scatters aligned without a media-presence collective.
        Unlisted encoders are skipped but retained in checkpoints.

        TODO: Build only required encoders once checkpoint saving can preserve
        weights for omitted encoders. Model-only loading supports omitted
        encoders, but saving without an encoder drops its weights from subsequent
        checkpoints. A decoder-only fine-tune would therefore lose pretrained
        encoder weights needed for a later multimodal fine-tune.
        """

        local_compile_regions: list[str] = field(
            default_factory=lambda: ["loss", "fp32_to_bf16_split"]
        )
        # pyrefly: ignore [bad-override]
        tok_embeddings: MuseSparkEmbeddingWithNorm.Config
        layers: list[MuseSparkTransformerBlock.Config]
        residual_dtype: torch.dtype = torch.float32
        final_norm_input_dtype: torch.dtype | None = None
        dataset_modalities: tuple[Literal["vision", "audio"], ...] = ()
        audio_encoder: MuseSparkAudioEncoder.Config | None = None
        vision_encoder: NaViTEncoder.Config | None = None
        vision_adapter: MuseSparkNaViTAdapter.Config | None = None
        vision_projection: Linear.Config | None = None
        perception_emb_norm: MuseSparkRMSNorm.Config | None = None

        def __post_init__(self) -> None:
            if self.enable_weight_tying:
                raise ValueError(
                    "Muse Spark does not support token embedding weight tying"
                )
            encoders = {"vision": self.vision_encoder, "audio": self.audio_encoder}
            for modality in self.dataset_modalities:
                if modality not in encoders:
                    raise ValueError(f"Unsupported dataset modality: {modality!r}")
                if encoders[modality] is None:
                    raise ValueError(
                        f"dataset_modalities requires a configured {modality}_encoder"
                    )

        def get_nparams_and_flops(
            self,
            model: nn.Module,
            seq_len: int,
        ) -> tuple[int, int]:
            # Perception modules run per image or audio clip rather than per
            # text token, so this remains a decoder-only MFU estimate.
            muse_model = cast("MuseSparkModel", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(
                    muse_model.vision_encoder,
                    muse_model.vision_adapter,
                    muse_model.vision_projection,
                    muse_model.perception_emb_norm,
                    muse_model.audio_encoder,
                ),
            )
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
                    sliding_window_size=attention.window_size,
                )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.residual_dtype = config.residual_dtype
        self.final_norm_input_dtype = config.final_norm_input_dtype
        self.vision_encoder: NaViTEncoder | None = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        self.vision_adapter: MuseSparkNaViTAdapter | None = (
            config.vision_adapter.build() if config.vision_adapter is not None else None
        )
        self.vision_projection: Linear | None = (
            config.vision_projection.build()
            if config.vision_projection is not None
            else None
        )
        self.perception_emb_norm: MuseSparkRMSNorm | None = (
            config.perception_emb_norm.build()
            if config.perception_emb_norm is not None
            else None
        )
        self.audio_encoder: MuseSparkAudioEncoder | None = (
            config.audio_encoder.build() if config.audio_encoder is not None else None
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        from torchtitan.models.muse_spark.sharding import (
            multimodal_bank_indices_placement,
        )

        del kwargs
        batch: dict[str, Any] = dict(input_dict)
        loss_kwargs = batch.pop("loss_kwargs", {})
        special_tokens = batch.get("special_tokens")
        for input_name, placeholder_key, bank_indices_name in (
            ("pixel_values", "image_id", "image_bank_indices_T"),
            ("pixel_values_videos", "video_id", "video_bank_indices_T"),
            ("waveforms", "audio_id", "audio_bank_indices_T"),
        ):
            if batch.get(input_name) is None:
                continue
            if special_tokens is None or placeholder_key not in special_tokens:
                raise ValueError(
                    f"{input_name} require special_tokens with a "
                    f"{placeholder_key} entry"
                )
            if self.tok_embeddings is not None:
                batch[bank_indices_name] = build_vision_bank_indices(
                    batch["input"],
                    placeholder_id=special_tokens[placeholder_key],
                )
        batch.pop("special_tokens", None)

        positions = batch.get("positions")
        padding_mask = batch.get("padding_mask")
        if positions is not None:
            batch["attention_metadata"] = self._get_attention_metadata(
                positions=positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

        perception_input_sharding = multimodal_input_sharding()
        perception_layout = perception_input_sharding["pixel_values"]
        input_sharding = {
            **decoder_input_sharding(),
            **perception_input_sharding,
            "waveforms": perception_layout,
            "waveform_lengths": perception_layout,
            "waveform_padded_lengths": perception_layout,
        }
        bank_indices_placement = multimodal_bank_indices_placement(
            enable_sp=parallelism.enable_sequence_parallel
        )
        for bank_indices_name in (
            "image_bank_indices_T",
            "video_bank_indices_T",
            "audio_bank_indices_T",
        ):
            input_sharding[bank_indices_name] = bank_indices_placement
        input_sharding.update(dict.fromkeys(loss_kwargs, input_sharding["labels"]))
        batch.update(loss_kwargs)
        if parallelism_context.cp_enabled:
            batch = self._cp_shard(
                batch,
                input_shardings=input_sharding,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
        if parallelism.enable_sequence_parallel and parallelism_context.tp_enabled:
            tp_group = parallelism_context.get_dense_tp_mesh().get_group()
            for bank_indices_name in (
                "image_bank_indices_T",
                "video_bank_indices_T",
                "audio_bank_indices_T",
            ):
                if bank_indices_name in batch:
                    batch[bank_indices_name] = spmd.shard(
                        batch[bank_indices_name],
                        tp_group,
                        src=spmd.I,
                        dst=spmd.S(0),
                    )
        batch = annotate_input_spmd_types(
            parallelism_context,
            batch,
            input_sharding,
        )

        loss_kwargs = {name: batch.pop(name) for name in loss_kwargs}
        inputs = batch.pop("input")
        labels = batch.pop("labels")
        if next(self.config.traverse(AuxLoss.Config), None) is not None:
            batch["aux_loss_denominators"] = global_routing_token_counts
        return inputs, labels, batch, loss_kwargs

    def _get_visual_bank(
        self,
        *,
        pixel_values_TP: torch.Tensor,
        grid_thw_I3: torch.Tensor,
    ) -> torch.Tensor:
        vision_encoder = self.vision_encoder
        vision_adapter = self.vision_adapter
        assert vision_encoder is not None and vision_adapter is not None
        vision_projection = self.vision_projection
        perception_emb_norm = self.perception_emb_norm
        assert vision_projection is not None and perception_emb_norm is not None
        tok_embeddings = self.tok_embeddings
        assert tok_embeddings is not None
        vision_features_TO = vision_adapter(
            vision_encoder(pixel_values_TP, grid_thw=grid_thw_I3)
        )
        vision_embeds_TD = (
            perception_emb_norm(vision_projection(vision_features_TO))
            * tok_embeddings.multiplier
        )
        return vision_embeds_TD

    def _get_audio_bank(
        self,
        *,
        waveforms_SC: torch.Tensor,
        waveform_padded_lengths_N: torch.Tensor,
    ) -> torch.Tensor:
        audio_encoder = self.audio_encoder
        assert audio_encoder is not None
        perception_emb_norm = self.perception_emb_norm
        assert perception_emb_norm is not None
        tok_embeddings = self.tok_embeddings
        assert tok_embeddings is not None
        audio_features_TD = audio_encoder(
            waveforms_SC,
            waveform_padded_lengths_N=waveform_padded_lengths_N,
        )
        audio_embeds_TD = (
            perception_emb_norm(audio_features_TD) * tok_embeddings.multiplier
        )
        return audio_embeds_TD

    def _prepare_visual_embeds(
        self,
        hidden_TD: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        image_bank_indices_T: torch.Tensor | None,
        video_bank_indices_T: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pack images and videos into one forward and fuse their output banks."""
        vision_encoder = self.vision_encoder
        assert vision_encoder is not None
        # Each set of indices refers to its own image or video bank.
        vision_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for pixels_TP, grid_I3, indices_T in (
            (pixel_values, grid_thw, image_bank_indices_T),
            (pixel_values_videos, grid_thw_videos, video_bank_indices_T),
        ):
            if pixels_TP is not None:
                assert grid_I3 is not None and indices_T is not None
                vision_inputs.append((pixels_TP, grid_I3, indices_T))
        # All enabled ranks enter the encoder once, even with no local media.
        is_dummy = not vision_inputs
        if is_dummy:
            grid_size = vision_encoder.downsample_factor
            pixels_TP, grid_I3 = build_dummy_vision_inputs(
                patch_dim=vision_encoder.conv1_linear.in_features,
                grid_thw=(1, grid_size, grid_size),
                device=hidden_TD.device,
            )
        else:
            # Preserve image/video order when packing pixels and their grids.
            patches, grids, _ = zip(*vision_inputs)
            pixels_TP = patches[0] if len(patches) == 1 else torch.cat(patches)
            grid_I3 = grids[0] if len(grids) == 1 else torch.cat(grids)

        vision_bank_VD = self._get_visual_bank(
            pixel_values_TP=pixels_TP,
            grid_thw_I3=grid_I3,
        )
        if is_dummy:
            # Keep the encoder in FSDP backward without changing text embeddings.
            return add_zero_perception_dependency(hidden_TD, vision_bank_VD)

        # Split full banks before gathering CP/TP-local tokens. Their sizes come
        # from unsharded media patches, and indices remain relative to each bank.
        bank_sizes = [
            patches_TP.shape[0] // vision_encoder.downsample_factor**2
            for patches_TP, _, _ in vision_inputs
        ]
        for bank_VD, (_, _, indices_T) in zip(
            vision_bank_VD.split(bank_sizes), vision_inputs, strict=True
        ):
            hidden_TD = gather_vision_embeds(
                hidden_TD,
                vision_bank_VD=bank_VD,
                vision_bank_indices_T=indices_T,
            )
        return hidden_TD

    def _prepare_audio_embeds(
        self,
        hidden_TD: torch.Tensor,
        *,
        waveforms: torch.Tensor | None,
        waveform_padded_lengths: torch.Tensor | None,
        audio_bank_indices_T: torch.Tensor | None,
    ) -> torch.Tensor:
        is_dummy = waveforms is None
        if waveforms is None:
            audio_config = cast(MuseSparkModel.Config, self.config).audio_encoder
            assert audio_config is not None
            waveforms, waveform_padded_lengths = build_dummy_audio_inputs(
                patch_size=audio_config.patch_size,
                input_channels=audio_config.input_channels,
                device=hidden_TD.device,
            )
        assert waveform_padded_lengths is not None
        audio_bank_VD = self._get_audio_bank(
            waveforms_SC=waveforms,
            waveform_padded_lengths_N=waveform_padded_lengths,
        )
        if is_dummy:
            return add_zero_perception_dependency(hidden_TD, audio_bank_VD)
        assert audio_bank_indices_T is not None
        return gather_vision_embeds(
            hidden_TD,
            vision_bank_VD=audio_bank_VD,
            vision_bank_indices_T=audio_bank_indices_T,
        )

    def _prepare_multimodal_embeds(
        self,
        hidden_TD: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        waveforms: torch.Tensor | None,
        waveform_padded_lengths: torch.Tensor | None,
        image_bank_indices_T: torch.Tensor | None,
        video_bank_indices_T: torch.Tensor | None,
        audio_bank_indices_T: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fuse active perception banks, using dummies for absent local inputs."""
        dataset_modalities = cast(
            "MuseSparkModel.Config", self.config
        ).dataset_modalities
        if (
            pixel_values is not None or pixel_values_videos is not None
        ) and "vision" not in dataset_modalities:
            raise ValueError(
                "Image and video inputs require 'vision' in dataset_modalities"
            )
        if waveforms is not None and "audio" not in dataset_modalities:
            raise ValueError("Audio inputs require 'audio' in dataset_modalities")
        if "vision" in dataset_modalities:
            hidden_TD = self._prepare_visual_embeds(
                hidden_TD,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                pixel_values_videos=pixel_values_videos,
                grid_thw_videos=grid_thw_videos,
                image_bank_indices_T=image_bank_indices_T,
                video_bank_indices_T=video_bank_indices_T,
            )

        if "audio" in dataset_modalities:
            hidden_TD = self._prepare_audio_embeds(
                hidden_TD,
                waveforms=waveforms,
                waveform_padded_lengths=waveform_padded_lengths,
                audio_bank_indices_T=audio_bank_indices_T,
            )

        return hidden_TD

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        waveforms: torch.Tensor | None = None,
        waveform_lengths: torch.Tensor | None = None,
        waveform_padded_lengths: torch.Tensor | None = None,
        image_bank_indices_T: torch.Tensor | None = None,
        video_bank_indices_T: torch.Tensor | None = None,
        audio_bank_indices_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del waveform_lengths
        if self.tok_embeddings is not None:
            hidden_TD = self.tok_embeddings(tokens)
            with spmd_local_context("dp"):
                hidden_TD = self._prepare_multimodal_embeds(
                    hidden_TD,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw=grid_thw,
                    grid_thw_videos=grid_thw_videos,
                    waveforms=waveforms,
                    waveform_padded_lengths=waveform_padded_lengths,
                    image_bank_indices_T=image_bank_indices_T,
                    video_bank_indices_T=video_bank_indices_T,
                    audio_bank_indices_T=audio_bank_indices_T,
                )
        else:
            hidden_TD = tokens

        hidden_TD = hidden_TD.to(self.residual_dtype)

        with spmd.no_typecheck():
            aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            hidden_TD = layer(
                hidden_TD,
                layer_attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )

        if self.norm is not None:
            final_norm_input_dtype = (
                self.final_norm_input_dtype
                if self.final_norm_input_dtype is not None
                else torch.bfloat16
            )
            hidden_TD = self.norm(hidden_TD.to(final_norm_input_dtype)).to(
                torch.bfloat16
            )

        if self._skip_lm_head:
            return hidden_TD
        lm_head = self.lm_head
        if lm_head is None:
            return hidden_TD
        return lm_head(hidden_TD.to(lm_head.weight.dtype))
