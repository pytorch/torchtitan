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
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_local_context,
)
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    FlexAttentionMetadata,
    FlexInnerAttention,
    GQAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.multimodal import (
    add_zero_vision_dependency,
    build_dummy_vision_inputs,
    build_vision_bank_indices,
    gather_vision_embeds,
    MultimodalModel,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module
from .state_dict_adapter import MuseGlimmerStateDictAdapter

from .vision_encoder import MuseGlimmerVisionAdapter, MuseGlimmerVisionEncoder


class RMSGainCenterNorm(RMSNorm):
    """RMSNorm whose effective scale is ``weight + gain_center``.

    Pre/post norms initialize ``weight`` to 0 with ``gain_center=1.0``.
    The final output norm initializes ``weight`` to 1 with ``gain_center=0.0``,
    so all of these norms start with unit effective scale.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        gain_center: float

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.gain_center = config.gain_center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w: torch.Tensor = self.weight + self.gain_center
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


class Attention(GQAttention):
    """Muse Glimmer GQA attention.

    Adds, on top of :class:`GQAttention`:
    - a tuned query scaling applied after q-norm (``scale_query_by``),
    - a sigmoid output gate (``o_gate``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        scale_query_by: float
        o_gate: Linear.Config | None = None
        # None = global attention (no sliding window) for this layer.
        window_size: int | None = None

        @property
        def sliding_window_size(self) -> int | None:
            # Alias: the vLLM generator wrapper reads ``sliding_window_size`` to
            # configure per-layer paged-attention windows; the flex path uses
            # ``window_size``. Keep both in sync via this alias (mirrors gpt_oss's
            # field name without renaming the flex-path usages).
            return self.window_size

    def __init__(self, config: Config):
        super().__init__(config)
        self.scale_query_by: float = config.scale_query_by
        self.window_size: int | None = config.window_size
        self.o_gate: Linear | None = None
        if config.o_gate is not None:
            self.o_gate = config.o_gate.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # qkv and the output gate both consume x, so gather once at their
        # common attention boundary.
        x_TD = maybe_gather_tp_input(self, x_TD)

        num_tokens = x_TD.shape[0]
        xq, xk, xv = self.qkv_linear(x_TD)

        # QK normalization before RoPE. Query is additionally scaled by a
        # tuned constant (k is only normalized).
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq = self.q_norm(xq) * self.scale_query_by
            xk = self.k_norm(xk)

        # iRoPE: RoPE is skipped on NoPE layers (config-driven per layer).
        if self.rope is not None:
            xq, xk = self.rope(xq, xk, positions)

        output = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            xq,
            xk,
            xv,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        # The copy below reads the inner_attention output with bare ops.
        remat.recompute_needs_tensor(output)
        output = output.contiguous().view(num_tokens, -1)

        if self.o_gate is not None:
            gate = self.o_gate(x_TD)
            # The gating reads the o_gate projection output with bare ops.
            remat.recompute_needs_tensor(gate)
            output = output * torch.sigmoid(gate)

        return self.wo(output)


class MuseGlimmerTransformerBlock(TransformerBlock):
    """Muse Glimmer transformer block with post-norm residuals.

    ``h = x + post_attention_norm(attn(attention_norm(x)))``
    ``out = h + post_ffn_norm(ffn(ffn_norm(h)))``
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        post_attention_norm: RMSNorm.Config
        post_ffn_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.post_attention_norm = config.post_attention_norm.build()
        self.post_ffn_norm = config.post_ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
    ):
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # post_attention_norm reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        h = x + self.post_attention_norm(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm(h))
        # post_ffn_norm reads the feed-forward output with bare ops.
        remat.recompute_needs_tensor(ffn_out)
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h, self.post_ffn_norm(ffn_out))


class SoftCappedLinear(Linear):
    """Output head that applies Muse Glimmer's output multiplier and optional tanh
    soft-cap on top of a plain linear projection.

    Keeping the transform in the ``lm_head`` (rather than in the model's
    ``forward``) means it runs wherever ``lm_head`` runs: the full forward for
    ``CrossEntropyLoss``, or per-chunk inside ``ChunkedLossWrapper`` (which applies
    ``lm_head`` itself after the model returns hidden states). The transform is
    elementwise per logit, so it composes with sequence chunking and vocab
    (loss-parallel) sharding.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        output_multiplier: float = 1.0
        output_soft_cap_temp: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_multiplier = config.output_multiplier
        self.output_soft_cap_temp = config.output_soft_cap_temp

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = super().forward(input).float()
        if self.output_soft_cap_temp is not None:
            logits = self.output_soft_cap_temp * torch.tanh(
                logits * self.output_multiplier / self.output_soft_cap_temp
            )
        else:
            logits = logits * self.output_multiplier
        return logits


class EmbeddingWithNorm(Module):
    """Token embedding bundled with a scaleless RMSNorm on the looked-up
    embeddings.

    Bundling keeps the embedding and its norm together as one
    pipeline-relocatable unit, so the norm travels with ``tok_embeddings`` under
    the default PP module split instead of being pruned to ``None``.

    The norm is a sibling child that runs *after* the embedding child so that,
    under TP, it sees the embedding's already-reduced output. (Vocab-parallel
    embedding emits a ``Partial`` result; the embedding child's sharding
    all-reduces it at the child boundary before the norm runs -- normalizing a
    partial sum would be incorrect.)

    Contrast with :class:`SoftCappedLinear` at the other end of the model. That
    transform is *elementwise per logit*, so it commutes with both vocab
    (loss-parallel) sharding and sequence chunking and can stay fused inside the
    ``lm_head`` wherever it runs. The norm here is the opposite: it *reduces
    across the feature dim*, so it is not valid on a sharded/partial activation
    and must instead be ordered after the embedding's reduction completes. The
    two classes solve composability differently for that reason -- one relies on
    elementwise independence, the other on explicit ordering relative to the
    collective.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        embedding: Embedding.Config
        norm: RMSNorm.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embedding = config.embedding.build()
        self.norm = config.norm.build()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embedding(tokens))


class MuseGlimmerModel(MultimodalModel):
    state_dict_adapter_cls = MuseGlimmerStateDictAdapter
    multimodal_encoder_fqns = ("vision_encoder",)
    pipeline_first_stage_module_fqns = (
        "vision_encoder",
        "vision_adapter",
        "vision_projection",
        "perception_emb_norm",
    )

    """Muse Glimmer decoder-only language model.

    Args:
        config (MuseGlimmerModel.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 6656
        vocab_size: int = 202048
        local_compile_regions: list[str] = field(
            default_factory=lambda: ["loss", "fused_binary_activation"]
        )
        # Narrows the base Decoder.Config.tok_embeddings (Embedding.Config) to the
        # bundled embedding+norm unit that sharding.py indexes via .embedding/.norm.
        # Dataclass fields are invariant, so pyrefly flags the (intentional) override.
        # pyrefly: ignore [bad-override]
        tok_embeddings: EmbeddingWithNorm.Config
        # Optional LLM-side multimodal injection. Preprocessing builds absolute
        # packed-bank indices before CP; forward gathers the corresponding vision
        # rows into the TP-replicated token embeddings.
        vision_projection: Linear.Config | None = None
        perception_emb_norm: RMSNorm.Config | None = None
        # Optional owned vision stack. When set, ``MuseGlimmerModel`` builds the encoder
        # + adapter as submodules and runs them inside ``forward`` (from padded
        # ``pixel_values`` + ``grid_thw``), mirroring qwen3_5's
        # ``Qwen35Model.vision_encoder``. The adapter output dim must match
        # ``vision_projection`` in_features. Both default to None (text-only model).
        vision_encoder: MuseGlimmerVisionEncoder.Config | None = None
        vision_adapter: MuseGlimmerVisionAdapter.Config | None = None

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # Vision modules run per image rather than per text token.
            muse_model = cast("MuseGlimmerModel", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(
                    muse_model.vision_encoder,
                    muse_model.vision_adapter,
                    muse_model.vision_projection,
                    muse_model.perception_emb_norm,
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

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import set_muse_glimmer_sharding_config

            set_muse_glimmer_sharding_config(
                self, enable_sp=parallelism.enable_sequence_parallel
            )

    def __init__(self, config: "MuseGlimmerModel.Config") -> None:
        super().__init__(config)
        # LLM-side multimodal injection modules (None for the text-only model).
        self.vision_projection = (
            config.vision_projection.build()
            if config.vision_projection is not None
            else None
        )
        self.perception_emb_norm = (
            config.perception_emb_norm.build()
            if config.perception_emb_norm is not None
            else None
        )
        # Owned vision stack (None unless a multimodal flavor configured it). When
        # present, ``forward`` runs encoder->adapter on packed pixel_values.
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        self.vision_adapter = (
            config.vision_adapter.build() if config.vision_adapter is not None else None
        )

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile_regions: list[str],
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
    ) -> MuseGlimmerModel:
        if self.vision_encoder is not None and parallelism_context.tp_enabled:
            assert self.vision_encoder.num_heads % parallelism_context.tp == 0, (
                f"vision num_heads ({self.vision_encoder.num_heads}) must be "
                f"divisible by TP degree ({parallelism_context.tp})"
            )

        return super().parallelize(
            parallelism_context=parallelism_context,
            training=training,
            parallelism=parallelism,
            local_compile_regions=local_compile_regions,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build first-stage vision-bank indices and masks, then shard the batch."""
        del kwargs
        from .sharding import vision_bank_indices_placement

        pixel_values = input_dict.get("pixel_values")
        grid_thw = input_dict.get("grid_thw")
        pixel_values_videos = input_dict.get("pixel_values_videos")
        grid_thw_videos = input_dict.get("grid_thw_videos")
        special_tokens = input_dict.get("special_tokens")
        has_images = pixel_values is not None
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError(
                "Muse Glimmer vision encoder does not support video inputs."
            )
        if has_images:
            vision_encoder_config = cast(
                MuseGlimmerModel.Config, self.config
            ).vision_encoder
            if vision_encoder_config is None:
                raise ValueError(
                    "pixel_values were provided but the model config has no "
                    "vision_encoder configured."
                )
            if grid_thw is None:
                raise ValueError(
                    "pixel_values were provided but grid_thw was not provided."
                )
            if special_tokens is None or "image_id" not in special_tokens:
                raise ValueError(
                    "pixel_values were provided but special_tokens with an "
                    "'image_id' entry was not provided."
                )
            if self.tok_embeddings is not None:
                input_dict["vision_bank_indices_T"] = build_vision_bank_indices(
                    input_dict["input"],
                    placeholder_id=special_tokens["image_id"],
                )
        input_dict.pop("special_tokens", None)

        positions = input_dict.get("positions", None)
        padding_mask = input_dict.pop("padding_mask", None)
        if positions is not None:
            inner = getattr(self.config.first_base_attention, "inner_attention", None)
            if isinstance(
                inner, (FlexInnerAttention.Config, VarlenInnerAttention.Config)
            ):
                input_dict["attention_metadata"] = self._get_attention_metadata(
                    positions=positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )

        input_shardings = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(),
        }
        input_shardings["vision_bank_indices_T"] = vision_bank_indices_placement(
            enable_sp=parallelism.enable_sequence_parallel
        )
        if parallelism_context.cp_enabled:
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
        if (
            parallelism.enable_sequence_parallel
            and parallelism_context.tp_enabled
            and "vision_bank_indices_T" in input_dict
        ):
            input_dict["vision_bank_indices_T"] = spmd.shard(
                input_dict["vision_bank_indices_T"],
                parallelism_context.get_dense_tp_mesh().get_group(),
                src=spmd.I,
                dst=spmd.S(0),
            )
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )

        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        return inputs, labels, input_dict

    def _get_vision_features(
        self,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode packed pixels into the normalized LLM-dimension vision bank.

        ``pixel_values`` contains all visual patches packed into one sequence,
        and ``grid_thw`` describes each visual item's contiguous segment.
        """
        assert self.vision_encoder is not None and self.vision_adapter is not None
        assert self.vision_projection is not None
        assert self.perception_emb_norm is not None
        assert pixel_values is not None and grid_thw is not None
        vision_features_VD = self.vision_adapter(
            self.vision_encoder(pixel_values, grid_thw=grid_thw)
        )
        return self.perception_emb_norm(self.vision_projection(vision_features_VD))

    def _prepare_multimodal_embeds(
        self,
        h_TD: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        vision_bank_indices_T: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build and inject image embeddings on the embedding pipeline stage."""
        if self.vision_encoder is None:
            return h_TD

        image_is_dummy = pixel_values is None
        if image_is_dummy:
            grid_size = self.vision_encoder.downsample_factor
            pixel_values, grid_thw = build_dummy_vision_inputs(
                patch_dim=self.vision_encoder.conv1_linear.in_features,
                grid_thw=(1, grid_size, grid_size),
                device=h_TD.device,
            )
        vision_bank_VD = self._get_vision_features(pixel_values, grid_thw)
        if image_is_dummy:
            return add_zero_vision_dependency(h_TD, vision_bank_VD)

        assert grid_thw is not None
        if vision_bank_indices_T is None:
            raise ValueError("vision_bank_indices_T is required for image inputs")
        return gather_vision_embeds(
            h_TD,
            vision_bank_VD=vision_bank_VD,
            vision_bank_indices_T=vision_bank_indices_T,
        )

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
        vision_bank_indices_T: torch.Tensor | None = None,
    ):
        # Video inputs are rejected by preprocess_inputs.
        del aux_loss_denominators, padding_mask, pixel_values_videos, grid_thw_videos

        # Embedding stage: embed tokens (the scaleless norm is bundled inside
        # tok_embeddings) and inject vision features before the decoder layers.
        # On non-embedding pipeline stages tok_embeddings is None and the input
        # is already hidden states, so injection is skipped there.
        if self.tok_embeddings is not None:
            h_TD = self.tok_embeddings(tokens)
            with spmd_local_context("dp"):
                h_TD = self._prepare_multimodal_embeds(
                    h_TD,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    vision_bank_indices_T=vision_bank_indices_T,
                )
        else:
            h_TD = tokens

        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(
                    cast(TransformerBlock, layer).attention.attention_metadata_key
                )
            )
            h_TD = layer(
                h_TD,
                layer_attention_metadata,
                positions,
            )

        h_TD = self.norm(h_TD) if self.norm is not None else h_TD

        # _skip_lm_head is an attribute (not a kwarg) because PP backward calls
        # .requires_grad on all stage inputs, which fails on bool kwargs.
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD
