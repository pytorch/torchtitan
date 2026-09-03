# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
    GQAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
    multimodal_context,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

from .vision_encoder import MuseGlimmerVisionAdapter, MuseGlimmerVisionEncoder


def _window_mask_key(window_size: int | None) -> str:
    """Mask-dict key for a layer's attention window (``"global"`` or ``"swa_<n>"``)."""
    return "global" if window_size is None else f"swa_{window_size}"


class RMSGainCenterNorm(RMSNorm):
    """RMSNorm whose effective scale is ``weight + gain_center``.

    The learnable ``weight`` is initialized to 0 so the norm starts centered
    on ``gain_center`` (1.0 for pre/post norms, 0.0 for the final output norm).
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
    - a sigmoid output gate (``o_gate``),
    - per-layer sliding-window selection from a window-keyed mask dict.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        # Muse Glimmer-specific per-layer iRoPE flag: the shared GQAttention always
        # applies RoPE, so Muse Glimmer carries its own flag and guards the call in
        # forward (NoPE layers still build a rope module so max_context_length
        # discovery/resize in the base Decoder works uniformly).
        use_rope: bool = True
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
        self.use_rope: bool = config.use_rope
        self.scale_query_by: float = config.scale_query_by
        self.window_size: int | None = config.window_size
        self.o_gate: Linear | None = None
        if config.o_gate is not None:
            self.o_gate = config.o_gate.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        xq, xk, xv = self.qkv_linear(x_TD)

        # QK normalization before RoPE. Query is additionally scaled by a
        # tuned constant (k is only normalized).
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq = self.q_norm(xq) * self.scale_query_by
            xk = self.k_norm(xk)

        # iRoPE: RoPE is skipped on NoPE layers (config-driven per layer).
        if self.use_rope:
            xq, xk = self.rope(xq, xk, positions)

        # Select this layer's mask by its window ("global" key = full attention).
        # Only flex passes a window-keyed dict of BlockMasks to index into. Varlen
        # passes a single VarlenMetadata shared by every layer (each layer's window is
        # a kernel arg, baked in at build time), so it goes straight through to the
        # kernel (mirrors gpt_oss).
        if isinstance(attention_masks, dict):
            attention_masks = attention_masks[_window_mask_key(self.window_size)]

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        output = output.view(num_tokens, -1)

        if self.o_gate is not None:
            output = output * torch.sigmoid(self.o_gate(x_TD))

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
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.post_attention_norm(
            self.attention(self.attention_norm(x), attention_masks, positions)
        )
        out = h + self.post_ffn_norm(self.feed_forward(self.ffn_norm(h)))
        return out


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


class MuseGlimmerModel(Decoder):
    """Muse Glimmer decoder-only language model.

    Args:
        config (MuseGlimmerModel.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 6656
        vocab_size: int = 202048
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

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            from .sharding import set_muse_glimmer_sharding_config

            set_muse_glimmer_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )

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

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build first-stage vision-bank indices and masks, then shard the batch."""
        # Function-local import avoids a circular import.
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        from .sharding import vision_bank_indices_placement

        batch: dict[str, Any] = dict(input_dict)
        pixel_values = batch.get("pixel_values")
        grid_thw = batch.get("grid_thw")
        pixel_values_videos = batch.get("pixel_values_videos")
        grid_thw_videos = batch.get("grid_thw_videos")
        special_tokens = batch.get("special_tokens")
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError(
                "Muse Glimmer vision encoder does not support video inputs."
            )
        if pixel_values is not None:
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
                batch["vision_bank_indices_T"] = build_vision_bank_indices(
                    batch["input"],
                    placeholder_id=special_tokens["image_id"],
                )
        batch.pop("special_tokens", None)

        positions = batch.get("positions", None)
        if positions is not None:
            inner = getattr(self.config.first_attention, "inner_attention", None)
            if isinstance(inner, (FlexAttention.Config, VarlenAttention.Config)):
                batch["attention_masks"] = self.get_attention_masks(positions=positions)

        input_sharding = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=True),
        }
        input_sharding["vision_bank_indices_T"] = vision_bank_indices_placement(
            enable_sp=parallelism.enable_sequence_parallel
        )
        if parallel_dims.cp_enabled:
            batch = prepare_context_parallel_input(
                batch,
                input_sharding,
                parallel_dims.get_mesh("cp"),
                parallelism.context_parallel_load_balancer,
                parallelism.context_parallel_ptrr_mask_key,
            )
        if parallelism.spmd_backend == "spmd_types":
            if (
                parallelism.enable_sequence_parallel
                and parallel_dims.tp_enabled
                and "vision_bank_indices_T" in batch
            ):
                batch["vision_bank_indices_T"] = spmd.shard(
                    batch["vision_bank_indices_T"],
                    parallel_dims.get_dense_tp_mesh().get_group(),
                    src=spmd.I,
                    dst=spmd.S(0),
                )
            batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)

        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def _get_vision_features(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode packed ``pixel_values`` and adapter-project into features.

        Mirrors qwen3_5's ``_get_vision_embeds``: runs the owned encoder +
        adapter and returns ``[T, adapter_dim]``. ``pixel_values`` contains all
        visual patches packed into one sequence, and ``grid_thw`` describes each
        visual item's contiguous segment.
        """
        assert self.vision_encoder is not None and self.vision_adapter is not None
        feats = self.vision_adapter(
            self.vision_encoder(pixel_values, grid_thw=grid_thw)
        )
        return feats

    def _prepare_multimodal_embeds(
        self,
        h_TD: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        vision_bank_indices_T: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build and inject image embeddings on the embedding pipeline stage."""
        if pixel_values is None:
            return h_TD
        assert grid_thw is not None
        assert vision_bank_indices_T is not None
        assert self.vision_projection is not None
        assert self.perception_emb_norm is not None

        vision_features_VD = self._get_vision_features(pixel_values, grid_thw)
        vision_bank_VD = self.perception_emb_norm(
            self.vision_projection(vision_features_VD)
        )
        return gather_vision_embeds(
            h_TD,
            vision_bank_VD=vision_bank_VD,
            vision_bank_indices_T=vision_bank_indices_T,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        vision_bank_indices_T: torch.Tensor | None = None,
    ):
        # Video inputs are rejected by preprocess_inputs.
        del pixel_values_videos, grid_thw_videos

        # Embedding stage: embed tokens (the scaleless norm is bundled inside
        # tok_embeddings) and inject vision features before the decoder layers.
        # On non-embedding pipeline stages tok_embeddings is None and the input
        # is already hidden states, so injection is skipped there.
        if self.tok_embeddings is not None:
            h_TD = self.tok_embeddings(tokens)
            with multimodal_context():
                h_TD = self._prepare_multimodal_embeds(
                    h_TD,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    vision_bank_indices_T=vision_bank_indices_T,
                )
        else:
            h_TD = tokens

        for layer in self.layers.values():
            h_TD = layer(h_TD, attention_masks, positions)

        h_TD = self.norm(h_TD) if self.norm is not None else h_TD

        # _skip_lm_head is an attribute (not a kwarg) because PP backward calls
        # .requires_grad on all stage inputs, which fails on bool kwargs.
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        attn_config = self.config.first_attention
        assert attn_config is not None
        inner_attn = attn_config.inner_attention
        # Varlen carries each layer's sliding window in its own kernel arg (baked at
        # build time), so all layers share one document-varlen metadata; only the
        # flex path needs the per-window BlockMask dict built below.
        if isinstance(inner_attn, VarlenAttention.Config):
            return create_varlen_metadata_for_document(positions)
        if not isinstance(inner_attn, FlexAttention.Config):
            raise TypeError(
                "Muse Glimmer requires FlexAttention or VarlenAttention for "
                f"sliding-window masks, got {type(inner_attn).__name__}"
            )

        # Language models always use block-causal (per-document) masking: the
        # dataloaders emit per-document positions, and the efficient packed-doc
        # mask ANDed with the causal mask yields same-document causal attention.
        seq_len = positions.shape[0]
        base_mods = [
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        ]

        # Match the base Decoder mask-building so the configured block size and
        # batch-invariance handling are honored.
        block_size = inner_attn.block_size
        separate_full_blocks = not is_in_batch_invariant_mode()

        def _build_mask(mask_mods: list) -> BlockMask:
            return create_attention_mask(
                and_masks(*mask_mods),
                1,
                None,
                seq_len,
                seq_len,
                device=positions.device,
                BLOCK_SIZE=block_size,
                separate_full_blocks=separate_full_blocks,
            )

        # "global" mask (no sliding window) plus one mask per distinct sliding
        # window size across the layers; Attention.forward selects by window.
        # All masks are built from the same ``base_mods`` so the global and
        # windowed variants cannot drift apart.
        masks: dict[str, BlockMask] = {_window_mask_key(None): _build_mask(base_mods)}
        window_sizes = {
            layer.attention.window_size
            for layer in self.config.layers
            if layer.attention.window_size is not None
        }
        for window_size in window_sizes:
            masks[_window_mask_key(window_size)] = _build_mask(
                [*base_mods, get_sliding_window_mask_mod(window_size)]
            )

        return masks
