# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.distributed.utils import get_spmd_backend, is_in_batch_invariant_mode
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
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.multimodal import multimodal_context
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.observability import tensor_logging
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
        # forward (NoPE layers still build a rope module so max_seq_len
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
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Param name must stay ``x_BLD`` to match the base GQAttention.forward and
        # the sharding-config key set by set_gqa_attention_sharding: the per-arg
        # input redistribution (SP Shard(1) -> Replicate) is looked up by the
        # forward's actual parameter name, so renaming this drops the gather.
        bs, seqlen, _ = x_BLD.shape
        xq, xk, xv = self.qkv_linear(x_BLD)
        tensor_logging.log_fwd_bwd_stats(self, xq=xq, xk=xk, xv=xv)

        # QK normalization before RoPE. Query is additionally scaled by a
        # tuned constant (k is only normalized).
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq = self.q_norm(xq) * self.scale_query_by
            xk = self.k_norm(xk)
            tensor_logging.log_fwd_bwd_stats(
                self,
                xq_normed=xq,
                xk_normed=xk,
            )

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
        output = output.view(bs, seqlen, -1)

        if self.o_gate is not None:
            output = output * torch.sigmoid(self.o_gate(x_BLD))

        tensor_logging.log_fwd_bwd_stats(self, head_out=output)
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
        tensor_logging.register_fwd_bwd(
            self,
            ["attn_stream", "attn_out", "ffn_stream", "ffn_out"],
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        attn_stream = x
        attn_out = self.post_attention_norm(
            self.attention(self.attention_norm(attn_stream), attention_masks, positions)
        )
        tensor_logging.log_fwd_bwd_stats(
            self,
            attn_stream=attn_stream,
            attn_out=attn_out,
        )

        ffn_stream = attn_stream + attn_out
        ffn_out = self.post_ffn_norm(self.feed_forward(self.ffn_norm(ffn_stream)))
        tensor_logging.log_fwd_bwd_stats(
            self,
            ffn_stream=ffn_stream,
            ffn_out=ffn_out,
        )
        return ffn_stream + ffn_out


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
        # Optional LLM-side multimodal injection. When set, encoded vision
        # features (already adapter-projected to ``vision_projection`` in_features)
        # are projected to ``dim``, scaleless-normed, and scattered into the token
        # embeddings at masked positions. Both default to None for the text-only
        # model, leaving the text path untouched.
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

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            from .sharding import set_muse_glimmer_sharding_config

            set_muse_glimmer_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layers[0].attention, GQAttention.Config)
            assert self.layers[0].attention.head_dim is not None
            nparams, num_flops_per_token = get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * self.layers[0].attention.head_dim,
                seq_len=seq_len,
                enable_weight_tying=False,
            )
            # get_dense_model_nparams_and_flops excludes embedding params from
            # the matmul FLOP count by scanning the model's *immediate* children
            # for nn.Embedding. Muse Glimmer nests its nn.Embedding inside
            # EmbeddingWithNorm, so that scan finds nothing and the embedding
            # FLOPs (6 * params) are not subtracted. Correct for it here (Muse Glimmer
            # does not tie embeddings). tok_embeddings is None on non-embedding
            # pipeline stages, where there is nothing to subtract.
            tok_embeddings = getattr(model, "tok_embeddings", None)
            if tok_embeddings is not None:
                nparams_embedding = sum(p.numel() for p in tok_embeddings.parameters())
                num_flops_per_token -= 6 * nparams_embedding
            return nparams, num_flops_per_token

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
        # present, ``forward`` runs encoder->adapter on padded pixel_values.
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        self.vision_adapter = (
            config.vision_adapter.build() if config.vision_adapter is not None else None
        )
        if self.vision_projection is not None:
            tensor_logging.register_fwd_bwd(
                self,
                ["vision_embeddings_after_projection"],
            )

    def _get_vision_features(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode padded ``pixel_values`` and adapter-project into features.

        Mirrors qwen3_5's ``_get_vision_embeds``: runs the owned encoder +
        adapter and returns ``[1, n_vision_tokens, adapter_dim]`` -- the shape
        ``_inject_vision`` expects. ``pixel_values`` is the padded
        ``[N, P, patch_dim]`` tensor (one row per image, zero-padded to the
        batch's max patch count) and ``grid_thw`` the ``[N, 3]`` per-image grid;
        the encoder unpads each row and casts inputs to its parameter dtype.
        """
        assert self.vision_encoder is not None and self.vision_adapter is not None
        feats = self.vision_adapter(
            self.vision_encoder(pixel_values, grid_thw=grid_thw)
        )
        return feats.unsqueeze(0)

    def _vision_spans(self, vision_mask: torch.Tensor) -> list[tuple[int, int, int]]:
        """Find contiguous vision spans as ``(sample_idx, start, n_tokens)``.

        Scans ``vision_mask`` (``[batch, seq_len]``) and returns the contiguous
        runs of True entries in row-major order -- the same order in which the
        flat ``vision_features`` tokens are laid out.
        """
        # Compute start/end transitions for the whole batch on-device, then do a
        # single .tolist() sync. Doing .tolist() per-row (per sample) would force
        # a device->host sync for every batch element on the vision path.
        zero_col = torch.zeros(
            vision_mask.shape[0], 1, dtype=torch.bool, device=vision_mask.device
        )
        prev = torch.cat([zero_col, vision_mask[:, :-1]], dim=1)
        nxt = torch.cat([vision_mask[:, 1:], zero_col], dim=1)
        # nonzero returns indices in row-major (sample-then-column) order, matching
        # the flat vision_features token layout; start/end rows align pairwise.
        start_idx = (vision_mask & ~prev).nonzero(as_tuple=False)
        end_cols = (vision_mask & ~nxt).nonzero(as_tuple=False)[:, 1]
        spans = torch.stack(
            [start_idx[:, 0], start_idx[:, 1], end_cols - start_idx[:, 1] + 1],
            dim=1,
        )
        return [(s, start, n) for s, start, n in spans.tolist()]

    def _inject_vision(
        self,
        h: torch.Tensor,
        vision_features: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project + scaleless-norm vision features and scatter them into the
        token-embedding stream at ``vision_mask`` positions.

        ``vision_features`` is ``[1, n_vision_tokens, adapter_dim]`` and
        ``vision_mask`` is ``[batch, seq_len]`` with exactly ``n_vision_tokens``
        True entries.

        Boolean-mask assignment (``h[vision_mask] = vision_features``) has no
        DTensor sharding rule, so the features are scattered by explicit
        integer/slice assignment instead.
        """
        if self.vision_projection is None or self.perception_emb_norm is None:
            raise ValueError(
                "vision_features were provided but the model has no "
                "vision_projection/perception_emb_norm configured."
            )
        v = self.vision_projection(vision_features)
        v = self.perception_emb_norm(v)
        v = v.squeeze(0).to(h.dtype)
        tensor_logging.log_fwd_bwd_stats(
            self,
            vision_embeddings_after_projection=v,
        )
        # Scatter via integer/slice assignment rather than boolean-mask
        # index_put: the latter has no DTensor sharding rule, the former does.
        # Under TP both ``h`` and ``v`` are Replicate (the embedding output is
        # overridden to Replicate so the full sequence is local), so the slice
        # assignment writes straight through DTensor -- no unwrap/clone/rewrap.
        # The plain path (single-GPU / FSDP) writes directly. ``v`` is consumed
        # in row-major order to match the boolean-mask semantics it replaces.
        v_offset = 0
        spans = self._vision_spans(vision_mask)
        total_span = sum(n_tokens for _, _, n_tokens in spans)
        if total_span != v.shape[0]:
            raise ValueError(
                f"vision_mask selects {total_span} positions but "
                f"vision_features has {v.shape[0]} tokens; counts must match."
            )
        for sample_idx, start, n_tokens in spans:
            h[sample_idx, start : start + n_tokens, :] = v[
                v_offset : v_offset + n_tokens, :
            ]
            v_offset += n_tokens
        return h

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
        special_tokens: dict[str, int] | None = None,
    ):
        # Embedding stage: embed tokens (the scaleless norm is bundled inside
        # tok_embeddings) and inject vision features before the decoder layers.
        # On non-embedding pipeline stages tok_embeddings is None and the input
        # is already hidden states, so injection is skipped there.
        with multimodal_context():
            if get_spmd_backend() == "spmd_types":
                from .sharding import annotate_muse_glimmer_input_spmd_types

                annotate_muse_glimmer_input_spmd_types(
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                )

            if self.tok_embeddings is not None:
                h = self.tok_embeddings(tokens)
                tensor_logging.log_fwd_bwd_stats(self, input=h)
                # The model owns the encoder: when padded pixel_values are passed,
                # run encoder->adapter here to produce the features for injection.
                # The placeholder mask is derived from tokens + special_tokens.
                # TODO: Video is not implemented in the training forward. The
                # encoder itself is video-capable; this path just lacks the
                # video-specific glue that the image path (above) doesn't need:
                #   1. Temporal frame packing -- group `patch_temporal` frames per
                #      patch.
                #   2. Spatial avg-pool compression between encoder and adapter
                #      (pool_factor from compression_ratio); the image path goes
                #      encoder->adapter directly with no compression.
                #   3. Video grid sizing (with compression_ratio / max_num_tokens)
                #      vs image grid sizing.
                #   4. A separate video placeholder token + mask instead of the
                #      image_id used below.
                if pixel_values_videos is not None or grid_thw_videos is not None:
                    raise NotImplementedError(
                        "Muse Glimmer vision encoder does not support video inputs."
                    )
                if pixel_values is not None:
                    if self.vision_encoder is None:
                        raise ValueError(
                            "pixel_values were provided but the model has no "
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
                    vision_mask = tokens == special_tokens["image_id"]
                    vision_features = self._get_vision_features(pixel_values, grid_thw)
                    h = self._inject_vision(h, vision_features, vision_mask)
            else:
                h = tokens

        if get_spmd_backend() == "spmd_types":
            # The scatter restores a token-aligned tensor, so text-model DP
            # resumes as global batch sharding after the multimodal region.

            # NOTE: Under PP + TP + SP, this is not a truly correct typeing.
            # In a later PP stage, h arrives as TP sharded activation,
            # so annotating it as R on TP is wrong. However,
            # PP + spmd typechecking is not supported currently and
            # the asserted type here is not used anywhere.
            spmd.assert_type(h, {"dp": spmd.S(0), "tp": spmd.R})

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h

        # _skip_lm_head is an attribute (not a kwarg) because PP backward calls
        # .requires_grad on all stage inputs, which fails on bool kwargs.
        if self._skip_lm_head:
            return h
        if self.lm_head is None:
            return h
        output = self.lm_head(h)
        tensor_logging.log_fwd_bwd_stats(self.lm_head, output=output)
        return output

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
        seq_len = positions.shape[1]
        B = positions.shape[0]
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
                B,
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
