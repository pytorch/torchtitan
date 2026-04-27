# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from dataclasses import dataclass, field

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger

from .vision_encoder import Qwen3VLVisionEncoder


class Qwen3VLModel(Qwen3Model):
    """Qwen3-VL: A Vision-Language Model based on Qwen3.

    Combines the Qwen3 language model with a Vision Transformer encoder
    for multimodal understanding of images and videos.

    Key features:
    - DeepStack: Vision embeddings from intermediate ViT layers are added to
      early LLM hidden states for better multimodal understanding
    - MRoPE: Multi-dimensional RoPE with interleaved temporal/height/width
      position encoding for vision tokens

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)           → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)  → vision embeddings + deepstack features
          │    │    └─ vision_encoder(pixel_values)  → per-layer features, merge patches
          │    ├─ _scatter_vision_embeds             → replace embeddings at <img>/<vid> positions with vision embeddings
          │    └─ combine image/video masks and deepstack features
          │
          ├─ _compute_mrope_freqs                   → build 3D position IDs, interleave into freqs_cis
          │
          └─ transformer layers
               └─ for each layer:
                    ├─ layer(hidden_states, freqs_cis, masks, positions)
                    └─ _deepstack_process: add intermediate ViT features to hidden states at vision positions (early layers only)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        vision_encoder: Qwen3VLVisionEncoder.Config

        # MRoPE section sizes for interleaved multi-dimensional RoPE
        # [temporal, height, width] - controls how position dimensions are interleaved
        mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                n_heads = self.layers[0].attention.n_heads
                n_kv_heads = self.layers[0].attention.n_kv_heads or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layers[0].attention, GQAttention.Config)
            assert self.layers[0].attention.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                2 * self.layers[0].attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)

        self.vision_encoder = config.vision_encoder.build()

        self.mrope_section = config.mrope_section
        self.spatial_merge_size = config.vision_encoder.spatial_merge_size

        # Number of early LLM layers that receive DeepStack visual features
        self.num_deepstack_layers = len(config.vision_encoder.deepstack_visual_indices)

    def _compute_mrope_freqs(
        self,
        tokens: torch.Tensor,
        *,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int],
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build 3D position IDs and compute interleaved MRoPE cos/sin frequencies.

        Constructs (temporal, height, width) position IDs for each token, then
        looks up cos/sin from the 1D RoPE table and overwrites H/W-assigned dims
        with their own position lookups.

        Args:
            tokens: (batch, seq_len) token IDs
            grid_thw: (num_images, 3) grid dimensions for images
            grid_thw_videos: (num_videos, 3) grid dimensions for videos
            special_tokens: Special token definitions
            positions: (batch, seq_len) per-token position IDs for packed
                sequences. When provided, document boundaries are detected
                where positions reset (positions[t] < positions[t-1]), and
                pos_id_offset resets to 0 at each boundary

        Returns:
            (batch, seq_len, 1, head_dim * 2) pre-computed MRoPE cos/sin
        """
        # --- Build 3D position IDs ---

        # Expand each video [T, H, W] into T rows of [1, H, W] so that
        # each frame is treated like an image in the MRoPE code below
        # Temporal position comes from frame ordering in the sequence
        if grid_thw_videos is not None:
            grid_thw_videos = torch.repeat_interleave(
                grid_thw_videos, grid_thw_videos[:, 0], dim=0
            )
            grid_thw_videos[:, 0] = 1

        spatial_merge_size = self.spatial_merge_size
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        batch_size, seq_len = tokens.shape
        position_ids = torch.zeros(
            3,
            batch_size,
            seq_len,
            dtype=tokens.dtype,
            device=tokens.device,
        )

        # Precompute document boundaries and vision token positions across batch
        if positions is not None:
            resets = positions[:, 1:] < positions[:, :-1]  # (batch, seq_len-1)
        # Find the first token of each consecutive vision region (image or video)
        # E.g. for [text, img, img, img, text, vid, vid] → positions [1, 5]
        vision_mask = (tokens == image_token_id) | (tokens == video_token_id)
        prev_vision = torch.cat(
            [torch.zeros_like(vision_mask[:, :1]), vision_mask[:, :-1]], dim=1
        )
        batch_vision_starts = vision_mask & ~prev_vision  # (batch, seq_len)
        # Cache vision grid indices by shape to avoid redundant construction
        grid_cache: dict[tuple[int, int, int], torch.Tensor] = {}

        image_index, video_index = 0, 0
        # Build MRoPE 3D position IDs per sample
        # With sample packing, each sample may contain multiple documents
        for sample_i in range(batch_size):
            llm_pos_ids_list: list[torch.Tensor] = []

            if positions is not None:
                # Detect document boundaries within one packed sample
                # pyrefly: ignore [unbound-name]
                reset_indices = torch.where(resets[sample_i])[0] + 1
                doc_starts = [0] + reset_indices.tolist()
                doc_ranges = [
                    (
                        doc_starts[d],
                        doc_starts[d + 1] if d + 1 < len(doc_starts) else seq_len,
                    )
                    for d in range(len(doc_starts))
                ]
            else:
                doc_ranges = [(0, seq_len)]

            sample_tokens = tokens[sample_i]
            sample_vision_starts = torch.where(batch_vision_starts[sample_i])[
                0
            ].tolist()
            vision_start_index = 0

            for doc_start, doc_end in doc_ranges:
                doc_pos_ids_list: list[torch.Tensor] = []

                # Advance pointer to collect vision region starts in this document
                doc_vision_starts: list[int] = []
                while (
                    vision_start_index < len(sample_vision_starts)
                    and sample_vision_starts[vision_start_index] < doc_end
                ):
                    doc_vision_starts.append(sample_vision_starts[vision_start_index])
                    vision_start_index += 1

                # Process [text tokens][vision tokens] pairs within this document
                pair_cursor = doc_start
                for vision_start in doc_vision_starts:
                    if sample_tokens[vision_start] == image_token_id:
                        # pyrefly: ignore [unsupported-operation]
                        t, h, w = grid_thw[image_index]
                        image_index += 1
                    else:
                        # pyrefly: ignore [unsupported-operation]
                        t, h, w = grid_thw_videos[video_index]
                        video_index += 1

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = vision_start - pair_cursor

                    # pos_id_offset may differ from pair_cursor due to compact
                    # spatial position IDs for vision regions
                    pos_id_offset = (
                        doc_pos_ids_list[-1].max() + 1
                        if len(doc_pos_ids_list) > 0
                        else 0
                    )
                    # [text tokens] — sequential positions, identical on all 3 axes
                    doc_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                    )
                    # [vision tokens] — 3D grid positions (T, H, W)
                    grid_key = (llm_grid_t, llm_grid_h, llm_grid_w)
                    if grid_key not in grid_cache:
                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            # pyrefly: ignore [no-matching-overload]
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                        h_index = (
                            torch.arange(llm_grid_h)
                            .view(1, -1, 1)
                            # pyrefly: ignore [no-matching-overload]
                            .expand(llm_grid_t, -1, llm_grid_w)
                            .flatten()
                        )
                        w_index = (
                            torch.arange(llm_grid_w)
                            .view(1, 1, -1)
                            # pyrefly: ignore [no-matching-overload]
                            .expand(llm_grid_t, llm_grid_h, -1)
                            .flatten()
                        )
                        # pyrefly: ignore [unsupported-operation]
                        grid_cache[grid_key] = torch.stack([t_index, h_index, w_index])
                    doc_pos_ids_list.append(
                        # pyrefly: ignore [bad-index]
                        grid_cache[grid_key]
                        + text_len
                        + pos_id_offset
                    )
                    pair_cursor = vision_start + llm_grid_t * llm_grid_h * llm_grid_w

                # Trailing [text tokens] after the last [text tokens][vision tokens] pair
                if pair_cursor < doc_end:
                    pos_id_offset = (
                        doc_pos_ids_list[-1].max() + 1
                        if len(doc_pos_ids_list) > 0
                        else 0
                    )
                    text_len = doc_end - pair_cursor
                    doc_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                    )

                llm_pos_ids_list.extend(doc_pos_ids_list)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[:, sample_i, :] = llm_positions.to(position_ids.device)

        # --- Compute interleaved MRoPE cos/sin from position IDs ---

        freqs_cis = self.freqs_cis
        head_dim = freqs_cis.shape[-1] // 2
        cos_cache = freqs_cis[:, :head_dim]
        sin_cache = freqs_cis[:, head_dim:]

        # Initialize with temporal positions, then overwrite H/W slices
        t_pos = position_ids[0].long()
        mrope_cos = cos_cache[t_pos]
        mrope_sin = sin_cache[t_pos]

        # Overwrite H and W slices with their own position lookups
        # Both halves of head_dim must be updated (head_dim = cat([freqs, freqs]))
        half = head_dim // 2
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            low = torch.arange(offset, length, 3, device=freqs_cis.device)
            col_indices = torch.cat([low, low + half])
            dim_pos = position_ids[dim].long()  # (batch, seq_len)
            mrope_cos[..., col_indices] = cos_cache[:, col_indices][dim_pos]
            mrope_sin[..., col_indices] = sin_cache[:, col_indices][dim_pos]

        return torch.cat([mrope_cos, mrope_sin], dim=-1).unsqueeze(2)

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run vision encoder and extract valid (non-padding) embeddings.

        Works for both images and videos — the ViT processes them identically.

        Args:
            pixel_values: Padded patches (num_items, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            vision_embeds: (total_valid_tokens, dim) concatenated valid tokens
            deepstack_embeds: List of (total_valid_tokens, dim) per intermediate layer
        """
        # Cast pixel values (float32 from data pipeline) to match the vision
        # encoder's param_dtype set by FSDP2's MixedPrecisionPolicy
        pixel_values = pixel_values.to(
            self.vision_encoder.patch_embed.proj.weight.dtype
        )
        merged_embeds, deepstack_features = self.vision_encoder(
            pixel_values, grid_thw=grid_thw
        )

        # Build mask to extract valid tokens (remove padding) across all items
        # merged_embeds: (num_items, max_tokens, dim) — padded to max_tokens
        # valid_mask: (num_items, max_tokens) bool — True for non-padding tokens
        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit  # (num_items,)
        max_tokens = merged_embeds.shape[1]
        valid_mask = torch.arange(max_tokens, device=merged_embeds.device).unsqueeze(
            0
        ) < num_tokens_per_item.unsqueeze(1)

        # Flatten valid tokens: (num_items, max_tokens, dim) → (total_valid_tokens, dim)
        vision_embeds = merged_embeds[valid_mask]
        deepstack_embeds = [ds_feat[valid_mask] for ds_feat in deepstack_features]

        return vision_embeds, deepstack_embeds

    def _scatter_vision_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        tokens: torch.Tensor,
        vision_embeds: torch.Tensor,
        vision_token_id: int,
    ) -> tuple[torch.Tensor, bool]:
        """Scatter vision embeddings into the text embeddings at placeholder positions.

        Args:
            inputs_embeds: Text embeddings (batch, seq_len, dim)
            tokens: Token IDs (batch, seq_len)
            vision_embeds: Vision embeddings to scatter (num_vision_tokens, dim)
            vision_token_id: Token ID for vision placeholder (image or video)

        Returns:
            Tuple of (updated embeddings, success flag)
        """
        vision_mask = tokens == vision_token_id
        num_placeholder_tokens = vision_mask.sum().item()
        num_vision_tokens = vision_embeds.shape[0]

        if num_placeholder_tokens == 0:
            # No placeholder tokens in input, skip scatter
            return inputs_embeds, False

        if num_placeholder_tokens != num_vision_tokens:
            raise ValueError(
                f"Number of vision placeholder tokens ({num_placeholder_tokens}) "
                f"does not match number of vision tokens ({num_vision_tokens}). "
                f"Vision token ID: {vision_token_id}"
            )

        vision_mask_expanded = vision_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            vision_mask_expanded,
            vision_embeds,
        )
        return inputs_embeds, True

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        *,
        vision_pos_masks: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Add vision embeddings to hidden states at vision token positions.

        This implements the DeepStack mechanism where intermediate vision
        embeddings are added to the LLM hidden states.

        Args:
            hidden_states: LLM hidden states (batch, seq_len, dim)
            vision_pos_masks: Boolean mask for vision positions (batch, seq_len)
            vision_embeds: Vision embeddings to add (num_vision_tokens, dim)

        Returns:
            Updated hidden states
        """
        num_mask_positions = vision_pos_masks.sum().item()
        num_vision_embeds = vision_embeds.shape[0]

        if num_mask_positions == 0:
            # No vision positions, skip DeepStack
            return hidden_states

        if num_mask_positions != num_vision_embeds:
            raise ValueError(
                f"DeepStack size mismatch: {num_mask_positions} vision mask positions "
                f"but {num_vision_embeds} vision embeddings."
            )

        # TODO(@shuhuayu): Replace boolean indexing with DTensor-compatible ops
        # so this clone() is no longer needed.
        # clone() is needed because hidden_states may be a view (from
        # PrepareModuleInputOutput's to_local()) and the in-place boolean
        # scatter below conflicts with autograd's custom backward on views.
        hidden_states = hidden_states.clone()

        hidden_states[vision_pos_masks] = (
            hidden_states[vision_pos_masks] + vision_embeds
        )
        return hidden_states

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Embed tokens, run vision encoder, scatter vision into text, prepare DeepStack.

        Args:
            tokens: Input token IDs (batch_size, seq_len)
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            inputs_embeds: (batch, seq_len, dim) with vision tokens scattered in
            vision_pos_masks: (batch, seq_len) bool mask or None
            deepstack_vision_embeds: list of (num_vision_tokens, dim) or None
        """
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        # Process image inputs
        image_mask = None
        deepstack_image_embeds = None
        if pixel_values is not None and grid_thw is not None:
            image_embeds, deepstack_image_embeds = self._get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            inputs_embeds, image_scatter_success = self._scatter_vision_embeds(
                inputs_embeds,
                tokens=tokens,
                vision_embeds=image_embeds,
                vision_token_id=image_token_id,
            )
            if image_scatter_success:
                image_mask = tokens == image_token_id
            else:
                deepstack_image_embeds = None

        # Process video inputs
        video_mask = None
        deepstack_video_embeds = None
        if pixel_values_videos is not None and grid_thw_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            inputs_embeds, video_scatter_success = self._scatter_vision_embeds(
                inputs_embeds,
                tokens=tokens,
                vision_embeds=video_embeds,
                vision_token_id=video_token_id,
            )
            if video_scatter_success:
                video_mask = tokens == video_token_id
            else:
                deepstack_video_embeds = None

        # Prepare DeepStack visual position masks and embeddings
        vision_pos_masks = None
        deepstack_vision_embeds = None
        if image_mask is not None and video_mask is not None:
            vision_pos_masks = image_mask | video_mask
            deepstack_vision_embeds = []
            image_mask_joint = image_mask[vision_pos_masks]
            video_mask_joint = video_mask[vision_pos_masks]
            # pyrefly: ignore [no-matching-overload]
            for image_embed, video_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = image_embed.new_zeros(
                    vision_pos_masks.sum(), image_embed.shape[-1]
                )
                embed_joint[image_mask_joint] = image_embed
                embed_joint[video_mask_joint] = video_embed
                deepstack_vision_embeds.append(embed_joint)
        elif image_mask is not None:
            vision_pos_masks = image_mask
            deepstack_vision_embeds = deepstack_image_embeds
        elif video_mask is not None:
            vision_pos_masks = video_mask
            deepstack_vision_embeds = deepstack_video_embeds

        return inputs_embeds, vision_pos_masks, deepstack_vision_embeds

    def forward(  # pyrefly: ignore [bad-override, bad-param-name-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        special_tokens: dict[str, int],
    ):
        """Forward pass of Qwen3-VL.

        Args:
            tokens: Input token IDs (batch_size, seq_len)
            pixel_values: Flattened image patches (num_images, max_num_patches, patch_dim)
            pixel_values_videos: Flattened video patches (num_videos, max_num_patches, patch_dim)
            grid_thw: Grid dimensions for images (num_images, 3)
            grid_thw_videos: Grid dimensions for videos (num_videos, 3)
            attention_masks: Attention masks for block_causal / flex attention
            positions: Per-token position IDs (batch_size, seq_len) for packed sequences.
                Each document's positions reset to 0. None means sequential positions.
            special_tokens: Special token definitions

        Returns:
            Output logits (batch_size, seq_len, vocab_size)
        """
        (
            inputs_embeds,
            vision_pos_masks,
            deepstack_vision_embeds,
        ) = self._prepare_multimodal_embeds(
            tokens,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            grid_thw=grid_thw,
            grid_thw_videos=grid_thw_videos,
            special_tokens=special_tokens,
        )

        # Compute MRoPE freqs when vision inputs are present
        if grid_thw is not None or grid_thw_videos is not None:
            # Per-position freqs_cis with 3D (T, H, W) positions baked in
            freqs_cis = self._compute_mrope_freqs(
                tokens,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,
                positions=positions,
            )
        else:
            # Standard freqs_cis indexed by positions in each layer
            freqs_cis = self.freqs_cis

        # Apply transformer layers with DeepStack
        hidden_states = inputs_embeds
        for layer_idx, layer in self.layers.items():
            hidden_states = layer(hidden_states, freqs_cis, attention_masks, positions)

            # Apply DeepStack: add visual features to early layer hidden states
            layer_idx_int = int(layer_idx)
            if (
                layer_idx_int < self.num_deepstack_layers
                and deepstack_vision_embeds is not None
                and vision_pos_masks is not None
                and layer_idx_int < len(deepstack_vision_embeds)
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    vision_pos_masks=vision_pos_masks,
                    vision_embeds=deepstack_vision_embeds[layer_idx_int],
                )

        hidden_states = (
            self.norm(hidden_states) if self.norm is not None else hidden_states
        )
        if getattr(self, "_skip_lm_head", False):
            return hidden_states
        output = (
            self.lm_head(hidden_states) if self.lm_head is not None else hidden_states
        )
        return output
