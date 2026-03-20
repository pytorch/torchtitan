# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL Model implementation.

This module combines the Qwen3 LLM with the Qwen3-VL Vision Encoder
to create a multimodal vision-language model with DeepStack support.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from torchtitan.hf_datasets import SpecialTokens
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger

from .vision_encoder import Qwen3VLVisionEncoder


class Qwen3VLModel(Qwen3Model):
    """
    Qwen3Model: The backbone language model for Qwen3-VL.

    Qwen3-VL: A Vision-Language Model based on Qwen3.

    Combines the Qwen3 language model with a Vision Transformer encoder
    for multimodal understanding of images and videos.

    Key features:
    - DeepStack: Visual features from intermediate ViT layers are added to
      early LLM hidden states for better multimodal understanding
    - MRoPE: Multi-dimensional RoPE with interleaved temporal/height/width
      position encoding for vision tokens
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        # Vision encoder configuration
        vision_encoder: Qwen3VLVisionEncoder.Config = field(
            default_factory=Qwen3VLVisionEncoder.Config
        )

        # MRoPE section sizes for interleaved multi-dimensional RoPE
        # [temporal, height, width] - controls how position dimensions are interleaved
        mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

        # Vision-language specific token IDs
        image_token_id: int = 151655
        video_token_id: int = 151656
        vision_start_token_id: int = 151652
        vision_end_token_id: int = 151653

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
            import dataclasses as _dc

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if self.layer.moe is not None:
                self.layer.moe.router._debug_force_load_balance = (
                    debug.moe_force_load_balance
                )

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA and FlexAttention."
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    f"Varlen attention is not supported with CP."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layer.attention, GQAttention.Config)
            assert self.layer.attention.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * self.layer.attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        # Initialize the base Qwen3 model
        super().__init__(config)

        # Vision encoder
        self.vision_encoder = Qwen3VLVisionEncoder(config.vision_encoder)

        # Store special token IDs for vision
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.vision_start_token_id = config.vision_start_token_id
        self.vision_end_token_id = config.vision_end_token_id

        # MRoPE section for interleaved multi-dimensional RoPE
        self.mrope_section = config.mrope_section

        # Number of early LLM layers that receive DeepStack visual features
        self.num_deepstack_layers = len(config.vision_encoder.deepstack_visual_indices)

    def _compute_mrope_freqs(
        self,
        input_ids: torch.Tensor,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: SpecialTokens | None,
    ) -> torch.Tensor:
        """Build 3D position IDs and compute interleaved MRoPE cos/sin frequencies.

        Constructs (temporal, height, width) position IDs for each token, then
        looks up cos/sin from the 1D RoPE table and overwrites H/W-assigned dims
        with their own position lookups.

        Args:
            input_ids: (batch, seq_len) token IDs.
            grid_thw: (num_images, 3) grid dimensions for images.
            grid_thw_videos: (num_videos, 3) grid dimensions for videos.
            special_tokens: Special token definitions.

        Returns:
            (batch, seq_len, 1, head_dim * 2) pre-computed MRoPE cos/sin.
        """
        # --- Step 1: Build 3D position IDs ---

        # For videos, split by temporal dimension (timestamps separate frames)
        if grid_thw_videos is not None:
            # pyrefly: ignore [no-matching-overload]
            grid_thw_videos = torch.repeat_interleave(
                grid_thw_videos, grid_thw_videos[:, 0], dim=0
            )
            grid_thw_videos[:, 0] = 1

        spatial_merge_size = self.vision_encoder.spatial_merge_size
        if special_tokens is not None:
            image_token_id = special_tokens.img_id
            video_token_id = special_tokens.vid_id
            vision_start_token_id = special_tokens.vision_start_id
        else:
            image_token_id = self.image_token_id
            video_token_id = self.video_token_id
            vision_start_token_id = self.vision_start_token_id

        batch_size, seq_len = input_ids.shape
        position_ids = torch.ones(
            3,
            batch_size,
            seq_len,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        # Find position ids for each sample in the batch
        # Process each sample in the format of [text][vision][text][vision]...
        for i in range(batch_size):
            tokens = input_ids[i]
            vision_start_indices = torch.argwhere(
                tokens == vision_start_token_id
            ).squeeze(1)
            vision_tokens = tokens[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum().item()
            video_nums = (vision_tokens == video_token_id).sum().item()
            input_tokens = tokens.tolist()
            llm_pos_ids_list: list = []
            # Iterate segment by segment
            # seg_start: token index in the sequence (absolute position in input_tokens)
            seg_start = 0
            remain_images, remain_videos = image_nums, video_nums
            # pyrefly: ignore [bad-assignment, no-matching-overload]
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    next_image_pos = input_tokens.index(image_token_id, seg_start)
                else:
                    next_image_pos = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    next_video_pos = input_tokens.index(video_token_id, seg_start)
                else:
                    next_video_pos = len(input_tokens) + 1

                if next_image_pos < next_video_pos:
                    # pyrefly: ignore [unsupported-operation]
                    t, h, w = grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    next_vision_pos = next_image_pos
                else:
                    # pyrefly: ignore [unsupported-operation]
                    t, h, w = grid_thw_videos[video_index]
                    video_index += 1
                    remain_videos -= 1
                    next_vision_pos = next_video_pos

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = next_vision_pos - seg_start

                # Each segment's position IDs start at max(previous segment) + 1
                # pos_id_offset: position ID for MRoPE (may differ from seg_start due
                #   to compact spatial position IDs for vision tokens)
                pos_id_offset = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                # Add position IDs for [text]
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                )
                # Add position IDs for [vision]
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
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + pos_id_offset
                )
                seg_start = next_vision_pos + llm_grid_t * llm_grid_h * llm_grid_w

            # After [text][vision] repetitions, if there remains a [text] segment
            if seg_start < len(input_tokens):
                pos_id_offset = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - seg_start
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + pos_id_offset
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[:, i, :] = llm_positions.to(position_ids.device)

        # --- Step 2: Compute interleaved MRoPE cos/sin from position IDs ---

        freqs_cis = self.freqs_cis
        head_dim_x2 = freqs_cis.shape[-1]
        head_dim = head_dim_x2 // 2
        cos_cache = freqs_cis[:, :head_dim]  # (max_seq_len, head_dim)
        sin_cache = freqs_cis[:, head_dim:]  # (max_seq_len, head_dim)

        # Start with T positions for all dims (T uses the most dims)
        t_pos = position_ids[0].long()  # (batch, seq_len)
        mrope_cos = cos_cache[t_pos]  # (batch, seq_len, head_dim)
        mrope_sin = sin_cache[t_pos]  # (batch, seq_len, head_dim)

        # Overwrite H and W slices with their own position lookups.
        # Only gather the specific columns each dimension needs.
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            col_indices = torch.arange(offset, length, 3, device=freqs_cis.device)
            dim_pos = position_ids[dim].long()  # (batch, seq_len)
            mrope_cos[..., col_indices] = cos_cache[:, col_indices][dim_pos]
            mrope_sin[..., col_indices] = sin_cache[:, col_indices][dim_pos]

        # Concatenate and add n_heads dimension: (batch, seq_len, 1, head_dim*2)
        # pyrefly: ignore [bad-return]
        return torch.cat([mrope_cos, mrope_sin], dim=-1).unsqueeze(2)

    def init_weights(
        self,
        *,
        buffer_device: torch.device | None = None,
        **kwargs,
    ):
        """Initialize all weights."""
        super().init_weights(buffer_device=buffer_device, **kwargs)

        if self.vision_encoder is not None:
            self.vision_encoder.init_weights()

    def _get_visual_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract visual features from the vision encoder.

        Works for both images and videos — the ViT processes them identically.

        Args:
            pixel_values: Padded patches (num_items, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            visual_embeds: List of embeddings per item (only valid tokens)
            deepstack_embeds: List of DeepStack features from intermediate layers
        """
        # Cast pixel values (float32 from data pipeline) to match the vision
        # encoder's param_dtype set by FSDP2's MixedPrecisionPolicy.
        # pyrefly: ignore [bad-assignment]
        pixel_values = pixel_values.to(
            self.vision_encoder.patch_embed.proj.weight.dtype
        )
        merged_embeds, deepstack_features = self.vision_encoder(
            pixel_values, grid_thw=grid_thw
        )

        # Compute valid sequence lengths per item (after merging)
        merge_unit = self.vision_encoder.spatial_merge_unit
        seq_lens = (grid_thw.prod(-1) // merge_unit).tolist()

        # Extract only valid tokens (remove padding) per item
        visual_embeds = []
        for i, seq_len in enumerate(seq_lens):
            visual_embeds.append(merged_embeds[i, :seq_len])

        # Extract valid tokens from DeepStack features
        deepstack_embeds_extracted = []
        for ds_feat in deepstack_features:
            ds_list = []
            for i, seq_len in enumerate(seq_lens):
                ds_list.append(ds_feat[i, :seq_len])
            deepstack_embeds_extracted.append(torch.cat(ds_list, dim=0))

        # pyrefly: ignore [bad-return]
        return visual_embeds, deepstack_embeds_extracted

    def _scatter_vision_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        vision_embeds: torch.Tensor,
        img_token_id: int,
    ) -> tuple[torch.Tensor, bool]:
        """Scatter vision embeddings into the text embeddings at placeholder positions.

        Args:
            inputs_embeds: Text embeddings (batch, seq_len, dim)
            input_ids: Token IDs (batch, seq_len)
            vision_embeds: Vision embeddings to scatter (num_vision_tokens, dim)
            img_token_id: Token ID for vision placeholder

        Returns:
            Tuple of (updated embeddings, success flag)
        """
        special_mask = input_ids == img_token_id
        num_placeholder_tokens = special_mask.sum().item()
        num_vision_tokens = vision_embeds.shape[0]

        if num_placeholder_tokens == 0:
            # No placeholder tokens in input, skip scatter
            return inputs_embeds, False

        if num_placeholder_tokens != num_vision_tokens:
            raise ValueError(
                f"Number of image placeholder tokens ({num_placeholder_tokens}) "
                f"does not match number of vision tokens ({num_vision_tokens}). "
                f"Image token ID: {img_token_id}"
            )

        # pyrefly: ignore [bad-argument-type]
        special_mask_expanded = special_mask.unsqueeze(-1).expand_as(inputs_embeds)
        # pyrefly: ignore [bad-assignment]
        inputs_embeds = inputs_embeds.masked_scatter(
            # pyrefly: ignore [bad-argument-type]
            special_mask_expanded,
            # pyrefly: ignore [bad-argument-type]
            vision_embeds,
        )
        return inputs_embeds, True

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Add visual features to hidden states at vision token positions.

        This implements the DeepStack mechanism where intermediate visual
        features are added to the LLM hidden states.

        Args:
            hidden_states: LLM hidden states (batch, seq_len, dim)
            visual_pos_masks: Boolean mask for vision positions (batch, seq_len)
            visual_embeds: Visual features to add (num_visual_tokens, dim)

        Returns:
            Updated hidden states
        """
        num_mask_positions = visual_pos_masks.sum().item()
        num_visual_embeds = visual_embeds.shape[0]

        if num_mask_positions == 0:
            # No vision positions, skip DeepStack
            return hidden_states

        if num_mask_positions != num_visual_embeds:
            raise ValueError(
                f"DeepStack size mismatch: {num_mask_positions} vision mask positions "
                f"but {num_visual_embeds} visual embeddings."
            )

        # pyrefly: ignore [bad-assignment]
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = (
            hidden_states[visual_pos_masks] + visual_embeds
        )
        return hidden_states

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: SpecialTokens | None,
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
            visual_pos_masks: (batch, seq_len) bool mask or None
            deepstack_visual_embeds: list of (num_visual_tokens, dim) or None
        """
        if special_tokens is not None:
            img_token_id = special_tokens.img_id
            vid_token_id = special_tokens.vid_id
        else:
            img_token_id = self.image_token_id
            vid_token_id = self.video_token_id

        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        # Process image inputs
        image_mask = None
        deepstack_image_embeds = None
        if (
            self.vision_encoder is not None
            and pixel_values is not None
            and grid_thw is not None
        ):
            image_embeds_list, deepstack_image_embeds = self._get_visual_features(
                pixel_values, grid_thw
            )
            # pyrefly: ignore [no-matching-overload]
            image_embeds = torch.cat(image_embeds_list, dim=0)
            inputs_embeds, image_scatter_success = self._scatter_vision_embeds(
                inputs_embeds, tokens, image_embeds, img_token_id=img_token_id
            )
            if image_scatter_success:
                image_mask = tokens == img_token_id
            else:
                deepstack_image_embeds = None

        # Process video inputs
        video_mask = None
        deepstack_video_embeds = None
        if (
            self.vision_encoder is not None
            and pixel_values_videos is not None
            and grid_thw_videos is not None
        ):
            video_embeds_list, deepstack_video_embeds = self._get_visual_features(
                pixel_values_videos, grid_thw_videos
            )
            # pyrefly: ignore [no-matching-overload]
            video_embeds = torch.cat(video_embeds_list, dim=0)
            inputs_embeds, video_scatter_success = self._scatter_vision_embeds(
                inputs_embeds, tokens, video_embeds, img_token_id=vid_token_id
            )
            if video_scatter_success:
                video_mask = tokens == vid_token_id
            else:
                deepstack_video_embeds = None

        # Prepare DeepStack visual position masks and embeddings
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            # pyrefly: ignore [no-matching-overload]
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                )
                embed_joint[image_mask_joint] = img_embed
                embed_joint[video_mask_joint] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # pyrefly: ignore [bad-return]
        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    # pyrefly: ignore [bad-param-name-override]
    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: SpecialTokens | None = None,
    ):
        """
        Forward pass of Qwen3-VL.

        Args:
            tokens: Input token IDs (batch_size, seq_len).
            pixel_values: Flattened image patches (num_images, max_num_patches, patch_dim)
            pixel_values_videos: Flattened video patches (num_videos, max_num_patches, patch_dim)
            grid_thw: Grid dimensions for images (num_images, 3)
            grid_thw_videos: Grid dimensions for videos (num_videos, 3)
            special_tokens: Special token definitions

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        (
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self._prepare_multimodal_embeds(
            tokens,
            pixel_values,
            pixel_values_videos,
            grid_thw,
            grid_thw_videos,
            special_tokens,
        )

        # Compute MRoPE freqs when vision inputs are present
        if grid_thw is not None or grid_thw_videos is not None:
            freqs_cis = self._compute_mrope_freqs(
                tokens, grid_thw, grid_thw_videos, special_tokens
            )
        else:
            freqs_cis = self.freqs_cis

        # Apply transformer layers with DeepStack
        hidden_states = inputs_embeds
        for layer_idx, layer in self.layers.items():
            hidden_states = layer(hidden_states, freqs_cis, None, None)

            # Apply DeepStack: add visual features to early layer hidden states
            layer_idx_int = int(layer_idx)
            if (
                layer_idx_int < self.num_deepstack_layers
                and deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx_int < len(deepstack_visual_embeds)
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx_int],
                )

        # Final normalization and output
        hidden_states = self.norm(hidden_states)
        output = self.output(hidden_states)

        return output
