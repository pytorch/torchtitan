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

import torch
from torch import nn

from torchtitan.models.qwen3.model.model import Qwen3Model

from .args import Qwen3VLModelArgs, SpecialTokens
from .vision_encoder import Qwen3VLVisionEncoder


class Qwen3VLModel(Qwen3Model):
    """
    Qwen3-VL: A Vision-Language Model based on Qwen3.

    Combines the Qwen3 language model with a Vision Transformer encoder
    for multimodal understanding of images and videos.

    Key features:
    - DeepStack: Visual features from intermediate ViT layers are added to
      early LLM hidden states for better multimodal understanding
    - MRoPE: Multi-dimensional RoPE with interleaved temporal/height/width
      position encoding for vision tokens
    """

    def __init__(self, model_args: Qwen3VLModelArgs):
        # Initialize the base Qwen3 model
        super().__init__(model_args)
        self.model_args = model_args

        # Vision encoder
        self.visual = Qwen3VLVisionEncoder(model_args.encoder)

        # Store special token IDs for vision
        self.image_token_id = model_args.image_token_id
        self.video_token_id = model_args.video_token_id
        self.vision_start_token_id = model_args.vision_start_token_id
        self.vision_end_token_id = model_args.vision_end_token_id

        # MRoPE section for interleaved multi-dimensional RoPE
        self.mrope_section = model_args.text_config.mrope_section

        # DeepStack layer indices (which LLM layers receive visual features)
        self.deepstack_layer_indices = list(range(len(model_args.encoder.deepstack_visual_indexes)))

    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize all weights."""
        super().init_weights(buffer_device=buffer_device)
        if self.visual is not None:
            self.visual.init_weights()

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract image features from the vision encoder.

        Args:
            pixel_values: Flattened image patches (total_patches, patch_dim)
            image_grid_thw: Grid dimensions (num_images, 3) for [t, h, w]

        Returns:
            image_embeds: List of image embeddings per image
            deepstack_embeds: List of DeepStack features from intermediate layers
        """
        pixel_values = pixel_values.type(self.visual.patch_embed.proj.weight.dtype)
        merged_embeds, deepstack_features = self.visual(pixel_values, grid_thw=image_grid_thw)

        # Split by image
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_unit
        ).tolist()
        image_embeds = torch.split(merged_embeds, split_sizes)

        return list(image_embeds), deepstack_features

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract video features from the vision encoder."""
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

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

        special_mask_expanded = special_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            special_mask_expanded, vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
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
            # Size mismatch - skip DeepStack for this layer
            return hidden_states

        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + visual_embeds
        return hidden_states

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        special_tokens: SpecialTokens | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass of Qwen3-VL.

        Args:
            tokens: Input token IDs (batch, seq_len)
            pixel_values: Flattened image patches (total_patches, patch_dim)
            pixel_values_videos: Flattened video patches
            grid_thw: Grid dimensions for images (num_images, 3)
            video_grid_thw: Grid dimensions for videos (num_videos, 3)
            special_tokens: Special token definitions
            positions: Position IDs for RoPE (optional)

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
        """
        # Alias for compatibility
        image_grid_thw = grid_thw

        # Use special tokens from input if provided, otherwise fall back to model defaults
        img_token_id = special_tokens.img_id if special_tokens is not None else self.image_token_id
        vid_token_id = self.video_token_id  # Video not yet in special_tokens

        # Get token embeddings (passthrough for nonexistent layers in PP)
        inputs_embeds = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        # Process image inputs
        image_mask = None
        deepstack_image_embeds = None
        if self.visual is not None and pixel_values is not None and image_grid_thw is not None:
            image_embeds_list, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
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
        if self.visual is not None and pixel_values_videos is not None and video_grid_thw is not None:
            video_embeds_list, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
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
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1])
                embed_joint[image_mask_joint] = img_embed
                embed_joint[video_mask_joint] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Apply transformer layers with DeepStack
        hidden_states = inputs_embeds
        for layer_idx, layer in self.layers.items():
            hidden_states = layer(hidden_states, self.rope_cache, None, positions)

            # Apply DeepStack: add visual features to early layer hidden states
            layer_idx_int = int(layer_idx)
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx_int in self.deepstack_layer_indices
            ):
                ds_idx = self.deepstack_layer_indices.index(layer_idx_int)
                if ds_idx < len(deepstack_visual_embeds):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[ds_idx],
                    )

        # Final normalization and output
        hidden_states = self.norm(hidden_states) if self.norm is not None else hidden_states
        output = self.output(hidden_states) if self.output is not None else hidden_states

        return output
