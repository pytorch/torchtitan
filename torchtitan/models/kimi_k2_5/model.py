# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reference (SGLang):
https://github.com/sgl-project/sglang/blob/e0c0c0a45cb1bda90392bfa2bba4184f5b0638a0/python/sglang/srt/models/kimi_k25.py
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

from .sharding import set_kimi_k2_5_sharding_config
from .vision_encoder import KimiK25VisionEncoder


class KimiK25Model(DeepSeekV3Model):
    """Kimi K2.5: DeepSeekV3 language model with a MoonViT3d vision encoder.

    Forward pass flow::

        forward(tokens, pixel_values[/videos], grid_thw, ...)
          |
          +-- tok_embeddings(tokens)               -> text embeddings
          +-- vision_encoder(pixels)               -> padded vision features
          +-- scatter at vision placeholder runs   -> multimodal embeddings
          +-- decoder layers (MLA + MoE)           -> hidden states
          +-- norm -> lm_head                      -> logits
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        vision_encoder: KimiK25VisionEncoder.Config | None = None

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            # TP must divide the attention head count.
            tp = parallelism.tensor_parallel_degree
            if tp > 1 and self.layers[0].attention.n_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"n_heads ({self.layers[0].attention.n_heads})."
                )

            set_kimi_k2_5_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int],
    ) -> torch.Tensor:
        """Embed tokens, run the vision encoder, scatter features into text.

        With kimi's single unified placeholder, a one-modality batch's runs map
        to visual items in order. Mixing images and videos in one batch is not
        yet supported (see the TODO below).
        """
        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        modalities = []
        if pixel_values is not None and grid_thw is not None:
            modalities.append((pixel_values, grid_thw))
        if pixel_values_videos is not None and grid_thw_videos is not None:
            modalities.append((pixel_values_videos, grid_thw_videos))

        if not modalities:
            return inputs_embeds
        # TODO: support mixed image+video batches. Upstream fix: when
        # image_id == video_id, emit one document-ordered vision stream so the
        # runs stay modality-agnostic and this branch goes away.
        assert len(modalities) == 1, "mixed image+video batches not yet supported"
        pixels, grid = modalities[0]

        placeholder_id = special_tokens["image_id"]
        assert placeholder_id == special_tokens["video_id"]

        # Patches arrive float32; match the encoder's compute dtype for the matmul.
        pixels = pixels.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixels, grid_thw=grid)
        # MoonViT collapses time (temporal pooling) and merges 2x2 spatially, so
        # the token count is (h/kh)*(w/kw), independent of t.
        kh, kw = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid[:, 1] // kh) * (grid[:, 2] // kw)
        vision_positions = get_vision_positions(
            tokens, num_tokens_per_item, placeholder_id
        )
        if vision_positions:
            inputs_embeds = scatter_vision_embeds(
                inputs_embeds,
                vision_embeds=vision_embeds,
                vision_positions=vision_positions,
            )
        return inputs_embeds

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """Forward pass for Kimi K2.5.

        Images and videos share one unified ``<|media_pad|>`` placeholder.

        Args:
            tokens: (batch, seq_len) token IDs.
            pixel_values: (num_images, max_num_patch, patch_dim) padded image
                patches, or None for text-only / video-only batches.
            grid_thw: (num_images, 3) patch counts ``[t, h, w]`` per image.
            pixel_values_videos: padded video patches, or None (mixing with
                ``pixel_values`` in one batch is not yet supported).
            grid_thw_videos: (num_videos, 3) patch counts per video.
            special_tokens: tokenizer-resolved ``image_id``/``video_id``;
                required for image/video batches, None for text-only.
            attention_masks: Decoder attention masks.
            positions: Per-token position IDs for packed sequences.

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        if self.tok_embeddings is not None:
            x = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                pixel_values_videos=pixel_values_videos,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,  # pyrefly: ignore [bad-argument-type]
            )
        else:
            x = tokens

        for layer in self.layers.values():
            x = layer(x, attention_masks, positions)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
