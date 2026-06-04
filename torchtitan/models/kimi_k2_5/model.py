# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KimiK25Model: a Vision-Language Model combining DeepSeekV3 with MoonViT3d.

Extends ``DeepSeekV3Model`` with a MoonViT3d vision encoder and overrides the
forward pass to scatter projected vision features into the text embedding
sequence at the vision placeholder positions — the tokenizer-resolved
image/video ids under the multimodal data pipeline, else the model's
``media_placeholder_token_id``. The language model keeps DeepSeekV3's standard
1D RoPE; the 2D RoPE lives entirely inside the vision encoder.
"""

import dataclasses
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.token_dispatcher import (
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)
from torchtitan.models.deepseek_v3.model import Attention, DeepSeekV3Model
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger

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
        vision_encoder: KimiK25VisionEncoder.Config

        media_placeholder_token_id: int = 163605

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            """Sync seq-len / RoPE onto the layers, validate the parallelism
            choices, and populate sharding configs (decoder + vision encoder).
            """
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug

            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum "
                    f"{self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            # Sync rope fields onto every attention config.
            for layer_cfg in self.layers:
                assert isinstance(layer_cfg.attention, Attention.Config)
                layer_cfg.attention.rope_max_seq_len = seq_len
                layer_cfg.attention.rope_factor = self.rope.rope_factor
                layer_cfg.attention.rope_original_seq_len = self.rope.original_seq_len

            # MoE debug flag + EP-required dispatcher validation.
            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )
                    token_dispatcher_cfg = layer_cfg.moe.experts.token_dispatcher
                    if (
                        isinstance(
                            token_dispatcher_cfg,
                            (
                                DeepEPTokenDispatcher.Config,
                                HybridEPTokenDispatcher.Config,
                            ),
                        )
                        and parallelism.expert_parallel_degree == 1
                    ):
                        raise ValueError(
                            "DeepEP/HybridEP token dispatcher requires expert "
                            "parallelism (expert_parallel_degree > 1)."
                        )

            # Context Parallel is not supported for the multimodal model: vision
            # scatter needs the full sequence before CP would shard it.
            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "Context Parallel is not yet supported for Kimi K2.5."
                )

            # TP must divide the attention head count.
            tp = parallelism.tensor_parallel_degree
            if tp > 1 and self.layers[0].attention.n_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"n_heads ({self.layers[0].attention.n_heads})."
                )

            set_kimi_k2_5_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """Parameter count and FLOP estimate via the shared MoE estimator."""
            assert isinstance(self.layers[0].attention, Attention.Config)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                self.layers[0].attention.qk_nope_head_dim
                + self.layers[0].attention.qk_rope_head_dim
                + self.layers[0].attention.v_head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = config.vision_encoder.build()
        self.media_placeholder_token_id = config.media_placeholder_token_id

    def _compute_vision_positions(
        self,
        tokens: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        placeholder_id: int,
    ) -> list[tuple[int, int, int, int]]:
        """Locate each visual item's placeholder run in the token sequence.

        Args:
            tokens: (batch, seq_len) token IDs.
            num_tokens_per_item: (num_items,) valid token count per visual item.
            placeholder_id: Token id whose contiguous runs mark vision spans.

        Returns:
            ``(item_idx, sample_idx, vision_start, n_tokens)`` per item, where
            ``vision_start`` is the position of the first placeholder token of
            that item within its sample.
        """
        vision_mask = tokens == placeholder_id
        flat_mask = vision_mask.view(-1)
        prev_mask = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=flat_mask.device), flat_mask[:-1]]
        )
        region_starts = torch.where(flat_mask & ~prev_mask)[0]
        seq_len = tokens.shape[1]

        positions: list[tuple[int, int, int, int]] = []
        for i in range(num_tokens_per_item.shape[0]):
            start = int(region_starts[i].item())
            n_tokens = int(num_tokens_per_item[i].item())
            positions.append((i, start // seq_len, start % seq_len, n_tokens))
        return positions

    def _scatter_vision_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        merged_embeds: torch.Tensor,
        vision_positions: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Copy padded vision features into the text sequence at placeholders.

        Args:
            inputs_embeds: (batch, seq_len, dim) text embeddings.
            merged_embeds: (num_items, max_tokens, dim) padded vision features.
            vision_positions: from ``_compute_vision_positions``.
        """
        for item_idx, sample_idx, vision_start, n_tokens in vision_positions:
            inputs_embeds[
                sample_idx, vision_start : vision_start + n_tokens, :
            ] = merged_embeds[item_idx, :n_tokens, :].to(inputs_embeds.dtype)
        return inputs_embeds

    def _placeholder_id(
        self, special_tokens: dict[str, int] | None, key: str
    ) -> int:
        """Resolve the scatter token id.

        With the multimodal pipeline the runtime ids come from the tokenizer via
        ``special_tokens`` (tokenizer-dependent). For direct calls without it,
        fall back to the model's hard-coded ``media_placeholder_token_id``.
        """
        if special_tokens is not None and key in special_tokens:
            return int(special_tokens[key])
        return self.media_placeholder_token_id

    def _embed_and_scatter(
        self,
        inputs_embeds: torch.Tensor,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        placeholder_id: int,
    ) -> torch.Tensor:
        """Run the vision encoder on one modality and scatter into text."""
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        merged_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        # MoonViT collapses time (temporal pooling) and merges 2x2 spatially,
        # so the token count is (h/kh)*(w/kw), independent of t.
        kh, kw = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kh) * (grid_thw[:, 2] // kw)
        vision_positions = self._compute_vision_positions(
            tokens, num_tokens_per_item, placeholder_id
        )
        if vision_positions:
            inputs_embeds = self._scatter_vision_embeds(
                inputs_embeds,
                merged_embeds=merged_embeds,
                vision_positions=vision_positions,
            )
        return inputs_embeds

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> torch.Tensor:
        """Embed tokens, run the vision encoder per modality, scatter into text."""
        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        if pixel_values is not None and grid_thw is not None and grid_thw.numel() > 0:
            inputs_embeds = self._embed_and_scatter(
                inputs_embeds,
                tokens,
                pixel_values,
                grid_thw,
                self._placeholder_id(special_tokens, "image_id"),
            )

        if (
            pixel_values_videos is not None
            and grid_thw_videos is not None
            and grid_thw_videos.numel() > 0
        ):
            inputs_embeds = self._embed_and_scatter(
                inputs_embeds,
                tokens,
                pixel_values_videos,
                grid_thw_videos,
                self._placeholder_id(special_tokens, "video_id"),
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

        Args:
            tokens: (batch, seq_len) token IDs.
            pixel_values: (num_images, max_num_patch, patch_dim) padded image
                patches, or None for text-only batches.
            grid_thw: (num_images, 3) patch counts ``[t, h, w]`` per image.
            pixel_values_videos: padded video patches, or None.
            grid_thw_videos: (num_videos, 3) patch counts per video.
            special_tokens: tokenizer-resolved ids (``image_id``/``video_id``)
                from the multimodal data pipeline; None for direct calls.
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
                special_tokens=special_tokens,
            )
        else:
            x = tokens

        freqs_cis = self.freqs_cis
        for layer in self.layers.values():
            x = layer(x, freqs_cis, attention_masks, positions)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
