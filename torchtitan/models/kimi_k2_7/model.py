# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reference (SGLang):
https://github.com/sgl-project/sglang/blob/e0c0c0a45cb1bda90392bfa2bba4184f5b0638a0/python/sglang/srt/models/kimi_k25.py
"""

# Tensor dimensions: T = text tokens, X = input patches, V = vision tokens,
# D = hidden size, P = flattened patch size, N = media items.

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import (
    decoder_input_sharding,
    token_id_placement,
)
from torchtitan.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
    multimodal_context,
)
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.deepseek_v3.model import (
    DeepSeekV3Model,
    get_deepseek_v3_nparams_and_flops as get_kimi_k2_7_nparams_and_flops,
)

from .sharding import set_kimi_k2_5_sharding_config
from .vision_encoder import KimiK25VisionEncoder


class KimiK25Model(DeepSeekV3Model):
    """Kimi K2.5: DeepSeekV3 language model with a MoonViT3d vision encoder.

    Forward pass flow::

        forward(tokens, pixel_values[/videos], grid_thw, ...)
          |
          +-- tok_embeddings(tokens)               -> text embeddings
          +-- vision_encoder(pixels)               -> packed vision features
          +-- gather by vision-bank indices        -> multimodal embeddings
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

            # Decoder.Config validates the text attention heads. Vision attention
            # is also head-sharded, so validate its head count independently.
            tp = parallelism.tensor_parallel_degree
            if (
                tp > 1
                and self.vision_encoder is not None
                and self.vision_encoder.num_heads % tp != 0
            ):
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"vision num_heads ({self.vision_encoder.num_heads})."
                )

            set_kimi_k2_5_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            kimi_model = cast("KimiK25Model", model)
            return get_kimi_k2_7_nparams_and_flops(
                self,
                model,
                seq_len,
                modules_excluded_from_active_params=(kimi_model.vision_encoder,),
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build masks, CP-shard, SPMD-wrap, and return the batch."""
        # Function-local import avoids a circular import.
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        batch: dict[str, Any] = dict(input_dict)
        pixel_values = batch.get("pixel_values")
        grid_thw = batch.get("grid_thw")
        pixel_values_videos = batch.get("pixel_values_videos")
        grid_thw_videos = batch.get("grid_thw_videos")
        special_tokens = batch.pop("special_tokens", None)

        if self.tok_embeddings is not None:
            placeholder_id = None
            if pixel_values is not None and grid_thw is not None:
                placeholder_id = special_tokens["image_id"]
            elif pixel_values_videos is not None and grid_thw_videos is not None:
                placeholder_id = special_tokens["video_id"]
            if placeholder_id is not None:
                batch["vision_bank_indices_T"] = build_vision_bank_indices(
                    batch["input"], placeholder_id=placeholder_id
                )

        positions = batch.get("positions", None)
        if positions is not None:
            inner = getattr(self.config.first_attention, "inner_attention", None)
            if isinstance(inner, (FlexAttention.Config, VarlenAttention.Config)):
                batch["attention_masks"] = self.get_attention_masks(positions=positions)

        input_sharding = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=True),
            "vision_bank_indices_T": token_id_placement(),
        }
        if parallel_dims.cp_enabled:
            batch = prepare_context_parallel_input(
                batch,
                input_sharding,
                parallel_dims.get_mesh("cp"),
                parallelism.context_parallel_load_balancer,
                parallelism.context_parallel_ptrr_mask_key,
            )
        if parallelism.spmd_backend == "spmd_types":
            batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)

        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def _prepare_multimodal_embeds(
        self,
        h_TD: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        vision_bank_indices_T: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode one media stream and gather it into token embeddings."""
        modalities = []
        if pixel_values is not None and grid_thw is not None:
            modalities.append((pixel_values, grid_thw))
        if pixel_values_videos is not None and grid_thw_videos is not None:
            modalities.append((pixel_values_videos, grid_thw_videos))

        if not modalities:
            return h_TD
        assert len(modalities) == 1, "mixed image+video batches not yet supported"
        pixels_XP, grid_N3 = modalities[0]

        assert self.vision_encoder is not None
        assert vision_bank_indices_T is not None

        pixels_XP = pixels_XP.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_bank_VD = self.vision_encoder(pixels_XP, grid_thw=grid_N3)
        return gather_vision_embeds(
            h_TD,
            vision_bank_VD=vision_bank_VD,
            vision_bank_indices_T=vision_bank_indices_T,
        )

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        vision_bank_indices_T: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """Forward pass for Kimi K2.5.

        Images and videos share one unified ``<|media_pad|>`` placeholder.

        Args:
            tokens: ``(num_tokens,)`` packed token IDs.
            pixel_values: ``(total_num_patches, patch_dim)`` packed image
                patches, or None for text-only / video-only batches.
            grid_thw: (num_images, 3) patch counts ``[t, h, w]`` per image.
            pixel_values_videos: Packed video patches, or None (mixing with
                ``pixel_values`` in one batch is not yet supported).
            grid_thw_videos: (num_videos, 3) patch counts per video.
            vision_bank_indices_T: Packed vision-bank row for each placeholder
                token, or -1 for text tokens.
            attention_masks: Decoder attention masks.
            positions: Per-token position IDs for packed sequences.

        Returns:
            ``(num_tokens, vocab_size)`` logits.
        """
        if self.tok_embeddings is not None:
            h_TD = self.tok_embeddings(tokens)
            with multimodal_context():
                h_TD = self._prepare_multimodal_embeds(
                    h_TD,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw_videos=grid_thw_videos,
                    vision_bank_indices_T=vision_bank_indices_T,
                )
        else:
            h_TD = tokens

        for layer in self.layers.values():
            h_TD = layer(h_TD, attention_masks, positions)

        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD
