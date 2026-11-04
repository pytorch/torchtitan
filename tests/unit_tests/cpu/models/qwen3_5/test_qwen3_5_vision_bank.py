# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch-provided vision bank rows (RL batches encode each distinct image once).

No decoder layers, so this file runs without ``fla``.
"""

from dataclasses import replace
from typing import cast

import torch
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config, Qwen35Model


class _LinearVisionEncoder(nn.Module):
    def __init__(self, patch_dim: int, output_dim: int) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, output_dim, bias=False)
        self.spatial_merge_size = 1
        self.spatial_merge_unit = 1

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        return self.patch_embed(pixel_values)


def _embeddings_only_model() -> Qwen35Model:
    config = cast(
        Qwen35Model.Config,
        build_model_config("debugmodel", seq_len=8, parallelism=ParallelismConfig()),
    )
    config = replace(
        config,
        vocab_size=8,
        tok_embeddings=replace(config.tok_embeddings, num_embeddings=8),
        lm_head=replace(config.lm_head, out_features=8),
        layers=[],
        vision_encoder=None,
    )
    model = config.build()
    model.init_states()
    model.vision_encoder = _LinearVisionEncoder(  # pyrefly: ignore [bad-assignment]
        4, config.dim
    )
    return model


def test_bank_rows_place_one_encoded_image_on_several_placeholder_runs() -> None:
    model = _embeddings_only_model()
    image, grid = torch.randn(1, 4), torch.ones(1, 3, dtype=torch.int64)
    tokens = torch.tensor([3, 1, 3, 1])  # two placeholder runs show one image
    no_videos = {
        "pixel_values_videos": None,
        "grid_thw_videos": None,
        "video_vision_bank_indices_T": None,
    }

    # Encoded once; the batch's bank rows place it on both runs, without special_tokens.
    inputs_embeds = model._prepare_multimodal_embeds(
        tokens,
        pixel_values=image,
        grid_thw=grid,
        special_tokens=None,
        image_vision_bank_indices_T=torch.tensor([-1, 0, -1, 0]),
        **no_videos,
    )

    expected = model.tok_embeddings(tokens).detach().clone()
    expected[[1, 3]] = model.vision_encoder(image, grid_thw=grid).detach()
    torch.testing.assert_close(inputs_embeds, expected, rtol=0, atol=0)
