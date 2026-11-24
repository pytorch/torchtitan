# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import (
    build_model_config,
    vision_encoder as vision_encoder_module,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexInnerAttention tiling needs a GPU"
)


def test_batch_invariant_features_do_not_depend_on_the_pack(monkeypatch):
    """In batch-invariant mode an image's features are bitwise the same alone, after
    another image, and between two others.

    Packed together, image B would start at patch 140 or 12 of the sequence, and
    FlexInnerAttention's tiles, which start at the sequence's start, would cut B
    differently each time.
    """
    monkeypatch.setattr(
        vision_encoder_module, "is_in_batch_invariant_mode", lambda: True
    )
    torch.manual_seed(0)
    config = build_model_config(
        "debugmodel_mm", seq_len=1024, parallelism=ParallelismConfig()
    ).vision_encoder
    encoder = config.build().cuda()
    encoder.init_states()
    for parameter in encoder.parameters():
        parameter.data = parameter.data.bfloat16()
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    # [t, h, w] patch grids: 140, 60 and 12 patches.
    grids = {"A": [1, 10, 14], "B": [1, 6, 10], "C": [1, 2, 6]}
    pixel_values = {
        name: torch.randn(t * h * w, patch_dim, device="cuda", dtype=torch.bfloat16)
        for name, (t, h, w) in grids.items()
    }
    merged_tokens = {
        name: t * h * w // encoder.spatial_merge_unit
        for name, (t, h, w) in grids.items()
    }

    def encode(*names):
        return encoder(
            torch.cat([pixel_values[name] for name in names]),
            grid_thw=torch.tensor([grids[name] for name in names], device="cuda"),
        )

    with torch.no_grad():
        alone = encode("B")
        after_a = encode("A", "B")[merged_tokens["A"] :]
        between = encode("C", "B", "A")[
            merged_tokens["C"] : merged_tokens["C"] + merged_tokens["B"]
        ]
    torch.testing.assert_close(after_a, alone, atol=0, rtol=0)
    torch.testing.assert_close(between, alone, atol=0, rtol=0)
