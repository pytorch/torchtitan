# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.components.data.multimodal.vision_processor import (
    PixelBudgetResize,
    VisionProcessor,
)


def _vision_processor_config(**overrides) -> VisionProcessor.Config:
    fields = dict(
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        resize=PixelBudgetResize.Config(min_pixels=65_536, max_pixels=16_777_216),
        patch_order="block",
    )
    fields.update(overrides)
    return VisionProcessor.Config(**fields)


@pytest.mark.parametrize(
    "field_name",
    ["patch_size", "temporal_patch_size", "spatial_merge_size"],
)
def test_zero_size_raises_value_error(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name) as exc_info:
        _vision_processor_config(**{field_name: 0})
    assert "0" in str(exc_info.value)


def test_negative_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="patch_size") as exc_info:
        _vision_processor_config(patch_size=-1)
    assert "-1" in str(exc_info.value)


@pytest.mark.parametrize("max_pixels", [0, -1])
def test_non_positive_max_pixels_raises_value_error(max_pixels: int) -> None:
    with pytest.raises(ValueError, match="max_pixels") as exc_info:
        PixelBudgetResize.Config(min_pixels=1, max_pixels=max_pixels)
    assert str(max_pixels) in str(exc_info.value)


def test_positive_vision_processor_config_constructs() -> None:
    config = _vision_processor_config()
    assert config.patch_size == 16
    assert config.temporal_patch_size == 2
    assert config.spatial_merge_size == 2
    assert config.image_mean == (0.5, 0.5, 0.5)
    assert config.image_std == (0.5, 0.5, 0.5)
    assert config.patch_order == "block"
    assert config.resize == PixelBudgetResize.Config(
        min_pixels=65_536, max_pixels=16_777_216
    )
