# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Job configuration extensions for Qwen3-VL training."""

from dataclasses import dataclass, field


@dataclass
class Data:
    max_images_per_batch: int = 10
    """Vision encoder batch size (N)"""
    max_patches_per_image: int = 256
    """Vision encoder sequence length (L)"""
    patch_size: int = 14
    """Patch size of the vision encoder.
    For example, image size 224x224, patch size 14
        Number of visual tokens is: (224/14)**2=256
    """
    temporal_patch_size: int = 2
    """Temporal patch size for video processing."""
    spatial_merge_size: int = 2
    """Spatially merge visual tokens after encoder. Default 2 means 2x2=4 patches merged.
    For example: image size 224x224, patch size 14, spatial merge size is 2
        Number of visual tokens for the LLM: (224/14/2)**2 = 64
    """
    packing_buffer_size: int = 0
    """Set to a value >0 to enable sample packing.
    This controls the buffer used to store training samples available for packing.
    """


@dataclass
class JobConfig:
    data: Data = field(default_factory=Data)
