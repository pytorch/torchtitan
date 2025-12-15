# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Data:
    max_images_per_batch: int = 10
    """Vision encoder batch size (N)"""
    max_patches_per_image: int = 256
    """Vision encoder sequence length (L)"""
    patch_size: int = 16
    """ Patch size of the vision encoder.
    For example, image size 256x256, patch size 16
        Number of visual tokens is: (256/16)**2=256
    """
    spatial_merge_size: int = 1
    """ Spatially merge visual tokens after encoder.  Default 1 means no merging.
    For example: image size 256x256, patch size 16, spaitl merge size is 2
        Number of visual tokens for the LLM: (256/16/2)**2 = 8
    """
    packing_buffer_size: int = 0
    """ Set to a value >0 to enable sample packing.
    This control the buffer uses to store training samples available for packing.
    """


@dataclass
class JobConfig:
    data: Data = field(default_factory=Data)
