# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Training:
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
    This control the buffer uses to store training samples avaliable for packing.
    """


# HACK: couldn't figure out how to modify the HF tokenizer's json
# to make these attribute accesible. Ideally these should be accesible from the tokenizer itself.
@dataclass
class SpecialTokens:
    img_token: str = "<|image|>"
    boi_token: str = "<|begin_of_image|>"
    eoi_token: str = "<|end_of_image|>"
    img_id: int = 1998
    boi_id: int = 1999
    eoi_id: int = 2000
    pad_id: int = 2001


@dataclass
class JobConfig:
    training: Training = field(default_factory=Training)
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)
