# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset utility functions for Qwen3-VL."""

from .image import (
    calculate_image_tokens,
    convert_to_patches,
    pad_empty_images_to_target_batch_size,
    pad_patches,
    process_image,
)
from .packing import SamplePacker
from .text import (
    pad_input_ids_and_labels_to_target_batch_size,
    pad_text_batch,
    process_text_with_images,
)

__all__ = [
    "calculate_image_tokens",
    "convert_to_patches",
    "pad_empty_images_to_target_batch_size",
    "pad_patches",
    "process_image",
    "SamplePacker",
    "pad_input_ids_and_labels_to_target_batch_size",
    "pad_text_batch",
    "process_text_with_images",
]
