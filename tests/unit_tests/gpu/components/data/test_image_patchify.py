# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io

import numpy as np
import torch
from PIL import Image

from torchtitan.components.data.multimodal.image_processor import ImageProcessor
from torchtitan.models.qwen3_5 import QWEN3_5_IMAGE_PROCESSOR

_PROCESSOR = ImageProcessor.Config(
    **{**QWEN3_5_IMAGE_PROCESSOR, "min_pixels": 32 * 32, "max_pixels": 256 * 256}
).build()


def test_pixel_values_on_gpu_match_cpu() -> None:
    # The generator and the trainer resize the image files on the CPU and compute pixel
    # values on the GPU; the GPU values must equal the CPU's.
    rng = np.random.default_rng(0)
    files = []
    for width, height in ((300, 200), (40, 900)):
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        files.append(buffer.getvalue())
    images = [_PROCESSOR.decode_and_resize(file) for file in files]

    on_gpu, grids_on_gpu = _PROCESSOR.normalize_and_patchify(
        [image.cuda() for image in images]
    )
    on_cpu, grids_on_cpu = _PROCESSOR.normalize_and_patchify(images)
    assert torch.equal(on_gpu.cpu(), on_cpu)
    assert torch.equal(grids_on_gpu.cpu(), grids_on_cpu)
