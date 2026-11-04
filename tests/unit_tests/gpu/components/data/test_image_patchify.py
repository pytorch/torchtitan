# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from PIL import Image

from torchtitan.components.data.multimodal.utils.image import (
    decode_image_files,
    decode_image_files_flat,
    ImageSpec,
    lookup_image,
    patchify,
    patchify_flat,
)

_SPEC = ImageSpec(min_pixels=32 * 32, max_pixels=256 * 256)


def _lookup_images(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for name, (width, height) in (("a.png", (300, 200)), ("b.png", (40, 900))):
        path = tmp_path / name
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return [lookup_image(path, _SPEC) for path in paths]


def test_patchify_on_gpu_matches_cpu(tmp_path) -> None:
    # The generator patchifies on the GPU; the trainer side must see the same values.
    image_infos = _lookup_images(tmp_path)
    images = decode_image_files([info.data for info in image_infos], _SPEC)

    on_gpu = patchify([image.cuda() for image in images], _SPEC)

    assert torch.equal(on_gpu.cpu(), patchify(images, _SPEC))


def test_trainer_decode_then_gpu_patchify_matches_cpu(tmp_path) -> None:
    # The trainer decodes the shipped files on the CPU and patchifies on the GPU
    # (`TrainingMicrobatch.to_input_dict`).
    image_infos = _lookup_images(tmp_path)
    files = torch.frombuffer(
        bytearray(b"".join(info.data for info in image_infos)), dtype=torch.uint8
    )
    sizes = [len(info.data) for info in image_infos]
    grids = [list(info.grid_thw) for info in image_infos]

    flat = decode_image_files_flat(files, sizes, grids, _SPEC)
    on_gpu = patchify_flat(flat.cuda(), grids, _SPEC)

    assert torch.equal(
        on_gpu.cpu(),
        patchify(decode_image_files([info.data for info in image_infos], _SPEC), _SPEC),
    )
