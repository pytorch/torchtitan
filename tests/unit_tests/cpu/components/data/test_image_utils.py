# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil

import numpy as np
import pytest
import torch
from PIL import Image

from torchtitan.components.data.multimodal.utils.image import (
    decode_image_files,
    decode_image_files_flat,
    ImageSpec,
    largest_image_grid,
    lookup_image,
    num_image_tokens,
    patchify,
    process_image,
    read_image_file,
    vision_to_patches,
)


@pytest.fixture
def image_paths(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for name, size in (
        ("a.png", (300, 200)),
        ("b.jpg", (64, 48)),
        ("c.png", (40, 900)),
    ):
        path = tmp_path / "imgs" / name
        path.parent.mkdir(exist_ok=True)
        pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
    return paths


def test_patches_and_header_grids_match_process_image(image_paths) -> None:
    spec = ImageSpec(min_pixels=32 * 32, max_pixels=256 * 256)

    images = [lookup_image(path, spec) for path in image_paths]
    pixel_values = patchify(
        decode_image_files([image.data for image in images], spec), spec
    )
    grid_thw = torch.tensor([image.grid_thw for image in images])

    expected = [
        vision_to_patches(
            process_image(path, min_pixels=spec.min_pixels, max_pixels=spec.max_pixels),
            spec.patch_size,
            spec.temporal_patch_size,
            spec.merge_size,
        )
        for path in image_paths
    ]
    assert torch.equal(pixel_values, torch.cat([patches for patches, _ in expected]))
    assert torch.equal(grid_thw, torch.stack([grid for _, grid in expected]))
    # The grids `lookup_image` read from the headers are the decoded images' grids.
    for image, (patches, grid) in zip(images, expected):
        assert image.grid_thw == tuple(grid.tolist())
        assert (
            num_image_tokens(image.grid_thw, spec)
            == patches.shape[0] // spec.merge_size**2
        )


def test_lookup_keys_an_image_by_content_from_any_path(
    tmp_path, image_paths, monkeypatch
) -> None:
    spec = ImageSpec()
    moved = tmp_path / "elsewhere" / "renamed.png"
    moved.parent.mkdir()
    shutil.copy(image_paths[0], moved)
    monkeypatch.chdir(moved.parent)
    found, original = lookup_image("renamed.png", spec), lookup_image(
        image_paths[0], spec
    )
    assert (found.key, found.grid_thw) == (original.key, original.grid_thw)
    assert len({lookup_image(path, spec).key for path in image_paths}) == 3


def test_read_image_file_shares_the_bytes_and_rejects_a_changed_file(
    image_paths,
) -> None:
    spec = ImageSpec()
    image = lookup_image(image_paths[0], spec)
    data = read_image_file(image.path, image.key)
    assert data == image.data
    # One object for every sample of a group, so a message holding them pickles it once.
    assert read_image_file(image.path, image.key) is data
    rendered = lookup_image(image_paths[2], spec)
    shutil.copy(image_paths[1], image_paths[2])
    with pytest.raises(ValueError, match="changed after it was rendered"):
        read_image_file(rendered.path, rendered.key)


def test_decode_flat_matches_decode_images_and_checks_grids(image_paths) -> None:
    spec = ImageSpec()
    images = [lookup_image(path, spec) for path in image_paths]
    files = torch.frombuffer(
        bytearray(b"".join(image.data for image in images)), dtype=torch.uint8
    )
    sizes = [len(image.data) for image in images]
    grids = [list(image.grid_thw) for image in images]
    expected = torch.cat(
        [
            image.flatten()
            for image in decode_image_files([image.data for image in images], spec)
        ]
    )
    assert torch.equal(decode_image_files_flat(files, sizes, grids, spec), expected)
    grids[1][1] += 1
    with pytest.raises(ValueError, match="decodes to grid"):
        decode_image_files_flat(files, sizes, grids, spec)


def test_largest_image_grid() -> None:
    spec = ImageSpec(min_pixels=32 * 32, max_pixels=256 * 256)
    # 256 x 256 pixels in 32-pixel merged tokens: at most 8 x 8 = 64 tokens.
    assert largest_image_grid(spec, 10_000) == (1, 16, 16)
    assert largest_image_grid(spec, 20) == (1, 8, 8)
    with pytest.raises(ValueError, match="fits"):
        largest_image_grid(spec, 0)
