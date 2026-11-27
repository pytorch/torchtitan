# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import torchtitan.quantization.mxfp8.tensor as mxfp8_tensor


# Copied from TorchAO's torchao/prototype/mx_formats/utils.py (BSD-3-Clause),
# retaining only the eager layout conversions:
# https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/utils.py
# Note: the copy is intended, as in the future we plan to remove the torchao
# dependency from torchtitan.
def _to_blocked(input_matrix):
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _from_blocked(blocked_tensor, original_rows, original_cols):
    n_row_blocks = (original_rows + 127) // 128
    n_col_blocks = (original_cols + 3) // 4

    rearranged = blocked_tensor.view(n_row_blocks * n_col_blocks, 32, 16)
    temp = rearranged.reshape(n_row_blocks * n_col_blocks, 32, 4, 4)
    temp = temp.transpose(1, 2)
    blocks = temp.reshape(n_row_blocks, n_col_blocks, 128, 4)
    padded_view = blocks.permute(0, 2, 1, 3)
    padded = padded_view.reshape(n_row_blocks * 128, n_col_blocks * 4)
    return padded[:original_rows, :original_cols]


def _make_scales(num_weight_rows, num_weight_cols):
    # Include every byte value, including E8M0's NaN encoding, without casts.
    tile_scales = torch.arange(
        (num_weight_rows // 32) * (num_weight_cols // 32),
        device="cuda",
        dtype=torch.int64,
    ).to(torch.uint8)
    scales = tile_scales.view(num_weight_rows // 32, num_weight_cols // 32)
    return _to_blocked(scales.repeat_interleave(32, dim=0))


def _reference(scales, num_weight_rows, num_weight_cols):
    unpacked = _from_blocked(scales, num_weight_rows, num_weight_cols // 32)
    return _to_blocked(unpacked[::32].t().repeat_interleave(32, dim=0)).view(1, -1).t()


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
        (32, 160),
        (160, 32),
        (96, 128),
        (128, 96),
        (128, 128),
        (160, 160),
        (4096, 4096),
        (14336, 4096),
        (4096, 14336),
        (8192, 4096),
    ],
)
def test_transpose_mxfp8_weight_scales_is_bitwise_equal(shape):
    scales = _make_scales(*shape)
    expected = _reference(scales, *shape)
    actual = mxfp8_tensor._transpose_mxfp8_weight_scales(scales, *shape)
    assert torch.equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.dtype == expected.dtype


def test_transpose_mxfp8_weight_scales_compile_and_capture():
    shape = (96, 160)
    scales = _make_scales(*shape)
    # Random padded bytes also catch accidental reads from input padding.
    scales = torch.randint(256, scales.shape, dtype=torch.uint8, device="cuda")
    expected = _reference(scales, *shape)

    def transpose_scales(scales):
        return mxfp8_tensor._transpose_mxfp8_weight_scales(
            scales.view(torch.uint8), *shape
        ).view(torch.float8_e8m0fnu)

    compiled = torch.compile(transpose_scales, fullgraph=True)
    actual = compiled(scales.view(torch.float8_e8m0fnu))
    assert torch.equal(actual.view(torch.uint8), expected)
    torch.library.opcheck(mxfp8_tensor._transpose_mxfp8_weight_scales, (scales, *shape))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mxfp8_tensor._transpose_mxfp8_weight_scales(scales, *shape)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = mxfp8_tensor._transpose_mxfp8_weight_scales(scales, *shape)
    for _ in range(2):
        scales.random_(0, 256)
        graph.replay()
        assert torch.equal(actual, _reference(scales, *shape))
