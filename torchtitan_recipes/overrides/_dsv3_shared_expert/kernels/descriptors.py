# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""TMA descriptors for native MXFP8 matrix and blocked scale layouts."""

import torch
import triton
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def _matrix(value, rows, columns, bm, bk, offset=0, dtype=gl.float8e4nv):
    matrix = torch.as_strided(
        value, (rows, columns), (columns, 1), value.storage_offset() + offset
    )
    shape = [bm, bk]
    return TensorDescriptor.from_tensor(
        matrix, shape, gl.NVMMASharedLayout.get_default_for(shape, dtype)
    )


def _scales(value, rows, columns, bm, bk, offset=0):
    nr, nk = triton.cdiv(rows, 128), triton.cdiv(columns, 128)
    scales = torch.as_strided(
        value.view(torch.uint8),
        (1, nr, nk, 2, 256),
        (nr * nk * 512, nk * 512, 512, 256, 1),
        value.storage_offset() + offset,
    )
    layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    return TensorDescriptor.from_tensor(
        scales, [1, bm // 128, bk // 128, 2, 256], layout
    )
