# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

import torch


def foreach_copy_(
    dst_tensors: list[torch.Tensor],
    src_tensors: list[torch.Tensor],
) -> None:
    """Copy tensors with one foreach runtime boundary when eager."""
    if len(dst_tensors) != len(src_tensors):
        raise AssertionError(
            f"Expected {len(dst_tensors)} destination tensors to match "
            f"{len(src_tensors)} source tensors."
        )
    if not dst_tensors:
        return
    if torch.compiler.is_compiling():
        for dst, src in zip(dst_tensors, src_tensors, strict=True):
            dst.copy_(src)
    else:
        torch._foreach_copy_(dst_tensors, src_tensors)


def copy_tensor_to_dtype(
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a contiguous tensor with dtype, using foreach copy for casts."""
    if tensor.is_contiguous() and tensor.dtype == dtype:
        return tensor

    out = torch.empty(
        tensor.shape,
        dtype=dtype,
        device=tensor.device,
    )
    if tensor.numel() > 0:
        foreach_copy_([out], [tensor])
    return out


def pack_segments_into_flat_buffer_triton_if_available(
    inputs: list[torch.Tensor],
    tensor_indices: Sequence[int],
    src_offsets: Sequence[int],
    numels: Sequence[int],
    dst_offsets: Sequence[int],
    out: torch.Tensor,
) -> list[torch.Tensor] | None:
    """Use the optional Triton descriptor pack path when it is available."""
    if torch.compiler.is_compiling():
        return None
    try:
        from ._pack_kernels import pack_segments_into_flat_buffer_triton
    except Exception:
        return None

    try:
        return pack_segments_into_flat_buffer_triton(
            inputs,
            tensor_indices,
            src_offsets,
            numels,
            dst_offsets,
            out,
        )
    except Exception:
        return None
