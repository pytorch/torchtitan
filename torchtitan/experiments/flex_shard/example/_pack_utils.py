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


def copy_tensors_to_dtype(
    tensors: list[torch.Tensor],
    dtypes: list[torch.dtype],
) -> list[torch.Tensor]:
    """Copy tensors to matching contiguous dtype buffers with one foreach call."""
    if len(tensors) != len(dtypes):
        raise AssertionError(
            f"Expected {len(tensors)} tensors to match {len(dtypes)} dtypes."
        )

    outputs: list[torch.Tensor] = []
    copy_srcs: list[torch.Tensor] = []
    copy_dsts: list[torch.Tensor] = []
    for tensor, dtype in zip(tensors, dtypes, strict=True):
        if tensor.is_contiguous() and tensor.dtype == dtype:
            outputs.append(tensor)
            continue

        out = torch.empty(
            tensor.shape,
            dtype=dtype,
            device=tensor.device,
        )
        outputs.append(out)
        if tensor.numel() > 0:
            copy_srcs.append(tensor)
            copy_dsts.append(out)

    foreach_copy_(copy_dsts, copy_srcs)
    return outputs


def pack_tensors_into_flat_buffer(
    tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pack tensors into one flat buffer using foreach copy/cast."""
    if not tensors:
        raise AssertionError("Expected at least one tensor to pack.")

    total_numel = sum(tensor.numel() for tensor in tensors)
    out = torch.empty(
        total_numel,
        dtype=dtype,
        device=tensors[0].device,
    )
    copy_tensors_into_flat_buffer(tensors, out)
    return out


def pack_tensors_into_flat_buffer_with_scratch(
    tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Pack tensors into one flat buffer and return async scratch to retain."""
    if not tensors:
        raise AssertionError("Expected at least one tensor to pack.")

    triton_result = _try_pack_tensors_into_flat_buffer_triton(tensors, dtype)
    if triton_result is not None:
        return triton_result

    return pack_tensors_into_flat_buffer(tensors, dtype), []


def copy_tensors_into_flat_buffer(
    tensors: list[torch.Tensor],
    out: torch.Tensor,
) -> None:
    """Copy tensors into preallocated flat out buffer with foreach copy/cast."""
    copy_srcs: list[torch.Tensor] = []
    copy_dsts: list[torch.Tensor] = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        if numel > 0:
            copy_srcs.append(tensor)
            copy_dsts.append(out.narrow(0, offset, numel).view(tensor.shape))
        offset += numel

    if offset != out.numel():
        raise AssertionError(
            f"Packed tensor numel {offset} does not match output numel {out.numel()}."
        )
    foreach_copy_(copy_dsts, copy_srcs)


def _try_pack_tensors_into_flat_buffer_triton(
    tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
    """Try the optional Triton copy-in path, falling back on any issue."""
    if torch.compiler.is_compiling():
        return None
    try:
        from ._pack_kernels import pack_tensors_into_flat_buffer_triton
    except Exception:
        return None

    try:
        return pack_tensors_into_flat_buffer_triton(tensors, dtype)
    except Exception:
        return None


def try_pack_segments_into_flat_buffer_triton(
    inputs: list[torch.Tensor],
    tensor_indices: Sequence[int],
    src_offsets: Sequence[int],
    numels: Sequence[int],
    dst_offsets: Sequence[int],
    out: torch.Tensor,
) -> list[torch.Tensor] | None:
    """Try the optional Triton descriptor pack path, falling back on any issue."""
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
