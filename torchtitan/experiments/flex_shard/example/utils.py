# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

import torch
import torch.distributed as dist

from ..flex_shard.bucket_storage import GradientReduceOp


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
        from ._copy_kernels import pack_tensors_into_flat_buffer_triton
    except Exception:
        return None

    try:
        return pack_tensors_into_flat_buffer_triton(tensors, dtype)
    except Exception:
        return None


def pack_segments_into_flat_buffer_triton_if_supported(
    inputs: list[torch.Tensor],
    tensor_indices: Sequence[int],
    src_offsets: Sequence[int],
    numels: Sequence[int],
    dst_offsets: Sequence[int],
    out: torch.Tensor,
) -> list[torch.Tensor] | None:
    """Use Triton descriptor packing for supported inputs.

    Triton is required for this backend. The function returns ``None`` only when
    the current inputs should use the deterministic foreach copy fallback.
    """
    if torch.compiler.is_compiling():
        return None
    if importlib.util.find_spec("triton") is None:
        raise AssertionError(
            "GroupedOwned Triton segment packing requires the triton package."
        )

    from ._copy_kernels import pack_segments_into_flat_buffer_triton

    return pack_segments_into_flat_buffer_triton(
        inputs,
        tensor_indices,
        src_offsets,
        numels,
        dst_offsets,
        out,
    )


def _to_dist_reduce_op(op: GradientReduceOp) -> dist.ReduceOp.RedOpType:
    if op == "avg":
        return dist.ReduceOp.AVG
    return dist.ReduceOp.SUM


__all__ = [
    "copy_tensor_to_dtype",
    "copy_tensors_into_flat_buffer",
    "foreach_copy_",
    "pack_segments_into_flat_buffer_triton_if_supported",
    "pack_tensors_into_flat_buffer",
    "pack_tensors_into_flat_buffer_with_scratch",
    "_to_dist_reduce_op",
]
