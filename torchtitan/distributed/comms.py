# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom intra-node all_reduce backend.

When ``comm.enable_custom_comm`` is set, ``maybe_enable_custom_comm`` registers a
CUDA kernel for ``_c10d_functional::all_reduce`` (the op every DTensor
``Partial -> Replicate`` redistribute lowers to). Eligible intra-node sum
reductions are routed through symmetric-memory kernels; everything else falls
back to the process-group NCCL all_reduce.

The kernel is registered on the CUDA dispatch key (the op ships only a
device-agnostic ``CompositeExplicitAutograd`` kernel, so a CUDA kernel takes
precedence for CUDA tensors). Because no Python is replaced, ``torch.compile``
sees the vanilla op and captures the collective as an ordinary graph node, so
turning this on does not introduce graph breaks. The symmetric-memory
acceleration is validated for eager use only; behavior inside a compiled region
is out of scope at this moment.

Two orthogonal axes pick the kernel per call:

- Transport: multimem (NVLink SHARP / NVLS multicast) where NVSwitch multicast is
  available (Hopper + NVSwitch), else P2P (direct peer reads).
- Pattern: one-shot (each rank reads and reduces every peer) at the smallest sizes for
  lowest latency, two-shot (reduce-scatter + all-gather) for medium size communication.
  The crossover is size-based and differs per transport, profiled on H100. A run-time
  profiling can be added as an improvement, but this version doesn't implement it.

Large sizes and every ineligible case (non-sum op, inter-node group, unsupported dtype,
deterministic mode, ...) fall back to NCCL.

A single persistent symm_buffer per (group, dtype) is allocated and rendezvous'd once.
"""

import functools
import os
from enum import auto, Enum

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group, GroupName

from torchtitan.config import CommConfig
from torchtitan.tools.logging import logger


class _Algo(Enum):
    NCCL = auto()
    ONE_SHOT = auto()
    TWO_SHOT = auto()
    MULTIMEM_ONE_SHOT = auto()
    MULTIMEM_TWO_SHOT = auto()


# Byte-size thresholds, profiled on 8xH100 (NVSwitch).
#
# Within each transport, one-shot wins at the smallest sizes (single barrier,
# lowest latency); two-shot takes over once its N/world data movement beats
# one-shot's N/rank peer reads. Measured crossovers on H100: ~64 KiB for
# multimem, ~128 KiB for P2P. Past the *_MAX bounds, NCCL wins and we fall back.
_MULTIMEM_ONE_SHOT_MAX_BYTES = 64 * 1024
_ONE_SHOT_MAX_BYTES = 128 * 1024
_TWO_SHOT_MAX_BYTES = 16 * 1024 * 1024
_MULTIMEM_MAX_BYTES = 32 * 1024 * 1024
_SYMM_BUFFER_MAX_BYTES = max(_TWO_SHOT_MAX_BYTES, _MULTIMEM_MAX_BYTES)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# Intra-node group sizes we support for symmetric memory.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)


# Holds the CUDA kernel registration; kept alive for the process (dropping the
# handle would deregister). Also serves as the idempotency guard.
_lib: torch.library.Library | None = None


def maybe_enable_custom_comm(comm: CommConfig) -> None:
    """Install the custom all_reduce backend if enabled. Idempotent."""
    global _lib
    if not comm.enable_custom_comm or _lib is not None:
        return
    if not _backend_supported():
        logger.warning(
            "comm.enable_custom_comm is set but symmetric-memory all_reduce is "
            "not supported on this platform; keeping the default NCCL all_reduce."
        )
        return

    _lib = torch.library.Library("_c10d_functional", "IMPL")
    _lib.impl("all_reduce", _custom_all_reduce, "CUDA")
    logger.info(
        "custom_comm: registered a CUDA _c10d_functional.all_reduce kernel using "
        "the symmetric-memory backend (one-shot/two-shot/multimem, NCCL fallback)"
    )


def _backend_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and hasattr(torch.ops.symm_mem, "one_shot_all_reduce_copy_out")
        and hasattr(torch.ops.symm_mem, "two_shot_all_reduce_out")
        and hasattr(torch.ops.symm_mem, "multimem_one_shot_all_reduce_out")
        and hasattr(torch.ops.symm_mem, "multimem_all_reduce_")
    )


@functools.lru_cache(maxsize=None)
def _local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))


@functools.lru_cache(maxsize=None)
def _group_world_size(group_name: str) -> int:
    return _resolve_process_group(GroupName(group_name)).size()


@functools.lru_cache(maxsize=None)
def _is_intra_node(group_name: str) -> bool:
    """True only if every rank of the group lives on the same node, which
    symmetric memory requires. A size check (world_size <= local_world_size) is
    necessary but not sufficient: a strided group (e.g. one rank per node) can
    pass it yet span nodes. Assumes node-contiguous global ranks (the torchrun /
    SLURM default), so rank r is on node r // local_world_size."""
    pg = _resolve_process_group(GroupName(group_name))
    local_world_size = _local_world_size()
    nodes = {r // local_world_size for r in dist.get_process_group_ranks(pg)}
    return len(nodes) == 1


@functools.lru_cache(maxsize=None)
def _has_multicast(device_index: int) -> bool:
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device_index)
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"custom_comm: multicast detection unavailable ({e}); using P2P "
            "instead of multimem. Update the detection path if torch moved it."
        )
        return False


@functools.lru_cache(maxsize=None)
def _get_symm_buffer(group_name: str, dtype: torch.dtype) -> torch.Tensor:
    """Persistent symmetric-memory buffer for a (group, dtype), rendezvous'd
    once. All ranks of ``group_name`` reach this together (SPMD), so the
    rendezvous collective is symmetric and scoped to that group only."""
    numel = _SYMM_BUFFER_MAX_BYTES // dtype.itemsize
    buf = symm_mem.empty(numel, device=torch.device("cuda"), dtype=dtype)
    symm_mem.rendezvous(buf, group=GroupName(group_name))
    return buf


def _select_algo(input: torch.Tensor, reduce_op: str, group_name: str) -> _Algo:
    if reduce_op != "sum":
        return _Algo.NCCL
    if input.dtype not in _SUPPORTED_DTYPES or not input.is_contiguous():
        return _Algo.NCCL
    if torch.are_deterministic_algorithms_enabled():
        return _Algo.NCCL

    world_size = _group_world_size(group_name)
    if world_size not in _SUPPORTED_WORLD_SIZES or not _is_intra_node(group_name):
        return _Algo.NCCL

    numel = input.numel()
    if numel == 0:
        return _Algo.NCCL
    # Conservative alignment for vectorized (16-byte) per-rank access.
    vec = 16 // input.element_size()
    if numel % (world_size * vec) != 0:
        return _Algo.NCCL

    nbytes = numel * input.element_size()
    if _has_multicast(input.device.index):
        if nbytes > _MULTIMEM_MAX_BYTES:
            return _Algo.NCCL
        if nbytes <= _MULTIMEM_ONE_SHOT_MAX_BYTES:
            return _Algo.MULTIMEM_ONE_SHOT
        return _Algo.MULTIMEM_TWO_SHOT
    if nbytes > _TWO_SHOT_MAX_BYTES:
        return _Algo.NCCL
    return _Algo.ONE_SHOT if nbytes <= _ONE_SHOT_MAX_BYTES else _Algo.TWO_SHOT


def _nccl_fallback(
    input: torch.Tensor, reduce_op: str, group_name: str
) -> torch.Tensor:
    """Out-of-place NCCL all_reduce, matching the op's Composite kernel (clone +
    process-group all_reduce). Used when a call is ineligible for symmetric
    memory. We cannot re-enter the op (it would re-dispatch to this kernel), so
    the process group is driven directly."""
    out = input.clone()
    dist.all_reduce(
        out,
        op=getattr(dist.ReduceOp, reduce_op.upper()),
        group=_resolve_process_group(GroupName(group_name)),
    )
    return out


def _custom_all_reduce(
    input: torch.Tensor, reduce_op: str, group_name: str
) -> torch.Tensor:
    algo = _select_algo(input, reduce_op, group_name)
    if algo is _Algo.NCCL:
        return _nccl_fallback(input, reduce_op, group_name)

    symm_buffer = _get_symm_buffer(group_name, input.dtype)
    view = symm_buffer[: input.numel()].view_as(input)
    out = torch.empty_like(input)

    match algo:
        case _Algo.ONE_SHOT:
            # Fused: copies input into the symmetric buffer, reduces, writes out.
            torch.ops.symm_mem.one_shot_all_reduce_copy_out(
                view, input, "sum", group_name, out
            )
        case _Algo.TWO_SHOT:
            view.copy_(input)
            torch.ops.symm_mem.two_shot_all_reduce_out(view, "sum", group_name, out)
        case _Algo.MULTIMEM_ONE_SHOT:
            view.copy_(input)
            torch.ops.symm_mem.multimem_one_shot_all_reduce_out(
                view, "sum", group_name, out
            )
        case _Algo.MULTIMEM_TWO_SHOT:
            # In-place on symm_buffer, then copy out of it.
            view.copy_(input)
            torch.ops.symm_mem.multimem_all_reduce_(view, "sum", group_name)
            out.copy_(view)
        case _:
            raise AssertionError(f"unhandled algo: {algo}")

    return out
