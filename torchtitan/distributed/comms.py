# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom intra-node all_reduce backend.

When ``comm.enable_custom_comm`` is set, ``maybe_enable_custom_comm`` monkey
patches ``torch.ops._c10d_functional.all_reduce`` (the op every DTensor
``Partial -> Replicate`` redistribute lowers to in eager mode) to route eligible
intra-node sum reductions through symmetric-memory kernels, falling back to the
built-in NCCL all_reduce otherwise.

Two orthogonal axes pick the kernel per call:

- Transport: multimem (NVLink SHARP / NVLS multicast) where NVSwitch multicast
  is available (Hopper + NVSwitch), else P2P (direct peer reads). P2P also
  serves non-NVSwitch GPUs (e.g. A100).
- Pattern: one-shot (each rank reads and reduces every peer) at the smallest
  sizes for lowest latency, two-shot (reduce-scatter + all-gather) above the
  crossover for better bandwidth. The crossover is size-based and differs per
  transport (profiled on H100; re-tune per architecture).

Large sizes and every ineligible case (non-sum op, inter-node group, unsupported
dtype, deterministic mode, ...) fall back to NCCL.

A single persistent scratch buffer per (group, dtype) is allocated and
rendezvous'd once (on first eligible call, over the collective's group), then
sliced per call. So there is no per-activation rendezvous and no per-step
collective setup, and the path is robust to varying activation sizes.

Numerics: these kernels reduce in a different order than NCCL, so results are not
bit-identical to the NCCL baseline. The backend is disabled under
``debug.deterministic``. Eager only: the dispatcher has data-dependent control
flow and non-traceable calls (process-group resolution, multicast query, lazy
symmetric-buffer rendezvous, symm ops without a fake-tensor impl), so it must not
be traced. Under ``torch.compile`` the patch defers to the original op, which is
captured as a graph node and lowered by inductor (preserving comm-compute
fusion / async-TP). This path assumes the default CUDA symmetric-memory backend
(local allocation + multicast); it is not compatible with the NVSHMEM backend,
whose allocation is a global collective.
"""

import functools
import os
from enum import auto, Enum

import torch
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group

from torchtitan.config import CommConfig
from torchtitan.tools.logging import logger


class _Algo(Enum):
    NCCL = auto()
    ONE_SHOT = auto()
    TWO_SHOT = auto()
    MULTIMEM_ONE_SHOT = auto()
    MULTIMEM_TWO_SHOT = auto()


# Byte-size thresholds, profiled on 8xH100 (NVSwitch). Re-tune per architecture;
# in particular the P2P bounds serve non-multicast GPUs (e.g. A100) and should be
# re-profiled there.
#
# Within each transport, one-shot wins at the smallest sizes (single barrier,
# lowest latency); two-shot takes over once its N/world data movement beats
# one-shot's N/rank peer reads. Measured crossovers on H100: ~64 KiB for
# multimem, ~128 KiB for P2P. Past the *_MAX bounds, NCCL wins and we fall back.
_MULTIMEM_ONE_SHOT_MAX_BYTES = 64 * 1024
_ONE_SHOT_MAX_BYTES = 128 * 1024
_TWO_SHOT_MAX_BYTES = 16 * 1024 * 1024
_MULTIMEM_MAX_BYTES = 32 * 1024 * 1024
_SCRATCH_MAX_BYTES = max(_TWO_SHOT_MAX_BYTES, _MULTIMEM_MAX_BYTES)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# Intra-node group sizes we support for symmetric memory.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)

_installed = False


def maybe_enable_custom_comm(comm: CommConfig) -> None:
    """Install the custom all_reduce backend if enabled. Idempotent."""
    global _installed
    if not comm.enable_custom_comm or _installed:
        return
    if not _backend_supported():
        logger.warning(
            "comm.enable_custom_comm is set but symmetric-memory all_reduce is "
            "not supported on this platform; keeping the default NCCL all_reduce."
        )
        return

    c10d = torch.ops._c10d_functional
    original_all_reduce = c10d.all_reduce

    def all_reduce(input, reduce_op, group_name):
        # Under torch.compile, defer to the original op so it is captured as a
        # graph node (inductor lowers it, enabling comm-compute fusion / async-TP).
        # Tracing our eager dispatcher instead would graph-break at every
        # non-traceable call (process-group resolution, multicast query, lazy
        # symmetric-buffer rendezvous, symm ops without a fake-tensor impl).
        if torch.compiler.is_compiling():
            return original_all_reduce(input, reduce_op, group_name)
        return _custom_all_reduce(input, reduce_op, group_name, original_all_reduce)

    # Preserve the .default overload so code paths that access it keep working.
    all_reduce.default = original_all_reduce.default
    c10d.all_reduce = all_reduce
    _installed = True
    logger.info(
        "custom_comm: routed _c10d_functional.all_reduce through the "
        "symmetric-memory backend (one-shot/two-shot/multimem, NCCL fallback)"
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
    return _resolve_process_group(group_name).size()


@functools.lru_cache(maxsize=None)
def _has_multicast(device_index: int) -> bool:
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device_index)
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def _get_scratch(group_name: str, dtype: torch.dtype) -> torch.Tensor:
    """Persistent symmetric scratch buffer for a (group, dtype), rendezvous'd
    once. All ranks of ``group_name`` reach this together (SPMD), so the
    rendezvous collective is symmetric and scoped to that group only."""
    numel = _SCRATCH_MAX_BYTES // dtype.itemsize
    buf = symm_mem.empty(numel, device="cuda", dtype=dtype)
    symm_mem.rendezvous(buf, group=group_name)
    return buf


def _select_algo(input: torch.Tensor, reduce_op: str, group_name: str) -> _Algo:
    if reduce_op != "sum":
        return _Algo.NCCL
    if input.dtype not in _SUPPORTED_DTYPES or not input.is_contiguous():
        return _Algo.NCCL
    if torch.are_deterministic_algorithms_enabled():
        return _Algo.NCCL

    world_size = _group_world_size(group_name)
    if world_size not in _SUPPORTED_WORLD_SIZES or world_size > _local_world_size():
        return _Algo.NCCL

    numel = input.numel()
    if numel == 0:
        return _Algo.NCCL
    # Conservative alignment for vectorized (16-byte) per-rank access.
    vec = 16 // input.element_size()
    if numel % (world_size * vec) != 0:
        return _Algo.NCCL

    nbytes = numel * input.element_size()
    # multimem (NVLS) where multicast is available, else P2P. Within each
    # transport, one-shot at the smallest sizes, two-shot above the crossover.
    if _has_multicast(input.device.index):
        if nbytes > _MULTIMEM_MAX_BYTES:
            return _Algo.NCCL
        if nbytes <= _MULTIMEM_ONE_SHOT_MAX_BYTES:
            return _Algo.MULTIMEM_ONE_SHOT
        return _Algo.MULTIMEM_TWO_SHOT
    if nbytes > _TWO_SHOT_MAX_BYTES:
        return _Algo.NCCL
    return _Algo.ONE_SHOT if nbytes <= _ONE_SHOT_MAX_BYTES else _Algo.TWO_SHOT


def _custom_all_reduce(input, reduce_op, group_name, fallback):
    algo = _select_algo(input, reduce_op, group_name)
    if algo is _Algo.NCCL:
        return fallback(input, reduce_op, group_name)

    scratch = _get_scratch(group_name, input.dtype)
    view = scratch[: input.numel()].view_as(input)
    out = torch.empty_like(input)

    if algo is _Algo.ONE_SHOT:
        # Fused: copies input into the symmetric buffer, reduces, writes out.
        torch.ops.symm_mem.one_shot_all_reduce_copy_out(
            view, input, "sum", group_name, out
        )
    elif algo is _Algo.TWO_SHOT:
        view.copy_(input)
        torch.ops.symm_mem.two_shot_all_reduce_out(view, "sum", group_name, out)
    elif algo is _Algo.MULTIMEM_ONE_SHOT:
        view.copy_(input)
        torch.ops.symm_mem.multimem_one_shot_all_reduce_out(
            view, "sum", group_name, out
        )
    else:  # MULTIMEM_TWO_SHOT: in-place on scratch, then copy out of the buffer.
        view.copy_(input)
        torch.ops.symm_mem.multimem_all_reduce_(view, "sum", group_name)
        out.copy_(view)

    # A fully-reduced plain tensor; the follow-up wait_tensor is a no-op.
    return out
