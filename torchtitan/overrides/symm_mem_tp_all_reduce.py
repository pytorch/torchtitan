# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override TP Partial -> Replicate/Identity redistributions with symm_mem."""

import functools
import os
from dataclasses import dataclass
from enum import auto, Enum

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group, GroupName

from torchtitan.config import derive, override
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.protocols.sharding import PerAxisRedistribution
from torchtitan.tools.logging import logger


class _Algo(Enum):
    NCCL = auto()
    ONE_SHOT = auto()
    TWO_SHOT = auto()
    MULTIMEM_ONE_SHOT = auto()
    MULTIMEM_TWO_SHOT = auto()


_MULTIMEM_ONE_SHOT_MAX_BYTES = 64 * 1024
_ONE_SHOT_MAX_BYTES = 128 * 1024
_TWO_SHOT_MAX_BYTES = 16 * 1024 * 1024
_MULTIMEM_MAX_BYTES = 32 * 1024 * 1024
_SYMM_BUFFER_MAX_BYTES = max(_TWO_SHOT_MAX_BYTES, _MULTIMEM_MAX_BYTES)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SUPPORTED_WORLD_SIZES = (2, 4, 8)


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
            f"symm_mem_tp_all_reduce: multicast detection unavailable ({e}); "
            "using P2P instead of multimem."
        )
        return False


@functools.lru_cache(maxsize=None)
def _get_symm_buffer(group_name: str, dtype: torch.dtype) -> torch.Tensor:
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
    out = input.clone()
    dist.all_reduce(
        out,
        op=getattr(dist.ReduceOp, reduce_op.upper()),
        group=_resolve_process_group(GroupName(group_name)),
    )
    return out


def _symm_mem_all_reduce(
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
            view.copy_(input)
            torch.ops.symm_mem.multimem_all_reduce_(view, "sum", group_name)
            out.copy_(view)
        case _:
            raise AssertionError(f"unhandled algo: {algo}")

    return out


class SymmMemTPAllReduce(PerAxisRedistribution):
    @dataclass(kw_only=True, slots=True)
    class Config(PerAxisRedistribution.Config):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if (
            torch.is_grad_enabled()
            or self.config.fwd_op_dtype is not None
            or self.config.fwd_out_dtype is not None
        ):
            return super().__call__(x)

        axis = self.config.axis.value
        if self.config.src == self.config.dst or spmd_mesh_size(axis) == 1:
            return x

        mesh = current_spmd_mesh()
        assert mesh is not None, "SPMD redistribution requires an active DeviceMesh"
        if not _backend_supported():
            return super().__call__(x)

        return _symm_mem_all_reduce(
            x,
            "sum",
            mesh.get_group(axis).group_name,
        )


def _is_tp_all_reduce(cfg: PerAxisRedistribution.Config) -> bool:
    return (
        cfg.axis == MeshAxisName.TP
        and cfg.src == spmd.P
        and cfg.dst in (spmd.R, spmd.I)
    )


@override(
    "symm_mem_tp_all_reduce",
    target=PerAxisRedistribution.Config,
    predicate=_is_tp_all_reduce,
    description="Use symm_mem for TP Partial -> Replicate/Identity redistributions.",
)
def symm_mem_tp_all_reduce(
    cfg: PerAxisRedistribution.Config,
) -> SymmMemTPAllReduce.Config:
    return derive(cfg, SymmMemTPAllReduce.Config)
