# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async implementations of tensor-parallel projection boundaries."""

from __future__ import annotations

import logging

from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.distributed.linear import AsyncAllGatherLinear, AsyncLinearReduceScatter
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.common.linear import ColumnParallelLinear, RowParallelLinear


logger = logging.getLogger(__name__)


_WARNED_NO_TP = False


def _warn_once_no_tp_overlap() -> None:
    """Warn when async TP was selected but no multi-rank TP group is active."""
    global _WARNED_NO_TP
    if not _WARNED_NO_TP:
        _WARNED_NO_TP = True
        logger.warning(
            "Async tensor parallelism was selected but tensor parallelism is not "
            "active; running the synchronous linear implementation."
        )


def _tp_group_from_context() -> dist.ProcessGroup | None:
    """The TP process group from the current spmd_types mesh context, or None.

    Resolved per forward rather than captured at parallelize time. The mesh
    context is only entered inside the trainer's ``train_context``, so it is
    unavailable during ``__init__`` and ``parallelize``. Resolving it here keeps
    process-group state out of the modules.

    None means "run the stock projection": either no mesh context or TP is degree
    1, in which case there is no collective to fuse.
    """
    mesh = current_spmd_mesh()
    if mesh is None or "tp" not in (mesh.mesh_dim_names or ()):
        return None
    tp_group = mesh.get_group("tp")
    return tp_group if tp_group.size() > 1 else None


def validate_async_tp_preconditions(*, enable_sp: bool) -> None:
    """Reject configurations the fused modules cannot serve.

    Called from the sharding setup, which is the first point that sees both the
    selected modules and the parallelism settings. Neither condition is detectable
    from inside a module at runtime: under spmd_types an activation is a plain
    local tensor with no placements to inspect.
    """
    if not enable_sp:
        raise ValueError(
            "Async tensor parallelism requires "
            "parallelism.enable_sequence_parallel; its fused kernels implement "
            "an all-gather before column-parallel GEMMs and a reduce-scatter "
            "after row-parallel GEMMs."
        )


class AsyncColumnParallelLinear(ColumnParallelLinear):
    """Overlap an input all-gather with a column-parallel GEMM."""

    @dataclass(kw_only=True, slots=True)
    class Config(ColumnParallelLinear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if type(self) is not AsyncColumnParallelLinear:
            raise RuntimeError(
                "AsyncColumnParallelLinear does not support converted linear modules"
            )
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)

        weight, bias = self._flatten_weight_and_bias()
        output = AsyncAllGatherLinear.apply(
            input,
            weight,
            bias,
            tp_group,
            tp_group.group_name,
        )
        return self._unflatten_output(output)


class AsyncRowParallelLinear(RowParallelLinear):
    """Overlap a row-parallel GEMM with its output reduce-scatter."""

    @dataclass(kw_only=True, slots=True)
    class Config(RowParallelLinear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if type(self) is not AsyncRowParallelLinear:
            raise RuntimeError(
                "AsyncRowParallelLinear does not support converted linear modules"
            )
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)
        return AsyncLinearReduceScatter.apply(
            input,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )


__all__ = [
    "AsyncColumnParallelLinear",
    "AsyncRowParallelLinear",
    "validate_async_tp_preconditions",
]
