# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async implementations of the TP communication-aware linear modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.distributed.linear import (
    AsyncAllGatherLinear as AsyncAllGatherLinearFunction,
    AsyncLinearReduceScatter as AsyncLinearReduceScatterFunction,
)
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.linear import AllGatherLinear, LinearReduceScatter
from torchtitan.tools.logging import logger


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
    """Return the active multi-rank TP process group, if one exists."""
    mesh = current_spmd_mesh()
    if mesh is None or "tp" not in (mesh.mesh_dim_names or ()):
        return None
    tp_group = mesh.get_group("tp")
    return tp_group if tp_group.size() > 1 else None


def validate_async_tp_preconditions(*, enable_sp: bool) -> None:
    """Reject configurations unsupported by the async TP linear modules."""
    backend = get_spmd_backend()
    if backend != "spmd_types":
        raise ValueError(
            "Async tensor parallelism requires "
            f"parallelism.spmd_backend='spmd_types', got {backend!r}."
        )
    if not enable_sp:
        raise ValueError(
            "Async tensor parallelism requires "
            "parallelism.enable_sequence_parallel; its fused kernels implement "
            "an all-gather before column-parallel GEMMs and a reduce-scatter "
            "after row-parallel GEMMs."
        )


class AsyncAllGatherLinear(AllGatherLinear):
    """Overlap the input all-gather with a column-parallel GEMM."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllGatherLinear.Config):
        pass

    def _uses_explicit_tp_input_redistribution(self) -> bool:
        return type(self) is AsyncAllGatherLinear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._uses_explicit_tp_input_redistribution():
            return super().forward(input)
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)
        return AsyncAllGatherLinearFunction.apply(
            input,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )


class AsyncLinearReduceScatter(LinearReduceScatter):
    """Overlap a row-parallel GEMM with its output reduce-scatter."""

    @dataclass(kw_only=True, slots=True)
    class Config(LinearReduceScatter.Config):
        pass

    def _uses_explicit_tp_output_redistribution(self) -> bool:
        return type(self) is AsyncLinearReduceScatter

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._uses_explicit_tp_output_redistribution():
            return super().forward(input)
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)
        return AsyncLinearReduceScatterFunction.apply(
            input,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )


__all__ = [
    "AsyncAllGatherLinear",
    "AsyncLinearReduceScatter",
    "validate_async_tp_preconditions",
]
