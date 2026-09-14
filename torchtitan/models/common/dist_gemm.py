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

from torchtitan.distributed.linear import (
    AsyncAllGatherLinear as AsyncAllGatherLinearFunction,
    AsyncLinearReduceScatter as AsyncLinearReduceScatterFunction,
)
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.linear import AllGatherLinear, Linear, LinearReduceScatter


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


class AsyncAllGatherLinear(AllGatherLinear):
    """Overlap an input all-gather with a column-parallel GEMM."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllGatherLinear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if type(self) is not AsyncAllGatherLinear:
            raise RuntimeError(
                "AsyncAllGatherLinear does not support converted linear modules"
            )
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


class AsyncAllGatherQKVLinear(QKVLinear):
    """Overlap the input all-gather with a fused QKV projection."""

    @dataclass(kw_only=True, slots=True)
    class Config(QKVLinear.Config):
        pass

    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if type(self) is not AsyncAllGatherQKVLinear or type(self.wqkv) is not Linear:
            raise RuntimeError(
                "AsyncAllGatherQKVLinear does not support converted QKV projections"
            )
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(x)

        qkv = AsyncAllGatherLinearFunction.apply(
            x,
            self.wqkv.weight,
            self.wqkv.bias,
            tp_group,
            tp_group.group_name,
        )
        num_tokens = qkv.shape[0]
        qkv = qkv.view(num_tokens, -1, self.r_dim, self.head_dim)
        xq, xk, xv = torch.split(qkv, [self.heads_per_kv, 1, 1], dim=-2)
        return (
            xq.reshape(num_tokens, -1, self.head_dim).contiguous(),
            xk.reshape(num_tokens, -1, self.head_dim).contiguous(),
            xv.reshape(num_tokens, -1, self.head_dim).contiguous(),
        )


class AsyncLinearReduceScatter(LinearReduceScatter):
    """Overlap a row-parallel GEMM with its output reduce-scatter."""

    @dataclass(kw_only=True, slots=True)
    class Config(LinearReduceScatter.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if type(self) is not AsyncLinearReduceScatter:
            raise RuntimeError(
                "AsyncLinearReduceScatter does not support converted linear modules"
            )
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
    "AsyncAllGatherQKVLinear",
    "AsyncLinearReduceScatter",
    "validate_async_tp_preconditions",
]
