# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model components that fold the TP collectives into their GEMMs.

:class:`AsyncColumnParallelLinear` and :class:`AsyncRowParallelLinear` are drop-in
replacements for stock linear projections. They move the TP collective into
the GEMM over the autograd Functions in ``torchtitan/distributed/linear.py``.
``AsyncRowParallelLinear`` serves both attention's ``wo`` and the FFN's ``w2``;
``AsyncColumnParallelLinear`` serves both attention's ``wqkv`` and the FFN's
``w13``. Nothing about the primitives is attention-specific, and MoE
projections could use the same pair.

What lives here is the wiring around each collective and the fallbacks, while
``torchtitan/distributed/linear.py`` holds the collective+GEMM math itself.

Selected by passing ``tp_gemm_backend="dist_gemm"`` to ``make_gqa_config`` or
``make_ffn_config`` (see ``config_utils.py``), which also drops the boundary
all-gather these modules take over.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.distributed.linear import (
    AllGatherLinear as AllGatherLinearFunction,
    LinearReduceScatter,
)

from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.common.linear import Linear


logger = logging.getLogger(__name__)

# Shape suffix legend:
#   T = token dimensions, F = feed-forward hidden dimension


_WARNED_NO_TP = False


def _warn_once_no_tp_overlap() -> None:
    """Say so when the dist-GEMM modules were selected but TP is not on.

    Otherwise the fallback is indistinguishable from the feature working: the run
    succeeds, the loss looks fine, and nothing ran fused. The preconditions cover
    the wrong-backend and SP-disabled cases with hard errors, but TP=1 has to stay
    runnable, so it warns instead.
    """
    global _WARNED_NO_TP
    if not _WARNED_NO_TP:
        _WARNED_NO_TP = True
        logger.warning(
            "tp_gemm_backend='dist_gemm' selected but tensor parallelism is not "
            "active; running the standard projection path without collective "
            "overlap."
        )


def _tp_group_from_context() -> dist.ProcessGroup | None:
    """The TP process group from the current spmd_types mesh context, or None.

    Resolved per forward rather than captured at parallelize time. The mesh
    context is only entered inside the trainer's ``train_context``, so it is
    unavailable during ``__init__`` and ``parallelize`` -- and reading it here
    means these modules need no ``parallelize`` override and hold no group state.

    None means "run the stock projection": either no mesh context or TP is degree
    1, in which case there is no collective to fuse.
    """
    mesh = current_spmd_mesh()
    if mesh is None or "tp" not in (mesh.mesh_dim_names or ()):
        return None
    tp_group = mesh.get_group("tp")
    return tp_group if tp_group.size() > 1 else None


def validate_dist_gemm_preconditions(*, enable_sp: bool) -> None:
    """Reject configurations the fused modules cannot serve.

    Called from the sharding setup, which is the first point that sees both the
    selected modules and the parallelism settings. Neither condition is detectable
    from inside a module at runtime: under spmd_types an activation is a plain
    local tensor with no placements to inspect.
    """
    if not enable_sp:
        raise ValueError(
            "tp_gemm_backend='dist_gemm' requires "
            "parallelism.enable_sequence_parallel; the fused GEMMs replace the SP "
            "all-gather and reduce-scatter, so there is nothing for them to fuse "
            "with SP disabled."
        )


class AsyncColumnParallelLinear(Linear):
    """Column-parallel linear that all-gathers its TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)

        return AllGatherLinearFunction.apply(
            input,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )


class AsyncRowParallelLinear(Linear):
    """Attention output projection: matmul fused with the TP reduce-scatter.

    Named for the role it fills rather than the collective it performs, so it does
    not read like the :class:`LinearReduceScatter` autograd Function it calls. The
    class itself is a plain rowwise linear and would work for any row-parallel
    projection; today it is only wired in as ``wo``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Same fields as a stock Linear. The subclass exists because it is what
        binds ``Config.build()`` to this module rather than the stock one, so it
        cannot be deleted as empty."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(input)

        return LinearReduceScatter.apply(
            input,
            self.weight,
            self.bias,
            tp_group,
            tp_group.group_name,
        )


__all__ = [
    "validate_dist_gemm_preconditions",
    "AsyncColumnParallelLinear",
    "AsyncRowParallelLinear",
]
