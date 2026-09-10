# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model components that fold the TP collectives into their GEMMs.

:class:`AllGatherFusedQKVLinear`, :class:`RowParallelLinear` and
:class:`DistGEMMFeedForward` are drop-in replacements for the stock QKV,
output and SwiGLU projections. They move the TP collective into the GEMM over
the autograd Functions in ``torchtitan/distributed/linear.py``.
``RowParallelLinear`` serves both attention's ``wo`` and the FFN's ``w2``;
nothing about the primitives is attention-specific, and MoE projections could
use the same pair.

What lives here is the wiring around each collective and the fallbacks, while
``torchtitan/distributed/linear.py`` holds the collective+GEMM math itself.

Selected by passing ``tp_gemm_backend="dist_gemm"`` to ``make_gqa_config`` or
``make_ffn_config`` (see ``config_utils.py``), which also drops the boundary
all-gather these modules take over.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_remat as remat

from torchtitan.distributed.linear import AllGatherLinear, LinearReduceScatter

from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.distributed.utils import get_spmd_backend

from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger

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
            "active; running the standard feed-forward path without collective "
            "overlap."
        )


def _tp_group_from_context() -> dist.ProcessGroup | None:
    """The TP process group from the current spmd_types mesh context, or None.

    Resolved per forward rather than captured at parallelize time. The mesh
    context is only entered inside the trainer's ``train_context``, so it is
    unavailable during ``__init__`` and ``parallelize`` -- and reading it here
    means these modules need no ``parallelize`` override and hold no group state.

    None means "run the stock projection": either no mesh context (non-spmd_types
    caller) or TP is degree 1, in which case there is no collective to fuse.
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
    backend = get_spmd_backend()
    if backend != "spmd_types":
        raise ValueError(
            "tp_gemm_backend='dist_gemm' requires "
            f"parallelism.spmd_backend='spmd_types', got {backend!r}. The fused "
            "modules take and return plain local tensors; the DTensor backends are "
            "being deprecated and are not supported."
        )
    if not enable_sp:
        raise ValueError(
            "tp_gemm_backend='dist_gemm' requires "
            "parallelism.enable_sequence_parallel; the fused GEMMs replace the SP "
            "all-gather and reduce-scatter, so there is nothing for them to fuse "
            "with SP disabled."
        )


class AllGatherFusedQKVLinear(QKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(QKVLinear.Config):
        """Same fields as the stock fused QKV. The subclass exists because it is
        what binds ``Config.build()`` to this module rather than the stock one, so
        it cannot be deleted as empty."""

    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(x)

        qkv = AllGatherLinear.apply(
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


class RowParallelLinear(Linear):
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


class DistGEMMFeedForward(FeedForward):
    """SwiGLU feed-forward with both TP collectives folded into its GEMMs.

    The fused ``w13`` projection consumes an all-gather of the sequence shard;
    ``w2`` is row-parallel and reduce-scatters back to a sequence shard. Logical
    checkpoint FQNs remain the logical ``w1``/``w2``/``w3``.

    Falls back to the standard forward when TP is off.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        """Same fields as the standard FFN. The subclass exists because it is what
        binds ``Config.build()`` to this module rather than the standard one, so it
        cannot be deleted as empty."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_group = _tp_group_from_context()
        if tp_group is None:
            _warn_once_no_tp_overlap()
            return super().forward(x)

        gate_up_TF = remat.region(
            AllGatherLinear.apply,
            self.remat_region_name("w13"),
            recompute=self.remat_should_recompute("w13"),
        )(
            x,
            self.w13.weight,
            self.w13.bias,
            tp_group,
            tp_group.group_name,
        )
        gate_TF, up_TF = gate_up_TF.unflatten(-1, (-1, 2)).unbind(-1)
        # Elementwise on feature-sharded activations: no collective.
        remat.recompute_needs_tensor(gate_TF, up_TF)
        h_TF = self._activation(gate_TF, up_TF)
        out_TD = remat.region(
            LinearReduceScatter.apply,
            self.remat_region_name("w2"),
            recompute=self.remat_should_recompute("w2"),
        )(
            h_TF,
            self.w2.weight,
            self.w2.bias,
            tp_group,
            tp_group.group_name,
        )
        remat.recompute_needs_tensor(out_TD)
        return out_TD


__all__ = [
    "validate_dist_gemm_preconditions",
    "DistGEMMFeedForward",
    "AllGatherFusedQKVLinear",
    "RowParallelLinear",
]
