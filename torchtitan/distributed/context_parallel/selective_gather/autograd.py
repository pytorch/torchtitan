# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Single autograd-aware API for the selective K/V gather.

``selective_gather`` is the one entry point a model calls. It is a differentiable
op (forward gather + backward scatter-reduce) that dispatches to whichever
backend the context selected:

  * ``"lsa"`` -- CuTeDSL LSA kernels. Fast path.
  * ``"p2p"`` -- portable ``batch_isend_irecv`` baseline. Fallback (AMD / no-GIN
    / no-CuTeDSL / pre-Hopper).

The ``"gin"`` inter-node backend is auto-selected for capable multi-node CP
groups but is not yet wired into this differentiable API (the gin_kernel is
driven directly for now); ``selective_gather`` warns and falls back to ``"p2p"``
for a ``"gin"`` context -- correct over an inter-node group, just slower.

The backend is chosen once at ``SelectiveGatherContext`` construction (see
``backend.select_backend`` for the capability check); callers do not pick it.
Backend-specific modules (CuTeDSL / nccl4py for LSA) are imported lazily inside
the Functions so the package stays importable on hosts that only support P2P.

Usage (set up once when the plan is known, then call every forward)::

    meta = build_plan_metadata(plan, group, blocks_per_rank=...)
    ctx = SelectiveGatherContext(group, shard_numel=..., block_numel=...,
                                 batch_size=B, max_consumers=meta.max_consumers, ...)
    k_full = selective_gather(k_local, ctx, meta)

One tensor (K or V) per call; gather K and V with separate contexts.
"""

import warnings

import torch
from torch.autograd.function import once_differentiable


def _empty_out(sg_ctx, ref):
    # torch.empty (not zeros): the gather fills only the planned slots; the rest
    # sit outside the attention window and are masked downstream, never read.
    # Zeroing the full buffer every step would be wasted work.
    return torch.empty(
        sg_ctx.cp_size * sg_ctx.shard_numel, dtype=sg_ctx.dtype, device=ref.device
    )


def _empty_grad(sg_ctx, ref):
    return torch.empty(sg_ctx.shard_numel, dtype=sg_ctx.dtype, device=ref.device)


class _LSAGatherFn(torch.autograd.Function):
    """LSA backend: signal-pad forward gather, staging-reduce backward."""

    @staticmethod
    def forward(actx, kv_local, sg_ctx, meta):  # pyrefly: ignore[bad-override]
        from .lsa_kernel import run_lsa_gather

        if meta.max_consumers > sg_ctx.max_consumers:
            raise ValueError(
                f"ctx.max_consumers ({sg_ctx.max_consumers}) < meta.max_consumers "
                f"({meta.max_consumers}): the LSA backward staging would overflow "
                "into a peer's window. Build the context with "
                "max_consumers=meta.max_consumers."
            )
        out = _empty_out(sg_ctx, kv_local)
        run_lsa_gather(sg_ctx, meta, kv_local.contiguous(), out, synchronize=False)
        actx.sg_ctx, actx.meta = sg_ctx, meta
        actx.kv_shape = kv_local.shape
        return out

    @staticmethod
    @once_differentiable
    def backward(actx, grad_out):  # pyrefly: ignore[bad-override]
        from .lsa_kernel import run_lsa_gather_backward

        d_kv = _empty_grad(actx.sg_ctx, grad_out)
        run_lsa_gather_backward(
            actx.sg_ctx, actx.meta, grad_out.contiguous(), d_kv, synchronize=False
        )
        return d_kv.view(actx.kv_shape), None, None


class _P2PGatherFn(torch.autograd.Function):
    """Portable backend: grouped batch_isend_irecv forward + backward."""

    @staticmethod
    def forward(actx, kv_local, sg_ctx, meta):  # pyrefly: ignore[bad-override]
        from .p2p import run_p2p_gather

        out = _empty_out(sg_ctx, kv_local)
        run_p2p_gather(sg_ctx, meta, kv_local.contiguous(), out)
        actx.sg_ctx, actx.meta = sg_ctx, meta
        # The transport works on element counts, so kv_local may have any shape;
        # the gradient has to come back in that shape.
        actx.kv_shape = kv_local.shape
        return out

    @staticmethod
    @once_differentiable
    def backward(actx, grad_out):  # pyrefly: ignore[bad-override]
        from .p2p import run_p2p_gather_backward

        d_kv = _empty_grad(actx.sg_ctx, grad_out)
        run_p2p_gather_backward(actx.sg_ctx, actx.meta, grad_out.contiguous(), d_kv)
        return d_kv.view(actx.kv_shape), None, None


def selective_gather(kv_local, sg_ctx, meta):
    """Differentiable selective K/V gather; dispatches on ``sg_ctx.backend``.

    Args:
        kv_local: this rank's local K/V shard, ``shard_numel`` elements (the
            activation to differentiate).
        sg_ctx: a ``SelectiveGatherContext`` (its ``backend`` picks the impl).
        meta: ``PlanMetadata`` from ``build_plan_metadata``, which carries the
            plan and is where its entries were validated. Its plan must name
            this rank's own blocks (``include_own=True``).

    Returns:
        The full-sequence gathered buffer (``cp_size * shard_numel``), with only
        the planned blocks filled (the rest masked downstream).

    Raises:
        ValueError: if the plan omits any of this rank's own blocks.
    """
    if meta.ranks_missing_own:
        raise ValueError(
            "selective_gather needs every rank's plan to name the blocks that "
            f"rank owns; ranks {list(meta.ranks_missing_own)} do not. Both "
            "backwards read that whole output region, so a plan built with "
            "include_own=False measures transport only and has no gradient."
        )
    backend = sg_ctx.backend
    if backend == "gin":
        # GIN is not yet wired into the differentiable API (gin_kernel is driven
        # directly for now). Don't silently run a different transport: warn and
        # fall back to p2p, which is correct on an inter-node group too.
        warnings.warn(
            "selective_gather: the 'gin' backend is not yet wired into the "
            "autograd API; falling back to 'p2p' (correct but slower "
            "inter-node). Drive gin_kernel directly for the GIN path.",
            stacklevel=2,
        )
        backend = "p2p"
    fn = _LSAGatherFn if backend == "lsa" else _P2PGatherFn
    return fn.apply(kv_local, sg_ctx, meta)
