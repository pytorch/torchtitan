# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""GroupedMuon: a communication-efficient per-expert Muon optimizer for MoE experts.

The Muon optimizer for "whole experts on this rank (``Shard(0)``) -> run per-expert
Newton-Schulz locally, no gather" over 3D ``[E, m, n]`` expert stacks. The within-matrix-sharded
counterpart (``Shard(>=1)``) is :class:`gather_muon.GatherGroupedMuon`; the routing that
chooses between them lives in :mod:`containers`.
"""

from __future__ import annotations

import torch
from torch.optim._muon import (
    _adjust_lr,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
)

from ..example.shard import Shard

from ..flex_shard.sharded_param import get_placements
from .newton_schulz import _zeropower_via_newtonschulz_batched


class GroupedMuon(torch.optim.Optimizer):
    """Muon for 3D MoE expert stacks via per-expert (batched) Newton-Schulz.

    A rank's local expert parameter is ``[local_E, m, n]``. When expert parallelism shards
    the expert (leading) dim -- ``Shard(0)``, which holds whenever ``efsdp <=
    num_local_experts`` (equivalently world_size <= num_experts) -- every ``[m, n]`` is a
    *whole* expert matrix, so Newton-Schulz runs per expert **locally and comm-efficient** (NS is
    per-matrix and a rank's experts are disjoint from other ranks'). One batched ``baddbmm``
    NS (:func:`_zeropower_via_newtonschulz_batched`) orthogonalizes the whole local stack at
    once -- same FLOPs as per-expert NS, far fewer launches.

    If an expert matrix is itself sharded (``Shard(dim>=1)``, i.e. ``efsdp >
    num_local_experts``), NS on a partial matrix is invalid; that needs a gather over the
    expert mesh before NS (as :class:`gather_muon.GatherMuon` does for 2D) and is not handled
    here -- such a placement raises with guidance to use :class:`gather_muon.GatherGroupedMuon`.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        ns_steps: int = DEFAULT_NS_STEPS,
        eps: float = EPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            coeffs = group["ns_coefficients"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            for p in group["params"]:
                if p.grad is None or p.numel() == 0:
                    continue
                if p.grad.ndim != 3:
                    raise ValueError(
                        "GroupedMuon expects 3D [num_experts, m, n] expert stacks, got "
                        f"ndim={p.grad.ndim}."
                    )
                placements = get_placements(p)
                if placements is not None and any(
                    isinstance(pl, Shard) and pl.dim >= 1 for pl in placements
                ):
                    raise NotImplementedError(
                        "GroupedMuon: experts are sharded within the matrix (efsdp > "
                        "num_local_experts, i.e. world_size > num_experts). NS on a partial "
                        "matrix is invalid -- use GatherGroupedMuon (gather-before-NS over "
                        "the expert mesh) for these; the FlexShard gather/owned Muon "
                        "containers route Shard(dim>=1) experts there automatically."
                    )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - momentum)
                update = p.grad.lerp(buf, momentum) if nesterov else buf
                # One batched NS over the local [local_E, m, n] stack (per-expert ortho).
                ortho = _zeropower_via_newtonschulz_batched(
                    update, coeffs, ns_steps, eps
                )
                # lr adjustment uses the per-expert matrix shape (whole here).
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape[1:])
                p.mul_(1 - lr * weight_decay)
                p.add_(ortho, alpha=-adjusted_lr)
        return loss
