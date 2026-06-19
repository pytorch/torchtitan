# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""DTensorMuon: the fsdp2 (core ``fully_shard``) Muon optimizer (gather-in-step).

Per param, lets DTensor do the all-gather (``full_tensor()``) inside ``step()``, runs
Newton-Schulz on the full tensor (one NS for a 2D matrix; batched per-expert NS for a 3D
``[E, m, n]`` MoE expert stack), and redistributes back to the param's sharding. The fsdp2
baseline against comm-efficient ``Owned`` Muon; the container that routes to it
(:class:`containers.FSDP2MuonOptimizers`) lives in :mod:`containers`.
"""

from __future__ import annotations

import torch
from torch.optim._muon import (
    _adjust_lr,
    _zeropower_via_newtonschulz,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
)

from .newton_schulz import _zeropower_via_newtonschulz_batched


class DTensorMuon(torch.optim.Optimizer):
    """Muon for core ``fully_shard`` (FSDP2) DTensor params: gather each tensor in step().

    The "let DTensor do the all-gather in opt.step" baseline (vs comm-efficient ``Owned``):
    params are ``Shard`` DTensors from ``fully_shard``. Per param, the element-wise
    momentum update is computed on the local shard (still a DTensor), ``full_tensor()``
    all-gathers it to the full tensor on every rank, Newton-Schulz runs on the full tensor
    (redundant across ranks) -- one NS for a 2D matrix, batched per-expert NS for a 3D
    ``[E, m, n]`` MoE expert stack -- and the result is redistributed back to the param's
    sharding so only this rank's shard is written. Bit-exact with single-device Muon; cost
    is one all-gather + redundant NS per tensor per step (the comm ``Owned`` avoids).

    TODO(checkpoint): callers build this with a flat param list and no "param_names", so
    DCP save/load is not yet supported (see FSDP2MuonOptimizers and checkpoint_utils.py:140).
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
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
        from torch.distributed.tensor import DTensor, Replicate

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            adjust_lr_fn = group["adjust_lr_fn"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if not isinstance(param.data, DTensor) or not isinstance(grad, DTensor):
                    raise ValueError(
                        "DTensorMuon requires DTensor params/grads (core fully_shard); "
                        f"got param={type(param.data).__name__} "
                        f"grad={type(grad).__name__}."
                    )
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                # DTensor does the all-gather: reconstruct the full tensor on every rank.
                # 2D matrix -> one NS; 3D [E, m, n] MoE expert stack -> batched per-expert
                # NS, with the lr adjusted by the per-expert (m, n) shape.
                full_pre = update.full_tensor()
                if full_pre.ndim == 3:
                    full_update = _zeropower_via_newtonschulz_batched(
                        full_pre, ns_coefficients, ns_steps, eps
                    )
                    lr_shape = tuple(full_update.shape[1:])
                else:
                    full_update = _zeropower_via_newtonschulz(
                        full_pre, ns_coefficients, ns_steps, eps
                    )
                    lr_shape = tuple(full_update.shape)
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)

                # Scatter back to the param's sharding (local slice, no collective): the
                # full update is identical on every rank, so treat it as Replicate and
                # redistribute to the param's placements to keep only this rank's shard.
                mesh = param.device_mesh
                update_dt = DTensor.from_local(
                    full_update,
                    mesh,
                    [Replicate()] * mesh.ndim,
                    run_check=False,
                ).redistribute(mesh, param.placements)

                param.mul_(1 - lr * weight_decay)
                param.add_(update_dt, alpha=-adjusted_lr)
        return loss
