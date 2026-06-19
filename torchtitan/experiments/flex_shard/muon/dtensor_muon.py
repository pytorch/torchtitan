# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""DTensorMuon: the fsdp2 (core ``fully_shard``) Muon optimizer.

Muon for ``fully_shard`` DTensor params. **2D dense matrices** are all-gathered in
``step()`` (``full_tensor()``), get one Newton-Schulz on the full matrix, and are scattered
back. **3D MoE expert stacks are EP-aware:** under ``fully_shard`` + EP the experts are
sharded on the expert dim across the ep/efsdp axes, so each rank already holds whole local
experts -- DTensorMuon runs batched per-expert NS on the local shard with **no all-gather**
(matching the EP-local GroupedMuon path, not a full expert-stack gather). The fsdp2 baseline
against comm-efficient ``Owned`` Muon; the container that routes to it
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
    """Muon for core ``fully_shard`` (FSDP2) DTensor params.

    Params are ``Shard`` DTensors from ``fully_shard``. Per param the element-wise momentum
    update is computed on the local shard (still a DTensor), then:

    * **2D dense matrix** -- ``full_tensor()`` all-gathers it to the full matrix on every
      rank (in bf16), one Newton-Schulz runs (redundant across ranks), and the result is
      scattered back to the param's shard. This is the "let DTensor gather in opt.step" cost
      that comm-efficient ``Owned`` avoids.
    * **3D ``[E, m, n]`` MoE expert stack -- EP-aware** -- under ``fully_shard`` + EP the
      experts are sharded on the *expert* dim (dim 0) across both the ep and efsdp axes, so
      each rank already holds whole local experts. Batched per-expert NS runs on the local
      shard and is written back in place with **no all-gather** -- matching EP-local
      GroupedMuon, not a full expert-stack gather (which is what an EP-unaware ``full_tensor``
      would do: gather the entire stack onto every rank + redundant NS).

    Bit-exact with single-device Muon (per-expert NS is independent, so local == slice of
    global).

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
        from torch.distributed.tensor import DTensor, Replicate, Shard

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

                mesh = param.device_mesh
                if param.data.ndim == 3:
                    # 3D MoE expert stack -- EP-aware. Under core fully_shard + EP the experts
                    # are sharded on the *expert* dim (dim 0) across both the ep and efsdp mesh
                    # axes, so every rank already holds WHOLE local experts (complete [m, n]
                    # matrices, just fewer of them). Run per-expert (batched) Newton-Schulz on
                    # the local shard and write it back in place -- no all-gather. This matches
                    # the EP-local GroupedMuon path; a plain full_tensor() would instead gather
                    # the entire expert stack onto every rank (EP-unaware) and run redundant NS.
                    if any(
                        isinstance(pl, Shard) and pl.dim >= 1
                        for pl in update.placements
                    ):
                        raise NotImplementedError(
                            "EP-aware DTensorMuon expects experts sharded on the expert dim "
                            f"(dim 0); got placements {update.placements}. Experts sharded "
                            "within a matrix dim (gather-before-NS) are future work."
                        )
                    local = update.to_local().to(torch.bfloat16)
                    local_update = _zeropower_via_newtonschulz_batched(
                        local, ns_coefficients, ns_steps, eps
                    )
                    lr_shape = tuple(local_update.shape[1:])
                    update_dt = DTensor.from_local(
                        local_update, mesh, param.placements, run_check=False
                    )
                else:
                    # 2D dense matrix: DTensor all-gathers the full matrix on every rank (in
                    # bf16 -- NS casts to bf16 as its first op, and the fp32->bf16 cast commutes
                    # with the lossless all-gather, so this is bit-identical while halving the
                    # gathered bytes; momentum state stays fp32), runs one Newton-Schulz, then
                    # scatters the (Replicate) full update back to the param's shard (a local
                    # slice, no extra collective).
                    full_pre = update.to(torch.bfloat16).full_tensor()
                    full_update = _zeropower_via_newtonschulz(
                        full_pre, ns_coefficients, ns_steps, eps
                    )
                    lr_shape = tuple(full_update.shape)
                    update_dt = DTensor.from_local(
                        full_update, mesh, [Replicate()] * mesh.ndim, run_check=False
                    ).redistribute(mesh, param.placements)

                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)
                param.mul_(1 - lr * weight_decay)
                param.add_(update_dt, alpha=-adjusted_lr)
        return loss
