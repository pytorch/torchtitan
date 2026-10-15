# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unsharded reference for Algorithm 1 of the DeepSeek V4.1 report."""

import math

import torch
from torch.distributed.tensor import DTensor


class SinkhornMomentum(torch.optim.Optimizer):
    """Nesterov momentum followed by alternating row/column normalization.

    This dense reference deliberately rejects sparse and distributed gradients:
    column normalization requires statistics from the entire logical matrix.
    """

    def __init__(
        self,
        params,
        *,
        lr=2.6e-4,
        momentum=0.95,
        lr_scale=0.18,
        num_steps=11,
        eps=1e-20,
        row_threshold=1e-3,
        fused=False,
        foreach=False,
    ):
        if lr < 0 or lr_scale < 0 or not 0 <= momentum < 1:
            raise ValueError("Invalid learning rate or momentum")
        if num_steps < 1 or num_steps % 2 != 1:
            raise ValueError("Sinkhorn normalization requires a positive odd num_steps")
        if eps <= 0 or row_threshold < 0 or fused or foreach:
            raise ValueError("Invalid normalization options; this reference is unfused")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                lr_scale=lr_scale,
                num_steps=num_steps,
                eps=eps,
                row_threshold=row_threshold,
            ),
        )
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2 or (
                    isinstance(param, DTensor) and param.device_mesh.size() != 1
                ):
                    raise ValueError(
                        "SinkhornMomentum requires a full 2D matrix on one rank"
                    )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        saved_ids = [
            key for group in state_dict["param_groups"] for key in group["params"]
        ]
        params = [param for group in self.param_groups for param in group["params"]]
        for key, param in zip(saved_ids, params, strict=True):
            saved = state_dict["state"].get(key, {})
            if "momentum_buffer" in saved:
                self.state[param]["momentum_buffer"] = (
                    saved["momentum_buffer"]
                    .to(
                        device=param.device,
                        dtype=torch.float32,
                    )
                    .clone()
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise ValueError(
                        "Sparse or distributed Sinkhorn updates are not implemented"
                    )
                local_param = param.to_local() if isinstance(param, DTensor) else param
                local_grad = (
                    param.grad.to_local()
                    if isinstance(param.grad, DTensor)
                    else param.grad
                )
                grad = local_grad.float()
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum = state["momentum_buffer"]
                beta = group["momentum"]
                momentum.mul_(beta).add_(grad, alpha=1 - beta)
                update = beta * momentum + (1 - beta) * grad
                row_norm = update.norm(dim=1, keepdim=True)
                update.masked_fill_(
                    row_norm <= group["row_threshold"] * row_norm.mean(), 0
                )
                for index in range(group["num_steps"]):
                    axis = 1 if index % 2 == 0 else 0
                    update.div_(update.norm(dim=axis, keepdim=True) + group["eps"])
                update.mul_(math.sqrt(param.shape[1]))
                local_param.add_(
                    update.to(param.dtype), alpha=-group["lr"] * group["lr_scale"]
                )
        return loss
