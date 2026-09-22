# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP primitives for MoE expert parallelism.

MoonEP (https://github.com/MoonshotAI/MoonEP) moves expert weights over NVLink
instead of moving every token to its expert's owner, so each rank receives a
fixed ``S x K`` tokens whatever the routing does. The dispatcher that drives
these lives in ``models/common/token_dispatcher.py`` and the expert side in
``models/common/moe.py``.
"""

import torch

DEFAULT_NUM_SMS = 32
"""SMs MoonEP's kernels may occupy; MoonEP's own default."""


def _import_moonep():
    try:
        import moonep  # pyrefly: ignore [missing-import]
    except ImportError as err:
        raise ImportError(
            "MoonEP is not installed. It is an optional dependency, like "
            "DeepEP: install from https://github.com/MoonshotAI/MoonEP, and "
            "note that it requires NVLink-connected GPUs. Use another "
            "comm_backend on hardware without that topology."
        ) from err
    return moonep


# Routing weights ride along as a second output so their gradient reaches the router.
class _MoonEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, plan_out, x_SH, weights_SK, ids_SK, counts_E):
        hidden_nvsh, weights_nvs, cu_seqlens, plan = buffer.dispatch(
            x_SH, weights_SK, ids_SK, counts_E, zero_copy=False
        )
        # The plan is not a tensor, so it leaves through the caller's list.
        plan_out.append(plan)
        ctx.buffer = buffer
        ctx.plan = plan
        ctx.shape_nvsh = hidden_nvsh.shape
        return hidden_nvsh, weights_nvs, cu_seqlens

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_hidden_nvsh, grad_weights_nvs, _grad_cu
    ):
        if grad_hidden_nvsh is None and grad_weights_nvs is None:
            return None, None, None, None, None, None
        if grad_hidden_nvsh is None:
            grad_hidden_nvsh = torch.zeros(
                ctx.shape_nvsh, dtype=torch.bfloat16, device=grad_weights_nvs.device
            )
        grad_x_SH, grad_weights_SK, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden_nvsh.to(torch.bfloat16).contiguous(),
            route_weights_nvs=(
                None
                if grad_weights_nvs is None
                else grad_weights_nvs.to(torch.float32).contiguous()
            ),
        )
        return None, None, grad_x_SH, grad_weights_SK, None, None


class _MoonEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, plan, hidden_nvsh):
        out_SH, _, _ = buffer.combine(plan=plan, hidden_nvsh=hidden_nvsh)
        ctx.buffer = buffer
        ctx.plan = plan
        return out_SH

    @staticmethod
    def backward(ctx, grad_out_SH):  # pyrefly: ignore[bad-override]
        grad_hidden_nvsh, _, _, _ = ctx.buffer.dispatch(
            grad_out_SH.to(torch.bfloat16), plan=ctx.plan
        )
        return None, None, grad_hidden_nvsh


def dispatch(buffer, plan_out, x_SH, weights_SK, ids_SK, counts_E):
    """Forward transport; its backward is ``combine`` on the same plan."""
    return _MoonEPDispatch.apply(buffer, plan_out, x_SH, weights_SK, ids_SK, counts_E)


def combine(buffer, plan, hidden_nvsh):
    """Forward transport; its backward is ``dispatch`` on the same plan."""
    return _MoonEPCombine.apply(buffer, plan, hidden_nvsh)


def make_buffer(*, S, H, K, E, num_ep_ranks, group):
    """MoonEP's persistent buffer, a collective on the EP group."""
    return _import_moonep().Buffer(
        S=S,
        H=H,
        K=K,
        E=E,
        num_ep_ranks=num_ep_ranks,
        num_sms=DEFAULT_NUM_SMS,
        group=group,
    )


class MoonEPTableBackendNVLink:
    """Expert rows over NVLink through MoonEP's public prefetch and reduce.

    Each projection gets one pool, an NVL-distributed ``[R, E / R, in, out]``
    tensor whose ``rank`` chunk holds this rank's prefetch slots and, for the
    gradients, the slot gradients its peers read back.
    """

    def __init__(self, ep_mesh, dispatcher):
        self.ep_mesh = ep_mesh
        self.rank, self.size = ep_mesh.get_local_rank(), ep_mesh.size()
        self.group = ep_mesh.get_group()
        self.dispatcher = dispatcher
        self.own_rows = 0
        self._prefetch: dict[str, torch.Tensor] = {}
        self._reduce: dict[str, torch.Tensor] = {}

    def configure(self, *, num_experts: int, num_slots: int) -> None:
        if num_slots != num_experts // self.size:
            raise ValueError(
                f"MoonEP gives every rank E / R = {num_experts // self.size} "
                f"slots; got {num_slots}."
            )
        self.own_rows = num_slots

    def _pool(self, in_dim: int, out_dim: int, dtype) -> torch.Tensor:
        moonep = _import_moonep()
        rows = moonep.buffer.pad_dim0_for_alignment(
            [self.own_rows, in_dim, out_dim], dtype
        )
        if rows != self.own_rows:
            raise ValueError(
                f"MoonEP wants the pool chunk [{self.own_rows}, {in_dim}, "
                f"{out_dim}] on its VMM granularity, which pads to {rows}."
            )
        pool = moonep.buffer.create_nvl_dist_tensor(
            [self.own_rows, in_dim, out_dim],
            dtype,
            self.rank,
            self.size,
            group=self.group,
        ).view(self.size, self.own_rows, in_dim, out_dim)
        pool[self.rank].zero_()
        return pool

    def alloc_prefetch_rows(self, name: str, in_dim: int, out_dim: int):
        """This rank's slot rows, which peers write and the experts read."""
        self._prefetch[name] = self._pool(in_dim, out_dim, torch.bfloat16)
        return self._prefetch[name][self.rank]

    def alloc_grad_rows(self, name: str, in_dim: int, out_dim: int):
        """This rank's own-row gradients and the slot gradients peers read."""
        self._reduce[name] = self._pool(in_dim, out_dim, torch.float32)
        own = torch.zeros(
            self.own_rows,
            in_dim,
            out_dim,
            dtype=torch.float32,
            device=self._reduce[name].device,
        )
        return own, self._reduce[name][self.rank]

    def prefetch(self, plan, local_rows: dict[str, torch.Tensor]) -> None:
        """Fill the slots for *plan* from the rows their owners hold."""
        self.dispatcher.buffer.prefetch_weight(
            plan=plan,
            local_gate_weight=local_rows["gate"],
            local_up_weight=local_rows["up"],
            local_down_weight=local_rows["down"],
            gate_prefetch_buffer=self._prefetch["gate"],
            up_prefetch_buffer=self._prefetch["up"],
            down_prefetch_buffer=self._prefetch["down"],
        )

    def reduce_grad(self, plan, own_grads: dict[str, torch.Tensor]) -> None:
        """Send each slot's gradient home and add the ones homed here."""
        self.dispatcher.buffer.reduce_grad(
            plan=plan,
            local_gate_grad=own_grads["gate"],
            local_up_grad=own_grads["up"],
            local_down_grad=own_grads["down"],
            gate_reduce_buffer=self._reduce["gate"],
            up_reduce_buffer=self._reduce["up"],
            down_reduce_buffer=self._reduce["down"],
        )
