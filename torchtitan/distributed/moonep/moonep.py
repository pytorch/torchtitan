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
import torch.distributed as dist

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


def padded_slot_count(base_slots: int, in_dim: int, out_dim: int) -> int:
    """``B`` raised to the VMM granularity MoonEP's reduce buffers are cut on."""
    moonep = _import_moonep()
    return int(
        moonep.buffer.pad_dim0_for_alignment(
            [base_slots, in_dim, out_dim], torch.float32
        )
    )


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


def make_buffer(*, S, H, K, E, num_ep_ranks, B, group):
    """MoonEP's persistent buffer, a collective on the EP group."""
    return _import_moonep().Buffer(
        S=S,
        H=H,
        K=K,
        E=E,
        num_ep_ranks=num_ep_ranks,
        num_sms=DEFAULT_NUM_SMS,
        B=B,
        group=group,
    )


def barrier_handles(buffer) -> dict:
    """The barrier handles ``launch_grad_reduce`` needs.

    MoonEP exposes no accessor for them, so they come off the Buffer's
    context; an upstream accessor would remove this reach.
    """
    ctx = buffer._require_ctx()
    return {
        "meta_buf": ctx["meta_buf"],
        "meta_stride": int(ctx["meta_chunk_padded"]),
        "barrier_off": int(ctx["BARRIER_OFF"]),
        "grid_sync_bar": ctx["grid_sync_bar"],
    }


class _MappedRows:
    """Per-owner NVLink mappings of one projection's ``[P + B, in, out]`` rows."""

    def __init__(self, ep_mesh, rank: int, size: int):
        self.ep_mesh, self.rank, self.size = ep_mesh, rank, size
        self.group = ep_mesh.get_group()
        self.owners: dict[str, list[torch.Tensor]] = {}

    def alloc(self, name: str, rows: int, in_dim: int, out_dim: int) -> torch.Tensor:
        moonep = _import_moonep()
        padded = moonep.buffer.pad_dim0_for_alignment(
            [rows, in_dim, out_dim], torch.bfloat16
        )
        mapped = []
        for owner in range(self.size):
            t = moonep.buffer.create_nvl_single_owner_tensor(
                [padded, in_dim, out_dim],
                torch.bfloat16,
                owner_rank=owner,
                local_rank=self.rank,
                group=self.group,
            )
            if owner == self.rank:
                t.zero_()
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            mapped.append(t[:rows])
        self.owners[name] = mapped
        return mapped[self.rank]


class MoonEPTableBackendNVLink:
    """Table backend on MoonEP's ``launch_prefetch`` and ``launch_grad_reduce``.

    ``plan.experts_to_copy`` is ``[R, B]`` int32: the global expert id in rank
    r's slot b, negative when unused.
    """

    def __init__(self, ep_mesh, dispatcher):
        self.ep_mesh = ep_mesh
        self.rank, self.size = ep_mesh.get_local_rank(), ep_mesh.size()
        self.group = ep_mesh.get_group()
        self.dispatcher = dispatcher
        self.num_experts = 0
        self.num_slots = 0
        self.own_rows = 0
        self._rows = _MappedRows(ep_mesh, self.rank, self.size)
        self._full_grads: dict[str, torch.Tensor] = {}
        self._reduce: dict[str, torch.Tensor] = {}

    def configure(self, *, num_experts: int, num_slots: int) -> None:
        self.num_experts, self.num_slots = num_experts, num_slots
        self.own_rows = num_experts // self.size

    def alloc_expert_rows(self, name, in_dim, out_dim):
        return self._rows.alloc(name, self.own_rows + self.num_slots, in_dim, out_dim)

    def alloc_grad_rows(self, name, in_dim, out_dim):
        moonep = _import_moonep()
        device = self._rows.owners[name][self.rank].device
        # launch_grad_reduce addresses grad rows by global expert id.
        full = torch.zeros(
            self.num_experts, in_dim, out_dim, dtype=torch.float32, device=device
        )
        reduce_full = moonep.buffer.create_nvl_dist_tensor(
            [self.num_slots, in_dim, out_dim],
            torch.float32,
            self.rank,
            self.size,
            group=self.group,
        )
        self._full_grads[name] = full
        self._reduce[name] = reduce_full.view(
            self.size, self.num_slots, in_dim, out_dim
        )
        lo = self.rank * self.own_rows
        return full[lo : lo + self.own_rows], self._reduce[name][self.rank]

    def prefetch(self, plan, tables):
        moonep = _import_moonep()
        ids = plan.experts_to_copy[self.rank]
        # Peers read this rank's rows, which its forward has just refreshed.
        dist.barrier(group=self.group)
        for owner in range(self.size):
            lo = owner * self.own_rows
            owned = (ids >= lo) & (ids < lo + self.own_rows)
            local_ids = torch.where(owned, ids - lo, torch.full_like(ids, -1))
            for name, table in tables.items():
                moonep.prefetch.launch_prefetch(
                    self._rows.owners[name][owner][: self.own_rows],
                    table[self.own_rows :],
                    local_ids.contiguous(),
                    num_sms=DEFAULT_NUM_SMS,
                )

    def reduce_grad(self, plan, grads):
        moonep = _import_moonep()
        handles = barrier_handles(self.dispatcher.buffer)
        # The kernel fences peers after its reads; the writes need this one.
        dist.barrier(group=self.group)
        for name in grads:
            moonep.grad_reduce.launch_grad_reduce(
                self._full_grads[name],
                self._reduce[name],
                plan.experts_to_copy,
                rank=self.rank,
                num_sms=DEFAULT_NUM_SMS,
                **handles,
            )
