# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP's expert side for the Kimi K3 latent MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.kimi_k3.moon_ep_dispatcher import _import_moonep

# Shape suffixes: R = received (VM-group-ordered) tokens, D = latent width,
# F = expert hidden, P = this rank's experts (E / ep size), B = prefetch slots.

_PROJECTIONS = ("gate", "up", "down")


class MoonEPTableBackend(Protocol):
    """Allocates this rank's bf16 ``[P + B]`` rows and fp32 grad rows, and moves
    slot weights in and slot gradients home."""

    def configure(self, *, num_experts: int, num_slots: int, num_sms: int) -> None: ...

    def alloc_expert_rows(
        self, name: str, in_dim: int, out_dim: int
    ) -> torch.Tensor: ...

    def alloc_grad_rows(
        self, name: str, in_dim: int, out_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def prefetch(self, plan, tables: dict[str, torch.Tensor]) -> None: ...

    def reduce_grad(self, plan, grads: dict[str, torch.Tensor]) -> None: ...


class _MoonEPExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, experts, x_RD, w1_l, w2_l, w3_l, offsets, plan
    ):
        experts._refresh_own_rows(w1_l, w2_l, w3_l)
        experts._backend.prefetch(plan, experts._tables)
        with torch.no_grad():
            out_RD = experts._compute(x_RD, experts._tables, offsets)
        ctx.experts = experts
        ctx.plan = plan
        ctx.save_for_backward(x_RD, offsets)
        return out_RD

    @staticmethod
    def backward(ctx, grad_out_RD):  # pyrefly: ignore[bad-override]
        experts = ctx.experts
        x_RD, offsets = ctx.saved_tensors
        # Recompute with the tables as leaves to get the [P + B] row grads.
        x_leaf = x_RD.detach().requires_grad_(True)
        leaves = {
            n: experts._tables[n].detach().requires_grad_(True) for n in _PROJECTIONS
        }
        with torch.enable_grad():
            out_RD = experts._compute(x_leaf, leaves, offsets)
        grads = torch.autograd.grad(
            out_RD,
            [x_leaf, *(leaves[n] for n in _PROJECTIONS)],
            grad_out_RD,
        )
        grad_x = grads[0]
        rows = experts.num_own_experts
        for name, row_grad in zip(_PROJECTIONS, grads[1:]):
            own_grad, slot_grad = experts._grads[name]
            own_grad.copy_(row_grad[:rows])
            slot_grad.copy_(row_grad[rows:])
        experts._backend.reduce_grad(
            ctx.plan, {n: experts._grads[n][0] for n in _PROJECTIONS}
        )
        # Rows are [in, out]; w1_EFD / w3_EFD are [P, F, D] and w2_EDF [P, D, F].
        grad_w1 = experts._grads["gate"][0].transpose(-2, -1).contiguous()
        grad_w3 = experts._grads["up"][0].transpose(-2, -1).contiguous()
        grad_w2 = experts._grads["down"][0].transpose(-2, -1).contiguous()
        return None, grad_x, grad_w1, grad_w2, grad_w3, None, None


class MoonEPGroupedExperts(GroupedExperts):
    """``GroupedExperts`` over this rank's experts and MoonEP's prefetch slots."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self._dispatcher = None
        self._backend: MoonEPTableBackend | None = None
        self._tables: dict[str, torch.Tensor] = {}
        self._grads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._row_span: tuple[int, int] = (0, 0)
        self.num_own_experts = 0
        self.num_prefetch_slots = 0

    def attach(self, dispatcher, backend: MoonEPTableBackend, ep_mesh) -> None:
        """Bind the dispatcher and the table backend and allocate this rank's rows."""
        rank, size = ep_mesh.get_local_rank(), ep_mesh.size()
        if self.num_experts % size != 0:
            raise ValueError(
                f"MoonEP needs num_experts ({self.num_experts}) divisible by "
                f"the EP size ({size})."
            )
        own = self.num_experts // size
        D, F = self.w1_EFD.shape[-1], self.w1_EFD.shape[-2]
        # init_buffer set the plan's slot count on MoonEP's VMM granularity.
        slots = dispatcher.num_prefetch_slots or own
        if slots < own:
            raise ValueError(
                f"MoonEP needs at least E / R = {own} prefetch slots to place "
                f"every duplicated expert, got B={slots}."
            )
        self.num_own_experts = own
        self.num_prefetch_slots = slots
        self._row_span = (rank * own, (rank + 1) * own)
        self._dispatcher = dispatcher
        self._backend = backend
        backend.configure(
            num_experts=self.num_experts,
            num_slots=slots,
            num_sms=dispatcher.num_sms,
        )
        self._tables = {
            "gate": backend.alloc_expert_rows("gate", D, F),
            "up": backend.alloc_expert_rows("up", D, F),
            "down": backend.alloc_expert_rows("down", F, D),
        }
        self._grads = {
            "gate": backend.alloc_grad_rows("gate", D, F),
            "up": backend.alloc_grad_rows("up", D, F),
            "down": backend.alloc_grad_rows("down", F, D),
        }

    def _refresh_own_rows(self, w1_l, w2_l, w3_l) -> None:
        rows = self.num_own_experts
        with torch.no_grad():
            self._tables["gate"][:rows].copy_(
                w1_l.transpose(-2, -1).to(torch.bfloat16)
            )
            self._tables["up"][:rows].copy_(w3_l.transpose(-2, -1).to(torch.bfloat16))
            self._tables["down"][:rows].copy_(w2_l.transpose(-2, -1).to(torch.bfloat16))

    def _offsets(self, cu_seqlens_R: torch.Tensor) -> torch.Tensor:
        """Token ends of this rank's expert rows followed by its slot rows."""
        lo, hi = self._row_span
        # A token reaches its expert's home rank or a rank holding a slot copy.
        if lo:
            torch._assert_async(cu_seqlens_R[lo - 1] == 0)
        torch._assert_async(cu_seqlens_R[self.num_experts - 1] == cu_seqlens_R[hi - 1])
        return torch.cat((cu_seqlens_R[lo:hi], cu_seqlens_R[self.num_experts :]))

    def _compute(self, x_RD, tables, offsets) -> torch.Tensor:
        # Rows are [in, out]; the grouped-mm seam takes [row, out, in].
        x_b = x_RD.to(torch.bfloat16)
        gate_RF = self._grouped_mm(
            A=x_b, weight_EOI=tables["gate"].transpose(-2, -1), offs=offsets
        )
        up_RF = self._grouped_mm(
            A=x_b, weight_EOI=tables["up"].transpose(-2, -1), offs=offsets
        )
        h_RF = self.activation_fn(gate_RF, up_RF)
        return self._grouped_mm(
            A=h_RF, weight_EOI=tables["down"].transpose(-2, -1), offs=offsets
        )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        # With MoonEP the token counts come from the plan's cu_seqlens instead.
        if self._dispatcher is None:
            return super().forward(x_RD, num_tokens_per_expert_E)
        plan, cu_seqlens_R = self._dispatcher.current_plan()
        w1 = self.w1_EFD.to_local() if isinstance(self.w1_EFD, DTensor) else self.w1_EFD
        w2 = self.w2_EDF.to_local() if isinstance(self.w2_EDF, DTensor) else self.w2_EDF
        w3 = self.w3_EFD.to_local() if isinstance(self.w3_EFD, DTensor) else self.w3_EFD
        out_RD = _MoonEPExpertFunction.apply(
            self, x_RD, w1, w2, w3, self._offsets(cu_seqlens_R), plan
        )
        return out_RD.type_as(x_RD)


def check_moonep_mesh(parallel_dims) -> None:
    """The first version keeps expert parameters whole per EP rank."""
    if parallel_dims.dp_replicate_enabled:
        raise NotImplementedError(
            "moe_comm_backend='moonep' with dp_replicate is not supported yet: "
            "duplicated-expert grads are reduced by MoonEP, not by the "
            "framework, and the replicate reduction is not wired around that."
        )
    efsdp = parallel_dims.get_optional_mesh("efsdp")
    degree = (
        efsdp.size()
        if efsdp is not None
        else parallel_dims.dp_shard
        * parallel_dims.cp
        * parallel_dims.tp
        // parallel_dims.ep
    )
    if degree != 1:
        raise NotImplementedError(
            "moe_comm_backend='moonep' needs efsdp == 1, i.e. "
            "data_parallel_shard_degree * context_parallel_degree * "
            f"tensor_parallel_degree == expert_parallel_degree (got efsdp "
            f"{degree}): MoonEP maps each rank's whole expert chunk over "
            "NVLink, which an FSDP-sharded expert cannot offer."
        )


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
        self.num_sms = 32
        self.own_rows = 0
        self._rows = _MappedRows(ep_mesh, self.rank, self.size)
        self._full_grads: dict[str, torch.Tensor] = {}
        self._reduce: dict[str, torch.Tensor] = {}

    def configure(self, *, num_experts: int, num_slots: int, num_sms: int) -> None:
        self.num_experts, self.num_slots, self.num_sms = (
            num_experts,
            num_slots,
            num_sms,
        )
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
                    num_sms=self.num_sms,
                )

    def reduce_grad(self, plan, grads):
        moonep = _import_moonep()
        handles = self.dispatcher.grad_reduce_handles()
        # The kernel fences peers after its reads; the writes need this one.
        dist.barrier(group=self.group)
        for name in grads:
            moonep.grad_reduce.launch_grad_reduce(
                self._full_grads[name],
                self._reduce[name],
                plan.experts_to_copy,
                rank=self.rank,
                num_sms=self.num_sms,
                **handles,
            )


