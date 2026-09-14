# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP's expert side for the Kimi K3 latent MoE.

``MoonEPGroupedExperts`` computes the routed experts over tables of ``E + B``
rows per projection (every expert, then ``B`` prefetch slots), and
``MoonEPTableBackendNVLink`` moves slot weights in and slot gradients home
over MoonEP's NVLink primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.kimi_k3.moon_ep_dispatcher import _import_moonep

# Shape suffixes: R = received (VM-group-ordered) tokens, D = latent width,
# F = expert hidden, E = experts, B = prefetch slots.

_PROJECTIONS = ("gate", "up", "down")


class MoonEPTableBackend(Protocol):
    """Allocates the tables MoonEP addresses across ranks."""

    def configure(self, *, num_experts: int, num_slots: int) -> None:
        """``E`` and ``B``: how the ``rows`` of every table split."""
        ...

    def alloc_weight_table(
        self, name: str, rows: int, in_dim: int, out_dim: int
    ) -> torch.Tensor:
        """bf16 ``[rows, in, out]``: rows ``[0, E)`` the experts, ``[E, E+B)``
        the prefetch slots."""
        ...

    def alloc_grad_table(
        self, name: str, rows: int, in_dim: int, out_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 ``[rows, in, out]`` plus this rank's ``[B, in, out]`` reduce
        buffer, the one other ranks read the slot grads from."""
        ...

    def prefetch(self, plan, tables: dict[str, torch.Tensor]) -> None:
        """Fill rows ``[E, E+B)`` of every table with the experts the plan
        copied onto this rank, reading them from their home ranks."""
        ...

    def reduce_grad(
        self, plan, grads: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Add every rank's slot grads for this rank's experts into its rows."""
        ...


def _ep_coords(ep_mesh) -> tuple[int, int]:
    return ep_mesh.get_local_rank(), ep_mesh.size()


class _MoonEPExpertFunction(torch.autograd.Function):
    """Prefetch, the three grouped GEMMs, and the grad routing around them."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, experts, x_RD, w1_l, w2_l, w3_l, cu_seqlens, plan):
        experts._refresh_local_rows(w1_l, w2_l, w3_l)
        experts._backend.prefetch(plan, experts._tables)
        with torch.no_grad():
            out_RD = experts._compute(x_RD, experts._tables, cu_seqlens)
        ctx.experts = experts
        ctx.plan = plan
        ctx.save_for_backward(x_RD, cu_seqlens)
        return out_RD

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_out_RD):
        experts = ctx.experts
        x_RD, cu_seqlens = ctx.saved_tensors
        # Recompute with the tables as leaves, so the grouped GEMM's own
        # backward yields the [E+B] table grads.
        x_leaf = x_RD.detach().requires_grad_(True)
        leaves = {
            n: experts._tables[n].detach().requires_grad_(True) for n in _PROJECTIONS
        }
        with torch.enable_grad():
            out_RD = experts._compute(x_leaf, leaves, cu_seqlens)
        grad_x, g_gate, g_up, g_down = torch.autograd.grad(
            out_RD, [x_leaf, leaves["gate"], leaves["up"], leaves["down"]], grad_out_RD
        )
        table_grads = {"gate": g_gate, "up": g_up, "down": g_down}
        # Route: local rows are ours, slot rows belong to other ranks' experts.
        E, B = experts.num_experts, experts.num_prefetch_slots
        lo, hi = experts._local_rows
        for name in _PROJECTIONS:
            full_grad, _ = experts._grad_tables[name]
            full_grad.zero_()
            full_grad[lo:hi].copy_(table_grads[name][lo:hi])
            full_grad[E : E + B].copy_(table_grads[name][E : E + B])
        experts._backend.reduce_grad(ctx.plan, experts._grad_tables)
        # Back to parameter orientation: tables are [row, in, out], the
        # parameters are w1_EFD/w3_EFD = [E, F, D] and w2_EDF = [E, D, F].
        grad_w1 = experts._grad_tables["gate"][0][lo:hi].transpose(-2, -1)
        grad_w3 = experts._grad_tables["up"][0][lo:hi].transpose(-2, -1)
        grad_w2 = experts._grad_tables["down"][0][lo:hi].transpose(-2, -1)
        return (
            None,
            grad_x,
            grad_w1.contiguous(),
            grad_w2.contiguous(),
            grad_w3.contiguous(),
            None,
            None,
        )


class MoonEPGroupedExperts(GroupedExperts):
    """``GroupedExperts`` computed over MoonEP's ``[E+B]`` tables.

    Without an EP mesh it is exactly the parent: local experts, local counts.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self._dispatcher = None
        self._backend: MoonEPTableBackend | None = None
        self._tables: dict[str, torch.Tensor] = {}
        self._grad_tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._local_rows: tuple[int, int] = (0, 0)
        self.num_prefetch_slots = 0

    # -- wiring --------------------------------------------------------------
    def attach(self, dispatcher, backend: MoonEPTableBackend, ep_mesh) -> None:
        """Bind the dispatcher whose plan and slot count this module uses and
        allocate the tables; called from the MoE's parallelize."""
        rank, size = _ep_coords(ep_mesh)
        if self.num_experts % size != 0:
            raise ValueError(
                f"MoonEP needs num_experts ({self.num_experts}) divisible by "
                f"the EP size ({size})."
            )
        local = self.num_experts // size
        self._local_rows = (rank * local, (rank + 1) * local)
        slots = dispatcher.num_prefetch_slots
        self.num_prefetch_slots = local if slots is None else slots
        self._dispatcher = dispatcher
        self._backend = backend
        backend.configure(
            num_experts=self.num_experts, num_slots=self.num_prefetch_slots
        )
        rows = self.num_experts + self.num_prefetch_slots
        D, F = self.w1_EFD.shape[-1], self.w1_EFD.shape[-2]
        self._tables = {
            "gate": backend.alloc_weight_table("gate", rows, D, F),
            "up": backend.alloc_weight_table("up", rows, D, F),
            "down": backend.alloc_weight_table("down", rows, F, D),
        }
        self._grad_tables = {
            "gate": backend.alloc_grad_table("gate", rows, D, F),
            "up": backend.alloc_grad_table("up", rows, D, F),
            "down": backend.alloc_grad_table("down", rows, F, D),
        }

    def _refresh_local_rows(self, w1_l, w2_l, w3_l) -> None:
        lo, hi = self._local_rows
        with torch.no_grad():
            self._tables["gate"][lo:hi].copy_(w1_l.transpose(-2, -1).to(torch.bfloat16))
            self._tables["up"][lo:hi].copy_(w3_l.transpose(-2, -1).to(torch.bfloat16))
            self._tables["down"][lo:hi].copy_(w2_l.transpose(-2, -1).to(torch.bfloat16))

    def _compute(self, x_RD, tables, cu_seqlens) -> torch.Tensor:
        # Tables are [row, in, out]; the grouped-mm seam takes [row, out, in].
        x_b = x_RD.to(torch.bfloat16)
        gate_RF = self._grouped_mm(
            A=x_b, weight_EOI=tables["gate"].transpose(-2, -1), offs=cu_seqlens
        )
        up_RF = self._grouped_mm(
            A=x_b, weight_EOI=tables["up"].transpose(-2, -1), offs=cu_seqlens
        )
        h_RF = self.activation_fn(gate_RF, up_RF)
        return self._grouped_mm(
            A=h_RF, weight_EOI=tables["down"].transpose(-2, -1), offs=cu_seqlens
        )

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        # Under MoonEP the second argument is the token count of each of the
        # [E+B] rows; without an EP mesh it is the parent's per-expert count.
        if self._dispatcher is None:
            return super().forward(x_RD, num_tokens_per_expert_E)
        plan, cu_seqlens = self._dispatcher.current_plan()
        w1 = self.w1_EFD.to_local() if isinstance(self.w1_EFD, DTensor) else self.w1_EFD
        w2 = self.w2_EDF.to_local() if isinstance(self.w2_EDF, DTensor) else self.w2_EDF
        w3 = self.w3_EFD.to_local() if isinstance(self.w3_EFD, DTensor) else self.w3_EFD
        out_RD = _MoonEPExpertFunction.apply(self, x_RD, w1, w2, w3, cu_seqlens, plan)
        return out_RD.type_as(x_RD)


def check_moonep_mesh(parallel_dims) -> None:
    """The first version keeps expert parameters whole per EP rank."""
    if parallel_dims.dp_replicate_enabled:
        raise NotImplementedError(
            "moe_comm_backend='moonep' with dp_replicate is not supported yet: "
            "duplicated-expert grads are reduced by MoonEP, not by the "
            "framework, and the replicate reduction is not wired around that."
        )
    dp_shard, cp, tp, ep = (
        parallel_dims.dp_shard,
        parallel_dims.cp,
        parallel_dims.tp,
        parallel_dims.ep,
    )
    if dp_shard * cp * tp != ep:
        raise NotImplementedError(
            "moe_comm_backend='moonep' needs efsdp == 1, i.e. "
            "data_parallel_shard_degree * context_parallel_degree * "
            f"tensor_parallel_degree == expert_parallel_degree (got {dp_shard} * "
            f"{cp} * {tp} vs {ep}): MoonEP maps each rank's whole expert chunk "
            "over NVLink, which an FSDP-sharded expert cannot offer."
        )


class MoonEPTableBackendNVLink:
    """The table backend over ``moonep.buffer``'s public primitives.

    The tables are ordinary local tensors. Every rank owns an NVLink-mapped copy
    of its expert chunk (bf16) and of its slot grads (fp32), made with
    ``create_nvl_single_owner_tensor``, and maps every other rank's;
    ``plan.experts_to_copy`` ([R, B] int32, the global expert id in rank r's
    slot b, negative when empty) says what moves where. Per MoE layer per step
    this costs three barriers on the EP group (one in ``prefetch``, two in
    ``reduce_grad``) and two host reads of ``experts_to_copy``.
    """

    def __init__(self, ep_mesh):
        self.ep_mesh = ep_mesh
        self.rank, self.size = _ep_coords(ep_mesh)
        self.group = ep_mesh.get_group()
        self.num_experts = 0
        self.num_slots = 0
        self._owned: dict[str, torch.Tensor] = {}
        self._mapped: dict[str, list[torch.Tensor]] = {}
        self._reduce_owned: dict[str, torch.Tensor] = {}
        self._reduce_mapped: dict[str, list[torch.Tensor]] = {}

    def configure(self, *, num_experts: int, num_slots: int) -> None:
        self.num_experts, self.num_slots = num_experts, num_slots

    def _map_all_owners(self, rows: int, in_dim: int, out_dim: int, dtype):
        """One mapped tensor per owner, allocated collectively in rank order as
        MoonEP's e2e test does it."""
        import torch.distributed as dist

        moonep = _import_moonep()
        padded = moonep.buffer.pad_dim0_for_alignment([rows, in_dim, out_dim], dtype)
        mapped = []
        for owner in range(self.size):
            t = moonep.buffer.create_nvl_single_owner_tensor(
                [padded, in_dim, out_dim],
                dtype,
                owner_rank=owner,
                local_rank=self.rank,
                group=self.group,
            )
            if owner == self.rank:
                t.zero_()
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            mapped.append(t[:rows])
        return mapped

    def _local_rows(self, rows: int) -> int:
        assert rows == self.num_experts + self.num_slots, (
            rows,
            self.num_experts,
            self.num_slots,
        )
        return self.num_experts // self.size

    def alloc_weight_table(self, name, rows, in_dim, out_dim):
        local = self._local_rows(rows)
        self._mapped[name] = self._map_all_owners(
            local, in_dim, out_dim, torch.bfloat16
        )
        self._owned[name] = self._mapped[name][self.rank]
        return torch.zeros(
            rows, in_dim, out_dim, dtype=torch.bfloat16, device=self._owned[name].device
        )

    def alloc_grad_table(self, name, rows, in_dim, out_dim):
        self._local_rows(rows)
        self._reduce_mapped[name] = self._map_all_owners(
            self.num_slots, in_dim, out_dim, torch.float32
        )
        self._reduce_owned[name] = self._reduce_mapped[name][self.rank]
        full = torch.zeros(
            rows,
            in_dim,
            out_dim,
            dtype=torch.float32,
            device=self._reduce_owned[name].device,
        )
        return full, self._reduce_owned[name]

    def prefetch(self, plan, tables):
        import torch.distributed as dist

        local = self.num_experts // self.size
        # Publish this rank's chunk, then read what the plan copied onto it.
        for name, table in tables.items():
            self._owned[name].copy_(table[self.rank * local : (self.rank + 1) * local])
        dist.barrier(group=self.group)
        for b, e in enumerate(plan.experts_to_copy[self.rank].tolist()):
            if e < 0:
                continue
            home, row = divmod(int(e), local)
            for name, table in tables.items():
                table[self.num_experts + b].copy_(self._mapped[name][home][row])

    def reduce_grad(self, plan, grads):
        import torch.distributed as dist

        local = self.num_experts // self.size
        E, B = self.num_experts, self.num_slots
        for name, (full_grad, _) in grads.items():
            self._reduce_owned[name].copy_(full_grad[E : E + B])
        dist.barrier(group=self.group)
        experts_to_copy = plan.experts_to_copy.tolist()
        for r in range(self.size):
            for b, e in enumerate(experts_to_copy[r]):
                if e >= 0 and int(e) // local == self.rank:
                    for name, (full_grad, _) in grads.items():
                        full_grad[int(e)].add_(self._reduce_mapped[name][r][b])
        # No rank overwrites a reduce buffer another rank is still reading.
        dist.barrier(group=self.group)
