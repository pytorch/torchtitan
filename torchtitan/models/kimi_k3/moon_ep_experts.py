# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The expert side of MoonEP (report sec 5.2.1): experts computed over a
weight table that spans every rank, not only the local shard.

MoonEP balances by duplicating hot experts onto other ranks for one step.
That is only possible if the grouped GEMM can address ANY expert by row, so
each projection is one table of ``E + B`` rows: rows ``[0, E)`` are every
rank's experts (``E/R`` per rank, each chunk physically the home rank's
memory mapped over NVLink), rows ``[E, E+B)`` are local prefetch slots that
``buffer.prefetch_weight`` fills with the experts this rank was handed.
``cu_seqlens[E+B]`` from dispatch says how many received tokens each row
serves, which is exactly the ``offs`` argument the grouped GEMM takes.

What lives where:

* master parameters stay what torchtitan gave them (fp32, EP-sharded
  DTensors, optimizer-owned). The tables are the bf16 compute copy, refreshed
  from the local rows before every prefetch -- the same copy FSDP's mixed
  precision makes, only into NVLink-mapped memory.
* gradients come back as ``[E+B]`` tables: the local rows become the local
  parameters' grads, the slot rows go to this rank's reduce buffer, and
  ``buffer.reduce_grad`` pulls every rank's slot grads for OUR experts back
  into our rows. Duplicated-expert grads therefore never touch the framework's
  own gradient reduction.
* the backward recomputes the expert forward (as activation checkpointing
  does) so the grouped GEMM's own backward produces the table grads.

Supported mesh, first version: experts are not FSDP-sharded beyond EP, i.e.
``efsdp == 1`` and no ``dp_replicate``; anything else raises at parallelize.

The table allocation is the one piece that needs MoonEP's VMM primitives
(``moonep.buffer.create_nvl_dist_tensor`` and the slot pages that must follow
the ``E`` rows contiguously); ``MoonEPTableBackend`` is that seam, and the
tests drive the whole unit through an in-process double of it and of
``Buffer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import GroupedExperts

# Shape suffixes: R = received (VM-group-ordered) tokens, D = latent width,
# F = expert hidden, E = experts, B = prefetch slots.

_PROJECTIONS = ("gate", "up", "down")


class MoonEPTableBackend(Protocol):
    """Allocates the tables MoonEP addresses across ranks."""

    def alloc_weight_table(
        self, name: str, rows: int, in_dim: int, out_dim: int
    ) -> torch.Tensor:
        """bf16 ``[rows, in, out]``: the first ``E`` rows NVLink-mapped so a
        remote ``prefetch_weight`` can read them, the slot rows local."""
        ...

    def alloc_grad_table(
        self, name: str, rows: int, in_dim: int, out_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 ``[rows, in, out]`` plus the ``[R, B, in, out]`` view of every
        rank's slot rows that ``reduce_grad`` reads."""
        ...


def _grouped_mm(A: torch.Tensor, B_t: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``torch._grouped_mm`` on CUDA; a per-group loop elsewhere (the tests).

    ``offs`` are cumulative end offsets, one per row of ``B_t``, so an empty
    row contributes nothing and a padded row multiplies zeros.
    """
    if A.is_cuda:
        return torch._grouped_mm(A, B_t, offs=offs)
    out = A.new_zeros(A.shape[0], B_t.shape[-1])
    start = 0
    for g, end in enumerate(offs.tolist()):
        if end > start:
            out[start:end] = A[start:end] @ B_t[g]
        start = end
    return out


def _ep_coords(ep_mesh) -> tuple[int, int]:
    return ep_mesh.get_local_rank(), ep_mesh.size()


class _MoonEPExpertFunction(torch.autograd.Function):
    """Prefetch, the three grouped GEMMs, and the grad routing around them."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, experts, x_RD, w1_l, w2_l, w3_l, cu_seqlens, plan):
        experts._refresh_local_rows(w1_l, w2_l, w3_l)
        experts._dispatcher._buffer.prefetch_weight(
            plan=plan,
            full_gate_weight=experts._tables["gate"],
            full_up_weight=experts._tables["up"],
            full_down_weight=experts._tables["down"],
        )
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
        experts._dispatcher._buffer.reduce_grad(
            plan=ctx.plan,
            full_gate_grad=experts._grad_tables["gate"][0],
            full_up_grad=experts._grad_tables["up"][0],
            full_down_grad=experts._grad_tables["down"][0],
            gate_reduce_buffer=experts._grad_tables["gate"][1],
            up_reduce_buffer=experts._grad_tables["up"][1],
            down_reduce_buffer=experts._grad_tables["down"][1],
        )
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
        num_prefetch_slots: int | None = None
        """MoonEP's ``B``; None is ``E // ep_size``, which training requires."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._config_prefetch_slots = config.num_prefetch_slots
        self._dispatcher = None
        self._backend: MoonEPTableBackend | None = None
        self._tables: dict[str, torch.Tensor] = {}
        self._grad_tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._local_rows: tuple[int, int] = (0, 0)
        self.num_prefetch_slots = 0

    # -- wiring --------------------------------------------------------------
    def attach(self, dispatcher, backend: MoonEPTableBackend, ep_mesh) -> None:
        """Bind the dispatcher whose plan this module consumes, the table
        allocator, and the EP coordinates. Called from the MoE's parallelize;
        the tests call it directly with their doubles."""
        rank, size = _ep_coords(ep_mesh)
        if self.num_experts % size != 0:
            raise ValueError(
                f"MoonEP needs num_experts ({self.num_experts}) divisible by "
                f"the EP size ({size})."
            )
        local = self.num_experts // size
        self._local_rows = (rank * local, (rank + 1) * local)
        self.num_prefetch_slots = (
            local
            if self._config_prefetch_slots is None
            else self._config_prefetch_slots
        )
        self._dispatcher = dispatcher
        self._backend = backend
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
        x_b = x_RD.to(torch.bfloat16)
        gate_RF = _grouped_mm(x_b, tables["gate"], cu_seqlens)
        up_RF = _grouped_mm(x_b, tables["up"], cu_seqlens)
        h_RF = self.activation_fn(gate_RF, up_RF)
        return _grouped_mm(h_RF, tables["down"], cu_seqlens)

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
    """The first version keeps expert params whole per EP rank."""
    if parallel_dims.dp_replicate_enabled:
        raise NotImplementedError(
            "moe_comm_backend='moonep' with dp_replicate is not supported yet: "
            "duplicated-expert grads are reduced by MoonEP, not by the "
            "framework, and the replicate reduction is not wired around that."
        )
    if parallel_dims.dp_shard != parallel_dims.ep:
        raise NotImplementedError(
            "moe_comm_backend='moonep' needs data_parallel_shard_degree == "
            "expert_parallel_degree (efsdp == 1): MoonEP maps each rank's whole "
            "expert chunk over NVLink, which an FSDP-sharded expert cannot offer."
        )


class MoonEPTableBackendNVLink:
    """The hardware allocator, over ``moonep.buffer``'s VMM primitives.

    ON-BOX. MoonEP's contract needs, per projection, ONE contiguous virtual
    range of ``E + B`` rows in which chunk ``r`` of the first ``E`` rows is
    rank ``r``'s physical memory mapped everywhere and the last ``B`` rows are
    local slot pages; ``prefetch_weight`` slices that single tensor. MoonEP's
    own e2e test fakes it with a local ``torch.empty`` because it only tests
    the communication, and ``create_nvl_dist_tensor`` maps equal chunks
    without a slot tail, so the composition is the one allocation this unit
    still has to establish against the installed package -- either a
    ``create_nvl_dist_tensor`` chunk of ``E/R (padded) + B`` rows with the
    plan's row numbering adjusted, or a VMM reserve that maps the ``R`` chunks
    and the slot pages back to back. Until then this raises with that note
    rather than guessing a layout the GEMM would silently misaddress.
    """

    def __init__(self, ep_mesh):
        self.ep_mesh = ep_mesh

    def alloc_weight_table(self, name, rows, in_dim, out_dim):
        raise NotImplementedError(
            "MoonEP weight tables need one contiguous VMM range of E + B rows "
            "(remote-mapped expert chunks followed by local slot pages); build "
            "it over moonep.buffer's primitives on NVLink hardware. The unit is "
            "exercised end to end on CPU through tests/moonep_fake.py."
        )

    def alloc_grad_table(self, name, rows, in_dim, out_dim):
        raise NotImplementedError("see alloc_weight_table")
