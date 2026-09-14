# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""An in-process double of MoonEP's ``Buffer`` and of the table allocator.

``R`` ranks run as ``R`` threads. Every MoonEP call that is a collective on
hardware is a barrier here: each thread deposits its operand, one thread
computes the exchange, everyone reads back. What it reproduces faithfully is
the CONTRACT -- ``cu_seqlens`` over ``E+B`` rows, received tokens in row
order with per-row padding, duplicated experts landing in prefetch slots,
slot grads reduced to the home rank -- not MoonEP's planner: which experts get
duplicated is handed in by the test (``dup_map``), and a duplicated expert's
tokens alternate between its home and its copies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch


@dataclass
class FakePlan:
    """Where every (source rank, token, k) copy landed."""

    placements: dict[tuple[int, int, int], tuple[int, int]] = field(
        default_factory=dict
    )
    cu_seqlens: dict[int, torch.Tensor] = field(default_factory=dict)
    num_rows_total: dict[int, int] = field(default_factory=dict)
    dup_map: dict[tuple[int, int], int] = field(default_factory=dict)


class _Collective:
    """deposit -> barrier -> rank 0 computes -> barrier -> read."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(world_size)
        self._inputs: dict = {}
        self._outputs: dict = {}

    def run(self, rank: int, payload, compute):
        with self._lock:
            self._inputs[rank] = payload
        self._barrier.wait()
        if rank == 0:
            outs = compute(self._inputs)
            with self._lock:
                self._outputs = outs
        self._barrier.wait()
        result = self._outputs[rank]
        self._barrier.wait()  # nobody clears before everyone read
        if rank == 0:
            with self._lock:
                self._inputs = {}
                self._outputs = {}
        return result


class FakeMoonEPWorld:
    def __init__(
        self,
        *,
        num_ranks: int,
        num_experts: int,
        top_k: int,
        tokens_per_rank: int,
        hidden_dim: int,
        num_prefetch_slots: int,
        dup_map: dict[tuple[int, int], int] | None = None,
        token_padding: int = 4,
    ):
        self.R = num_ranks
        self.E = num_experts
        self.K = top_k
        self.S = tokens_per_rank
        self.H = hidden_dim
        self.B = num_prefetch_slots
        self.dup_map = dup_map or {}
        self.token_padding = token_padding
        self.local = num_experts // num_ranks
        self._coll = {
            name: _Collective(num_ranks)
            for name in ("dispatch", "redispatch", "combine", "prefetch", "reduce")
        }
        self._tables: dict[tuple[int, str], torch.Tensor] = {}
        self._grads: dict[tuple[int, str], torch.Tensor] = {}
        self._reduce: dict[tuple[int, str], torch.Tensor] = {}

    def home(self, expert: int) -> int:
        return expert // self.local

    # -- what the tests hand to the dispatcher / experts -------------------
    def buffer_for(self, rank: int):
        return FakeBuffer(self, rank)

    def backend_for(self, rank: int):
        return FakeTableBackend(self, rank)

    def mesh_for(self, rank: int):
        return _FakeMesh(rank, self.R)

    # -- planning ---------------------------------------------------------
    def _plan(self, inputs: dict) -> FakePlan:
        plan = FakePlan(dup_map=dict(self.dup_map))
        copies_of = {
            e: [self.home(e)] + [r for (r, _b), e2 in self.dup_map.items() if e2 == e]
            for e in range(self.E)
        }
        slot_of = {(r, e): b for (r, b), e in self.dup_map.items()}
        rows: dict[int, dict[int, list]] = {r: {} for r in range(self.R)}
        for src in range(self.R):
            _x, _w, ids, _c = inputs[src]
            for t in range(ids.shape[0]):
                for k in range(ids.shape[1]):
                    e = int(ids[t, k])
                    dests = copies_of[e]
                    dest = dests[(t + k) % len(dests)]
                    row = e if dest == self.home(e) else self.E + slot_of[(dest, e)]
                    rows[dest].setdefault(row, []).append((src, t, k))
        for dest in range(self.R):
            cu = []
            pos = 0
            for row in range(self.E + self.B):
                entries = rows[dest].get(row, [])
                for entry in entries:
                    plan.placements[entry] = (dest, pos)
                    pos += 1
                if entries and pos % self.token_padding:
                    pos += self.token_padding - pos % self.token_padding
                cu.append(pos)
            plan.cu_seqlens[dest] = torch.tensor(cu, dtype=torch.int32)
            plan.num_rows_total[dest] = pos
        return plan

    def dispatch(self, rank, x, w, ids, counts):
        def compute(inputs):
            plan = self._plan(inputs)
            outs = {}
            for dest in range(self.R):
                n = plan.num_rows_total[dest]
                hidden = torch.zeros(n, self.H, dtype=torch.bfloat16)
                weights = torch.zeros(n, dtype=torch.float32)
                for (src, t, k), (d, pos) in plan.placements.items():
                    if d == dest:
                        hidden[pos] = inputs[src][0][t]
                        weights[pos] = inputs[src][1][t, k]
                outs[dest] = (hidden, weights, plan.cu_seqlens[dest].clone(), plan)
            return outs

        return self._coll["dispatch"].run(rank, (x, w, ids, counts), compute)

    def redispatch(self, rank, grad_sh, plan):
        def compute(inputs):
            outs = {}
            for dest in range(self.R):
                out = torch.zeros(
                    plan.num_rows_total[dest], self.H, dtype=torch.bfloat16
                )
                for (src, t, _k), (d, pos) in plan.placements.items():
                    if d == dest:
                        out[pos] = inputs[src][t]
                outs[dest] = (out, None, None, None)
            return outs

        return self._coll["redispatch"].run(rank, grad_sh, compute)

    def combine(self, rank, plan, hidden_nvsh, weights_nvs):
        def compute(inputs):
            outs = {}
            for src in range(self.R):
                out = torch.zeros(self.S, self.H, dtype=torch.float32)
                gathered = (
                    torch.zeros(self.S, self.K, dtype=torch.float32)
                    if inputs[src][1] is not None
                    else None
                )
                for (s, t, k), (d, pos) in plan.placements.items():
                    if s == src:
                        out[t] += inputs[d][0][pos].float()
                        if gathered is not None:
                            gathered[t, k] = inputs[d][1][pos]
                outs[src] = (out.to(torch.bfloat16), gathered, None)
            return outs

        return self._coll["combine"].run(rank, (hidden_nvsh, weights_nvs), compute)

    def prefetch(self, rank, plan):
        def compute(_inputs):
            for (r, b), e in plan.dup_map.items():
                for name in ("gate", "up", "down"):
                    self._tables[(r, name)][self.E + b].copy_(
                        self._tables[(self.home(e), name)][e]
                    )
            return {r: None for r in range(self.R)}

        return self._coll["prefetch"].run(rank, None, compute)

    def reduce_grad(self, rank, plan):
        def compute(_inputs):
            for (r, b), e in plan.dup_map.items():
                for name in ("gate", "up", "down"):
                    self._grads[(self.home(e), name)][e] += self._reduce[(r, name)][b]
                    self._reduce[(r, name)][b].zero_()
            return {r: None for r in range(self.R)}

        return self._coll["reduce"].run(rank, None, compute)


class _FakeMesh:
    def __init__(self, rank: int, size: int):
        self._rank, self._size = rank, size

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank

    def get_group(self):
        return None


class FakeBuffer:
    """``moonep.Buffer``'s call surface, backed by the world above."""

    def __init__(self, world: FakeMoonEPWorld, rank: int, **_kwargs):
        self.world, self.rank = world, rank

    def dispatch(
        self,
        hidden_sh,
        route_weights_sk=None,
        topk_experts_sk=None,
        tokens_per_expert=None,
        *,
        plan=None,
        zero_copy=False,
    ):
        assert hidden_sh.dtype == torch.bfloat16
        if plan is not None:
            return self.world.redispatch(self.rank, hidden_sh, plan)
        assert hidden_sh.shape[0] == self.world.S, "MoonEP's S is a static shape"
        return self.world.dispatch(
            self.rank, hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert
        )

    def combine(self, *, plan, hidden_nvsh, route_weights_nvs=None, zero_copy=False):
        assert hidden_nvsh.dtype == torch.bfloat16
        return self.world.combine(self.rank, plan, hidden_nvsh, route_weights_nvs)

    def prefetch_weight(
        self, *, plan, full_gate_weight, full_up_weight, full_down_weight
    ):
        for name, t in (
            ("gate", full_gate_weight),
            ("up", full_up_weight),
            ("down", full_down_weight),
        ):
            assert (
                t is self.world._tables[(self.rank, name)]
            ), "prefetch must see the allocated table"
        return self.world.prefetch(self.rank, plan)

    def reduce_grad(
        self,
        *,
        plan,
        full_gate_grad,
        full_up_grad,
        full_down_grad,
        gate_reduce_buffer,
        up_reduce_buffer,
        down_reduce_buffer,
    ):
        for name, t in (
            ("gate", full_gate_grad),
            ("up", full_up_grad),
            ("down", full_down_grad),
        ):
            assert t is self.world._grads[(self.rank, name)]
        return self.world.reduce_grad(self.rank, plan)

    def destroy(self) -> None:
        pass


class FakeTableBackend:
    def __init__(self, world: FakeMoonEPWorld, rank: int):
        self.world, self.rank = world, rank

    def alloc_weight_table(self, name, rows, in_dim, out_dim):
        t = torch.zeros(rows, in_dim, out_dim, dtype=torch.bfloat16)
        self.world._tables[(self.rank, name)] = t
        return t

    def alloc_grad_table(self, name, rows, in_dim, out_dim):
        full = torch.zeros(rows, in_dim, out_dim, dtype=torch.float32)
        self.world._grads[(self.rank, name)] = full
        # The slot rows ARE this rank's reduce buffer, as on hardware.
        self.world._reduce[(self.rank, name)] = full[self.world.E :]
        return full, full[self.world.E :]
