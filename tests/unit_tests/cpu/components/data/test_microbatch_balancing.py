# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import socket
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.components.data.balancing import (
    balanced_step_iterator,
    document_attention_cost,
    MicrobatchBalancer,
    plan_balanced_assignment,
)
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

SEQ = 32


def _microbatch(doc_lens: list[int], tag: int) -> TokenizedTrainingMicrobatch:
    """A SEQ-token microbatch with the given documents, then padding.

    ``tag`` is written into every input id so tests can identify the microbatch.
    """
    num_tokens = sum(doc_lens)
    positions = torch.arange(SEQ)
    if doc_lens:
        positions[:num_tokens] = torch.cat([torch.arange(n) for n in doc_lens])
    padding_mask = torch.ones(SEQ, dtype=torch.bool)
    padding_mask[:num_tokens] = False
    return TokenizedTrainingMicrobatch(
        input=torch.full((SEQ,), tag, dtype=torch.int64),
        labels=torch.full((SEQ,), tag, dtype=torch.int64),
        positions=positions,
        padding_mask=padding_mask,
        loss_token_counts=torch.tensor(num_tokens),
        routing_token_counts=torch.tensor([num_tokens]),
    )


class TestDocumentAttentionCost(unittest.TestCase):
    def test_sum_of_squared_document_lengths(self):
        self.assertEqual(document_attention_cost(_microbatch([3, 2, 7], 0)), 9 + 4 + 49)

    def test_padding_is_ignored(self):
        # Padding positions continue as 0..k but must not count as a document.
        self.assertEqual(document_attention_cost(_microbatch([5], 0)), 25)
        self.assertEqual(document_attention_cost(_microbatch([], 0)), 0)

    def test_full_length_document(self):
        self.assertEqual(document_attention_cost(_microbatch([SEQ], 0)), SEQ * SEQ)


class TestPlanBalancedAssignment(unittest.TestCase):
    def _check_plan(self, costs):
        plan = plan_balanced_assignment(costs)
        num_ranks, num_slots = len(costs), len(costs[0])
        assigned = [item for row in plan for item in row]
        # Every microbatch is assigned exactly once.
        self.assertEqual(
            sorted(assigned),
            [(r, i) for r in range(num_ranks) for i in range(num_slots)],
        )
        # Slot g holds the g-th group of num_ranks microbatches by cost.
        ranked = sorted(
            (-costs[r][i] for r in range(num_ranks) for i in range(num_slots))
        )
        for g, row in enumerate(plan):
            got = sorted(-costs[src][idx] for src, idx in row)
            self.assertEqual(got, ranked[g * num_ranks : (g + 1) * num_ranks])
        return plan

    def test_invariants_on_random_costs(self):
        rng = random.Random(0)
        for num_ranks, num_slots in [(1, 1), (2, 3), (4, 4), (8, 2), (5, 7)]:
            costs = [
                [rng.choice([1.0, 2.0, 50.0, rng.random()]) for _ in range(num_slots)]
                for _ in range(num_ranks)
            ]
            self._check_plan(costs)

    def test_never_worse_than_unbalanced(self):
        # Heaviest-first striping minimizes the sum over slots of the slowest
        # rank's cost, so it never exceeds the as-read assignment.
        rng = random.Random(1)
        for _ in range(200):
            num_ranks, num_slots = rng.randint(1, 6), rng.randint(1, 6)
            costs = [
                [rng.random() ** 4 for _ in range(num_slots)] for _ in range(num_ranks)
            ]
            plan = plan_balanced_assignment(costs)
            balanced = sum(max(costs[s][i] for s, i in row) for row in plan)
            as_read = sum(
                max(costs[r][g] for r in range(num_ranks)) for g in range(num_slots)
            )
            self.assertLessEqual(balanced, as_read + 1e-12)

    def test_keeps_microbatches_on_owner_when_possible(self):
        # Already balanced: nothing moves.
        costs = [[3.0, 1.0], [3.0, 1.0], [3.0, 1.0]]
        plan = self._check_plan(costs)
        for row in plan:
            for r, (src, _) in enumerate(row):
                self.assertEqual(src, r)

    def test_heavy_microbatches_share_a_slot(self):
        costs = [[100.0, 1.0], [1.0, 1.0]]
        plan = self._check_plan(costs)
        # The heavy microbatch stays on rank 0 in slot 0; rank 1 runs a light
        # one alongside it, so the second slot holds only light microbatches.
        self.assertEqual(plan[0][0], (0, 0))
        self.assertEqual(costs[plan[0][1][0]][plan[0][1][1]], 1.0)
        self.assertEqual(plan[1][1], (1, 0))

    def test_rejects_ragged_costs(self):
        with self.assertRaises(ValueError):
            plan_balanced_assignment([[1.0, 2.0], [1.0]])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _doc_lens(dp_rank: int, i: int) -> list[int]:
    # One heavy microbatch (a full-length document) on some ranks, many short
    # documents elsewhere, so the as-read assignment is imbalanced.
    if (dp_rank + i) % 3 == 0:
        return [SEQ]
    return [4] * (SEQ // 4 - (dp_rank + i) % 2)


def _worker(rank, world_size, port, mesh_shape, group_size, num_slots, result_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        names = ("dp", "cp")[: len(mesh_shape)]
        mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=names)
        dp_mesh = mesh["dp"]
        dp_rank = dp_mesh.get_local_rank()
        balancer = MicrobatchBalancer.Config(group_size=group_size).build(
            dp_mesh=dp_mesh, num_microbatches_per_step=num_slots
        )
        # Two steps through the iterator wrapper; cp peers read identical data.
        source = (
            _microbatch(
                _doc_lens(dp_rank + step, i), tag=1000 * step + 10 * dp_rank + i
            )
            for step in range(2)
            for i in range(num_slots)
        )
        out = list(balanced_step_iterator(iter(source), balancer))
        tags = [int(mb.input[0]) for mb in out]
        costs = [document_attention_cost(mb) for mb in out]
        # The exchanged microbatch keeps every field.
        for mb in out:
            tag = int(mb.input[0])
            assert torch.equal(mb.labels, torch.full((SEQ,), tag))
            assert int(mb.loss_token_counts) == int((~mb.padding_mask).sum())
        torch.save(
            {"dp_rank": dp_rank, "tags": tags, "costs": costs},
            os.path.join(result_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class TestMicrobatchBalancerDistributed(unittest.TestCase):
    def _run(self, mesh_shape, group_size, num_slots):
        world_size = 1
        for d in mesh_shape:
            world_size *= d
        with tempfile.TemporaryDirectory() as result_dir:
            mp.spawn(
                _worker,
                args=(
                    world_size,
                    _free_port(),
                    mesh_shape,
                    group_size,
                    num_slots,
                    result_dir,
                ),
                nprocs=world_size,
                join=True,
            )
            return [
                torch.load(os.path.join(result_dir, f"rank{r}.pt"), weights_only=True)
                for r in range(world_size)
            ]

    def _check(self, results, dp_size, group_size, num_slots):
        cp_size = len(results) // dp_size
        by_dp = {}
        for res in results:
            by_dp.setdefault(res["dp_rank"], []).append(res)
        # cp peers of a data-parallel rank receive identical microbatches.
        for peers in by_dp.values():
            self.assertEqual(len(peers), cp_size)
            for p in peers[1:]:
                self.assertEqual(p["tags"], peers[0]["tags"])
        group_size = group_size or dp_size
        for step in range(2):
            lo, hi = step * num_slots, (step + 1) * num_slots
            for start in range(0, dp_size, group_size):
                members = range(start, start + group_size)
                got = sorted(t for d in members for t in by_dp[d][0]["tags"][lo:hi])
                want = sorted(
                    1000 * step + 10 * d + i for d in members for i in range(num_slots)
                )
                # Microbatches stay within their balancing group and step.
                self.assertEqual(got, want)
                # Slots hold microbatches in non-increasing cost order.
                slot_max = [
                    max(by_dp[d][0]["costs"][lo + g] for d in members)
                    for g in range(num_slots)
                ]
                self.assertEqual(slot_max, sorted(slot_max, reverse=True))

    def test_whole_dp_mesh(self):
        self._check(self._run((4,), 0, 3), dp_size=4, group_size=0, num_slots=3)

    def test_groups_of_two(self):
        self._check(self._run((4,), 2, 3), dp_size=4, group_size=2, num_slots=3)

    def test_context_parallel_peers(self):
        self._check(self._run((2, 2), 0, 4), dp_size=2, group_size=0, num_slots=4)

    def test_rejects_group_size_not_dividing_dp(self):
        # Validated before any collective, so a stand-in mesh is enough.
        dp_mesh = mock.Mock(ndim=1)
        dp_mesh.size.return_value = 4
        with self.assertRaisesRegex(ValueError, "must divide"):
            MicrobatchBalancer.Config(group_size=3).build(
                dp_mesh=dp_mesh, num_microbatches_per_step=2
            )


if __name__ == "__main__":
    unittest.main()
