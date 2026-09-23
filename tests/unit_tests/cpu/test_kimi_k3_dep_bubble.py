# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The bubble planner's invariants, where the runtime fires, and the deferred backward.

No GPU and no model: all three are scheduling logic, and the property that matters
for the runtime, that the encode happens between two of the rank's actions, is an
ordering fact a fake schedule checks exactly.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.pipeline_parallel.dep_backward import (
    cut_for_deferred_backward,
    GradQueue,
)
from torchtitan.models.kimi_k3.pipeline_parallel.dep_bubble_plan import plan_for_rank
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep import (
    VisionDepRuntime,
    VisionFeatureCache,
)


class _FakeAction:
    __slots__ = ("computation_type", "microbatch_index", "stage_index")

    def __init__(self, kind: str, stage: int, mb: int | None) -> None:
        self.computation_type = kind
        self.stage_index = stage
        self.microbatch_index = mb


def _interleaved_actions(pp_size: int, vp: int, n_microbatches: int) -> dict[int, list]:
    """Every rank's action list for Interleaved1F1B, without stages or a process group.

    The schedule's __init__ validates and wires real stages, and the placement
    question has no model in it, so everything the action generation reads is set
    here instead.
    """
    from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

    num_stages = pp_size * vp
    sched = ScheduleInterleaved1F1B.__new__(ScheduleInterleaved1F1B)
    sched._num_stages = num_stages
    sched.pp_group_size = pp_size
    sched._n_microbatches = n_microbatches
    sched.n_microbatches = n_microbatches
    sched.stage_index_to_group_rank = {s: s % pp_size for s in range(num_stages)}
    sched.number_of_rounds = max(1, n_microbatches // pp_size)
    sched.microbatches_per_round = n_microbatches // sched.number_of_rounds

    class _FakeStage:
        def __init__(self, index: int) -> None:
            self.stage_index = index
            self.num_stages = num_stages
            self.group_rank = index % pp_size
            self.is_first = index == 0
            self.is_last = index == num_stages - 1

    orders = {}
    for rank in range(pp_size):
        sched._stages = [_FakeStage(s) for s in range(rank, num_stages, pp_size)]
        sched.n_local_stages = len(sched._stages)
        sched.rank = rank
        orders[rank] = sched._calculate_single_rank_operations(rank)
    return orders


def _plans(pp_size: int, vp: int, n_microbatches: int, cost_ratio: float):
    orders = _interleaved_actions(pp_size, vp, n_microbatches)
    return {
        rank: plan_for_rank(
            actions,
            rank=rank,
            vision_microbatches=n_microbatches,
            cost_ratio=cost_ratio,
            upfront=pp_size,
        )
        for rank, actions in orders.items()
    }


class TestBubblePlan(unittest.TestCase):
    def test_no_encode_is_placed_after_its_own_consumer(self):
        for vp in (1, 2, 4):
            plans = _plans(pp_size=8, vp=vp, n_microbatches=32, cost_ratio=0.493)
            for rank, plan in plans.items():
                for p in plan.placed:
                    kind, stage, anchor_mb = p.anchor
                    if "FORWARD" in kind and stage == 0 and anchor_mb >= 0:
                        # The anchor is stage 0's forward of anchor_mb, which runs
                        # at anchor_mb's consumption point.
                        self.assertGreaterEqual(
                            p.microbatch,
                            anchor_mb,
                            f"vp={vp} rank={rank}: encode for mb {p.microbatch} "
                            f"placed at mb {anchor_mb}'s consumption point",
                        )

    def test_every_microbatch_is_accounted_for_exactly_once(self):
        for plan in _plans(8, 2, 32, 0.493).values():
            seen = (
                list(plan.upfront)
                + [p.microbatch for p in plan.placed]
                + list(plan.synchronous)
            )
            self.assertEqual(sorted(seen), list(range(32)))
            self.assertEqual(len(seen), len(set(seen)))

    def test_the_plan_is_a_function_of_its_inputs(self):
        """Every rank derives the same placements, which is what makes the
        vision collectives safe to issue from a bubble."""
        first = _plans(8, 2, 32, 0.493)
        second = _plans(8, 2, 32, 0.493)
        self.assertEqual(
            {r: (p.upfront, p.placed, p.synchronous) for r, p in first.items()},
            {r: (p.upfront, p.placed, p.synchronous) for r, p in second.items()},
        )

    def test_a_bubble_run_too_short_to_pay_places_nothing(self):
        actions = [_FakeAction("FORWARD", 0, 0), None, _FakeAction("FORWARD", 0, 1)]
        plan = plan_for_rank(
            actions, rank=0, vision_microbatches=2, cost_ratio=5.0, upfront=0
        )
        self.assertEqual(plan.placed, ())
        self.assertEqual(sorted(plan.synchronous), [0, 1])
        self.assertEqual(plan.slots_starved, 1)

    def test_a_bubble_after_every_consumer_is_counted_as_exhausted(self):
        actions = [
            _FakeAction("FORWARD", 0, 0),
            _FakeAction("FORWARD", 0, 1),
            None,
            None,
        ]
        plan = plan_for_rank(
            actions, rank=0, vision_microbatches=2, cost_ratio=1.0, upfront=0
        )
        self.assertEqual(plan.placed, ())
        self.assertEqual(sorted(plan.synchronous), [0, 1])
        self.assertEqual(plan.slots_exhausted, 2)

    def test_a_backward_can_anchor_a_placement(self):
        """The idle time after a backward is a bubble like any other."""
        actions = [
            _FakeAction("FORWARD", 1, 0),
            _FakeAction("FULL_BACKWARD", 1, 0),
            None,
            _FakeAction("FORWARD", 0, 1),
        ]
        plan = plan_for_rank(
            actions, rank=0, vision_microbatches=2, cost_ratio=1.0, upfront=0
        )
        self.assertEqual([p.microbatch for p in plan.placed], [0])
        self.assertIn("BACKWARD", plan.placed[0].anchor[0])


class _FakeTower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoded: list[int] = []

    def encode_images(self, pixel_values, grid_thw):
        self.encoded.append(int(pixel_values.item()))
        return torch.zeros(1)


def _runtime(order, *, prefetch=0, bubble=True, tower_stage=0, queue=None):
    tower = _FakeTower()
    runtime = VisionDepRuntime(
        VisionFeatureCache(tower),
        tower_stage_index=tower_stage,
        rank=0,
        pp_size=1,
        prefetch=prefetch,
        cost_ratio=1.0,
        pipeline_order={0: order} if bubble else None,
        queue=queue,
    )
    return runtime, tower


def _kwarg_mbs(n: int) -> list[dict]:
    return [
        {"pixel_values": torch.tensor(float(mb)), "grid_thw": torch.zeros(1, 3)}
        for mb in range(n)
    ]


class TestVisionDepRuntime(unittest.TestCase):
    def _warm(self, runtime):
        """The first step encodes inline; the plan starts with the second."""
        runtime.begin_step(_kwarg_mbs(2))
        runtime.after_forward(0, 0)
        runtime.end_step()

    def test_the_encode_fires_at_the_action_it_is_anchored_to(self):
        order = [
            _FakeAction("FORWARD", 0, 0),
            None,
            None,
            _FakeAction("FORWARD", 1, 0),
        ]
        runtime, tower = _runtime(order)
        self._warm(runtime)
        tower.encoded.clear()
        runtime.begin_step(_kwarg_mbs(2))
        # upfront is min(pp_size, n) = 1, so micro-batch 0 is encoded at step entry.
        self.assertEqual(tower.encoded, [0])
        runtime.after_forward(1, 0)
        self.assertEqual(tower.encoded, [0], "no placement anchors on stage 1")
        runtime.after_forward(0, 0)
        self.assertEqual(tower.encoded, [0, 1])

    def test_a_placement_anchored_on_a_backward_fires(self):
        order = [
            _FakeAction("FORWARD", 1, 0),
            _FakeAction("FULL_BACKWARD", 1, 0),
            None,
            _FakeAction("FORWARD", 0, 1),
        ]
        runtime, tower = _runtime(order)
        self._warm(runtime)
        tower.encoded.clear()
        runtime.begin_step(_kwarg_mbs(2))
        self.assertEqual(tower.encoded, [0])
        runtime.after_backward(1, 0, weight_only=False)
        self.assertEqual(tower.encoded, [0, 1])

    def test_the_tower_stage_reads_what_was_encoded_ahead(self):
        order = [_FakeAction("FORWARD", 0, 0), None, _FakeAction("FORWARD", 0, 1)]
        runtime, tower = _runtime(order)
        self._warm(runtime)
        runtime.begin_step(_kwarg_mbs(2))
        kwargs = runtime.forward_kwargs(0, 0, {"tokens": 1})
        self.assertIn("vision_embeds", kwargs)
        self.assertEqual(kwargs["tokens"], 1)
        # A stage that does not hold the tower is handed its kwargs unchanged.
        self.assertEqual(runtime.forward_kwargs(1, 0, {"tokens": 1}), {"tokens": 1})

    def test_a_micro_batch_that_was_not_encoded_ahead_is_left_to_the_forward(self):
        runtime, _ = _runtime([_FakeAction("FORWARD", 0, 0)], bubble=False)
        runtime.begin_step(_kwarg_mbs(2))
        self.assertEqual(runtime.forward_kwargs(0, 1, None), None)

    def test_the_run_ahead_encodes_the_next_micro_batches(self):
        runtime, tower = _runtime([], prefetch=2, bubble=False)
        runtime.begin_step(_kwarg_mbs(4))
        runtime.after_forward(0, 0)
        self.assertEqual(tower.encoded, [1, 2])
        runtime.after_forward(0, 1)
        self.assertEqual(tower.encoded, [1, 2, 3])


class TestDeferredVisionGrad(unittest.TestCase):
    """The deferred backward must be exact, and must never lose a gradient."""

    def _tower(self):
        torch.manual_seed(0)
        return torch.nn.Linear(4, 4, bias=False)

    def test_deferred_backward_matches_the_inline_one_exactly(self):
        x = torch.randn(3, 4)

        inline = self._tower()
        inline(x).sum().backward()
        expected = inline.weight.grad.clone()

        deferred = self._tower()
        queue = GradQueue()
        out = cut_for_deferred_backward(deferred(x), queue, 0)
        out.sum().backward()
        self.assertIsNone(deferred.weight.grad, "the text backward must not reach in")
        self.assertEqual(queue.pending_count(), 1)
        self.assertTrue(queue.run_one(0))
        torch.testing.assert_close(deferred.weight.grad, expected, rtol=0, atol=0)

    def test_nothing_is_lost_when_no_slot_ever_comes(self):
        x = torch.randn(3, 4)
        expected_model = self._tower()
        expected_model(x).sum().backward()
        expected = expected_model.weight.grad.clone()

        model = self._tower()
        queue = GradQueue()
        cut_for_deferred_backward(model(x), queue, 7).sum().backward()
        self.assertEqual(queue.drain(), 1)
        torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)
        self.assertEqual(queue.pending_count(), 0)

    def test_a_slot_before_the_gradient_arrives_is_not_an_error(self):
        queue = GradQueue()
        self.assertFalse(queue.run_one(3))
        self.assertEqual(queue.pending_count(), 0)

    def test_a_tower_with_no_gradient_path_is_left_alone(self):
        features = torch.randn(2, 4)
        queue = GradQueue()
        self.assertIs(cut_for_deferred_backward(features, queue, 0), features)

    def test_two_microbatches_accumulate_like_one_pass(self):
        xs = [torch.randn(2, 4), torch.randn(2, 4)]
        expected_model = self._tower()
        for x in xs:
            expected_model(x).sum().backward()
        expected = expected_model.weight.grad.clone()

        model = self._tower()
        queue = GradQueue()
        for mb, x in enumerate(xs):
            cut_for_deferred_backward(model(x), queue, mb).sum().backward()
        # Out of order deliberately: parameter gradients accumulate, so a deferred
        # backward may run in any bubble after its gradient arrives.
        queue.run_one(1)
        queue.run_one(0)
        torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)


class TestPendingBound(unittest.TestCase):
    """Each pending entry keeps one micro-batch's tower graph alive, so the bound is
    the backward half's memory window; what it must not change is that every
    gradient runs exactly once."""

    def _tower(self):
        torch.manual_seed(0)
        return torch.nn.Linear(4, 4, bias=False)

    def test_the_bound_runs_the_earliest_instead_of_growing(self):
        xs = [torch.randn(2, 4) for _ in range(3)]
        model = self._tower()
        queue = GradQueue(max_pending=1)
        for mb, x in enumerate(xs):
            cut_for_deferred_backward(model(x), queue, mb).sum().backward()
            self.assertLessEqual(queue.pending_count(), 1)
        self.assertEqual(queue.forced, 2)
        self.assertEqual(queue.pending_count(), 1)

    def test_the_bound_changes_when_not_whether_a_gradient_runs(self):
        xs = [torch.randn(2, 4) for _ in range(3)]
        expected_model = self._tower()
        for x in xs:
            expected_model(x).sum().backward()
        expected = expected_model.weight.grad.clone()

        model = self._tower()
        queue = GradQueue(max_pending=1)
        for mb, x in enumerate(xs):
            cut_for_deferred_backward(model(x), queue, mb).sum().backward()
        queue.drain()
        torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)
        self.assertEqual(queue.pending_count(), 0)

    def test_zero_means_unbounded(self):
        xs = [torch.randn(2, 4) for _ in range(3)]
        model = self._tower()
        queue = GradQueue(max_pending=0)
        for mb, x in enumerate(xs):
            cut_for_deferred_backward(model(x), queue, mb).sum().backward()
        self.assertEqual(queue.pending_count(), 3)
        self.assertEqual(queue.forced, 0)

    def test_a_slot_that_finds_nothing_is_counted(self):
        queue = GradQueue()
        self.assertFalse(queue.run_next())
        self.assertFalse(queue.run_next())
        self.assertEqual(queue.idle_slots, 2)


if __name__ == "__main__":
    unittest.main()
