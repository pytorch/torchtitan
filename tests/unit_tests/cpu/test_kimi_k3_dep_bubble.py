# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The bubble planner's invariants, and that the runtime fires before the wait.

No GPU and no model: both pieces are scheduling logic, and the property that matters
for the runtime -- that the encode happens BEFORE the rank waits on its receive -- is
an ordering fact that a fake schedule can check exactly.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.dep_bubble_backward import (
    cut_for_deferred_backward,
    GradQueue,
)
from torchtitan.models.kimi_k3.dep_bubble_plan import build_plans, plan_for_rank
from torchtitan.models.kimi_k3.dep_bubble_runtime import install_bubble_runtime


class _FakeAction:
    def __init__(self, kind: str, stage: int, mb: int | None) -> None:
        self.computation_type = kind
        self.stage_index = stage
        self.microbatch_index = mb


class TestBubblePlan(unittest.TestCase):
    def test_no_encode_is_placed_after_its_own_consumer(self):
        """The constraint that halved the first version's claimed placements.

        A bubble after micro-batch j's features are consumed cannot pay for encoding
        them, however much budget has accumulated.
        """
        for vp in (1, 2, 4):
            plans = build_plans(
                pp_size=8, vp=vp, n_microbatches=32, cost_ratio=0.493
            )
            for rank, plan in plans.items():
                for p in plan.placed:
                    kind, stage, anchor_mb = p.anchor
                    if "FORWARD" in kind and stage == 0 and anchor_mb >= 0:
                        # The anchor is stage 0's forward of anchor_mb, which runs at
                        # anchor_mb's consumption point, so the placed micro-batch must
                        # not be earlier than it.
                        self.assertGreaterEqual(
                            p.microbatch,
                            anchor_mb,
                            f"vp={vp} rank={rank}: encode for mb {p.microbatch} placed "
                            f"at mb {anchor_mb}'s consumption point",
                        )

    def test_every_microbatch_is_accounted_for_exactly_once(self):
        plans = build_plans(pp_size=8, vp=2, n_microbatches=32, cost_ratio=0.493)
        for plan in plans.values():
            seen = (
                list(plan.upfront)
                + [p.microbatch for p in plan.placed]
                + list(plan.synchronous)
            )
            self.assertEqual(sorted(seen), list(range(32)))
            self.assertEqual(len(seen), len(set(seen)))

    def test_all_ranks_derive_the_same_plan_shape(self):
        """Consistency is what makes the vision collectives safe to issue here.

        Ranks own different stages so their action lists differ, but the plan must be a
        function of values every rank agrees on -- so recomputing it must be
        deterministic, and the per-rank counts must not depend on call order.
        """
        a = build_plans(pp_size=8, vp=2, n_microbatches=32, cost_ratio=0.493)
        b = build_plans(pp_size=8, vp=2, n_microbatches=32, cost_ratio=0.493)
        self.assertEqual(
            {r: (p.upfront, p.placed, p.synchronous) for r, p in a.items()},
            {r: (p.upfront, p.placed, p.synchronous) for r, p in b.items()},
        )

    def test_a_bubble_run_too_short_to_pay_places_nothing(self):
        actions = [_FakeAction("FORWARD", 0, 0), None, _FakeAction("FORWARD", 0, 1)]
        plan = plan_for_rank(
            actions, rank=0, vision_microbatches=2, cost_ratio=5.0, upfront=0
        )
        self.assertEqual(plan.placed, ())
        self.assertEqual(sorted(plan.synchronous), [0, 1])

    def test_a_bubble_after_the_consumer_is_not_used(self):
        """Trailing idle time is usable in general, since the preceding action anchors
        it, so what rules a bubble out is the consumption point rather than its
        position. The earlier version of this test asserted trailing bubbles were
        unusable, which held only while placements anchored on the FOLLOWING action.
        """
        actions = [
            _FakeAction("FORWARD", 0, 0),
            _FakeAction("FORWARD", 0, 1),
            None,
            None,
        ]
        plan = plan_for_rank(
            actions, rank=0, vision_microbatches=2, cost_ratio=1.0, upfront=0
        )
        self.assertEqual([p.microbatch for p in plan.placed], [])
        self.assertEqual(sorted(plan.synchronous), [0, 1])


class _FakeStage:
    """A stage whose forward records itself, so ordering can be asserted."""

    def __init__(self, stage_index: int, trace: list[str]) -> None:
        self.stage_index = stage_index
        self._trace = trace

    def forward_one_chunk(self, fwd_chunk_id, *args, **kwargs):
        self._trace.append(f"fwd({self.stage_index},{fwd_chunk_id})")
        return "out"


class _FakeSchedule:
    """Enough of _PipelineScheduleRuntime to test the ordering property.

    The idle interval is what happens between two of this rank's actions, so the encode
    has to land after the action it is anchored to and before the next one.
    """

    def __init__(self, order: list) -> None:
        self.trace: list[str] = []
        stages: dict[int, _FakeStage] = {}
        for a in order:
            if a is not None and a.stage_index not in stages:
                stages[a.stage_index] = _FakeStage(a.stage_index, self.trace)
        self._stages = list(stages.values())
        self._by_index = stages
        self._order = order

    def step(self, *args, **kwargs):
        for action in self._order:
            if action is None:
                self.trace.append("idle")
                continue
            if "FORWARD" in action.computation_type:
                self._by_index[action.stage_index].forward_one_chunk(
                    action.microbatch_index
                )
        return "stepped"


class TestBubbleRuntime(unittest.TestCase):
    def _install(self, order, plan):
        sched = _FakeSchedule(order)
        install_bubble_runtime(
            sched,
            plan_for_step=lambda: plan,
            encode_now=lambda mbs: sched.trace.append(f"encode{list(mbs)}"),
            upfront_encode=lambda mbs: sched.trace.append(f"upfront{list(mbs)}"),
        )
        return sched

    def test_the_encode_lands_in_the_idle_interval(self):
        """The whole design in one assertion.

        The encode must come after the action it is anchored to and before the next real
        action, i.e. inside the gap. Anchoring on the FOLLOWING action instead put the
        hook on a receive wait that never happens for pipeline stage 0 -- exactly the
        rank that owns the tower.
        """
        order = [
            _FakeAction("FORWARD", 0, 0),
            None,
            None,
            _FakeAction("FORWARD", 1, 0),
        ]
        plan = plan_for_rank(
            order, rank=0, vision_microbatches=2, cost_ratio=1.0, upfront=0
        )
        self.assertTrue(plan.placed, "fixture must place at least one encode")
        sched = self._install(order, plan)
        sched.step()
        real = [i for i, t in enumerate(sched.trace) if t.startswith("fwd")]
        enc = next(i for i, t in enumerate(sched.trace) if t.startswith("encode"))
        self.assertGreater(enc, real[0], f"trace={sched.trace}")
        self.assertLess(enc, real[1], f"trace={sched.trace}")

    def test_no_plan_leaves_the_schedule_untouched(self):
        order = [_FakeAction("FORWARD", 0, 0)]
        sched = self._install(order, None)
        self.assertEqual(sched.step(), "stepped")
        self.assertEqual([t for t in sched.trace if "encode" in t], [])

    def test_installing_twice_is_a_no_op(self):
        order = [_FakeAction("FORWARD", 0, 0)]
        plan = plan_for_rank(
            order, rank=0, vision_microbatches=1, cost_ratio=1.0, upfront=1
        )
        sched = self._install(order, plan)
        first = sched.step
        install_bubble_runtime(
            sched,
            plan_for_step=lambda: plan,
            encode_now=lambda mbs: None,
            upfront_encode=lambda mbs: None,
        )
        self.assertIs(sched.step, first)


if __name__ == "__main__":
    unittest.main()


class TestDeferredVisionGrad(unittest.TestCase):
    """The deferred backward must be exact, and must never lose a gradient."""

    def _tower(self):
        torch.manual_seed(0)
        return torch.nn.Linear(4, 4, bias=False)

    def test_deferred_backward_matches_the_inline_one_exactly(self):
        """Cutting the graph and re-running it later is only sound if it is identical."""
        x = torch.randn(3, 4)

        inline = self._tower()
        inline(x).sum().backward()
        expected = inline.weight.grad.clone()

        deferred = self._tower()
        queue = GradQueue()
        out = cut_for_deferred_backward(deferred(x), queue, 0)
        out.sum().backward()
        self.assertIsNone(deferred.weight.grad, "the text backward must not reach in")
        self.assertTrue(queue.has(0))
        self.assertTrue(queue.run_one(0))
        torch.testing.assert_close(deferred.weight.grad, expected, rtol=0, atol=0)

    def test_nothing_is_lost_when_no_slot_ever_comes(self):
        """The drain is the correctness guarantee, not a tidiness measure.

        A deferred backward that never runs leaves the tower without that
        micro-batch's gradient and raises nothing, so the step-end drain has to be
        unconditional.
        """
        x = torch.randn(3, 4)
        expected_model = self._tower()
        expected_model(x).sum().backward()
        expected = expected_model.weight.grad.clone()

        model = self._tower()
        queue = GradQueue()
        cut_for_deferred_backward(model(x), queue, 7).sum().backward()
        self.assertEqual(queue.drain(), 1)
        torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)
        queue.assert_empty("after drain")

    def test_a_slot_before_the_gradient_arrives_is_not_an_error(self):
        queue = GradQueue()
        self.assertFalse(queue.run_one(3))
        queue.assert_empty("nothing was ever stashed")

    def test_assert_empty_refuses_to_let_a_leak_through(self):
        x = torch.randn(2, 4)
        model = self._tower()
        queue = GradQueue()
        cut_for_deferred_backward(model(x), queue, 1).sum().backward()
        with self.assertRaises(AssertionError):
            queue.assert_empty("before the optimizer step")

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
        # Deliberately out of order: parameter gradients accumulate, so a deferred
        # backward may run in any bubble after its gradient arrives.
        queue.run_one(1)
        queue.run_one(0)
        torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)


class TestPendingBound(unittest.TestCase):
    """The memory window of the backward half, as a configured quantity.

    Each pending entry keeps one micro-batch's tower forward graph alive from the
    encode until the replay. Unbounded, the plan decides how many that is; bounded,
    the earliest runs early and the window is known. What must not change either way
    is that every gradient runs exactly once.
    """

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
        # Two were forced out by the bound; the third is still waiting for a slot.
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
        queue.assert_empty("after drain under a pending bound")

    def test_zero_means_unbounded(self):
        xs = [torch.randn(2, 4) for _ in range(3)]
        model = self._tower()
        queue = GradQueue(max_pending=0)
        for mb, x in enumerate(xs):
            cut_for_deferred_backward(model(x), queue, mb).sum().backward()
        self.assertEqual(queue.pending_count(), 3)
        self.assertEqual(queue.forced, 0)

    def test_a_slot_that_finds_nothing_is_counted(self):
        """The greedy placement assumes the earliest micro-batch's grad arrives first.

        A high idle count is how that assumption failing becomes visible, since the
        step-end drain keeps it correct and therefore silent.
        """
        queue = GradQueue()
        self.assertFalse(queue.run_next())
        self.assertFalse(queue.run_next())
        self.assertEqual(queue.idle_slots, 2)
