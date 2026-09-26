# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pipeline memory plan on hand-built action orders and profiles."""

import unittest

from torch.distributed.pipelining.schedules import _Action, _ComputationType

from torchtitan.models.kimi_k3.pipeline_parallel.activations import (
    compute_actions,
    MemoryPlan,
    RankProfile,
)

F, B = _ComputationType.FORWARD, _ComputationType.FULL_BACKWARD
MB, GIB = 4, 2**30


def _order(stage: int) -> list:
    """All forwards of ``stage``, then all backwards: every save is held across the rank's peak."""
    return [_Action(stage, F, m) for m in range(MB)] + [
        _Action(stage, B, m) for m in range(MB)
    ]


def _profile(
    order: list, base: int, per_item: int, seconds: float = 1.0
) -> RankProfile:
    """A rank that holds each micro-batch's saves from its forward to its backward, over ``base``."""
    actions = compute_actions(order)
    position = {key: i for i, key in enumerate(actions)}
    items = {(key[1], key[2]): per_item for key in actions if key[0] == "F"}
    peaks = [base] * len(actions)
    for stage, mb in items:
        for i in range(position[("F", stage, mb)], position[("B", stage, mb)] + 1):
            peaks[i] += per_item
    return RankProfile(peaks=peaks, seconds=[seconds] * len(actions), items=items)


class TestMemoryPlan(unittest.TestCase):
    def _plan(self, profiles, orders, **kwargs):
        options = dict(
            target_bytes=None,
            offload=False,
            balance=False,
            host_bps=100 * GIB,
            peer_bps=100 * GIB,
        )
        options.update(kwargs)
        return MemoryPlan(orders, profiles, **options)

    def test_only_a_rank_over_the_target_moves_and_what_it_moves_covers_its_peak(self):
        orders = {0: _order(0), 1: _order(1)}
        profiles = {
            0: _profile(orders[0], 4 * GIB, GIB),
            1: _profile(orders[1], 0, GIB),
        }
        plan = self._plan(profiles, orders, target_bytes=6 * GIB, offload=True)
        self.assertTrue(all(rank == 0 for rank, _, _ in plan.backend))
        self.assertLessEqual(plan.peaks[0], 6 * GIB)
        self.assertEqual(plan.peaks[1], plan.profiled[1])
        # the peak is the last forward, which every moved micro-batch's window covers
        actions = compute_actions(orders[0])
        peak_at = actions.index(("F", 0, MB - 1))
        for _, stage, mb in plan.backend:
            back = next(
                key for key, items in plan.due[0].items() if (stage, mb) in items
            )
            self.assertGreater(actions.index(back), peak_at)

    def test_balance_parks_on_the_lightest_rank_with_a_pool_sized_by_what_is_parked_at_once(
        self,
    ):
        orders = {r: _order(r) for r in range(3)}
        profiles = {
            0: _profile(orders[0], 5 * GIB, GIB),
            1: _profile(orders[1], 3 * GIB, GIB),
            2: _profile(orders[2], 0, GIB),
        }
        plan = self._plan(profiles, orders, balance=True)
        self.assertEqual(set(plan.dests.values()), {2})
        self.assertFalse(set(plan.dests) & set(plan.dests.values()))
        self.assertLessEqual(plan.peaks[2], plan.target)
        self.assertLess(plan.peaks[0], plan.profiled[0])
        for source, span in plan.spans.items():
            moved = sum(1 for rank, _, _ in plan.backend if rank == source)
            self.assertLessEqual(span, int(moved * GIB * 1.1))
            self.assertGreaterEqual(span, GIB)

    def test_a_window_shorter_than_the_link_time_keeps_the_save(self):
        orders = {0: _order(0), 1: _order(1)}
        profiles = {
            0: _profile(orders[0], 4 * GIB, GIB),
            1: _profile(orders[1], 0, GIB),
        }
        plan = self._plan(
            profiles, orders, target_bytes=GIB, offload=True, host_bps=GIB / 100
        )
        self.assertEqual(plan.backend, {})
        self.assertEqual(plan.peaks, plan.profiled)

    def test_every_rank_derives_the_same_plan(self):
        orders = {r: _order(r) for r in range(3)}
        profiles = {r: _profile(orders[r], (3 - r) * GIB, GIB) for r in range(3)}
        first = self._plan(profiles, orders, offload=True, balance=True)
        second = self._plan(profiles, orders, offload=True, balance=True)
        self.assertEqual(first.backend, second.backend)
        self.assertEqual(first.dests, second.dests)
        self.assertEqual(first.spans, second.spans)
        self.assertEqual(dict(first.due), dict(second.due))


if __name__ == "__main__":
    unittest.main()
