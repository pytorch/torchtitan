# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import (
    _slot_times,
    anchor_of,
    START,
    STEP_END,
    STEP_START,
    VisionDepPlan,
)


def _interleaved_order(pp: int, vp: int, n_microbatches: int) -> dict[int, list]:
    from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

    num_stages = pp * vp
    schedule = ScheduleInterleaved1F1B.__new__(ScheduleInterleaved1F1B)
    schedule._num_stages = num_stages
    schedule.pp_group_size = pp
    schedule._n_microbatches = n_microbatches
    schedule.n_microbatches = n_microbatches
    schedule.stage_index_to_group_rank = {s: s % pp for s in range(num_stages)}
    schedule.number_of_rounds = max(1, n_microbatches // pp)
    schedule.microbatches_per_round = n_microbatches // schedule.number_of_rounds

    class _Stage:
        def __init__(self, index: int) -> None:
            self.stage_index = index
            self.num_stages = num_stages
            self.group_rank = index % pp
            self.is_first = index == 0
            self.is_last = index == num_stages - 1

    orders = {}
    for rank in range(pp):
        schedule._stages = [_Stage(s) for s in range(rank, num_stages, pp)]
        schedule.n_local_stages = len(schedule._stages)
        schedule.rank = rank
        orders[rank] = schedule._calculate_single_rank_operations(rank)
    return orders


def _action(kind: str, stage: int, mb: int):
    from torch.distributed.pipelining.schedules import _Action, _ComputationType

    types = {"F": _ComputationType.FORWARD, "B": _ComputationType.FULL_BACKWARD}
    return _Action(stage, types[kind], mb)


def _step_end(order: dict[int, list]) -> int:
    return max(
        max(j for j, a in enumerate(actions) if a is not None) + 1
        for actions in order.values()
    )


def _stage0_slots(order: dict[int, list]) -> tuple[dict[int, int], dict[int, int]]:
    consume, ready = {}, {}
    for slot, action in enumerate(order[0]):
        if action is None or action.stage_index != 0:
            continue
        kind, _, mb = anchor_of(action)
        if kind == "F":
            consume[mb] = slot
        else:
            ready[mb] = slot + 1
    return consume, ready


def _hooks(order: dict[int, list], rank: int) -> list:
    hooks = [STEP_START]
    for action in order[rank]:
        if action is not None:
            hooks += [("before", anchor_of(action)), ("after", anchor_of(action))]
    return hooks + [STEP_END]


def _interval(order: dict[int, list], rank: int, hook) -> tuple[int, int]:
    step_end = max(
        max(j for j, a in enumerate(actions) if a is not None) + 1
        for actions in order.values()
    )
    slots = [j for j, a in enumerate(order[rank]) if a is not None]
    if hook == STEP_START:
        return 0, slots[0]
    if hook == STEP_END:
        return slots[-1] + 1, step_end
    position, anchor = hook
    j = next(j for j in slots if anchor_of(order[rank][j]) == anchor)
    if position == "before":
        return j, j
    return j + 1, next((s for s in slots if s > j), step_end)


def _posted(plan, order: dict[int, list], rank: int) -> list:
    hooks = _hooks(order, rank)
    assert set(plan.posts[rank]) <= set(hooks), set(plan.posts[rank]) - set(hooks)
    return [(hook, t) for hook in hooks for t in plan.posts[rank].get(hook, ())]


def _uneven(n_microbatches: int) -> dict[int, int]:
    patches = (64, 0, 1024, 256, 64, 256, 0, 1024)
    return {
        mb: patches[mb % len(patches)]
        for mb in range(n_microbatches)
        if patches[mb % len(patches)]
    }


def _stalls(pp: int, vp: int, m: int, plan=None, loads=()) -> dict:
    """Where each rank stops when no kernel runs while a send or receive it posted is
    unmatched, the order a first kernel load imposes; empty when every rank finishes."""
    from torch.distributed.pipelining.schedules import _add_send_recv

    compute = _interleaved_order(pp, vp, m)
    trimmed = {
        r: acts[: max(i for i, a in enumerate(acts) if a is not None) + 1]
        for r, acts in compute.items()
    }
    comms = _add_send_recv(trimmed, lambda stage: stage % pp, pp * vp)
    events = {r: [] for r in range(pp)}
    for r, ev in events.items():
        if plan is not None:
            ev += [("compute", ("encode", mb)) for mb in plan.prologue[r]]
            ev += [("post", t) for t in plan.posts[r].get(STEP_START, ())]
            ev += [("compute", w) for w in plan.anchored[r].get(("START", -1, -1), ())]
        acts = comms[r]
        i = 0
        while i < len(acts):
            action = acts[i]
            name = action.computation_type.name
            if name.startswith(("SEND", "RECV")):
                ev.append(("p2p", action))
                i += 1
                continue
            anchor = anchor_of(action)
            if plan is not None:
                ev += [("post", t) for t in plan.posts[r].get(("before", anchor), ())]
                if name == "FORWARD" and anchor[1] == 0 and anchor[2] in loads:
                    ev.append(("features", anchor[2]))
            ev.append(("action", anchor))
            nxt = acts[i + 1] if i + 1 < len(acts) else None
            if nxt is not None and nxt.computation_type.name.startswith("SEND"):
                ev.append(("p2p", nxt))
                i += 1
            if plan is not None:
                ev += [("post", t) for t in plan.posts[r].get(("after", anchor), ())]
                ev += [("compute", w) for w in plan.anchored[r].get(anchor, ())]
            i += 1
        if plan is not None:
            ev += [("post", t) for t in plan.posts[r].get(STEP_END, ())]
            ev += [("compute", ("backward", mb)) for mb in plan.epilogue[r]]

    def p2p_ends(action):
        name, s, mb = (
            action.computation_type.name,
            action.stage_index,
            action.microbatch_index,
        )
        edge = {"SEND_F": ("F", s), "RECV_F": ("F", s - 1), "SEND_B": ("B", s)}
        key = edge.get(name, ("B", s + 1)) + (mb,)
        side = "send" if name.startswith("SEND") else "recv"
        return (key, side), (key, "recv" if side == "send" else "send")

    posted, done = set(), set()
    pending = {r: set() for r in range(pp)}
    at = {r: 0 for r in range(pp)}
    moved = True
    while moved:
        moved = False
        for r, ev in events.items():
            while at[r] < len(ev):
                kind, obj = ev[at[r]]
                if kind in ("post", "p2p"):
                    if kind == "post":
                        side, what = obj[0].split("_")
                        mine = (("dep", what, obj[1]), side)
                        other = (mine[0], "recv" if side == "send" else "send")
                    else:
                        mine, other = p2p_ends(obj)
                    posted.add(mine)
                    pending[r].add(other)
                else:
                    pending[r] -= posted
                    if pending[r]:
                        break
                    if kind == "features" and not (
                        (("dep", "feature", obj), "send") in posted
                        or ("encode", obj) in done
                    ):
                        break
                    if kind == "action":
                        what, s, mb = obj
                        needs = ("F", s - 1, mb) if what == "F" else ("B", s + 1, mb)
                        if needs[1] in range(pp * vp) and needs not in done:
                            break
                        obj = ("F" if what == "F" else "B", s, mb)
                    if kind == "compute" and obj[0] == "backward":
                        remote = r != 0
                        sent = (("dep", "gradient", obj[1]), "send") in posted
                        if ("B", 0, obj[1]) not in done or (remote and not sent):
                            break
                    done.add(obj)
                at[r] += 1
                moved = True
    return {r: ev[at[r]] for r, ev in events.items() if at[r] < len(ev)}


SHAPES = ((4, 2, 8), (4, 2, 16), (8, 4, 32))
_MATCH = {
    "send_feature": "recv_feature",
    "recv_feature": "send_feature",
    "send_gradient": "recv_gradient",
    "recv_gradient": "send_gradient",
}


class TestVisionDepPlan(unittest.TestCase):
    def test_without_the_schedule_the_work_is_balanced_before_and_after_it(self):
        loads = {0: 10, 1: 30, 2: 20, 3: 40, 5: 5}
        plan = VisionDepPlan(
            loads, num_microbatches=6, num_ranks=3, stage0_rank=0, trainable=True
        )
        for split in (plan.prologue, plan.epilogue):
            self.assertEqual(
                sorted(mb for mbs in split.values() for mb in mbs), [0, 1, 2, 3, 5]
            )
            busy = [sum(loads[mb] for mb in split[r]) for r in range(3)]
            self.assertLessEqual(max(busy) - min(busy), max(loads.values()))
        self.assertEqual(plan.placed, {})
        for rank in range(3):
            self.assertLessEqual(set(plan.posts[rank]), {STEP_START, STEP_END})
            for kind, _, _ in plan.posts[rank].get(STEP_START, ()):
                self.assertIn(kind, ("send_feature", "recv_feature"))
            for kind, _, _ in plan.posts[rank].get(STEP_END, ()):
                self.assertIn(kind, ("send_gradient", "recv_gradient"))
        remote = [mb for mb in loads if plan.encode_rank[mb] != 0]
        received = [mb for _, mb, _ in plan.posts[0].get(STEP_START, ())]
        self.assertEqual(sorted(received), sorted(remote))

    def test_bubble_work_meets_its_deadline_after_its_gradient_in_idle_slots(self):
        for pp, vp, m in SHAPES:
            for loads in ({mb: 100 for mb in range(m)}, _uneven(m)):
                with self.subTest(pp=pp, vp=vp, m=m, loads=loads):
                    order = _interleaved_order(pp, vp, m)
                    plan = VisionDepPlan(
                        loads,
                        num_microbatches=m,
                        num_ranks=pp,
                        stage0_rank=0,
                        trainable=True,
                        pipeline_order=order,
                        cost_ratio=0.5,
                    )
                    consume, ready = _stage0_slots(order)
                    step_end = _step_end(order)
                    times = _slot_times(order, step_end)
                    for (kind, mb), (rank, start, end) in plan.placed.items():
                        hop = 0.0 if rank == 0 else 1.0
                        if kind == "encode":
                            self.assertLessEqual(end, times[consume[mb]] - hop)
                        else:
                            self.assertGreaterEqual(start, times[ready[mb]] + hop)
                        for slot in range(min(step_end, len(order[rank]))):
                            if times[slot] < end and times[slot + 1] > start:
                                self.assertIsNone(order[rank][slot])
                    for rank in range(pp):
                        spans = sorted(
                            (start, end)
                            for (r, start, end) in plan.placed.values()
                            if r == rank
                        )
                        for (_, end), (start, _) in zip(spans, spans[1:]):
                            self.assertLessEqual(end, start)
                    self.assertEqual(sorted(plan.encode_rank), sorted(loads))
                    self.assertEqual(sorted(plan.backward_rank), sorted(loads))

    def test_the_first_pipeline_degree_microbatches_are_encoded_upfront(self):
        for pp, vp, m in SHAPES:
            order = _interleaved_order(pp, vp, m)
            plan = VisionDepPlan(
                {mb: 100 for mb in range(m)},
                num_microbatches=m,
                num_ranks=pp,
                stage0_rank=0,
                trainable=True,
                pipeline_order=order,
            )
            consume, _ = _stage0_slots(order)
            first = sorted(consume, key=consume.get)[:pp]
            upfront = {mb for mbs in plan.prologue.values() for mb in mbs}
            self.assertTrue(set(first) <= upfront)
            self.assertEqual(
                sorted(len(mbs) > 0 for mbs in plan.prologue.values()), [True] * pp
            )

    def test_both_ends_of_a_transfer_post_it_at_one_slot_boundary(self):
        for pp, vp, m in SHAPES:
            for ratio in (0.25, 1.0):
                with self.subTest(pp=pp, vp=vp, m=m, ratio=ratio):
                    order = _interleaved_order(pp, vp, m)
                    plan = VisionDepPlan(
                        _uneven(m),
                        num_microbatches=m,
                        num_ranks=pp,
                        stage0_rank=0,
                        trainable=True,
                        pipeline_order=order,
                        cost_ratio=ratio,
                    )
                    posted = {r: _posted(plan, order, r) for r in range(pp)}
                    for x in range(pp):
                        for y in range(x + 1, pp):
                            ours = [
                                (hook, kind, mb)
                                for hook, (kind, mb, peer) in posted[x]
                                if peer == y
                            ]
                            theirs = [
                                (hook, kind, mb)
                                for hook, (kind, mb, peer) in posted[y]
                                if peer == x
                            ]
                            self.assertEqual(
                                [(kind, mb) for _, kind, mb in ours],
                                [(_MATCH[kind], mb) for _, kind, mb in theirs],
                            )
                            for (hook_x, *_), (hook_y, *_) in zip(
                                ours, theirs, strict=True
                            ):
                                lo_x, hi_x = _interval(order, x, hook_x)
                                lo_y, hi_y = _interval(order, y, hook_y)
                                self.assertLessEqual(max(lo_x, lo_y), min(hi_x, hi_y))

    def test_a_transfer_leaves_after_its_data_exists_and_arrives_before_its_use(self):
        for pp, vp, m in SHAPES:
            order = _interleaved_order(pp, vp, m)
            plan = VisionDepPlan(
                _uneven(m),
                num_microbatches=m,
                num_ranks=pp,
                stage0_rank=0,
                trainable=True,
                pipeline_order=order,
                cost_ratio=0.25,
            )
            stage0 = {
                (kind, mb): (kind, stage, mb)
                for kind, stage, mb in (anchor_of(a) for a in order[0] if a is not None)
                if stage == 0
            }
            for rank in range(pp):
                hooks = _hooks(order, rank)
                where = {
                    (kind, mb): hooks.index(hook)
                    for hook, (kind, mb, _) in _posted(plan, order, rank)
                }
                items = {
                    work: hooks.index(("after", anchor))
                    for anchor, works in plan.anchored[rank].items()
                    for work in works
                }
                for (kind, mb), at in where.items():
                    if kind == "send_feature" and ("encode", mb) in items:
                        self.assertGreater(at, items[("encode", mb)])
                    if kind == "recv_feature":
                        use = ("before", stage0[("F", mb)])
                        self.assertLessEqual(at, hooks.index(use))
                    if kind == "send_gradient":
                        done = ("after", stage0[("B", mb)])
                        self.assertGreaterEqual(at, hooks.index(done))
                    if kind == "recv_gradient" and ("backward", mb) in items:
                        self.assertLessEqual(at, items[("backward", mb)])

    def test_no_wait_cycle_when_kernels_wait_for_the_ranks_posted_transfers(self):
        for pp, vp, m in ((2, 4, 4),) + SHAPES:
            self.assertEqual(_stalls(pp, vp, m), {})
            for bubble in (False, True):
                for loads in ({mb: 100 for mb in range(m)}, _uneven(m)):
                    plan = VisionDepPlan(
                        loads,
                        num_microbatches=m,
                        num_ranks=pp,
                        stage0_rank=0,
                        trainable=True,
                        pipeline_order=_interleaved_order(pp, vp, m)
                        if bubble
                        else None,
                        cost_ratio=0.25,
                    )
                    with self.subTest(pp=pp, vp=vp, m=m, bubble=bubble, n=len(loads)):
                        self.assertEqual(_stalls(pp, vp, m, plan, loads), {})

    def test_cheap_encodes_after_the_upfront_ones_all_land_in_idle_slots(self):
        order = _interleaved_order(8, 4, 32)
        plan = VisionDepPlan(
            {mb: 100 for mb in range(32)},
            num_microbatches=32,
            num_ranks=8,
            stage0_rank=0,
            trainable=True,
            pipeline_order=order,
        )
        encodes = [mb for kind, mb in plan.placed if kind == "encode"]
        self.assertEqual(len(encodes), 32 - 8)
        self.assertGreater(len({plan.encode_rank[mb] for mb in encodes}), 1)

    def test_work_too_costly_for_any_idle_slot_runs_before_and_after_the_schedule(self):
        order = _interleaved_order(4, 2, 8)
        plan = VisionDepPlan(
            {mb: 100 for mb in range(8)},
            num_microbatches=8,
            num_ranks=4,
            stage0_rank=0,
            trainable=True,
            pipeline_order=order,
            cost_ratio=100.0,
        )
        self.assertEqual(plan.placed, {})
        self.assertEqual(sum(len(mbs) for mbs in plan.prologue.values()), 8)
        self.assertEqual(sum(len(mbs) for mbs in plan.epilogue.values()), 8)

    def test_a_frozen_tower_and_text_only_microbatches_get_no_backward(self):
        order = _interleaved_order(4, 2, 8)
        plan = VisionDepPlan(
            {0: 10, 3: 10, 6: 10},
            num_microbatches=8,
            num_ranks=4,
            stage0_rank=0,
            trainable=False,
            pipeline_order=order,
        )
        self.assertEqual(sorted(plan.encode_rank), [0, 3, 6])
        self.assertEqual(plan.backward_rank, {})
        for rank in range(4):
            for transfers in plan.posts[rank].values():
                for kind, _, _ in transfers:
                    self.assertIn(kind, ("send_feature", "recv_feature"))
        self.assertFalse(any(kind == "backward" for kind, _ in plan.placed))

    def test_a_slot_lasts_as_long_as_its_longest_action(self):
        order = {
            0: [_action("F", 0, 0), None, _action("B", 0, 0)],
            1: [None, None, _action("F", 1, 1)],
        }
        self.assertEqual(_slot_times(order, 3), [0.0, 1.0, 1.0, 3.0])

    def test_a_backward_uses_the_rest_of_an_idle_run_that_began_before_its_gradient(
        self,
    ):
        order = {
            0: [_action("F", 0, mb) for mb in range(3)]
            + [_action("B", 0, 0)]
            + [_action("F", 0, mb) for mb in range(3, 6)]
            + [_action("B", 0, 1), _action("B", 0, 2)],
            1: [None, _action("F", 1, 0), _action("B", 1, 0)],
        }
        for ratio, placed in ((2.0, (1, 7.0, 13.0)), (2.1, None)):
            with self.subTest(ratio=ratio):
                plan = VisionDepPlan(
                    {0: 100},
                    num_microbatches=6,
                    num_ranks=2,
                    stage0_rank=0,
                    trainable=True,
                    pipeline_order=order,
                    cost_ratio=ratio,
                )
                self.assertEqual(plan.placed.get(("backward", 0)), placed)
                self.assertEqual(plan.epilogue[0], () if placed else (0,))

    def test_encodes_open_and_backwards_close_each_ranks_schedule(self):
        for pp, vp, m in SHAPES:
            for ratio in (0.1, 0.5, 1.0):
                with self.subTest(pp=pp, vp=vp, m=m, ratio=ratio):
                    order = _interleaved_order(pp, vp, m)
                    plan = VisionDepPlan(
                        _uneven(m),
                        num_microbatches=m,
                        num_ranks=pp,
                        stage0_rank=0,
                        trainable=True,
                        pipeline_order=order,
                        cost_ratio=ratio,
                    )
                    step_end = _step_end(order)
                    times = _slot_times(order, step_end)
                    for (kind, _), (rank, start, end) in plan.placed.items():
                        busy = [
                            slot
                            for slot, action in enumerate(order[rank])
                            if action is not None
                        ]
                        if kind == "encode":
                            self.assertLessEqual(end, times[busy[0]])
                        else:
                            self.assertGreaterEqual(start, times[busy[-1] + 1])

    def test_the_first_and_last_pipeline_degree_microbatches_run_outside_the_schedule(
        self,
    ):
        for pp, vp, m in SHAPES:
            with self.subTest(pp=pp, vp=vp, m=m):
                order = _interleaved_order(pp, vp, m)
                plan = VisionDepPlan(
                    {mb: 100 for mb in range(m)},
                    num_microbatches=m,
                    num_ranks=pp,
                    stage0_rank=0,
                    trainable=True,
                    pipeline_order=order,
                    cost_ratio=0.01,
                )
                consume, ready = _stage0_slots(order)
                first = sorted(consume, key=consume.get)[:pp]
                last = sorted(ready, key=ready.get)[-pp:]
                self.assertEqual(
                    sorted(plan.prologue.values()), [(mb,) for mb in first]
                )
                self.assertEqual(sorted(plan.epilogue.values()), [(mb,) for mb in last])
                self.assertEqual(len(plan.placed), 2 * (m - pp))

    def test_three_ranks_and_six_microbatches_lay_out_as_in_the_k3_report(self):
        order = _interleaved_order(3, 4, 6)
        last = {r: anchor_of([a for a in order[r] if a][-1]) for r in order}
        for ratio in (0.05, 0.1, 0.2, 0.3):
            with self.subTest(ratio=ratio):
                plan = VisionDepPlan(
                    {mb: 100 for mb in range(6)},
                    num_microbatches=6,
                    num_ranks=3,
                    stage0_rank=0,
                    trainable=True,
                    pipeline_order=order,
                    cost_ratio=ratio,
                )
                self.assertEqual(plan.prologue, {0: (0,), 1: (1,), 2: (2,)})
                self.assertEqual(plan.epilogue, {0: (3,), 1: (4,), 2: (5,)})
                self.assertEqual(
                    plan.anchored,
                    {
                        0: {},
                        1: {START: (("encode", 3),), last[1]: (("backward", 0),)},
                        2: {
                            START: (("encode", 4), ("encode", 5)),
                            last[2]: (("backward", 1), ("backward", 2)),
                        },
                    },
                )

    def test_the_plan_is_a_function_of_its_inputs(self):
        order = _interleaved_order(8, 4, 32)
        kwargs = dict(
            num_microbatches=32,
            num_ranks=8,
            stage0_rank=0,
            trainable=True,
            pipeline_order=order,
            cost_ratio=0.5,
        )
        first = VisionDepPlan(_uneven(32), **kwargs)
        second = VisionDepPlan(_uneven(32), **kwargs)
        for name in (
            "encode_rank",
            "backward_rank",
            "prologue",
            "epilogue",
            "anchored",
            "posts",
            "placed",
        ):
            self.assertEqual(getattr(first, name), getattr(second, name))


if __name__ == "__main__":
    unittest.main()
