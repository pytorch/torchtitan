# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.experiments.ft_fsdp.membership import (
    assign_slots,
    EVICT_INCOMPLETE_ROUNDS,
    MemberInfo,
    reconfigure_uuid,
)


def _host(
    name: str,
    *,
    slot: int = -1,
    gen: int = -1,
    procs: int = 2,
    skip=(),
    round: int = 0,
):
    return [
        MemberInfo(
            host=name,
            local_rank=l,
            uid=hash((name, l)),
            ident=f"{name}/{l}",
            handle=f"{name}/{l}",
            slot=slot,
            gen=gen,
            round=round,
        )
        for l in range(procs)
        if l not in skip
    ]


class AssignSlotsTest(unittest.TestCase):
    def test_initial(self) -> None:
        members = _host("c") + _host("a") + _host("b")
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.gen, 0)
        self.assertTrue(a.initial)
        self.assertEqual(a.host_slots, {"a": 0, "b": 1})
        self.assertEqual([m.handle for m in a.ranks], ["a/0", "a/1", "b/0", "b/1"])
        self.assertEqual(a.new_hosts, {"a", "b"})
        self.assertEqual(a.evict, set())

    def test_survivors_keep_slots(self) -> None:
        members = _host("z", slot=1, gen=3) + _host("a") + _host("y", slot=2, gen=3)
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.gen, 4)
        self.assertFalse(a.initial)
        self.assertEqual(a.host_slots, {"a": 0, "z": 1, "y": 2})
        self.assertEqual(a.new_hosts, {"a"})
        self.assertEqual(
            [m.handle for m in a.ranks], ["a/0", "a/1", "z/0", "z/1", "y/0", "y/1"]
        )

    def test_not_enough_spares(self) -> None:
        members = _host("a", slot=0, gen=1) + _host("s", skip=(1,))
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.ranks, ())
        self.assertEqual(a.host_slots, {})
        self.assertEqual(a.gen, 1)

    def test_waits_for_incomplete(self) -> None:
        members = (
            _host("a", slot=0, gen=2, skip=(0,), round=2)
            + _host("b", slot=2, gen=1)
            + _host("c", slot=1, gen=2)
            + _host("s1")
            + _host("s2", round=5)
        )
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.ranks, ())
        self.assertEqual(a.evict, {"b"})

    def test_evicts_incomplete_and_stale(self) -> None:
        members = (
            _host("a", slot=0, gen=2, skip=(0,), round=EVICT_INCOMPLETE_ROUNDS)
            + _host("b", slot=2, gen=1)
            + _host("c", slot=1, gen=2)
            + _host("d", slot=3, gen=2)
            + _host("s1")
            + _host("s2")
        )
        a = assign_slots(members, num_slots=4, procs_per_host=2)
        self.assertEqual(a.evict, {"a", "b"})
        self.assertEqual(a.host_slots, {"s1": 0, "c": 1, "s2": 2, "d": 3})
        self.assertEqual(a.gen, 3)

    def test_inconsistent_host_slots_evicted(self) -> None:
        members = (
            [
                MemberInfo("a", 0, 0, "a/0", "a/0", 0, 1),
                MemberInfo("a", 1, 1, "a/1", "a/1", 1, 1),
            ]
            + _host("b", slot=1, gen=1)
            + _host("s")
        )
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.evict, {"a"})
        self.assertEqual(a.host_slots, {"s": 0, "b": 1})

    def test_duplicate_slot_raises(self) -> None:
        members = _host("a", slot=0, gen=1) + _host("b", slot=0, gen=1)
        with self.assertRaisesRegex(ValueError, "both claim slot 0"):
            assign_slots(members, num_slots=2, procs_per_host=2)

    def test_slot_out_of_range_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "claims slot 5"):
            assign_slots(_host("a", slot=5, gen=0), num_slots=2, procs_per_host=2)

    def test_extra_spares_unassigned(self) -> None:
        members = _host("a", slot=0, gen=0) + _host("s1") + _host("s2")
        a = assign_slots(members, num_slots=2, procs_per_host=2)
        self.assertEqual(a.host_slots, {"a": 0, "s1": 1})
        self.assertNotIn("s2", a.host_slots)

    def test_order_independent(self) -> None:
        members = _host("b", slot=1, gen=0) + _host("x") + _host("y")
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        b = assign_slots(list(reversed(members)), num_slots=3, procs_per_host=2)
        self.assertEqual(a, b)

    def test_adjacent_free_slots_rejected(self) -> None:
        members = _host("c", slot=2, gen=1) + _host("s1") + _host("s2")
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.ranks, ())
        self.assertEqual(a.gen, 1)

    def test_free_slot_needs_successor(self) -> None:
        members = _host("a", slot=0, gen=1) + _host("b", slot=1, gen=1) + _host("s")
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.host_slots, {"a": 0, "b": 1, "s": 2})
        members = _host("a", slot=0, gen=1) + _host("c", slot=2, gen=1) + _host("s")
        a = assign_slots(members, num_slots=3, procs_per_host=2)
        self.assertEqual(a.host_slots, {"a": 0, "s": 1, "c": 2})

    def test_uuid(self) -> None:
        u = reconfigure_uuid("run", 1, 7)
        self.assertEqual(u, reconfigure_uuid("run", 1, 7))
        self.assertNotEqual(u, reconfigure_uuid("run", 2, 7))
        self.assertTrue(0 <= u < 1 << 63)


if __name__ == "__main__":
    unittest.main()
