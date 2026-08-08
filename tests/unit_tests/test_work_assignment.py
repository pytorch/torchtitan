# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

from torchtitan.distributed import assign_balanced_work


@dataclass(frozen=True)
class _Work:
    key: str
    weight: int


class TestWorkAssignment(unittest.TestCase):
    def test_balances_current_and_cumulative_load(self):
        work = (_Work("c", 4), _Work("b", 7), _Work("a", 8))
        assignments, cumulative_loads = assign_balanced_work(
            work,
            num_partitions=2,
            initial_loads=(0, 0),
            get_weight=lambda item: item.weight,
            get_stable_key=lambda item: item.key,
        )
        self.assertEqual(assignments, (1, 1, 0))
        self.assertEqual(cumulative_loads, (8, 11))

        assignments, cumulative_loads = assign_balanced_work(
            (_Work("d", 6), _Work("e", 6)),
            num_partitions=2,
            initial_loads=cumulative_loads,
            get_weight=lambda item: item.weight,
            get_stable_key=lambda item: item.key,
        )
        self.assertEqual(assignments, (0, 1))
        self.assertEqual(cumulative_loads, (14, 17))

    def test_stable_key_breaks_equal_weight_ties(self):
        assignments, _loads = assign_balanced_work(
            (_Work("z", 4), _Work("y", 4)),
            num_partitions=2,
            initial_loads=(0, 0),
            get_weight=lambda item: item.weight,
            get_stable_key=lambda item: item.key,
        )
        self.assertEqual(assignments, (1, 0))

    def test_requires_aligned_partitions_and_loads(self):
        with self.assertRaisesRegex(ValueError, "partitions and initial loads"):
            assign_balanced_work(
                (),
                num_partitions=2,
                initial_loads=(0,),
                get_weight=lambda item: item.weight,
                get_stable_key=lambda item: item.key,
            )
