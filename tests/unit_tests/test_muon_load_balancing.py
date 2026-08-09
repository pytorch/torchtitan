# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.components.distributed_optimizers.muon.load_balancing import (
    balance_loads_across_partitions,
)


class TestLoadBalancing(unittest.TestCase):
    def test_balances_current_and_cumulative_load(self):
        assignments, cumulative_loads = balance_loads_across_partitions(
            ((4, "c"), (7, "b"), (8, "a")),
            initial_partition_loads=(0, 0),
        )
        self.assertEqual(assignments, (1, 1, 0))
        self.assertEqual(cumulative_loads, (8, 11))

        assignments, cumulative_loads = balance_loads_across_partitions(
            ((6, "d"), (6, "e")),
            initial_partition_loads=cumulative_loads,
        )
        self.assertEqual(assignments, (0, 1))
        self.assertEqual(cumulative_loads, (14, 17))

    def test_stable_key_breaks_equal_weight_ties(self):
        assignments, _loads = balance_loads_across_partitions(
            ((4, "z"), (4, "y")),
            initial_partition_loads=(0, 0),
        )
        self.assertEqual(assignments, (1, 0))

    def test_requires_a_partition(self):
        with self.assertRaisesRegex(ValueError, "at least one partition"):
            balance_loads_across_partitions((), initial_partition_loads=())
