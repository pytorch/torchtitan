# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.components.distributed_optimizers.muon.load_balancing import (
    balance_loads_across_partitions,
)
from torchtitan.components.distributed_optimizers.muon.storage_to_compute import (
    _estimate_muon_compute_cost,
)


class TestLoadBalancing(unittest.TestCase):
    def test_balances_compute_bytes_and_cumulative_load(self):
        square_cost = _estimate_muon_compute_cost(torch.Size((8, 8)), ns_steps=2)
        wide_cost = _estimate_muon_compute_cost(torch.Size((4, 16)), ns_steps=2)
        assignments, cumulative_loads = balance_loads_across_partitions(
            (
                (square_cost, 256, "square"),
                (wide_cost, 256, "wide.0"),
                (wide_cost, 256, "wide.1"),
            ),
            initial_cumulative_primary_loads=(0, 0),
        )
        self.assertEqual(assignments, (0, 1, 1))
        self.assertEqual(cumulative_loads, (square_cost, 2 * wide_cost))

        assignments, _ = balance_loads_across_partitions(
            ((8, 1, "a"), (7, 100, "b"), (1, 1, "c"), (1, 1, "d")),
            initial_cumulative_primary_loads=(0, 0),
        )
        self.assertEqual(assignments, (0, 1, 1, 0))

        assignments, cumulative_loads = balance_loads_across_partitions(
            ((4, 10, "a"), (4, 10, "b")),
            initial_cumulative_primary_loads=(100, 0),
        )
        self.assertEqual(assignments, (1, 0))
        self.assertEqual(cumulative_loads, (104, 4))

        assignments, _ = balance_loads_across_partitions(
            ((4, 10, "z"), (4, 10, "y")),
            initial_cumulative_primary_loads=(0, 0),
        )
        self.assertEqual(assignments, (1, 0))


if __name__ == "__main__":
    unittest.main()
