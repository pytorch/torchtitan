# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for DP assignment of sorted packed bins."""

import pytest

from torchtitan.rl.components.batcher import assign_sorted_bins_to_dp_ranks


def test_single_pp_microbatch_preserves_global_zigzag() -> None:
    assert assign_sorted_bins_to_dp_ranks(
        [20, 10, 9, 8, 7, 1], dp_degree=2, num_pp_microbatches=1
    ) == [[0, 1], [3, 2], [4, 5]]


def test_two_pp_microbatches_preserve_zigzag_with_ties() -> None:
    assert assign_sorted_bins_to_dp_ranks(
        [20, 10, 9, 8, 7, 7, 6, 5], dp_degree=2, num_pp_microbatches=2
    ) == [[0, 1], [3, 2], [4, 5], [7, 6]]


def test_three_pp_microbatches_balance_accumulated_cost() -> None:
    workloads = [20, 10, 9, 8, 7, 1]
    assignments = assign_sorted_bins_to_dp_ranks(
        workloads, dp_degree=2, num_pp_microbatches=3
    )

    assert assignments == [[0, 1], [3, 2], [5, 4]]
    rank_costs = [sum(workloads[row[rank]] for row in assignments) for rank in range(2)]
    assert rank_costs == [29, 26]
    assert max(rank_costs) - min(rank_costs) == 3
    # Alternating reversal would put 7 on the already heavier rank: [35, 20].
    assert max(rank_costs) - min(rank_costs) < (20 + 8 + 7) - (10 + 9 + 1)


def test_greedy_can_have_a_higher_maximum_than_zigzag() -> None:
    workloads = [92, 77, 64, 61, 58, 47, 36, 36, 35, 31, 18, 8]

    assignments = assign_sorted_bins_to_dp_ranks(
        workloads, dp_degree=3, num_pp_microbatches=4
    )

    rank_costs = [sum(workloads[row[rank]] for row in assignments) for rank in range(3)]
    assert rank_costs == [182, 189, 192]
    assert max(rank_costs) > max([183, 189, 191])


def test_rank_costs_restart_at_each_gradient_accumulation_step() -> None:
    assert assign_sorted_bins_to_dp_ranks(
        [20, 10, 9, 8, 7, 1, 6, 5, 4, 3, 2, 1],
        dp_degree=2,
        num_pp_microbatches=3,
    ) == [[0, 1], [3, 2], [5, 4], [6, 7], [9, 8], [10, 11]]


def test_grid_requires_complete_accumulation_steps() -> None:
    with pytest.raises(AssertionError):
        assign_sorted_bins_to_dp_ranks([20, 10, 9], dp_degree=2, num_pp_microbatches=2)
