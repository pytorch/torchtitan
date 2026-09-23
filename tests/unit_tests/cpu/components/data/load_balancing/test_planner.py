# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.data.load_balancing.planner import (
    LoadBalancePlan,
    PackableItem,
    PackingBin,
    WholeMicrobatchBalancer,
)


def _bin(rank: int, accumulation: int, pp_microbatch: int = 0) -> PackingBin:
    return PackingBin(
        stable_id=(rank, accumulation, pp_microbatch),
        token_capacity=8,
        document_capacity=4,
        logical_dp_rank=rank,
        accumulation_index=accumulation,
        pp_microbatch_index=pp_microbatch,
    )


def _item(
    item_id: int,
    cost: int,
    original_bin: PackingBin,
    *,
    payload_bytes: int = 10,
) -> PackableItem:
    return PackableItem(
        stable_id=(item_id,),
        num_tokens=8,
        num_documents=1,
        cost=cost,
        payload_bytes=payload_bytes,
        original_bin_id=original_bin.stable_id,
    )


def _item_ids_by_slot(
    plan: LoadBalancePlan, bins: list[PackingBin]
) -> list[tuple[int, ...]]:
    assignments = {
        assignment.bin_id: assignment.item_ids for assignment in plan.assignments
    }
    return [
        assignments[bin_.stable_id][0]
        for bin_ in sorted(
            bins,
            key=lambda bin_: (
                bin_.accumulation_index,
                bin_.pp_microbatch_index,
                bin_.logical_dp_rank,
            ),
        )
    ]


def test_single_rank_orders_whole_microbatches_heavy_to_light_stably() -> None:
    bins = [_bin(0, accumulation) for accumulation in range(3)]
    items = [
        _item(0, 2, bins[0]),
        _item(1, 5, bins[1]),
        _item(2, 5, bins[2]),
    ]

    plan = WholeMicrobatchBalancer.Config().build().plan(items, bins)

    assert _item_ids_by_slot(plan, bins) == [(1,), (2,), (0,)]
    assert plan.predicted_cost <= plan.baseline_predicted_cost


def test_two_rank_plan_groups_heavy_microbatches_in_the_same_slots() -> None:
    bins = [_bin(rank, accumulation) for accumulation in range(2) for rank in range(2)]
    items = [
        _item(0, 10, bins[0]),
        _item(1, 8, bins[1]),
        _item(2, 9, bins[2]),
        _item(3, 1, bins[3]),
    ]

    plan = WholeMicrobatchBalancer.Config().build().plan(items, bins)

    assert plan.baseline_predicted_cost == 19
    assert plan.predicted_cost == 18
    assert _item_ids_by_slot(plan, bins) == [(0,), (2,), (3,), (1,)]


def test_pp_balances_total_rank_cost_within_the_accumulation_step() -> None:
    bins = [_bin(rank, 0, pp) for pp in range(2) for rank in range(2)]
    items = [
        _item(0, 64, bins[0]),
        _item(1, 32, bins[2]),
        _item(2, 16, bins[1]),
        _item(3, 8, bins[3]),
    ]

    plan = WholeMicrobatchBalancer.Config().build().plan(items, bins)

    assert plan.baseline_predicted_cost == 96
    assert plan.predicted_cost == 72
    assert _item_ids_by_slot(plan, bins) == [(0,), (1,), (3,), (2,)]


def test_already_balanced_plan_is_unchanged() -> None:
    bins = [_bin(rank, accumulation) for accumulation in range(2) for rank in range(2)]
    items = [
        _item(0, 10, bins[0]),
        _item(1, 9, bins[1]),
        _item(2, 8, bins[2]),
        _item(3, 1, bins[3]),
    ]

    plan = WholeMicrobatchBalancer.Config().build().plan(items, bins)

    assert plan.is_unchanged
    assert _item_ids_by_slot(plan, bins) == [(0,), (1,), (2,), (3,)]


def test_multi_rank_exact_metric_tie_keeps_baseline() -> None:
    bins = [_bin(rank, accumulation) for accumulation in range(2) for rank in range(2)]
    items = [
        _item(0, 5, bins[2]),
        _item(1, 5, bins[3]),
        _item(2, 5, bins[0]),
        _item(3, 5, bins[1]),
    ]

    plan = WholeMicrobatchBalancer.Config().build().plan(items, bins)

    assert plan.is_unchanged


def test_plan_is_independent_of_input_order() -> None:
    bins = [_bin(rank, accumulation) for accumulation in range(2) for rank in range(2)]
    items = [
        _item(0, 10, bins[0]),
        _item(1, 8, bins[1]),
        _item(2, 9, bins[2]),
        _item(3, 1, bins[3]),
    ]
    balancer = WholeMicrobatchBalancer.Config().build()

    forward = balancer.plan(items, bins)
    reverse = balancer.plan(list(reversed(items)), list(reversed(bins)))

    assert forward == reverse


def test_plan_prefers_to_keep_larger_payload_on_its_original_rank() -> None:
    bins = [_bin(rank, accumulation) for accumulation in range(2) for rank in range(2)]
    items = [
        _item(0, 10, bins[0], payload_bytes=100),
        _item(1, 8, bins[1], payload_bytes=100),
        _item(2, 9, bins[2], payload_bytes=1),
        _item(3, 1, bins[3], payload_bytes=1),
    ]

    plan = WholeMicrobatchBalancer.Config().build().plan(items, bins)

    assert _item_ids_by_slot(plan, bins) == [(0,), (2,), (3,), (1,)]
    assert plan.moved_payload_bytes == 2
