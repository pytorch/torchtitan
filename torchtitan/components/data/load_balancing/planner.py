# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-independent planning for optimizer-step load balancing."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from torchtitan.config import Configurable


StableId = tuple[int, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class PackableItem:
    """Metadata for one indivisible payload handled by the planner."""

    stable_id: StableId
    num_tokens: int
    num_documents: int
    cost: int
    payload_bytes: int
    original_bin_id: StableId

    def __post_init__(self) -> None:
        if not self.stable_id:
            raise ValueError("item stable_id must not be empty")
        if self.num_tokens <= 0:
            raise ValueError("item num_tokens must be greater than 0")
        if self.num_documents <= 0:
            raise ValueError("item num_documents must be greater than 0")
        if self.cost < 0:
            raise ValueError("item cost must be nonnegative")
        if self.payload_bytes < 0:
            raise ValueError("item payload_bytes must be nonnegative")
        if not self.original_bin_id:
            raise ValueError("item original_bin_id must not be empty")


@dataclass(frozen=True, kw_only=True, slots=True)
class PackingBin:
    """Capacity and execution coordinates for one output microbatch."""

    stable_id: StableId
    token_capacity: int
    document_capacity: int | None
    logical_dp_rank: int
    accumulation_index: int
    pp_microbatch_index: int

    def __post_init__(self) -> None:
        if not self.stable_id:
            raise ValueError("bin stable_id must not be empty")
        if self.token_capacity <= 0:
            raise ValueError("bin token_capacity must be greater than 0")
        if self.document_capacity is not None and self.document_capacity <= 0:
            raise ValueError("bin document_capacity must be greater than 0")
        if self.logical_dp_rank < 0:
            raise ValueError("bin logical_dp_rank must be nonnegative")
        if self.accumulation_index < 0:
            raise ValueError("bin accumulation_index must be nonnegative")
        if self.pp_microbatch_index < 0:
            raise ValueError("bin pp_microbatch_index must be nonnegative")


@dataclass(frozen=True, kw_only=True, slots=True)
class BinAssignment:
    """Stable item IDs assigned to one output bin."""

    bin_id: StableId
    item_ids: tuple[StableId, ...]


@dataclass(frozen=True, order=True, kw_only=True, slots=True)
class LoadBalancePlanObjective:
    """Lexicographic optimizer-step objective."""

    synchronized_cost: int
    worst_dp_skew: int
    moved_payload_bytes: int
    stable_tiebreaker: tuple[tuple[StableId, tuple[tuple[int, StableId], ...]], ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class LoadBalancePlan:
    """A validated assignment and its ordinary baseline."""

    assignments: tuple[BinAssignment, ...]
    baseline_assignments: tuple[BinAssignment, ...]
    objective: LoadBalancePlanObjective
    baseline_objective: LoadBalancePlanObjective

    @property
    def is_unchanged(self) -> bool:
        """Return whether the ordinary assignment was retained."""
        return self.assignments == self.baseline_assignments


def _bin_order(bin_: PackingBin) -> tuple[int, int, int, StableId]:
    return (
        bin_.accumulation_index,
        bin_.pp_microbatch_index,
        bin_.logical_dp_rank,
        bin_.stable_id,
    )


def _validate_inputs(
    items: Sequence[PackableItem], bins: Sequence[PackingBin]
) -> tuple[dict[StableId, PackableItem], dict[StableId, PackingBin]]:
    if not items:
        raise ValueError("planner requires at least one item")
    if not bins:
        raise ValueError("planner requires at least one bin")

    items_by_id = {item.stable_id: item for item in items}
    if len(items_by_id) != len(items):
        raise ValueError("item stable IDs must be unique")
    bins_by_id = {bin_.stable_id: bin_ for bin_ in bins}
    if len(bins_by_id) != len(bins):
        raise ValueError("bin stable IDs must be unique")

    unknown_original_bins = {
        item.original_bin_id for item in items if item.original_bin_id not in bins_by_id
    }
    if unknown_original_bins:
        raise ValueError(
            f"items reference unknown original bins: {unknown_original_bins}"
        )

    coordinates = {
        (
            bin_.logical_dp_rank,
            bin_.accumulation_index,
            bin_.pp_microbatch_index,
        )
        for bin_ in bins
    }
    if len(coordinates) != len(bins):
        raise ValueError("bin execution coordinates must be unique")

    ranks_by_slot: dict[tuple[int, int], set[int]] = defaultdict(set)
    for bin_ in bins:
        ranks_by_slot[(bin_.accumulation_index, bin_.pp_microbatch_index)].add(
            bin_.logical_dp_rank
        )
    expected_ranks = next(iter(ranks_by_slot.values()))
    if any(ranks != expected_ranks for ranks in ranks_by_slot.values()):
        raise ValueError("every execution slot must contain the same logical DP ranks")

    return items_by_id, bins_by_id


def _ordinary_assignments(
    items: Sequence[PackableItem], bins_by_id: dict[StableId, PackingBin]
) -> tuple[BinAssignment, ...]:
    item_ids_by_bin: dict[StableId, list[StableId]] = defaultdict(list)
    for item in items:
        item_ids_by_bin[item.original_bin_id].append(item.stable_id)
    return tuple(
        BinAssignment(
            bin_id=bin_.stable_id,
            item_ids=tuple(sorted(item_ids_by_bin[bin_.stable_id])),
        )
        for bin_ in sorted(bins_by_id.values(), key=_bin_order)
    )


def _validate_assignments(
    assignments: Sequence[BinAssignment],
    items_by_id: dict[StableId, PackableItem],
    bins_by_id: dict[StableId, PackingBin],
) -> None:
    assigned_bin_ids = [assignment.bin_id for assignment in assignments]
    if len(set(assigned_bin_ids)) != len(assigned_bin_ids):
        raise ValueError("plan assigns a bin more than once")
    if set(assigned_bin_ids) != set(bins_by_id):
        raise ValueError("plan must assign every bin exactly once")

    assigned_item_ids = [
        item_id for assignment in assignments for item_id in assignment.item_ids
    ]
    if len(set(assigned_item_ids)) != len(assigned_item_ids):
        raise ValueError("plan assigns an item more than once")
    if set(assigned_item_ids) != set(items_by_id):
        raise ValueError("plan must assign every item exactly once")

    for assignment in assignments:
        bin_ = bins_by_id[assignment.bin_id]
        assigned_items = [items_by_id[item_id] for item_id in assignment.item_ids]
        if sum(item.num_tokens for item in assigned_items) > bin_.token_capacity:
            raise ValueError(f"bin {bin_.stable_id} exceeds token capacity")
        if (
            bin_.document_capacity is not None
            and sum(item.num_documents for item in assigned_items)
            > bin_.document_capacity
        ):
            raise ValueError(f"bin {bin_.stable_id} exceeds document capacity")


def _evaluate_assignments(
    assignments: Sequence[BinAssignment],
    items_by_id: dict[StableId, PackableItem],
    bins_by_id: dict[StableId, PackingBin],
) -> LoadBalancePlanObjective:
    costs_by_slot: dict[tuple[int, int], list[int]] = defaultdict(list)
    moved_payload_bytes = 0
    original_ranks = {
        item.stable_id: bins_by_id[item.original_bin_id].logical_dp_rank
        for item in items_by_id.values()
    }
    assignment_by_bin = {assignment.bin_id: assignment for assignment in assignments}

    stable_tiebreaker: list[tuple[StableId, tuple[tuple[int, StableId], ...]]] = []
    for bin_ in sorted(bins_by_id.values(), key=_bin_order):
        assignment = assignment_by_bin[bin_.stable_id]
        assigned_items = [items_by_id[item_id] for item_id in assignment.item_ids]
        bin_cost = sum(item.cost for item in assigned_items)
        costs_by_slot[(bin_.accumulation_index, bin_.pp_microbatch_index)].append(
            bin_cost
        )
        moved_payload_bytes += sum(
            item.payload_bytes
            for item in assigned_items
            if original_ranks[item.stable_id] != bin_.logical_dp_rank
        )
        stable_tiebreaker.append(
            (
                bin_.stable_id,
                tuple((-item.cost, item.stable_id) for item in assigned_items),
            )
        )

    synchronized_cost = sum(max(costs) for costs in costs_by_slot.values())
    worst_dp_skew = max(max(costs) - min(costs) for costs in costs_by_slot.values())
    return LoadBalancePlanObjective(
        synchronized_cost=synchronized_cost,
        worst_dp_skew=worst_dp_skew,
        moved_payload_bytes=moved_payload_bytes,
        stable_tiebreaker=tuple(stable_tiebreaker),
    )


def validate_plan(
    plan: LoadBalancePlan,
    items: Iterable[PackableItem],
    bins: Iterable[PackingBin],
) -> None:
    """Validate exact coverage, capacities, baseline, and objective values."""
    item_list = list(items)
    bin_list = list(bins)
    items_by_id, bins_by_id = _validate_inputs(item_list, bin_list)
    _validate_assignments(plan.assignments, items_by_id, bins_by_id)
    _validate_assignments(plan.baseline_assignments, items_by_id, bins_by_id)

    expected_baseline = _ordinary_assignments(item_list, bins_by_id)
    if plan.baseline_assignments != expected_baseline:
        raise ValueError("plan baseline does not match the ordinary assignment")
    if plan.objective != _evaluate_assignments(
        plan.assignments, items_by_id, bins_by_id
    ):
        raise ValueError("plan objective does not match its assignment")
    if plan.baseline_objective != _evaluate_assignments(
        plan.baseline_assignments, items_by_id, bins_by_id
    ):
        raise ValueError("plan baseline objective does not match its assignment")
    if plan.objective > plan.baseline_objective:
        raise ValueError("planned objective is worse than the ordinary assignment")


class WholeMicrobatchBalancer(Configurable):
    """Assign one indivisible microbatch to every execution bin."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def plan(
        self,
        items: Iterable[PackableItem],
        bins: Iterable[PackingBin],
    ) -> LoadBalancePlan:
        """Return the better of the ordinary and canonical balanced plans."""
        item_list = list(items)
        bin_list = list(bins)
        items_by_id, bins_by_id = _validate_inputs(item_list, bin_list)
        if len(item_list) != len(bin_list):
            raise ValueError("whole-microbatch planning requires one item per bin")

        baseline_assignments = _ordinary_assignments(item_list, bins_by_id)
        if any(len(assignment.item_ids) != 1 for assignment in baseline_assignments):
            raise ValueError(
                "whole-microbatch planning requires one original item per bin"
            )
        _validate_assignments(baseline_assignments, items_by_id, bins_by_id)
        baseline_objective = _evaluate_assignments(
            baseline_assignments, items_by_id, bins_by_id
        )

        candidate_assignments = self._build_candidate(item_list, bin_list, bins_by_id)
        _validate_assignments(candidate_assignments, items_by_id, bins_by_id)
        candidate_objective = _evaluate_assignments(
            candidate_assignments, items_by_id, bins_by_id
        )
        if candidate_objective < baseline_objective:
            assignments = candidate_assignments
            objective = candidate_objective
        else:
            assignments = baseline_assignments
            objective = baseline_objective

        plan = LoadBalancePlan(
            assignments=assignments,
            baseline_assignments=baseline_assignments,
            objective=objective,
            baseline_objective=baseline_objective,
        )
        validate_plan(plan, item_list, bin_list)
        return plan

    def _build_candidate(
        self,
        items: Sequence[PackableItem],
        bins: Sequence[PackingBin],
        bins_by_id: dict[StableId, PackingBin],
    ) -> tuple[BinAssignment, ...]:
        ordered_bins = sorted(bins, key=_bin_order)
        logical_ranks = sorted({bin_.logical_dp_rank for bin_ in bins})
        num_ranks = len(logical_ranks)
        ordered_items = sorted(items, key=lambda item: (-item.cost, item.stable_id))
        assignments: list[BinAssignment] = []

        for start in range(0, len(ordered_bins), num_ranks):
            slot_bins = ordered_bins[start : start + num_ranks]
            slot_items = ordered_items[start : start + num_ranks]
            remaining_items = list(slot_items)
            assigned_by_rank: dict[int, PackableItem] = {}

            for logical_rank in logical_ranks:
                same_owner_items = [
                    item
                    for item in remaining_items
                    if bins_by_id[item.original_bin_id].logical_dp_rank == logical_rank
                ]
                if same_owner_items:
                    kept_item = min(
                        same_owner_items,
                        key=lambda item: (-item.payload_bytes, item.stable_id),
                    )
                    assigned_by_rank[logical_rank] = kept_item
                    remaining_items.remove(kept_item)

            remaining_ranks = [
                logical_rank
                for logical_rank in logical_ranks
                if logical_rank not in assigned_by_rank
            ]
            ordered_remaining_items = sorted(
                remaining_items, key=lambda remaining_item: remaining_item.stable_id
            )
            for logical_rank, item in zip(remaining_ranks, ordered_remaining_items):
                assigned_by_rank[logical_rank] = item

            assignments.extend(
                BinAssignment(
                    bin_id=bin_.stable_id,
                    item_ids=(assigned_by_rank[bin_.logical_dp_rank].stable_id,),
                )
                for bin_ in slot_bins
            )

        return tuple(assignments)
