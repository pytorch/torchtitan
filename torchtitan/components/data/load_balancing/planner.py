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


@dataclass(frozen=True, kw_only=True, slots=True)
class PackingBin:
    """Capacity and execution coordinates for one output microbatch."""

    stable_id: StableId
    token_capacity: int
    document_capacity: int | None
    logical_dp_rank: int
    accumulation_index: int
    pp_microbatch_index: int


@dataclass(frozen=True, kw_only=True, slots=True)
class BinAssignment:
    """Stable item IDs assigned to one output bin."""

    bin_id: StableId
    item_ids: tuple[StableId, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class LoadBalancePlan:
    """Selected assignments, ordinary baseline, and reporting metrics."""

    assignments: tuple[BinAssignment, ...]
    baseline_assignments: tuple[BinAssignment, ...]
    predicted_cost: int
    baseline_predicted_cost: int
    moved_payload_bytes: int

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


def _ordinary_assignments(
    items: Sequence[PackableItem], bins: Sequence[PackingBin]
) -> tuple[BinAssignment, ...]:
    item_ids_by_bin: dict[StableId, list[StableId]] = defaultdict(list)
    for item in items:
        item_ids_by_bin[item.original_bin_id].append(item.stable_id)
    return tuple(
        BinAssignment(
            bin_id=bin_.stable_id,
            item_ids=tuple(sorted(item_ids_by_bin[bin_.stable_id])),
        )
        for bin_ in sorted(bins, key=_bin_order)
    )


def _score_assignments(
    assignments: Sequence[BinAssignment],
    items_by_id: dict[StableId, PackableItem],
    bins_by_id: dict[StableId, PackingBin],
) -> tuple[int, int, int]:
    """Return accumulation cost, worst DP skew, and moved payload bytes.

    One gradient-accumulation iteration invokes the PP schedule once, so PP
    microbatch costs are summed per DP rank before taking the slowest rank.
    """
    costs_by_accumulation_and_rank: dict[tuple[int, int], int] = defaultdict(int)
    moved_payload_bytes = 0
    assignment_by_bin = {assignment.bin_id: assignment for assignment in assignments}

    for bin_ in sorted(bins_by_id.values(), key=_bin_order):
        assigned_items = [
            items_by_id[item_id]
            for item_id in assignment_by_bin[bin_.stable_id].item_ids
        ]
        bin_cost = sum(item.cost for item in assigned_items)
        costs_by_accumulation_and_rank[
            (bin_.accumulation_index, bin_.logical_dp_rank)
        ] += bin_cost
        moved_payload_bytes += sum(
            item.payload_bytes
            for item in assigned_items
            if bins_by_id[item.original_bin_id].logical_dp_rank != bin_.logical_dp_rank
        )

    logical_ranks = sorted({bin_.logical_dp_rank for bin_ in bins_by_id.values()})
    accumulation_indices = sorted(
        {bin_.accumulation_index for bin_ in bins_by_id.values()}
    )
    costs_by_accumulation = [
        [
            costs_by_accumulation_and_rank[(accumulation_index, logical_rank)]
            for logical_rank in logical_ranks
        ]
        for accumulation_index in accumulation_indices
    ]
    synchronized_cost = sum(max(costs) for costs in costs_by_accumulation)
    worst_dp_skew = max(max(costs) - min(costs) for costs in costs_by_accumulation)
    return synchronized_cost, worst_dp_skew, moved_payload_bytes


def _order_accumulation_steps_heavy_first(
    assignments: Sequence[BinAssignment],
    items_by_id: dict[StableId, PackableItem],
    bins: Sequence[PackingBin],
) -> tuple[BinAssignment, ...]:
    """Order complete accumulation steps by descending synchronized cost."""
    assignment_by_bin = {assignment.bin_id: assignment for assignment in assignments}
    bins_by_coordinate = {
        (
            bin_.accumulation_index,
            bin_.logical_dp_rank,
            bin_.pp_microbatch_index,
        ): bin_
        for bin_ in bins
    }
    costs_by_accumulation_and_rank: dict[tuple[int, int], int] = defaultdict(int)
    for bin_ in bins:
        costs_by_accumulation_and_rank[
            (bin_.accumulation_index, bin_.logical_dp_rank)
        ] += sum(
            items_by_id[item_id].cost
            for item_id in assignment_by_bin[bin_.stable_id].item_ids
        )

    accumulation_indices = sorted({bin_.accumulation_index for bin_ in bins})
    logical_ranks = sorted({bin_.logical_dp_rank for bin_ in bins})
    source_accumulations = sorted(
        accumulation_indices,
        key=lambda accumulation_index: (
            -max(
                costs_by_accumulation_and_rank[(accumulation_index, logical_rank)]
                for logical_rank in logical_ranks
            ),
            accumulation_index,
        ),
    )
    source_by_destination = dict(zip(accumulation_indices, source_accumulations))

    ordered_assignments = []
    for destination_bin in sorted(bins, key=_bin_order):
        source_bin = bins_by_coordinate[
            (
                source_by_destination[destination_bin.accumulation_index],
                destination_bin.logical_dp_rank,
                destination_bin.pp_microbatch_index,
            )
        ]
        ordered_assignments.append(
            BinAssignment(
                bin_id=destination_bin.stable_id,
                item_ids=assignment_by_bin[source_bin.stable_id].item_ids,
            )
        )
    return tuple(ordered_assignments)


class WholeMicrobatchBalancer(Configurable):
    """Balance indivisible microbatches across accumulation steps and DP ranks.

    Microbatches are sorted by descending cost and divided into one cohort per
    gradient-accumulation step. Within each cohort, capacity-constrained Longest
    Processing Time assignment places the next heaviest microbatch on the
    currently lightest DP rank. Each rank receives exactly one item for every
    PP microbatch position, but PP positions do not affect the cost objective.
    """

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
        """Return the better of the ordinary and balanced assignments."""
        item_list = list(items)
        bin_list = list(bins)
        items_by_id = {item.stable_id: item for item in item_list}
        bins_by_id = {bin_.stable_id: bin_ for bin_ in bin_list}

        baseline_assignments = _ordinary_assignments(item_list, bin_list)
        baseline_score = _score_assignments(
            baseline_assignments, items_by_id, bins_by_id
        )

        candidate_assignments = self._build_candidate(item_list, bin_list, bins_by_id)
        candidate_score = _score_assignments(
            candidate_assignments, items_by_id, bins_by_id
        )

        num_ranks = len({bin_.logical_dp_rank for bin_ in bin_list})
        # Scores compare accumulation cost, DP skew, then moved bytes. With one
        # visible rank these values always tie, so retain the canonical
        # heavy-first ordering to align independent ranks by accumulation step.
        use_candidate = candidate_score < baseline_score or (
            num_ranks == 1 and candidate_score == baseline_score
        )
        assignments, score = (
            (candidate_assignments, candidate_score)
            if use_candidate
            else (baseline_assignments, baseline_score)
        )
        # Independent balancing groups do not communicate their schedules. Give
        # every selected plan the same heavy-first temporal convention so costly
        # accumulation steps tend to overlap globally instead of serializing.
        assignments = _order_accumulation_steps_heavy_first(
            assignments,
            items_by_id,
            bin_list,
        )

        return LoadBalancePlan(
            assignments=assignments,
            baseline_assignments=baseline_assignments,
            predicted_cost=score[0],
            baseline_predicted_cost=baseline_score[0],
            moved_payload_bytes=score[2],
        )

    def _build_candidate(
        self,
        items: Sequence[PackableItem],
        bins: Sequence[PackingBin],
        bins_by_id: dict[StableId, PackingBin],
    ) -> tuple[BinAssignment, ...]:
        """Build a deterministic two-level accumulation and DP assignment."""
        ordered_bins = sorted(bins, key=_bin_order)
        ordered_items = sorted(items, key=lambda item: (-item.cost, item.stable_id))
        item_offset = 0
        item_id_by_bin: dict[StableId, StableId] = {}

        for accumulation_index in sorted(
            {bin_.accumulation_index for bin_ in ordered_bins}
        ):
            # Taking a full accumulation step at a time groups globally heavy
            # work into the same sequential trainer iteration.
            accumulation_bins = [
                bin_
                for bin_ in ordered_bins
                if bin_.accumulation_index == accumulation_index
            ]
            cohort = ordered_items[item_offset : item_offset + len(accumulation_bins)]
            item_offset += len(accumulation_bins)

            bins_by_rank: dict[int, list[PackingBin]] = defaultdict(list)
            for bin_ in accumulation_bins:
                bins_by_rank[bin_.logical_dp_rank].append(bin_)

            assigned_items_by_rank: dict[int, list[PackableItem]] = {
                logical_rank: [] for logical_rank in bins_by_rank
            }
            assigned_cost_by_rank = {logical_rank: 0 for logical_rank in bins_by_rank}

            # Capacity-constrained LPT: place the next heaviest item on the
            # lightest rank, preferring its original owner when loads tie.
            for item in cohort:
                original_rank = bins_by_id[item.original_bin_id].logical_dp_rank
                eligible_ranks = [
                    logical_rank
                    for logical_rank, assigned_items in assigned_items_by_rank.items()
                    if len(assigned_items) < len(bins_by_rank[logical_rank])
                ]
                selected_rank = min(
                    eligible_ranks,
                    key=lambda logical_rank: (
                        assigned_cost_by_rank[logical_rank],
                        logical_rank != original_rank,
                        logical_rank,
                    ),
                )
                assigned_items_by_rank[selected_rank].append(item)
                assigned_cost_by_rank[selected_rank] += item.cost

            # PP indices only provide deterministic positions within the
            # accumulation step; they are not part of the cost objective.
            for logical_rank, rank_bins in bins_by_rank.items():
                rank_items = sorted(
                    assigned_items_by_rank[logical_rank],
                    key=lambda item: item.stable_id,
                )
                for bin_, item in zip(sorted(rank_bins, key=_bin_order), rank_items):
                    item_id_by_bin[bin_.stable_id] = item.stable_id

        return tuple(
            BinAssignment(
                bin_id=bin_.stable_id,
                item_ids=(item_id_by_bin[bin_.stable_id],),
            )
            for bin_ in ordered_bins
        )
