# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `TrainingSample`s until a group-count training batch is ready, then packs it.
`Batcher` packs a `TrainerStepBatch` of `[num_microbatches][dp_degree]`
`TrainingMicrobatch`es;
"""

import heapq
import logging
import math
from dataclasses import dataclass, replace

import torch

from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.types import (
    TrainerStepBatch,
    TrainingMicrobatch,
    TrainingSample,
    TrainingSampleGroup,
)

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_UNTRAINABLE_BATCHES = 10

# Per-field pad values + tensor dtypes for a packed row.
_PAD_VALUES: dict[str, int | float | bool] = {
    "input_ids": 0,  # overwritten with pad_id in __init__-bound builds
    "labels": 0,
    "generator_logprobs": 0.0,
    "loss_mask": False,
    "advantages": 0.0,
}
_DTYPES: dict[str, torch.dtype] = {
    "input_ids": torch.long,
    "labels": torch.long,
    "generator_logprobs": torch.float,
    "loss_mask": torch.bool,
    "advantages": torch.float,
}


def assign_sorted_bins_to_dp_ranks(
    workloads: list[int], *, dp_degree: int, num_pp_microbatches: int
) -> list[list[int]]:
    """Assign D-wide rows of sorted bins to DP ranks within each G step.

    ``workloads`` is ordered from highest to lowest attention cost. For each
    row of D bins, give the heaviest bin to the rank with the lowest cost
    accumulated in this G step, the next to the next-lowest rank, and so on.
    Rank totals reset for the next G step; every rank receives exactly one
    bin from each of its M rows. This is a greedy heuristic: it does not
    guarantee a smaller maximum rank cost than alternating zig-zag. For M=1,
    keep the original global zig-zag rank order.

    Example:
        # D=2, M=3; the bins are already sorted by attention cost.
        assign_sorted_bins_to_dp_ranks(
            [20, 10, 9, 8, 7, 1], dp_degree=2, num_pp_microbatches=3
        )
        # -> [[0, 1], [3, 2], [5, 4]] (bin indices per rank, per row)
        # Rank totals are [29, 26], versus [35, 20] for zig-zag.
    """
    assert dp_degree > 0 and num_pp_microbatches > 0
    num_bins_per_step = dp_degree * num_pp_microbatches
    assert len(workloads) % num_bins_per_step == 0

    assignments: list[list[int]] = []
    for step_start in range(0, len(workloads), num_bins_per_step):
        rank_workloads = [0] * dp_degree
        for microbatch in range(num_pp_microbatches):
            row_start = step_start + microbatch * dp_degree
            rank_assignments = [0] * dp_degree
            # Preserve the original global zig-zag when M=1, and break ties
            # with the original within-step zig-zag when M>1.
            if num_pp_microbatches == 1:
                tie_direction = -1 if step_start // num_bins_per_step % 2 else 1
            else:
                tie_direction = -1 if microbatch % 2 else 1
            ranks_by_workload = sorted(
                range(dp_degree),
                key=lambda rank: (rank_workloads[rank], tie_direction * rank),
            )
            for offset, rank in enumerate(ranks_by_workload):
                bin_index = row_start + offset
                rank_assignments[rank] = bin_index
                rank_workloads[rank] += workloads[bin_index]
            assignments.append(rank_assignments)

    return assignments


class Batcher(Configurable):
    """Accumulate `num_prompts_per_train_step` groups and packs
    `[num_microbatches][dp_degree]` flat `TrainingMicrobatch`es.

    Example:
        # num_prompts_per_train_step=2, dp_degree=2, 256 tokens/rank
        # The trigger is 2 trainable GROUPS, regardless of how many samples/tokens each contains.
        batcher = Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=256,
            max_context_length=128,
            num_prompts_per_train_step=2,
            dp_degree=2,
            pad_id=0,
        )
        pending, _ = batcher.add_training_samples(training_sample_group=group0)
        batch, _ = batcher.add_training_samples(training_sample_group=group1)
        # pending is None; batch.microbatches: [num_microbatches][2 ranks]; each
        # TrainingMicrobatch.input: [256 tokens]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""
        max_num_documents: int | None = None
        """Maximum non-padding document segments per local trainer microbatch.

        When unset, the trainer uses the microbatch token capacity as the
        CUDA-graph-safe metadata bound without imposing a tighter packing cap.
        """

        def __post_init__(self) -> None:
            if self.max_num_documents is not None and self.max_num_documents <= 0:
                raise ValueError("max_num_documents must be positive")

    def __init__(
        self,
        config: Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        num_prompts_per_train_step: int,
        dp_degree: int,
        pad_id: int,
    ) -> None:
        self.seq_len = max_context_length
        self._num_rows_per_microbatch, remainder = divmod(
            num_tokens_per_microbatch_per_dp_rank,
            max_context_length,
        )
        if remainder:
            raise ValueError(
                "num_tokens_per_microbatch_per_dp_rank "
                f"({num_tokens_per_microbatch_per_dp_rank}) must be divisible by "
                f"max_context_length ({max_context_length})."
            )
        self.pad_id = pad_id
        self._per_sample_pad_multiple = config.per_sample_pad_multiple
        self._max_num_documents = config.max_num_documents
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._dp_degree = dp_degree
        self._groups_for_next_batch: list[TrainingSampleGroup] = []
        self._num_consecutive_zero_output_groups = 0

    def _record_untrainable_groups(self, *, group_is_trainable: bool) -> None:
        """Fail when consecutive groups cannot contribute one training sample.

        Each ``num_prompts_per_train_step`` consecutive untrainable groups
        represents one complete untrainable batch. For a target of 8 groups,
        warn after each block of 8 and fail after 10 such batches (80 groups).
        Any trainable group resets the count.
        """
        # A useful group proves the pipeline is making progress, even before
        # enough useful groups have accumulated to form a complete batch.
        if group_is_trainable:
            self._num_consecutive_zero_output_groups = 0
            return

        self._num_consecutive_zero_output_groups += 1
        # Report once per group count that would normally produce a training step.
        if self._num_consecutive_zero_output_groups % self._num_prompts_per_train_step:
            return

        num_untrainable_batches = (
            self._num_consecutive_zero_output_groups // self._num_prompts_per_train_step
        )
        logger.warning(
            "Consecutive untrainable batches: %d/%d (%d rollout groups "
            "produced no trainable samples).",
            num_untrainable_batches,
            _MAX_CONSECUTIVE_UNTRAINABLE_BATCHES,
            self._num_consecutive_zero_output_groups,
        )
        if num_untrainable_batches < _MAX_CONSECUTIVE_UNTRAINABLE_BATCHES:
            return

        raise RuntimeError(
            f"{num_untrainable_batches} consecutive untrainable batches "
            f"({self._num_consecutive_zero_output_groups} rollout groups); "
            "check reward diversity and training-sample filters."
        )

    def add_training_samples(
        self, *, training_sample_group: TrainingSampleGroup
    ) -> tuple[TrainerStepBatch | None, bool]:
        """Add one group and report whether any samples survive batcher filtering.

        Args:
            training_sample_group: One rollout group's trainable samples plus rollout metrics.

        Example:
            batcher = Batcher.Config().build(
                num_tokens_per_microbatch_per_dp_rank=16384,
                max_context_length=2048,
                num_prompts_per_train_step=2,
                dp_degree=1,
                pad_id=0,
            )
            batcher.add_training_samples(training_sample_group=group0)  # -> (None, True)
            batcher.add_training_samples(training_sample_group=group1)  # -> (TrainerStepBatch, True)
        """
        # Drop samples longer than seq_len: can't fill a row
        samples = training_sample_group.training_samples
        kept = [s for s in samples if self.num_tokens_to_pack(s) <= self.seq_len]
        num_dropped = len(samples) - len(kept)
        if num_dropped:
            logger.warning(
                "Batcher dropped %d/%d sample(s) exceeding seq_len=%d.",
                num_dropped,
                len(samples),
                self.seq_len,
            )
            training_sample_group = replace(
                training_sample_group,
                training_samples=kept,
                metrics=[
                    *training_sample_group.metrics,
                    m.Metric(
                        "batcher/num_samples_dropped_oversized",
                        m.Sum(float(num_dropped)),
                    ),
                ],
            )

        group_is_trainable = bool(training_sample_group.training_samples)
        self._groups_for_next_batch.append(training_sample_group)
        self._record_untrainable_groups(group_is_trainable=group_is_trainable)
        num_trainable_groups = sum(
            bool(group.training_samples) for group in self._groups_for_next_batch
        )
        if num_trainable_groups < self._num_prompts_per_train_step:
            return None, group_is_trainable  # accumulate until one full batch is ready
        return self._pack_one_training_batch(), group_is_trainable

    def _pack_one_training_batch(self) -> TrainerStepBatch:
        """Pack the oldest accumulated groups (up to `num_prompts_per_train_step` trainable groups) into one batch."""
        (
            training_samples,
            metrics,
            num_rollout_groups,
            num_metric_only_groups,
        ) = self._take_groups()
        baseline_assignments = self._assign_with_row_packing(training_samples)
        padding_frac_before = self._padding_fraction(
            num_microbatches=len(baseline_assignments),
            training_samples=training_samples,
        )
        assignments = self._assign_training_samples_to_microbatches(training_samples)
        microbatches = [
            [self._pack_training_samples(samples) for samples in rank_assignments]
            for rank_assignments in assignments
        ]
        num_global_valid_tokens = sum(
            int(
                (microbatch.loss_mask & torch.isfinite(microbatch.generator_logprobs))
                .sum()
                .item()
            )
            for rank_microbatches in microbatches
            for microbatch in rank_microbatches
        )
        num_response_tokens = sum(
            int(microbatch.loss_mask.sum().item())
            for rank_microbatches in microbatches
            for microbatch in rank_microbatches
        )
        return TrainerStepBatch(
            microbatches=microbatches,
            num_global_valid_tokens=num_global_valid_tokens,
            metrics=[
                *metrics,
                # Keep this response-level metric exact without adding a second
                # token-count field to TrainerStepBatch.
                m.Metric(
                    "loss/generator_logprob_nan_frac",
                    m.NoReduce(
                        (num_response_tokens - num_global_valid_tokens)
                        / max(num_response_tokens, 1)
                    ),
                ),
                *self._packing_metrics(
                    assignments,
                    training_samples,
                    num_rollout_groups,
                    num_metric_only_groups,
                    padding_frac_before=padding_frac_before,
                ),
            ],
            # Trainer computes policy_age from these at consume time (faithful to what it trains on).
            # min_policy_version is the oldest version this training_sample was sampled under.
            min_policy_versions=[
                training_sample.min_policy_version
                for training_sample in training_samples
            ],
        )

    def _take_groups(
        self,
    ) -> tuple[list[TrainingSample], list[m.Metric], int, int]:
        """Pop accumulated groups oldest-first until `num_prompts_per_train_step` are taken."""
        taken_training_samples: list[TrainingSample] = []
        taken_metrics: list[m.Metric] = []
        num_trainable_groups = 0
        cut = 0
        for group in self._groups_for_next_batch:
            if num_trainable_groups >= self._num_prompts_per_train_step:
                break
            cut += 1
            taken_metrics.extend(group.metrics)
            if group.training_samples:
                num_trainable_groups += 1

        # Pack in group-id order so on-policy runs stay reproducible whatever the finish order.
        for group in sorted(
            self._groups_for_next_batch[:cut], key=lambda taken: taken.group_id
        ):
            taken_training_samples.extend(group.training_samples)

        # surplus carried over
        self._groups_for_next_batch = self._groups_for_next_batch[cut:]
        num_metric_only_groups: int = cut - num_trainable_groups

        return (
            taken_training_samples,
            taken_metrics,
            num_trainable_groups,
            num_metric_only_groups,
        )

    def _assign_training_samples_to_microbatches(
        self,
        training_samples: list[TrainingSample],
        *,
        num_pp_microbatches: int = 1,
    ) -> list[list[list[TrainingSample]]]:
        """Pack samples into an FFD-derived grid and balance its attention work.

        T is the token capacity per DP rank, D the DP degree, and M the number
        of PP microbatches per scheduling step. First-fit decreasing (FFD)
        determines B bins subject to T and max_num_documents. Each step has
        M * D slots, so G = ceil(B / (M * D)) steps form a ``[G][M][D]`` grid.
        Try longest-processing-time (LPT) packing: place longest samples first in
        the feasible bin with the highest padding-aware attention work,
        breaking ties by fewer packed tokens. If any
        sample cannot fit, discard the partial LPT assignment and reuse FFD,
        splitting its bins to fill the extra slots where possible. Both paths
        have the same grid size and therefore the same total padding fraction.

        Sort the bins by padding-aware attention cost (see
        ``_attention_workload``), group consecutive bins into D-wide rows, and
        greedily assign each row's heaviest bin to the rank with the least
        cumulative cost in its G step. This heuristic does not guarantee an
        optimal or zig-zag-dominating maximum rank cost. With M=1 it preserves
        the original alternating rank order between G steps.

        This method returns the first two grid axes flattened as
        ``[G * M][D]``. The RL trainer currently uses M=1; M>1 is available
        for PP scheduling tests.
        """
        num_tokens_per_rank = self._num_rows_per_microbatch * self.seq_len

        # Step 1: find the bin count with FFD, retaining its fallback packing.
        ordered_samples = sorted(
            training_samples,
            key=self.num_tokens_to_pack,
            reverse=True,
        )
        ffd_bins: list[list[TrainingSample]] = []
        ffd_bin_num_tokens: list[int] = []

        for training_sample in ordered_samples:
            num_tokens = self.num_tokens_to_pack(training_sample)
            destination = next(
                (
                    index
                    for index, bin_ in enumerate(ffd_bins)
                    if ffd_bin_num_tokens[index] + num_tokens <= num_tokens_per_rank
                    and (
                        self._max_num_documents is None
                        or len(bin_) < self._max_num_documents
                    )
                ),
                None,
            )
            if destination is None:
                ffd_bins.append([])
                ffd_bin_num_tokens.append(0)
                destination = len(ffd_bins) - 1

            ffd_bins[destination].append(training_sample)
            ffd_bin_num_tokens[destination] += num_tokens

        # Step 2: make the [G][M][D] grid rectangular.
        num_bins_per_step = num_pp_microbatches * self._dp_degree
        assert num_bins_per_step > 0
        target_num_bins = (
            math.ceil(len(ffd_bins) / num_bins_per_step) * num_bins_per_step
        )

        # Step 3: repack across the entire grid, preserving the FFD bin count.
        bins = self._pack_with_lpt(ordered_samples, target_num_bins=target_num_bins)
        if bins is None:
            bins = ffd_bins
            self._expand_bins_by_splitting(bins, target_num_bins=target_num_bins)

        # Step 4: estimate attention work, including tail padding.
        workloads = [self._attention_workload(bin_) for bin_ in bins]
        sorted_bin_indices = sorted(
            range(len(bins)), key=workloads.__getitem__, reverse=True
        )

        # Steps 5-6: keep adjacent bins together and balance their rank totals.
        bin_indices_by_rank = assign_sorted_bins_to_dp_ranks(
            [workloads[index] for index in sorted_bin_indices],
            dp_degree=self._dp_degree,
            num_pp_microbatches=num_pp_microbatches,
        )
        return [
            [bins[sorted_bin_indices[index]] for index in rank_indices]
            for rank_indices in bin_indices_by_rank
        ]

    def _attention_workload(self, training_samples: list[TrainingSample]) -> int:
        """Estimate document and tail-padding attention as squared lengths.

        For packed sample lengths L and token capacity T, let
        ``q, r = divmod(T - sum(L), seq_len)``. The estimate is
        ``sum(L**2) + q * seq_len**2 + r**2``. Padding positions reset at
        seq_len, so even an empty bin costs ``(T // seq_len) * seq_len**2``.
        This models the existing padding segments without increasing their
        number or changing fixed-size varlen metadata.
        """
        # TODO(rl): Account for hybrid sliding-window or linear-attention layers.
        sample_lengths = [
            self.num_tokens_to_pack(sample) for sample in training_samples
        ]
        num_tokens_per_bin = self._num_rows_per_microbatch * self.seq_len
        num_full_padding_segments, remaining_padding = divmod(
            num_tokens_per_bin - sum(sample_lengths), self.seq_len
        )
        return (
            sum(length**2 for length in sample_lengths)
            + num_full_padding_segments * self.seq_len**2
            + remaining_padding**2
        )

    def _pack_with_lpt(
        self,
        ordered_samples: list[TrainingSample],
        *,
        target_num_bins: int,
    ) -> list[list[TrainingSample]] | None:
        """Put longest samples in the highest-workload feasible fixed bin.

        Adding a sample cannot increase padding-aware attention work. Break
        workload ties by fewer packed tokens to spread full-length documents
        whose work exactly replaces one padding segment.
        Return None if token or document capacity blocks a sample; the caller
        then discards this partial assignment and uses the FFD fallback.
        """
        num_tokens_per_bin = self._num_rows_per_microbatch * self.seq_len
        bins: list[list[TrainingSample]] = [[] for _ in range(target_num_bins)]
        bin_num_tokens = [0] * target_num_bins
        bin_workloads = [self._attention_workload([])] * target_num_bins

        for sample in ordered_samples:
            num_tokens = self.num_tokens_to_pack(sample)
            destination = max(
                (
                    index
                    for index, bin_ in enumerate(bins)
                    if bin_num_tokens[index] + num_tokens <= num_tokens_per_bin
                    and (
                        self._max_num_documents is None
                        or len(bin_) < self._max_num_documents
                    )
                ),
                key=lambda index: (bin_workloads[index], -bin_num_tokens[index]),
                default=None,
            )
            if destination is None:
                return None

            bins[destination].append(sample)
            bin_num_tokens[destination] += num_tokens
            bin_workloads[destination] = self._attention_workload(bins[destination])

        return bins

    def _expand_bins_by_splitting(
        self,
        bins: list[list[TrainingSample]],
        *,
        target_num_bins: int,
    ) -> None:
        """Fill extra FFD grid slots from multi-sample bins when this lowers cost.

        A max-heap tracks each bin with more than one sample by attention
        workload. For every required bin, move a sample from the current
        heaviest donor only if it reduces the higher of the two bins' estimated
        costs, including padding. Filling stops at the new bin's
        token or document limit. Keeping one sample in every donor avoids
        replacing one empty bin with another.

        Every successful inner-loop iteration moves one sample. The outer loop
        either appends a non-empty bin or stops and pads the remaining slots, so
        neither loop can stall when no further redistribution is possible.
        """
        num_tokens_per_bin = self._num_rows_per_microbatch * self.seq_len
        donor_heap = [
            (-self._attention_workload(bin_), -len(bin_), index)
            for index, bin_ in enumerate(bins)
            if len(bin_) > 1
        ]
        heapq.heapify(donor_heap)

        while len(bins) < target_num_bins:
            new_bin: list[TrainingSample] = []
            new_bin_num_tokens = 0
            skipped_donors: list[tuple[int, int, int]] = []

            while donor_heap and (
                self._max_num_documents is None
                or len(new_bin) < self._max_num_documents
            ):
                negative_workload, _, donor_index = heapq.heappop(donor_heap)
                donor = bins[donor_index]
                donor_workload = -negative_workload
                new_bin_workload = self._attention_workload(new_bin)
                current_max_workload = max(donor_workload, new_bin_workload)
                best_move: tuple[int, int, int, int] | None = None
                for sample_index, sample in enumerate(donor):
                    num_tokens = self.num_tokens_to_pack(sample)
                    if new_bin_num_tokens + num_tokens > num_tokens_per_bin:
                        continue
                    donor_after = self._attention_workload(
                        donor[:sample_index] + donor[sample_index + 1 :]
                    )
                    new_bin_after = self._attention_workload([*new_bin, sample])
                    max_after = max(donor_after, new_bin_after)
                    if max_after >= current_max_workload:
                        continue
                    move = (max_after, -num_tokens, sample_index, donor_after)
                    if best_move is None or move < best_move:
                        best_move = move

                if best_move is None:
                    skipped_donors.append((negative_workload, -len(donor), donor_index))
                    continue

                _, negative_num_tokens, sample_index, donor_after = best_move
                num_tokens = -negative_num_tokens
                sample = donor[sample_index]
                donor.pop(sample_index)
                new_bin.append(sample)
                new_bin_num_tokens += num_tokens

                if len(donor) > 1:
                    heapq.heappush(
                        donor_heap,
                        (-donor_after, -len(donor), donor_index),
                    )

                # Filling the new bin can make a previously skipped move useful.
                for donor_entry in skipped_donors:
                    heapq.heappush(donor_heap, donor_entry)
                skipped_donors.clear()

            for donor_entry in skipped_donors:
                heapq.heappush(donor_heap, donor_entry)

            if not new_bin:
                break

            bins.append(new_bin)
            if len(new_bin) > 1:
                heapq.heappush(
                    donor_heap,
                    (-self._attention_workload(new_bin), -len(new_bin), len(bins) - 1),
                )

        bins.extend([] for _ in range(target_num_bins - len(bins)))

    def _fill_empty_rank_assignments(
        self, assignments: list[list[list[TrainingSample]]]
    ) -> None:
        """Move samples from non-singleton cells into otherwise empty DP cells."""
        cells = [cell for microbatch in assignments for cell in microbatch]
        for empty_cell in [cell for cell in cells if not cell]:
            donor = max(
                (cell for cell in cells if len(cell) > 1),
                key=lambda cell: sum(
                    self.num_tokens_to_pack(sample) for sample in cell
                ),
                default=None,
            )
            if donor is None:
                break
            sample_index = min(
                range(len(donor)),
                key=lambda index: self.num_tokens_to_pack(donor[index]),
            )
            empty_cell.append(donor.pop(sample_index))

    def _assign_training_samples_to_rows(
        self, training_samples: list[TrainingSample]
    ) -> list[list[TrainingSample]]:
        """Build the previous next-fit assignment for comparison and fallback.

        Example:

            # seq_len=10, training_sample effective lengths [5, 5, 5]
            _assign_training_samples_to_rows([e5, e5, e5])  # -> [[e5, e5], [e5]]
        """
        rows: list[list[TrainingSample]] = []
        current_row: list[TrainingSample] = []
        current_len = 0
        for training_sample in training_samples:
            num_tokens_to_pack = self.num_tokens_to_pack(training_sample)

            # A row must fit in one local microbatch, so the full microbatch
            # document limit is also a valid upper bound for one row.
            if current_row and (
                current_len + num_tokens_to_pack > self.seq_len
                or (
                    self._max_num_documents is not None
                    and len(current_row) >= self._max_num_documents
                )
            ):
                rows.append(current_row)
                current_row, current_len = [], 0

            current_row.append(training_sample)
            current_len += num_tokens_to_pack

        if current_row:
            rows.append(current_row)

        return rows

    def _assign_with_row_packing(
        self, training_samples: list[TrainingSample]
    ) -> list[list[list[TrainingSample]]]:
        """Return assignments from the previous fixed-row policy."""
        rows = self._assign_training_samples_to_rows(training_samples)
        if self._max_num_documents is None:
            rows_per_microbatch = self._num_rows_per_microbatch * self._dp_degree
            num_microbatches = max(1, math.ceil(len(rows) / rows_per_microbatch))
            num_cells = num_microbatches * self._dp_degree
            cells: list[list[TrainingSample]] = [[] for _ in range(num_cells)]
            for index, row in enumerate(rows):
                cells[index % num_cells].extend(row)
            return [
                cells[start : start + self._dp_degree]
                for start in range(0, num_cells, self._dp_degree)
            ]

        cells: list[list[TrainingSample]] = []
        current_cell: list[TrainingSample] = []
        current_num_rows = 0
        current_num_documents = 0
        for row in rows:
            num_documents = len(row)
            assert num_documents <= self._max_num_documents
            if current_cell and (
                current_num_rows >= self._num_rows_per_microbatch
                or current_num_documents + num_documents > self._max_num_documents
            ):
                cells.append(current_cell)
                current_cell = []
                current_num_rows = 0
                current_num_documents = 0
            current_cell.extend(row)
            current_num_rows += 1
            current_num_documents += num_documents
        if current_cell:
            cells.append(current_cell)

        num_microbatches = max(1, math.ceil(len(cells) / self._dp_degree))
        cells.extend([] for _ in range(num_microbatches * self._dp_degree - len(cells)))
        assignments = [
            cells[start : start + self._dp_degree]
            for start in range(0, len(cells), self._dp_degree)
        ]
        self._fill_empty_rank_assignments(assignments)
        return assignments

    def num_tokens_to_pack(self, training_sample: TrainingSample) -> int:
        """Tokens this training_sample contributes to a packed input.

        The loss-target split drops the last token (``input_ids = raw[:-1]``), and batch-invariant
        mode rounds the length up to ``per_sample_pad_multiple``.

        Example:

            # token_ids of length 6, per_sample_pad_multiple=None  -> 5
            # token_ids of length 6, per_sample_pad_multiple=8     -> 8
        """
        num_tokens = len(training_sample.token_ids) - 1
        if self._per_sample_pad_multiple:
            multiple = self._per_sample_pad_multiple
            num_tokens = ((num_tokens + multiple - 1) // multiple) * multiple
        return num_tokens

    # TODO(async-rl): make packing pluggable -- a `Packer` protocol on `Batcher.Config` (e.g. `TextPacker`)
    #   so callers swap logic per modality (images, ...).
    def _pack_training_samples(
        self, training_samples: list[TrainingSample]
    ) -> TrainingMicrobatch:
        """Concatenate samples into one fixed-size local microbatch.

        - Labels and logits are shifted
        -`positions` restart at 0 per sample

        Example:

            # Two three-token samples in an eight-token local microbatch.
            input_ids = [10, 11, 20, 21, 0, 0, 0, 0]
            labels    = [11, 12, 21, 22, 0, 0, 0, 0]
            positions = [ 0,  1,  0,  1, 0, 1, 2, 3]
        """
        pad_values = {**_PAD_VALUES, "input_ids": self.pad_id, "labels": self.pad_id}
        keys = list(pad_values)
        packed_fields: dict[str, list] = {key: [] for key in keys}
        positions: list[int] = []
        padding_mask: list[bool] = []

        # Shift labels/logits and pad to per_sample_pad_multiple.
        for training_sample in training_samples:
            sample = {
                "input_ids": training_sample.token_ids[:-1],
                "labels": training_sample.token_ids[1:],
                "generator_logprobs": training_sample.logprobs[1:],
                "loss_mask": training_sample.loss_mask[1:],
                "advantages": training_sample.advantage[1:],
            }
            sample_len = len(sample["input_ids"])
            unpadded_len = sample_len

            # pad to multiple
            if self._per_sample_pad_multiple:
                align = self._per_sample_pad_multiple
                padded_len = ((sample_len + align - 1) // align) * align
                for key in keys:
                    sample[key] = sample[key] + [pad_values[key]] * (
                        padded_len - sample_len
                    )
                sample_len = padded_len

            # extend row
            for key in keys:
                packed_fields[key].extend(sample[key])
            positions.extend(range(sample_len))
            padding_mask.extend([False] * unpadded_len)
            padding_mask.extend([True] * (sample_len - unpadded_len))

        num_tokens_per_rank = self._num_rows_per_microbatch * self.seq_len
        pad_len = num_tokens_per_rank - len(positions)
        assert pad_len >= 0
        if pad_len > 0:
            for key in keys:
                packed_fields[key].extend([pad_values[key]] * pad_len)
            positions.extend(index % self.seq_len for index in range(pad_len))
            padding_mask.extend([True] * pad_len)

        generator_logprobs = torch.tensor(
            packed_fields["generator_logprobs"], dtype=_DTYPES["generator_logprobs"]
        )
        loss_mask = torch.tensor(packed_fields["loss_mask"], dtype=_DTYPES["loss_mask"])
        return TrainingMicrobatch(
            input=torch.tensor(
                packed_fields["input_ids"], dtype=_DTYPES["input_ids"]
            ),
            labels=torch.tensor(packed_fields["labels"], dtype=_DTYPES["labels"]),
            positions=torch.tensor(positions, dtype=torch.long),
            generator_logprobs=generator_logprobs,
            loss_mask=loss_mask,
            advantages=torch.tensor(
                packed_fields["advantages"], dtype=_DTYPES["advantages"]
            ),
            padding_mask=torch.tensor(padding_mask, dtype=torch.bool),
            num_valid_tokens=int(
                (loss_mask & torch.isfinite(generator_logprobs)).sum().item()
            ),
        )

    def _padding_fraction(
        self,
        *,
        num_microbatches: int,
        training_samples: list[TrainingSample],
    ) -> float:
        num_tokens_per_rank = self._num_rows_per_microbatch * self.seq_len
        total_slots = num_microbatches * self._dp_degree * num_tokens_per_rank
        num_real_tokens = sum(len(sample.token_ids) - 1 for sample in training_samples)
        return (total_slots - num_real_tokens) / total_slots

    def _packing_metrics(
        self,
        assignments: list[list[list[TrainingSample]]],
        training_samples: list[TrainingSample],
        num_rollout_groups: int,
        num_metric_only_groups: int,
        padding_frac_before: float,
    ) -> list[m.Metric]:
        """Per-training-batch packing + count metrics. (policy age is logged at trainer consume time.)"""
        padding_frac = self._padding_fraction(
            num_microbatches=len(assignments),
            training_samples=training_samples,
        )
        return [
            m.Metric(
                "train_batch/padding_frac_before_load_balance",
                m.NoReduce(padding_frac_before),
            ),
            m.Metric(
                "train_batch/padding_frac",
                m.NoReduce(padding_frac),
            ),
            m.Metric(
                "train_batch/num_microbatches",
                m.NoReduce(float(len(assignments))),
            ),
            m.Metric(
                "train_batch/num_rollout_groups", m.NoReduce(float(num_rollout_groups))
            ),
            m.Metric(
                "train_batch/num_metric_only_groups",
                m.NoReduce(float(num_metric_only_groups)),
            ),
            m.Metric(
                "train_batch/num_training_samples",
                m.NoReduce(float(len(training_samples))),
            ),
        ]
