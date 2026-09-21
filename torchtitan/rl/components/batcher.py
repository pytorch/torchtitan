# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `TrainingSample`s until a group-count training batch is ready, then packs it.
`Batcher` packs a `TrainerStepBatch` of `[num_microbatches][dp_degree]`
`TrainingMicrobatch`es;
"""

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
        self, training_samples: list[TrainingSample]
    ) -> list[list[list[TrainingSample]]]:
        """Pack and schedule samples using the following steps.

        The future PP layout is ``[G][M][D]``, where G is the number of outer
        gradient accumulation steps, M is the number of PP microbatches, and D
        is the DP degree. RL currently has M=1 and returns ``[G][D]``.

        1. Use first-fit decreasing (FFD) to pack samples into fixed-capacity
           bins, with each bin becoming one flat trainer input.
        2. Round the number of bins up to a multiple of D. With PP this target
           will be ``G * M * D``; the current M=1 target is ``G * D``.
        3. Fill added slots by recursively splitting the highest-workload
           multi-sample bin using two-way longest-processing-time scheduling.
           Keep an empty bin only when no bin can be split.
        4. Estimate each bin workload as ``sum(L**2)`` over its samples.
        5. Sort all bins by workload in descending order.
        6. Partition the sorted bins into consecutive D-wide groups, so the D
           inputs executed concurrently have similar workloads.
        7. Reverse every other D-wide group. In a ``[G][M][D]`` layout, this
           zig-zag pairs heavier and lighter PP microbatches on each replica.
        """
        num_tokens_per_rank = self._num_rows_per_microbatch * self.seq_len

        # Step 1: pack samples into the minimum number of bins found by FFD.
        ordered_samples = sorted(
            training_samples,
            key=self.num_tokens_to_pack,
            reverse=True,
        )
        bins: list[list[TrainingSample]] = []
        bin_num_tokens: list[int] = []

        for training_sample in ordered_samples:
            num_tokens = self.num_tokens_to_pack(training_sample)
            destination = next(
                (
                    index
                    for index, bin_ in enumerate(bins)
                    if bin_num_tokens[index] + num_tokens <= num_tokens_per_rank
                    and (
                        self._max_num_documents is None
                        or len(bin_) < self._max_num_documents
                    )
                ),
                None,
            )
            if destination is None:
                bins.append([])
                bin_num_tokens.append(0)
                destination = len(bins) - 1

            bins[destination].append(training_sample)
            bin_num_tokens[destination] += num_tokens

        # Step 2: make the current M=1 grid rectangular in D.
        target_num_bins = math.ceil(len(bins) / self._dp_degree) * self._dp_degree

        # Step 3: prefer balanced LPT splits over fully padded bins.
        self._expand_bins_by_splitting(bins, target_num_bins=target_num_bins)

        # Steps 4-5: estimate full-attention work and order bins by that cost.
        bins.sort(key=self._attention_workload, reverse=True)

        num_microbatches = len(bins) // self._dp_degree
        assignments = []
        for microbatch in range(num_microbatches):
            # Step 6: adjacent bins form one concurrently executed DP group.
            rank_assignments = bins[
                microbatch * self._dp_degree : (microbatch + 1) * self._dp_degree
            ]
            # Step 7: zig-zag adjacent groups across DP replicas.
            if microbatch % 2:
                rank_assignments.reverse()
            assignments.append(rank_assignments)
        return assignments

    def _attention_workload(self, training_samples: list[TrainingSample]) -> int:
        """Estimate packed full-attention work as the sum of squared lengths."""
        return sum(self.num_tokens_to_pack(sample) ** 2 for sample in training_samples)

    def _split_by_attention_workload(
        self, training_samples: list[TrainingSample]
    ) -> tuple[list[TrainingSample], list[TrainingSample]]:
        """Split one bin into two with longest-processing-time scheduling."""
        assert len(training_samples) > 1
        splits: tuple[list[TrainingSample], list[TrainingSample]] = ([], [])
        workloads = [0, 0]
        for training_sample in sorted(
            training_samples,
            key=lambda sample: self.num_tokens_to_pack(sample) ** 2,
            reverse=True,
        ):
            destination = 0 if workloads[0] <= workloads[1] else 1
            splits[destination].append(training_sample)
            workloads[destination] += self.num_tokens_to_pack(training_sample) ** 2
        assert splits[0] and splits[1]
        return splits

    def _expand_bins_by_splitting(
        self,
        bins: list[list[TrainingSample]],
        *,
        target_num_bins: int,
    ) -> None:
        """Split the heaviest multi-sample bins until reaching the target count."""
        while len(bins) < target_num_bins:
            candidates = [
                (self._attention_workload(bin_), index)
                for index, bin_ in enumerate(bins)
                if len(bin_) > 1
            ]
            if not candidates:
                break
            _, donor_index = max(candidates)
            bins[donor_index], new_bin = self._split_by_attention_workload(
                bins[donor_index]
            )
            bins.append(new_bin)

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
