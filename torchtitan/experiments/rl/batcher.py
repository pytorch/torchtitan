# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `TrainingSample`s until a rollout-group-count training batch is ready, then packs it.

`Batcher` accumulates `TrainingSampleBuilderOutput`s. When enough fresh rollouts groups have accumulated, it drops
stale groups against the LIVE trainer version and packs a `TrainingBatch` of `[num_microbatches][dp_degree]`
`TrainingMicrobatch`es; surplus training samples carry to the next batch.

TODO: move this module to components/
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilderOutput,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import TrainingMicrobatch, TrainingSample

logger = logging.getLogger(__name__)

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


@dataclass(kw_only=True, slots=True)
class BatchConfig:
    """Batch shape parameters for the RL batcher.

    TODO: Refactor the pre-training trainer to use an owned batch config
    instead of keeping batch shape fields directly on TrainingConfig.
    """

    local_batch_size: int = 8
    """Per-DP-rank batch size (rows per forward pass)."""

    seq_len: int = 2048
    """Tokens per row (packed sequence length)."""


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Packed microbatches for one optimizer step.

    Example:
        # local_batch_size=2, dp_degree=1, three 5-token training_samples, seq_len=10
        # -> microbatches [[row[e5,e5]], [row[e5]]], num_global_valid_tokens = trained-token count
    """

    microbatches: list[list[TrainingMicrobatch]]  # [num_microbatches][dp_degree]
    num_global_valid_tokens: int
    metrics: list[m.Metric]
    oldest_sampled_versions: list[
        int
    ]  # one per packed training_sample; trainer computes policy_age at consume time


class Batcher(Configurable):
    """Accumulate training_sample groups, drop stale ones at flush, and pack rollout-count training batches.

    Example (target = 2 rollouts; g1, g2 are TrainingSampleBuilderOutputs of one rollout each):
        batcher.add_training_sample_group(training_sample_builder_output=g1)  # -> None (1 < target)
        batcher.add_training_sample_group(training_sample_builder_output=g2)  # -> TrainingBatch (2 == target)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch: BatchConfig = field(default_factory=BatchConfig)
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""

        drop_rollout_group_if_any_stale: bool = False
        """When a group has any too-off-policy training_sample at flush: drop the WHOLE group (True,
        keeps the GRPO group intact) or only the stale training_samples (False)."""

    def __init__(
        self,
        config: Config,
        *,
        target_rollouts_per_training_batch: int,
        max_offpolicy_steps: int,
        trainer_policy_version: Callable[[], int],
        dp_degree: int,
        pad_id: int,
    ) -> None:
        self.local_batch_size = config.batch.local_batch_size
        self.seq_len = config.batch.seq_len
        self.pad_id = pad_id
        self._per_sample_pad_multiple = config.per_sample_pad_multiple
        self._drop_group_if_any_stale = config.drop_rollout_group_if_any_stale
        self._target = target_rollouts_per_training_batch
        self._max_offpolicy_steps = max_offpolicy_steps
        self._trainer_policy_version = trainer_policy_version  # read live at flush
        self._dp_degree = dp_degree
        self._groups_for_next_batch: list[TrainingSampleBuilderOutput] = []
        self._num_stale_training_samples_dropped_since_pack = 0

    def add_training_sample_group(
        self, *, training_sample_builder_output: TrainingSampleBuilderOutput
    ) -> TrainingBatch | None:
        """Add one group's training_samples to the batcher;

        Return a single training batch iff a full one is now ready, else None.
        """
        self._groups_for_next_batch.append(training_sample_builder_output)
        trainer_version = self._trainer_policy_version()
        self._drop_stale_groups(trainer_version)
        has_full_batch = self._num_fresh_rollouts() >= self._target
        if not has_full_batch:
            return None  # accumulate until one full batch is ready
        return self._pack_one_training_batch(trainer_version)

    def flush_remaining(self) -> TrainingBatch | None:
        """Pack whatever is left at clean close (may be under target)."""
        self._drop_stale_groups(self._trainer_policy_version())
        if not self._num_fresh_rollouts():
            return None
        return self._pack_one_training_batch(self._trainer_policy_version())

    # --- staleness (the only staleness site; against the live trainer version) ---

    # This drop is still needed with the strict-FIFO buffer: failed / zero-std / truncated groups make
    # "groups in the buffer == train steps" inexact, so a training_sample can exceed max_offpolicy_steps
    # even though the buffer never skipped one.
    # TODO(async-rl): make off-policy handling pluggable (drop vs mask vs reweight) behind a manager seam.
    def _drop_stale_groups(self, version: int) -> None:
        """Drop training_samples (or whole groups) too far off-policy from the accumulator, against `version`."""
        survivors: list[TrainingSampleBuilderOutput] = []
        for group in self._groups_for_next_batch:
            if (
                not group.training_samples
            ):  # metrics-only (failed / filtered) group: keep so its metrics ride out
                survivors.append(group)
                continue
            fresh = [
                training_sample
                for training_sample in group.training_samples
                if version
                - training_sample.min_policy_version  # min = the oldest version
                <= self._max_offpolicy_steps
            ]
            num_dropped = len(group.training_samples) - len(fresh)
            if self._drop_group_if_any_stale and num_dropped:
                self._num_stale_training_samples_dropped_since_pack += len(
                    group.training_samples
                )
                survivors.append(
                    TrainingSampleBuilderOutput(
                        training_samples=[], metrics=group.metrics
                    )
                )
            else:
                self._num_stale_training_samples_dropped_since_pack += num_dropped
                survivors.append(
                    TrainingSampleBuilderOutput(
                        training_samples=fresh, metrics=group.metrics
                    )
                )
        self._groups_for_next_batch = survivors

    def _num_fresh_rollouts(self) -> int:
        """Distinct rollouts (siblings) with at least one surviving training_sample (a rollout may branch into >1)."""
        return len(
            {
                (
                    training_sample.rollout_id.group_id,
                    training_sample.rollout_id.rollout_id,
                )
                for group in self._groups_for_next_batch
                for training_sample in group.training_samples
            }
        )

    # --- batch formation (count-triggered) + packing ---

    def _pack_one_training_batch(self, version: int) -> TrainingBatch:
        """Pack the oldest accumulated groups (up to `target` rollouts) into one training batch."""
        training_samples, metrics, rollouts = self._take_groups_up_to_target()
        # Next-fit all taken training_samples into rows.
        rows = self._pack_training_samples_into_rows(training_samples)
        packed_rows = [self._pack_training_sample_row(row) for row in rows]
        return TrainingBatch(
            microbatches=self._build_microbatch_grid(packed_rows),
            num_global_valid_tokens=sum(
                int(row["loss_mask"].sum().item()) for row in packed_rows
            ),
            metrics=[
                *metrics,
                *self._packing_metrics(packed_rows, training_samples, rollouts),
            ],
            # Trainer computes policy_age from these at consume time (faithful to what it trains on).
            # min_policy_version is the oldest version this training_sample was sampled under.
            oldest_sampled_versions=[
                training_sample.min_policy_version
                for training_sample in training_samples
            ],
        )

    def _take_groups_up_to_target(
        self,
    ) -> tuple[list[TrainingSample], list[m.Metric], set[tuple[str, int]]]:
        """Pop accumulated groups oldest-first until `target` rollouts are reached; carry the rest over."""
        taken_training_samples: list[TrainingSample] = []
        taken_metrics: list[m.Metric] = []
        taken_rollouts: set[tuple[str, int]] = set()
        remaining = list(self._groups_for_next_batch)
        while remaining and len(taken_rollouts) < self._target:
            group = remaining.pop(0)
            taken_training_samples.extend(group.training_samples)
            taken_metrics.extend(group.metrics)
            taken_rollouts.update(
                (
                    training_sample.rollout_id.group_id,
                    training_sample.rollout_id.rollout_id,
                )
                for training_sample in group.training_samples
            )
        self._groups_for_next_batch = remaining  # surplus carried over
        return taken_training_samples, taken_metrics, taken_rollouts

    def _build_microbatch_grid(
        self, packed_rows: list[dict]
    ) -> list[list[TrainingMicrobatch]]:
        """Build `[num_microbatches][dp_degree]` from however many rows packing produced (variable count).

        Example:
            # local_batch_size=2, dp_degree=2 -> 4 rows/microbatch; 5 rows -> pad to 8 -> 2 microbatches
        """
        rows_per_microbatch = self.local_batch_size * self._dp_degree
        num_microbatches = max(1, math.ceil(len(packed_rows) / rows_per_microbatch))
        while len(packed_rows) < num_microbatches * rows_per_microbatch:
            packed_rows.append(self._pack_training_sample_row([]))  # pad-only row
        grid: list[list[TrainingMicrobatch]] = []
        for microbatch in range(num_microbatches):
            ranks: list[TrainingMicrobatch] = []
            for rank in range(self._dp_degree):
                start = (microbatch * self._dp_degree + rank) * self.local_batch_size
                ranks.append(
                    self.collate(packed_rows[start : start + self.local_batch_size])
                )
            grid.append(ranks)
        return grid

    def num_packed_tokens(self, training_sample: TrainingSample) -> int:
        """Tokens this training_sample contributes to a packed row.

        The loss-target split drops the last token (``input_ids = raw[:-1]``), and batch-invariant
        mode rounds the length up to ``per_sample_pad_multiple``.

        Example:

            # token_ids of length 6, per_sample_pad_multiple=None  -> 5
            # token_ids of length 6, per_sample_pad_multiple=8      -> 8
        """
        length = len(training_sample.token_ids) - 1
        if self._per_sample_pad_multiple:
            align = self._per_sample_pad_multiple
            length = ((length + align - 1) // align) * align
        return length

    def _pack_training_samples_into_rows(
        self, training_samples: list[TrainingSample]
    ) -> list[list[TrainingSample]]:
        """Next-fit training_samples into rows of <= ``seq_len`` tokens (the caller already capped the count).

        Example:

            # seq_len=10, training_sample effective lengths [5, 5, 5]
            _pack_training_samples_into_rows([e5, e5, e5])  # -> [[e5, e5], [e5]]
        """
        # TODO(async-rl): packing is greedy next-fit; the seam for alternative algorithms (e.g. best-fit,
        # each training_sample scanning earlier rows' remaining slots) lives here.
        rows: list[list[TrainingSample]] = []
        current_row: list[TrainingSample] = []
        current_len = 0
        for training_sample in training_samples:
            length = self.num_packed_tokens(training_sample)
            if (
                current_row and current_len + length > self.seq_len
            ):  # doesn't fit -> close the row
                rows.append(current_row)
                current_row, current_len = [], 0
            current_row.append(training_sample)
            current_len += length
        if current_row:
            rows.append(current_row)
        return rows

    def _pack_training_sample_row(self, training_samples: list[TrainingSample]) -> dict:
        """Concatenate one row's training_samples into a single ``[1, seq_len]`` padded row.

        Each training_sample's raw tokens (length N) split into ``input_ids = raw[:-1]`` and
        ``labels = raw[1:]`` (length N-1); each sample is padded to ``per_sample_pad_multiple``; then
        the row is padded up to ``seq_len``. ``positions`` restart at 0 per sample; ``seq_lens`` keeps
        the per-sample lengths (for the pad-fraction metric and packed-attention).
        """
        pad_values = {**_PAD_VALUES, "input_ids": self.pad_id, "labels": self.pad_id}
        keys = list(pad_values)
        row: dict[str, list] = {key: [] for key in keys}
        positions: list[int] = []
        seq_lens: list[int] = []
        for training_sample in training_samples:
            sample = {
                "input_ids": training_sample.token_ids[:-1],
                "labels": training_sample.token_ids[1:],
                "generator_logprobs": training_sample.logprobs[1:],
                "loss_mask": training_sample.loss_mask[1:],
                "advantages": training_sample.advantage[1:],
            }
            sample_len = len(sample["input_ids"])
            if self._per_sample_pad_multiple:
                align = self._per_sample_pad_multiple
                padded_len = ((sample_len + align - 1) // align) * align
                for key in keys:
                    sample[key] = sample[key] + [pad_values[key]] * (
                        padded_len - sample_len
                    )
                sample_len = padded_len
            for key in keys:
                row[key].extend(sample[key])
            positions.extend(range(sample_len))
            seq_lens.append(sample_len)

        pad_len = self.seq_len - len(positions)
        if pad_len > 0:
            for key in keys:
                row[key].extend([pad_values[key]] * pad_len)
            positions.extend(range(pad_len))

        packed = {
            key: torch.tensor(row[key], dtype=_DTYPES[key]).unsqueeze(0) for key in keys
        }
        packed["positions"] = torch.tensor(positions, dtype=torch.long).unsqueeze(0)
        packed["seq_lens"] = seq_lens
        return packed

    # TODO: accept a collate_fn on Batcher.Config (like the pre-trainer's
    # dataloader) and wire a non-pretraining collate only when a caller actually
    # needs one.
    @staticmethod
    def collate(rows: list[dict]) -> TrainingMicrobatch:
        """Concatenate packed rows into a single ``[B, L]`` TrainingMicrobatch."""
        return TrainingMicrobatch(
            token_ids=torch.cat([row["input_ids"] for row in rows]),
            labels=torch.cat([row["labels"] for row in rows]),
            positions=torch.cat([row["positions"] for row in rows]),
            generator_logprobs=torch.cat([row["generator_logprobs"] for row in rows]),
            loss_mask=torch.cat([row["loss_mask"] for row in rows]),
            advantages=torch.cat([row["advantages"] for row in rows]),
        )

    def _packing_metrics(
        self,
        packed_rows: list[dict],
        training_samples: list[TrainingSample],
        rollouts: set[tuple[str, int]],
    ) -> list[m.Metric]:
        """Per-training-batch packing + count metrics. (policy age is logged at trainer consume time.)"""
        total_slots = len(packed_rows) * self.seq_len
        non_padded = sum(sum(row["seq_lens"]) for row in packed_rows)
        dropped, self._num_stale_training_samples_dropped_since_pack = (
            self._num_stale_training_samples_dropped_since_pack,
            0,
        )
        return [
            m.Metric(
                "train_batch/padding_frac",
                m.NoReduce(
                    (total_slots - non_padded) / total_slots if total_slots else 0.0
                ),
            ),
            m.Metric(
                "train_batch/num_microbatches",
                m.NoReduce(
                    float(len(packed_rows) // (self.local_batch_size * self._dp_degree))
                ),
            ),
            m.Metric("train_batch/num_rollouts", m.NoReduce(float(len(rollouts)))),
            m.Metric(
                "train_batch/num_training_samples",
                m.NoReduce(float(len(training_samples))),
            ),
            m.Metric(
                "train_batch/num_stale_training_samples_dropped",
                m.NoReduce(float(dropped)),
            ),
        ]
