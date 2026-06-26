# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `TrainingSample`s until a group-count training batch is ready, then packs it.
`Batcher` packs a `TrainingBatch` of `[num_microbatches][dp_degree]` `TrainingMicrobatch`es;
"""

import logging
import math
from dataclasses import dataclass, field

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import (
    TrainingBatch,
    TrainingMicrobatch,
    TrainingSample,
    TrainingSampleGroup,
)

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
    NOTE: in pretraining we would have global_batch_size. But now we have
    num_groups_per_train_step. This will need to be addressed.
    """

    local_batch_size: int = 8
    """Per-DP-rank microbatch size (rows per forward pass). If the number of tokens in the
    rollouts exceed the number of rows*seq_len, a new microbatch is started.
    If it is less, the remaining rows are padded to this size."""

    seq_len: int = 2048
    """Tokens per row (packed sequence length)."""


class Batcher(Configurable):
    """Accumulate `num_groups_per_train_step` groups and packs
    `[num_microbatches][dp_degree]` `TrainingMicrobatch`es of `[local_batch_size, seq_len]`.

    Example:
        # num_groups_per_train_step=2, dp_degree=2, local_batch_size=2
        # The trigger is 2 trainable GROUPS, regardless of how many samples/tokens each contains.
        batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=128)).build(
            num_groups_per_train_step=2, dp_degree=2, pad_id=0,
        )
        _ = batcher.add_training_samples(training_sample_group=group0)  # -> None (only 1 trainable group)
        batch = batcher.add_training_samples(training_sample_group=group1)  # -> TrainingBatch
        # batch.microbatches: [num_microbatches][2 ranks]; each TrainingMicrobatch.token_ids: [2 rows, 128 tokens]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch: BatchConfig = field(default_factory=BatchConfig)
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""

    def __init__(
        self,
        config: Config,
        *,
        num_groups_per_train_step: int,
        dp_degree: int,
        pad_id: int,
    ) -> None:
        self.local_batch_size = config.batch.local_batch_size
        self.seq_len = config.batch.seq_len
        self.pad_id = pad_id
        self._per_sample_pad_multiple = config.per_sample_pad_multiple
        self._num_groups_per_train_step = num_groups_per_train_step
        self._dp_degree = dp_degree
        self._groups_for_next_batch: list[TrainingSampleGroup] = []

    def add_training_samples(
        self, *, training_sample_group: TrainingSampleGroup
    ) -> TrainingBatch | None:
        """Add one rollout group and pack one train step once enough trainable groups are ready.

        Args:
            training_sample_group: One rollout group's trainable samples plus rollout metrics.

        Example:
            batcher = Batcher.Config().build(num_groups_per_train_step=2, dp_degree=1, pad_id=0)
            batcher.add_training_samples(training_sample_group=group0)  # -> None
            batcher.add_training_samples(training_sample_group=group1)  # -> TrainingBatch
        """
        self._groups_for_next_batch.append(training_sample_group)
        num_trainable_groups = sum(
            bool(group.training_samples) for group in self._groups_for_next_batch
        )
        if num_trainable_groups < self._num_groups_per_train_step:
            return None  # accumulate until one full batch is ready
        return self._pack_one_training_batch()

    def _pack_one_training_batch(self) -> TrainingBatch:
        """Pack the oldest accumulated groups (up to `num_groups_per_train_step` trainable groups) into one batch."""
        (
            training_samples,
            metrics,
            num_rollout_groups,
            num_metric_only_groups,
        ) = self._take_groups_for_train_step()
        # Next-fit all taken training_samples into rows.
        rows = self._assign_training_samples_to_rows(training_samples)
        packed_rows = [self._pack_training_sample_row(row) for row in rows]
        return TrainingBatch(
            microbatches=self._build_microbatch_grid(packed_rows),
            num_global_valid_tokens=sum(
                int(row["loss_mask"].sum().item()) for row in packed_rows
            ),
            metrics=[
                *metrics,
                *self._packing_metrics(
                    packed_rows,
                    training_samples,
                    num_rollout_groups,
                    num_metric_only_groups,
                ),
            ],
            # Trainer computes policy_age from these at consume time (faithful to what it trains on).
            # min_policy_version is the oldest version this training_sample was sampled under.
            min_policy_versions=[
                training_sample.min_policy_version
                for training_sample in training_samples
            ],
        )

    def _take_groups_for_train_step(
        self,
    ) -> tuple[list[TrainingSample], list[m.Metric], int, int]:
        """Pop accumulated groups oldest-first until `num_groups_per_train_step` are taken."""
        taken_training_samples: list[TrainingSample] = []
        taken_metrics: list[m.Metric] = []
        num_trainable_groups = 0
        cut = 0
        for group in self._groups_for_next_batch:
            if num_trainable_groups >= self._num_groups_per_train_step:
                break
            cut += 1

            taken_metrics.extend(group.metrics)
            if group.training_samples:
                num_trainable_groups += 1
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

    def _assign_training_samples_to_rows(
        self, training_samples: list[TrainingSample]
    ) -> list[list[TrainingSample]]:
        """Next-fit training_samples into rows of <= ``seq_len`` tokens (the caller already capped the count).

        Example:

            # seq_len=10, training_sample effective lengths [5, 5, 5]
            _assign_training_samples_to_rows([e5, e5, e5])  # -> [[e5, e5], [e5]]
        """
        # TODO(async-rl): assignment is greedy next-fit. Swap in smarter algorithms here -- e.g. best-fit,
        #   DP/CP/PP load balancing, or balancing tokens across DP rows on a seq_len**2 budget.
        rows: list[list[TrainingSample]] = []
        current_row: list[TrainingSample] = []
        current_len = 0
        for training_sample in training_samples:
            num_tokens_to_pack = self.num_tokens_to_pack(training_sample)

            # doesn't fit, close the row
            if current_row and current_len + num_tokens_to_pack > self.seq_len:
                rows.append(current_row)
                current_row, current_len = [], 0

            current_row.append(training_sample)
            current_len += num_tokens_to_pack

        if current_row:
            rows.append(current_row)

        return rows

    def num_tokens_to_pack(self, training_sample: TrainingSample) -> int:
        """Tokens this training_sample contributes to a packed row.

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

    def _build_microbatch_grid(
        self, packed_rows: list[dict]
    ) -> list[list[TrainingMicrobatch]]:
        """Build `[num_microbatches][dp_degree]` from however many rows packing produced (variable count).

        Each (microbatch, rank) takes its rows round-robin. Pad-only rows are appended last, so dealing them round-robin
        spreads them across microbatches/ranks instead of all landing on the last one.

        Example:
            # local_batch_size=2, dp_degree=2 -> 4 rows/microbatch; 5 real rows -> pad to 8 -> 2 microbatches.
            # The 3 pad rows land on 3 different (microbatch, rank) pairs; none is all padding.
        """
        rows_per_microbatch = self.local_batch_size * self._dp_degree
        num_microbatches = max(1, math.ceil(len(packed_rows) / rows_per_microbatch))

        # Pad up to a full grid
        while len(packed_rows) < num_microbatches * rows_per_microbatch:
            packed_rows.append(self._pack_training_sample_row([]))

        # [num_rows] -> [num_microbatches][dp_degree], dealing rows round-robin so padding spreads out
        grid: list[list[TrainingMicrobatch]] = []
        for microbatch in range(num_microbatches):
            ranks: list[TrainingMicrobatch] = []
            for rank in range(self._dp_degree):
                start = microbatch * self._dp_degree + rank
                # this (microbatch, rank)'s rows: every (num_microbatches * dp_degree)-th row from `start`
                ranks.append(
                    self.collate(
                        packed_rows[start :: num_microbatches * self._dp_degree]
                    )
                )
            grid.append(ranks)
        return grid

    # TODO(async-rl): make packing pluggable -- a `Packer` protocol on `Batcher.Config` (e.g. `TextPacker`)
    #   so callers swap logic per modality (images, ...).
    def _pack_training_sample_row(self, training_samples: list[TrainingSample]) -> dict:
        """Concatenate one row's samples into a `[1, seq_len]` padded row.
        - Labels and logits are shifted
        -`positions` restart at 0 per sample
        -`seq_lens` keeps per-sample lengths

        Example:

            # two 3-token samples [10, 11, 12] and [20, 21, 22], seq_len=8, pad_id=0
            # each sample drops one token via the raw[:-1]/raw[1:] split (3 -> 2), then the row pads to 8:
            input_ids = [10, 11, 20, 21, 0, 0, 0, 0]
            labels    = [11, 12, 21, 22, 0, 0, 0, 0]
            positions = [ 0,  1,  0,  1, 0, 0, 0, 0]   # restart at 0 per sample, then pad
            seq_lens  = [2, 2]                         # per-sample lengths after the split (4 real tokens, 4 pad)
        """
        pad_values = {**_PAD_VALUES, "input_ids": self.pad_id, "labels": self.pad_id}
        keys = list(pad_values)
        row: dict[str, list] = {key: [] for key in keys}
        positions: list[int] = []
        seq_lens: list[int] = []

        # Shift labals/logits + pad to per_sample_pad_multiple.
        for training_sample in training_samples:
            sample = {
                "input_ids": training_sample.token_ids[:-1],
                "labels": training_sample.token_ids[1:],
                "generator_logprobs": training_sample.logprobs[1:],
                "loss_mask": training_sample.loss_mask[1:],
                "advantages": training_sample.advantage[1:],
            }
            sample_len = len(sample["input_ids"])

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
                row[key].extend(sample[key])
            positions.extend(range(sample_len))
            seq_lens.append(sample_len)

        # Pad the row up to seq_len.
        pad_len = self.seq_len - len(positions)
        if pad_len > 0:
            for key in keys:
                row[key].extend([pad_values[key]] * pad_len)
            positions.extend(range(pad_len))

        # Stack lists into [1, L] tensors.
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
        num_rollout_groups: int,
        num_metric_only_groups: int,
    ) -> list[m.Metric]:
        """Per-training-batch packing + count metrics. (policy age is logged at trainer consume time.)"""
        total_slots = len(packed_rows) * self.seq_len
        non_padded = sum(sum(row["seq_lens"]) for row in packed_rows)
        return [
            m.Metric(
                "train_batch/padding_frac",
                m.NoReduce((total_slots - non_padded) / total_slots),
            ),
            m.Metric(
                "train_batch/num_microbatches",
                m.NoReduce(
                    float(len(packed_rows) // (self.local_batch_size * self._dp_degree))
                ),
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
