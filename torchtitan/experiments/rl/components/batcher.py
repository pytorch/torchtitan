# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `TrainingSample`s until a group-count training batch is ready, then packs it.
`Batcher` packs a `TrainingBatch` of `[num_microbatches][dp_degree]` `TrainingMicrobatch`es;
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated

import torch
import tyro

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import (
    TrainingBatch,
    TrainingMicrobatch,
    TrainingSample,
    TrainingSampleGroup,
)

logger = logging.getLogger(__name__)


def quadratic_sequence_cost(sequence_lengths: torch.Tensor) -> torch.Tensor:
    """Return the full-attention cost for each sequence length in a 1D tensor."""
    return sequence_lengths.square()


# Per-field pad values and tensor dtypes for a packed microbatch.
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
    num_prompts_per_train_step. This will need to be addressed.
    """

    local_batch_size: int = 8
    """Number of max-length sequences in each DP rank's token budget.

    The model input is flat with shape ``[local_batch_size * seq_len]``.
    """

    seq_len: int = 2048
    """Maximum length of one packed sequence."""


class Batcher(Configurable):
    """Accumulate `num_prompts_per_train_step` groups and packs
    `[num_microbatches][dp_degree]` flat `TrainingMicrobatch`es.

    Example:
        # num_prompts_per_train_step=2, dp_degree=2, local_batch_size=2
        # The trigger is 2 trainable GROUPS, regardless of how many samples/tokens each contains.
        batcher = Batcher.Config(batch=BatchConfig(local_batch_size=2, seq_len=128)).build(
            num_prompts_per_train_step=2, dp_degree=2, pad_id=0,
        )
        pending, _ = batcher.add_training_samples(training_sample_group=group0)
        batch, _ = batcher.add_training_samples(training_sample_group=group1)
        # pending is None; batch.microbatches: [num_microbatches][2 ranks]; each
        # TrainingMicrobatch.token_ids: [2 * 128 tokens]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch: BatchConfig = field(default_factory=BatchConfig)
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""
        cost_model: Annotated[
            Callable[[torch.Tensor], torch.Tensor], tyro.conf.Suppress
        ] = quadratic_sequence_cost
        """Map a 1D tensor of sequence lengths to one cost per sequence.

        The default models full attention with ``sequence_length**2``. Set this
        programmatically for architectures with a different compute profile.
        """

    def __init__(
        self,
        config: Config,
        *,
        num_prompts_per_train_step: int,
        dp_degree: int,
        pad_id: int,
    ) -> None:
        self.local_batch_size = config.batch.local_batch_size
        self.seq_len = config.batch.seq_len
        self.pad_id = pad_id
        self._per_sample_pad_multiple = config.per_sample_pad_multiple
        self._cost_model = config.cost_model
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._dp_degree = dp_degree
        self._groups_for_next_batch: list[TrainingSampleGroup] = []

    def add_training_samples(
        self, *, training_sample_group: TrainingSampleGroup
    ) -> tuple[TrainingBatch | None, bool]:
        """Add one group and report whether any samples survive batcher filtering.

        Args:
            training_sample_group: One rollout group's trainable samples plus rollout metrics.

        Example:
            batcher = Batcher.Config().build(num_prompts_per_train_step=2, dp_degree=1, pad_id=0)
            batcher.add_training_samples(training_sample_group=group0)  # -> (None, True)
            batcher.add_training_samples(training_sample_group=group1)  # -> (TrainingBatch, True)
        """
        # A single sample cannot exceed the configured context length.
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
        num_trainable_groups = sum(
            bool(group.training_samples) for group in self._groups_for_next_batch
        )
        if num_trainable_groups < self._num_prompts_per_train_step:
            return None, group_is_trainable  # accumulate until one full batch is ready
        return self._pack_one_training_batch(), group_is_trainable

    def _pack_one_training_batch(self) -> TrainingBatch:
        """Pack the oldest accumulated groups (up to `num_prompts_per_train_step` trainable groups) into one batch."""
        (
            training_samples,
            metrics,
            num_rollout_groups,
            num_metric_only_groups,
        ) = self._take_groups()
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
        return TrainingBatch(
            microbatches=microbatches,
            num_global_valid_tokens=num_global_valid_tokens,
            metrics=[
                *metrics,
                # Keep this response-level metric exact without adding a second
                # token-count field to TrainingBatch.
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
        """Pack samples into fixed-token-budget rank inputs, grouped by microbatch.

        Each DP rank receives one flat 1D input with capacity
        ``local_batch_size * seq_len``. Samples are ordered by estimated cost and
        assigned to the lowest-cost rank where they fit. If no rank has enough
        capacity, the current microbatch is emitted and packing continues in a
        fresh one.

        Example:

            # dp_degree=2, tokens_per_rank=10, lengths [6, 5, 5, 4]
            # -> one microbatch with rank assignments [[e6, e4], [e5, e5]]
        """
        num_tokens_per_rank = self.local_batch_size * self.seq_len
        sample_lengths = [
            self.num_tokens_to_pack(training_sample)
            for training_sample in training_samples
        ]
        sample_costs = self._sequence_costs(sample_lengths)
        ordered_samples = sorted(
            zip(training_samples, sample_lengths, sample_costs, strict=True),
            key=lambda item: (item[2], item[1]),
            reverse=True,
        )

        microbatches: list[list[list[TrainingSample]]] = []
        rank_assignments: list[list[TrainingSample]] = [
            [] for _ in range(self._dp_degree)
        ]
        rank_num_tokens = [0] * self._dp_degree
        rank_costs = [0.0] * self._dp_degree

        for training_sample, sequence_length, cost in ordered_samples:
            eligible_ranks = [
                rank
                for rank in range(self._dp_degree)
                if rank_num_tokens[rank] + sequence_length <= num_tokens_per_rank
            ]
            if not eligible_ranks:
                microbatches.append(rank_assignments)
                rank_assignments = [[] for _ in range(self._dp_degree)]
                rank_num_tokens = [0] * self._dp_degree
                rank_costs = [0.0] * self._dp_degree
                eligible_ranks = list(range(self._dp_degree))

            rank = min(
                eligible_ranks,
                key=lambda candidate: (
                    rank_costs[candidate],
                    rank_num_tokens[candidate],
                    candidate,
                ),
            )
            rank_assignments[rank].append(training_sample)
            rank_num_tokens[rank] += sequence_length
            rank_costs[rank] += cost

        microbatches.append(rank_assignments)
        return microbatches

    def _sequence_costs(self, sequence_lengths: list[int]) -> list[float]:
        """Evaluate the configured cost model on a 1D sequence-length tensor."""
        lengths = torch.tensor(sequence_lengths, dtype=torch.long)
        costs = self._cost_model(lengths)
        if costs.shape != lengths.shape:
            raise ValueError(
                "Batcher cost_model must return one cost per sequence length; "
                f"got input shape {tuple(lengths.shape)} and output shape "
                f"{tuple(costs.shape)}."
            )
        if not torch.isfinite(costs).all() or torch.any(costs < 0):
            raise ValueError(
                "Batcher cost_model must return finite, non-negative costs."
            )
        return [float(cost) for cost in costs]

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
        """Concatenate samples into one flat, fixed-token-budget microbatch.

        - Labels and logits are shifted
        -`positions` restart at 0 per sample

        Example:

            # two 3-token samples, num_tokens_per_rank=8, pad_id=0
            # each sample drops one token via the raw[:-1]/raw[1:] split:
            input_ids = [10, 11, 20, 21, 0, 0, 0, 0]
            labels    = [11, 12, 21, 22, 0, 0, 0, 0]
            positions = [ 0,  1,  0,  1, 0, 1, 2, 3]
        """
        pad_values = {**_PAD_VALUES, "input_ids": self.pad_id, "labels": self.pad_id}
        keys = list(pad_values)
        packed_fields: dict[str, list] = {key: [] for key in keys}
        positions: list[int] = []

        # Shift labels/logits + pad to per_sample_pad_multiple.
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

            # Extend the packed token fields.
            for key in keys:
                packed_fields[key].extend(sample[key])
            positions.extend(range(sample_len))

        num_tokens_per_rank = self.local_batch_size * self.seq_len
        pad_len = num_tokens_per_rank - len(positions)
        if pad_len > 0:
            for key in keys:
                packed_fields[key].extend([pad_values[key]] * pad_len)
            positions.extend(range(pad_len))

        return TrainingMicrobatch(
            token_ids=torch.tensor(
                packed_fields["input_ids"], dtype=_DTYPES["input_ids"]
            ),
            labels=torch.tensor(packed_fields["labels"], dtype=_DTYPES["labels"]),
            positions=torch.tensor(positions, dtype=torch.long),
            generator_logprobs=torch.tensor(
                packed_fields["generator_logprobs"],
                dtype=_DTYPES["generator_logprobs"],
            ),
            loss_mask=torch.tensor(
                packed_fields["loss_mask"], dtype=_DTYPES["loss_mask"]
            ),
            advantages=torch.tensor(
                packed_fields["advantages"], dtype=_DTYPES["advantages"]
            ),
        )

    def _cost_imbalance(self, assignments: list[list[list[TrainingSample]]]) -> float:
        """Return mean microbatch max-rank cost divided by mean-rank cost.

        1.0 is perfect DP-rank balance; higher means a straggler rank gates the
        lockstep forward/backward pass.
        """
        ratios: list[float] = []
        for microbatch_assignments in assignments:
            rank_costs = [
                sum(
                    self._sequence_costs(
                        [self.num_tokens_to_pack(sample) for sample in samples]
                    )
                )
                for samples in microbatch_assignments
            ]
            total = sum(rank_costs)
            if total > 0:
                ratios.append(max(rank_costs) * self._dp_degree / total)
        return sum(ratios) / len(ratios) if ratios else 1.0

    def _packing_metrics(
        self,
        assignments: list[list[list[TrainingSample]]],
        training_samples: list[TrainingSample],
        num_rollout_groups: int,
        num_metric_only_groups: int,
    ) -> list[m.Metric]:
        """Per-training-batch packing + count metrics. (policy age is logged at trainer consume time.)"""
        num_tokens_per_rank = self.local_batch_size * self.seq_len
        total_slots = len(assignments) * self._dp_degree * num_tokens_per_rank
        non_padded = sum(
            self.num_tokens_to_pack(sample)
            for microbatch_assignments in assignments
            for samples in microbatch_assignments
            for sample in samples
        )
        return [
            m.Metric(
                "train_batch/padding_frac",
                m.NoReduce((total_slots - non_padded) / total_slots),
            ),
            m.Metric(
                "train_batch/cost_imbalance",
                m.NoReduce(self._cost_imbalance(assignments)),
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
