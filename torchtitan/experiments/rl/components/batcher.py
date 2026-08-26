# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `TrainingSample`s until a group-count training batch is ready, then packs it.
`Batcher` packs a `TrainingBatch` of `[num_microbatches][dp_degree]` `TrainingMicrobatch`es;
"""

import logging
from dataclasses import dataclass, replace

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

# Per-field pad values + tensor dtypes for a packed microbatch.
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
        # TrainingMicrobatch.token_ids: [256 tokens]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""

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
        self.max_context_length = max_context_length
        self.num_tokens_per_microbatch_per_dp_rank = (
            num_tokens_per_microbatch_per_dp_rank
        )
        self.pad_id = pad_id
        self._per_sample_pad_multiple = config.per_sample_pad_multiple
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
            batcher = Batcher.Config().build(
                num_tokens_per_microbatch_per_dp_rank=16384,
                max_context_length=2048,
                num_prompts_per_train_step=2,
                dp_degree=1,
                pad_id=0,
            )
            batcher.add_training_samples(training_sample_group=group0)  # -> (None, True)
            batcher.add_training_samples(training_sample_group=group1)  # -> (TrainingBatch, True)
        """
        # Drop samples longer than max_context_length or the per-rank microbatch
        # token budget: a sample cannot be split across microbatches.
        samples = training_sample_group.training_samples
        max_sample_tokens = min(
            self.max_context_length,
            self.num_tokens_per_microbatch_per_dp_rank,
        )
        kept = [s for s in samples if self.num_tokens_to_pack(s) <= max_sample_tokens]
        num_dropped = len(samples) - len(kept)
        if num_dropped:
            logger.warning(
                "Batcher dropped %d/%d sample(s) exceeding its packing limit=%d "
                "(max_context_length=%d, num_tokens_per_microbatch_per_dp_rank=%d).",
                num_dropped,
                len(samples),
                max_sample_tokens,
                self.max_context_length,
                self.num_tokens_per_microbatch_per_dp_rank,
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
        # Next-fit all taken training samples into flat per-rank microbatches.
        microbatch_samples = self._assign_training_samples_to_microbatches(
            training_samples
        )
        packed_microbatches = [
            self._pack_training_samples(samples) for samples in microbatch_samples
        ]
        num_global_valid_tokens = sum(
            int(
                (microbatch.loss_mask & torch.isfinite(microbatch.generator_logprobs))
                .sum()
                .item()
            )
            for microbatch in packed_microbatches
        )
        num_response_tokens = sum(
            int(microbatch.loss_mask.sum().item()) for microbatch in packed_microbatches
        )
        return TrainingBatch(
            microbatches=self._build_microbatch_grid(packed_microbatches),
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
                    packed_microbatches,
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
    ) -> list[list[TrainingSample]]:
        """Next-fit complete samples into per-rank flat microbatches.

        Example:

            # num_tokens_per_microbatch_per_dp_rank=10, sample lengths [5, 5, 5]
            _assign_training_samples_to_microbatches([e5, e5, e5])
            # -> [[e5, e5], [e5]]
        """
        # TODO(async-rl): assignment is greedy next-fit. Swap in smarter algorithms
        # here, e.g. best-fit or DP/CP/PP load balancing.
        microbatches: list[list[TrainingSample]] = []
        current_microbatch: list[TrainingSample] = []
        current_num_tokens = 0
        for training_sample in training_samples:
            num_tokens_to_pack = self.num_tokens_to_pack(training_sample)
            assert (
                num_tokens_to_pack <= self.num_tokens_per_microbatch_per_dp_rank
            ), "Training samples must fit within one per-rank microbatch."

            # The sample does not fit, so close the current microbatch.
            if (
                current_microbatch
                and current_num_tokens + num_tokens_to_pack
                > self.num_tokens_per_microbatch_per_dp_rank
            ):
                microbatches.append(current_microbatch)
                current_microbatch, current_num_tokens = [], 0

            current_microbatch.append(training_sample)
            current_num_tokens += num_tokens_to_pack

        if current_microbatch:
            microbatches.append(current_microbatch)

        # Pad to a complete DP grid, then move whole samples from populated
        # microbatches into empty ranks when possible. This preserves sample
        # boundaries while avoiding all-padding ranks in the final step.
        num_global_microbatches = max(
            1, (len(microbatches) + self._dp_degree - 1) // self._dp_degree
        )
        num_per_rank_microbatches = num_global_microbatches * self._dp_degree
        microbatches.extend(
            [] for _ in range(num_per_rank_microbatches - len(microbatches))
        )
        for empty_microbatch in (batch for batch in microbatches if not batch):
            donor = max(
                (batch for batch in microbatches if len(batch) > 1),
                key=lambda batch: sum(self.num_tokens_to_pack(s) for s in batch),
                default=None,
            )
            if donor is None:
                break
            empty_microbatch.append(donor.pop())

        return microbatches

    def num_tokens_to_pack(self, training_sample: TrainingSample) -> int:
        """Tokens this training sample contributes to a packed microbatch.

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
        self, packed_microbatches: list[TrainingMicrobatch]
    ) -> list[list[TrainingMicrobatch]]:
        """Build `[num_microbatches][dp_degree]` from packed per-rank batches.

        Example:
            # 4 packed per-rank microbatches, dp_degree=2 -> 2 global
            # microbatches, each containing data for two ranks.
        """
        assert packed_microbatches and len(packed_microbatches) % self._dp_degree == 0
        num_microbatches = len(packed_microbatches) // self._dp_degree

        return [
            packed_microbatches[
                microbatch * self._dp_degree : (microbatch + 1) * self._dp_degree
            ]
            for microbatch in range(num_microbatches)
        ]

    # TODO(async-rl): make packing pluggable -- a `Packer` protocol on `Batcher.Config` (e.g. `TextPacker`)
    #   so callers swap logic per modality (images, ...).
    def _pack_training_samples(
        self, training_samples: list[TrainingSample]
    ) -> TrainingMicrobatch:
        """Concatenate samples directly into one flat padded microbatch.

        - Labels and logits are shifted
        - `positions` restart at 0 per sample

        Example:

            # two 3-token samples [10, 11, 12] and [20, 21, 22],
            # token budget=8, pad_id=0
            # Each sample drops one token via raw[:-1]/raw[1:] (3 -> 2),
            # then the flat microbatch pads to 8:
            input_ids = [10, 11, 20, 21, 0, 0, 0, 0]
            labels    = [11, 12, 21, 22, 0, 0, 0, 0]
            positions = [ 0,  1,  0,  1, 0, 0, 0, 0]   # restart at 0 per sample, then pad
        """
        pad_values = {**_PAD_VALUES, "input_ids": self.pad_id, "labels": self.pad_id}
        keys = list(pad_values)
        microbatch: dict[str, list] = {key: [] for key in keys}
        positions: list[int] = []

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

            # pad to multiple
            if self._per_sample_pad_multiple:
                align = self._per_sample_pad_multiple
                padded_len = ((sample_len + align - 1) // align) * align
                for key in keys:
                    sample[key] = sample[key] + [pad_values[key]] * (
                        padded_len - sample_len
                    )
                sample_len = padded_len

            # Extend the flat microbatch.
            for key in keys:
                microbatch[key].extend(sample[key])
            positions.extend(range(sample_len))

        pad_len = self.num_tokens_per_microbatch_per_dp_rank - len(positions)
        if pad_len > 0:
            for key in keys:
                microbatch[key].extend([pad_values[key]] * pad_len)
            positions.extend(range(pad_len))

        tensors = {
            key: torch.tensor(microbatch[key], dtype=_DTYPES[key]) for key in keys
        }
        return TrainingMicrobatch(
            token_ids=tensors["input_ids"],
            labels=tensors["labels"],
            positions=torch.tensor(positions, dtype=torch.long),
            generator_logprobs=tensors["generator_logprobs"],
            loss_mask=tensors["loss_mask"],
            advantages=tensors["advantages"],
        )

    def _packing_metrics(
        self,
        packed_microbatches: list[TrainingMicrobatch],
        training_samples: list[TrainingSample],
        num_rollout_groups: int,
        num_metric_only_groups: int,
    ) -> list[m.Metric]:
        """Per-training-batch packing + count metrics. (policy age is logged at trainer consume time.)"""
        total_slots = (
            len(packed_microbatches) * self.num_tokens_per_microbatch_per_dp_rank
        )
        non_padded = sum(
            self.num_tokens_to_pack(training_sample)
            for training_sample in training_samples
        )
        return [
            m.Metric(
                "train_batch/padding_frac",
                m.NoReduce((total_slots - non_padded) / total_slots),
            ),
            m.Metric(
                "train_batch/num_microbatches",
                m.NoReduce(float(len(packed_microbatches) // self._dp_degree)),
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
