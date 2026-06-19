# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `Episode`s until a rollout-count training batch is ready, then packs it.

`EpisodeBatcher` accumulates `EpisodeBuilderOutput`s. When enough fresh rollouts have accumulated, it drops
stale groups against the LIVE trainer version and packs a `PackedTrainingBatch` of `[num_microbatches][dp_degree]`
microbatches; surplus episodes carry to the next batch. The tensor packing (next-fit rows, the loss-target
`[:-1]/[1:]` split, per-sample padding, collation) is the same as before; only the step trigger changed
(token-ready fixed grid -> rollout-count variable grid). Tensor packing is meant to run under `asyncio.to_thread`.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.controller.episode_builder import EpisodeBuilderOutput
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Episode, TrainingBatch

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


def oldest_sampled_version(episode: Episode) -> int:
    """The oldest policy version the episode's tokens were sampled at (conservative staleness).

    Reads the min over `version_intervals` so a packed multi-turn episode is as stale as its oldest turn;
    falls back to `policy_version` when intervals are absent.

    Example:
        oldest_sampled_version(Episode(version_intervals=[(0, 5), (40, 6)], ...))  # -> 5
    """
    if not episode.version_intervals:
        return episode.policy_version
    return min(version for _start, version in episode.version_intervals)


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
class PackedTrainingBatch:
    """Packed microbatches for one optimizer step.

    Example:
        # local_batch_size=2, dp_degree=1, three 5-token episodes, seq_len=10
        # -> microbatches [[row[e5,e5]], [row[e5]]], num_global_valid_tokens = trained-token count
    """

    microbatches: list[list[TrainingBatch]]  # [num_microbatches][dp_degree]
    num_global_valid_tokens: int
    metrics: list[m.Metric]


class EpisodeBatcher(Configurable):
    """Accumulate episode groups, drop stale ones at flush, and pack rollout-count training batches.

    Example (target = 2 rollouts):
        batcher.add_episode_group(episode_builder_output=EpisodeBuilderOutput([e_a], []))  # 1 rollout  -> []
        batcher.add_episode_group(episode_builder_output=EpisodeBuilderOutput([e_b], []))  # 2 rollouts -> [batch]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch: BatchConfig = field(default_factory=BatchConfig)
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""

        drop_rollout_group_if_any_stale: bool = False
        """At flush, drop the whole group if ANY episode is stale (keeps GRPO groups intact), instead of dropping
        stale episodes individually. No effect when `max_offpolicy_steps` is None."""

    def __init__(
        self,
        config: Config,
        *,
        target_rollouts_per_training_batch: int,
        max_offpolicy_steps: int | None,
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
        self._episode_groups_for_next_training_batch: list[EpisodeBuilderOutput] = []
        self._num_stale_episodes_dropped_since_pack = 0

    def add_episode_group(
        self, *, episode_builder_output: EpisodeBuilderOutput
    ) -> list[PackedTrainingBatch]:
        """Add one group's episodes; return every training batch that is now ready (0, 1, or more)."""
        self._episode_groups_for_next_training_batch.append(episode_builder_output)
        return self._pack_ready_training_batches()

    def flush_remaining(self) -> PackedTrainingBatch | None:
        """Pack whatever is left at clean close (may be under target)."""
        self._drop_stale_groups(self._trainer_policy_version())
        if not self._num_fresh_rollouts():
            return None
        return self._pack_one_training_batch(self._trainer_policy_version())

    def _pack_ready_training_batches(self) -> list[PackedTrainingBatch]:
        packed_training_batches: list[PackedTrainingBatch] = []
        while True:
            version = self._trainer_policy_version()  # LIVE; no +1 (see TODO)
            self._drop_stale_groups(version)
            if self._num_fresh_rollouts() < self._target:
                return packed_training_batches  # not enough yet; surplus carries over
            # TODO(async-rl): staleness is filtered HERE, but the batch trains a bit later -- the trainer can
            # advance during the off-thread pack (the "pack-window" race) and while the batch waits in the
            # size-1 packed_training_batch_queue -- so the effective bound is ~max_offpolicy_steps + 1. Both are
            # the same post-filter race. Investigate accounting for it (filter against trainer_policy_version +
            # queue_depth, or re-validate after the pack) vs the exact trainer-pull on-policy mode. Left out for
            # now to match the simpler ex-post drop other libraries use.
            packed_training_batches.append(self._pack_one_training_batch(version))

    # --- staleness (the only staleness site; against the live trainer version) ---

    def _drop_stale_groups(self, version: int) -> None:
        """Drop stale episodes (or whole stale groups) from the accumulator, against `version`."""
        if self._max_offpolicy_steps is None:
            return
        survivors: list[EpisodeBuilderOutput] = []
        for group in self._episode_groups_for_next_training_batch:
            if (
                not group.episodes
            ):  # metrics-only (failed / filtered) group: keep so its metrics ride out
                survivors.append(group)
                continue
            fresh = [
                episode
                for episode in group.episodes
                if version - oldest_sampled_version(episode)
                <= self._max_offpolicy_steps
            ]
            num_dropped = len(group.episodes) - len(fresh)
            if self._drop_group_if_any_stale and num_dropped:
                self._num_stale_episodes_dropped_since_pack += len(group.episodes)
                survivors.append(
                    EpisodeBuilderOutput(episodes=[], metrics=group.metrics)
                )
            else:
                self._num_stale_episodes_dropped_since_pack += num_dropped
                survivors.append(
                    EpisodeBuilderOutput(episodes=fresh, metrics=group.metrics)
                )
        self._episode_groups_for_next_training_batch = survivors

    def _num_fresh_rollouts(self) -> int:
        """Distinct rollouts (siblings) with at least one surviving episode (a rollout may branch into >1)."""
        return len(
            {
                (episode.rollout_id.group_id, episode.rollout_id.rollout_id)
                for group in self._episode_groups_for_next_training_batch
                for episode in group.episodes
            }
        )

    # --- batch formation (count-triggered) + packing (the packing helpers are unchanged from the old Batcher) ---

    def _pack_one_training_batch(self, version: int) -> PackedTrainingBatch:
        """Take groups (oldest first) until `target` rollouts are reached, pack them, carry the remainder."""
        taken_episodes: list[Episode] = []
        taken_metrics: list[m.Metric] = []
        taken_rollouts: set[tuple[str, int]] = set()
        remaining = list(self._episode_groups_for_next_training_batch)
        while remaining and len(taken_rollouts) < self._target:
            group = remaining.pop(0)
            taken_episodes.extend(group.episodes)
            taken_metrics.extend(group.metrics)
            taken_rollouts.update(
                (episode.rollout_id.group_id, episode.rollout_id.rollout_id)
                for episode in group.episodes
            )
        self._episode_groups_for_next_training_batch = remaining  # surplus carried over

        # All taken episodes form this batch (next-fit into rows; max_rows is non-binding here).
        rows = self._pack_episodes_into_rows(
            taken_episodes, max_rows=len(taken_episodes)
        )
        packed_rows = [self._pack_episode_row(row) for row in rows]
        return PackedTrainingBatch(
            microbatches=self._build_microbatch_grid(packed_rows),
            num_global_valid_tokens=sum(
                int(row["loss_mask"].sum().item()) for row in packed_rows
            ),
            metrics=[
                *taken_metrics,
                *self._packing_metrics(
                    packed_rows, taken_episodes, taken_rollouts, version
                ),
            ],
        )

    def _build_microbatch_grid(
        self, packed_rows: list[dict]
    ) -> list[list[TrainingBatch]]:
        """Build `[num_microbatches][dp_degree]` from however many rows packing produced (variable count).

        Example:
            # local_batch_size=2, dp_degree=2 -> 4 rows/microbatch; 5 rows -> pad to 8 -> 2 microbatches
        """
        rows_per_microbatch = self.local_batch_size * self._dp_degree
        num_microbatches = max(1, math.ceil(len(packed_rows) / rows_per_microbatch))
        while len(packed_rows) < num_microbatches * rows_per_microbatch:
            packed_rows.append(self._pack_episode_row([]))  # pad-only row
        grid: list[list[TrainingBatch]] = []
        for microbatch in range(num_microbatches):
            ranks: list[TrainingBatch] = []
            for rank in range(self._dp_degree):
                start = (microbatch * self._dp_degree + rank) * self.local_batch_size
                ranks.append(
                    self.collate(packed_rows[start : start + self.local_batch_size])
                )
            grid.append(ranks)
        return grid

    def num_packed_tokens(self, episode: Episode) -> int:
        """Tokens this episode contributes to a packed row.

        The loss-target split drops the last token (``input_ids = raw[:-1]``), and batch-invariant
        mode rounds the length up to ``per_sample_pad_multiple``.

        Example:

            # token_ids of length 6, per_sample_pad_multiple=None  -> 5
            # token_ids of length 6, per_sample_pad_multiple=8      -> 8
        """
        length = len(episode.token_ids) - 1
        if self._per_sample_pad_multiple:
            align = self._per_sample_pad_multiple
            length = ((length + align - 1) // align) * align
        return length

    def _pack_episodes_into_rows(
        self, episodes: list[Episode], *, max_rows: int
    ) -> list[list[Episode]]:
        """Next-fit episodes into rows of <= ``seq_len`` tokens, up to ``max_rows`` rows.

        Surplus beyond ``max_rows`` is left for the next batch.

        Example:

            # seq_len=10, episode effective lengths [5, 5, 5], max_rows=2
            _pack_episodes_into_rows([e5, e5, e5], max_rows=2)  # -> [[e5, e5], [e5]]
        """
        rows: list[list[Episode]] = []
        current_row: list[Episode] = []
        current_len = 0
        for episode in episodes:
            length = self.num_packed_tokens(episode)
            if (
                current_row and current_len + length > self.seq_len
            ):  # doesn't fit -> close the row
                rows.append(current_row)
                if (
                    len(rows) == max_rows
                ):  # budget full -> leave the rest for the next batch
                    return rows
                current_row, current_len = [], 0
            current_row.append(episode)
            current_len += length
        if current_row and len(rows) < max_rows:
            rows.append(current_row)
        return rows

    def _pack_episode_row(self, episodes: list[Episode]) -> dict:
        """Concatenate one row's episodes into a single ``[1, seq_len]`` padded row.

        Each episode's raw tokens (length N) split into ``input_ids = raw[:-1]`` and
        ``labels = raw[1:]`` (length N-1); each sample is padded to ``per_sample_pad_multiple``; then
        the row is padded up to ``seq_len``. ``positions`` restart at 0 per sample; ``seq_lens`` keeps
        the per-sample lengths (for the pad-fraction metric and packed-attention).
        """
        pad_values = {**_PAD_VALUES, "input_ids": self.pad_id, "labels": self.pad_id}
        keys = list(pad_values)
        row: dict[str, list] = {key: [] for key in keys}
        positions: list[int] = []
        seq_lens: list[int] = []
        for episode in episodes:
            sample = {
                "input_ids": episode.token_ids[:-1],
                "labels": episode.token_ids[1:],
                "generator_logprobs": episode.logprobs[1:],
                "loss_mask": episode.loss_mask[1:],
                "advantages": episode.advantage[1:],
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

    # TODO: accept a collate_fn on Batcher.Config (like the pre-trainer's dataloader) and wire a
    # non-pretraining collate only when a caller actually needs one.
    @staticmethod
    def collate(rows: list[dict]) -> TrainingBatch:
        """Concatenate packed rows into a single ``[B, L]`` TrainingBatch."""
        return TrainingBatch(
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
        episodes: list[Episode],
        rollouts: set[tuple[str, int]],
        version: int,
    ) -> list[m.Metric]:
        """Per-training-batch packing, count, and staleness metrics."""
        total_slots = len(packed_rows) * self.seq_len
        non_padded = sum(sum(row["seq_lens"]) for row in packed_rows)
        staleness = [version - oldest_sampled_version(episode) for episode in episodes]
        dropped, self._num_stale_episodes_dropped_since_pack = (
            self._num_stale_episodes_dropped_since_pack,
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
            m.Metric("train_batch/num_episodes", m.NoReduce(float(len(episodes)))),
            m.Metric("train_batch/staleness", m.Mean.from_list(staleness)),
            m.Metric(
                "train_batch/staleness_max",
                m.NoReduce(float(max(staleness, default=0))),
            ),
            m.Metric(
                "train_batch/num_stale_episodes_dropped", m.NoReduce(float(dropped))
            ),
        ]
