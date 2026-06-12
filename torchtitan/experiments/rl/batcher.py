# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packs scored `Episode`s into the trainer's microbatch grid.

A `Batcher` turns a `list[Episode]` into `[grad_accum_steps][dp_degree]` microbatches, each a
`TrainingBatch` of shape `[local_batch_size, seq_len]`. It owns ALL packing decisions — next-fit row
assignment, the loss-target `[:-1]/[1:]` split, and per-sample padding — so callers (the episode
buffer) hand it raw episodes and never re-derive packing.
"""

import logging
from dataclasses import dataclass, field

import torch

from torchtitan.config import Configurable
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


@dataclass(kw_only=True, slots=True)
class BatchConfig:
    """Batch shape parameters for the RL batcher.

    TODO: Refactor the pre-training trainer to use an owned batch config
    instead of keeping batch shape fields directly on TrainingConfig.
    """

    local_batch_size: int = 8
    """Per-DP-rank batch size (rows per forward pass)."""

    global_batch_size: int = -1
    """Target number of rows per optimizer step. Defaults to
    ``local_batch_size * data-parallel degree`` when set to -1."""

    seq_len: int = 2048
    """Tokens per row (packed sequence length)."""


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """One training batch packed from the front of a list of episodes.

    `num_episodes_consumed` is how many leading episodes filled this batch; the caller drops exactly
    that many and keeps the rest for the next batch (the surplus is never truncated).

    Example:

        # global_batch_size=2, seq_len=10, three 5-token episodes
        result = batcher.pack_one_batch([e5, e5, e5], dp_degree=1)
        # -> microbatches [grad_accum][dp], num_episodes_consumed=3 (rows [e5,e5] and [e5])
    """

    microbatches: list[list[TrainingBatch]]  # [grad_accum_steps][dp_degree]
    num_global_valid_tokens: int
    num_episodes_consumed: int
    metrics: list[m.Metric]


class Batcher(Configurable):
    """Packs ``list[Episode]`` into ``[grad_accum_steps][dp_degree]`` microbatches.

    ``gradient_accumulation_steps = global_batch_size // (local_batch_size * dp_degree)``
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch: BatchConfig = field(default_factory=BatchConfig)
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""

    def __init__(self, config: Config, *, pad_id: int):
        self.local_batch_size = config.batch.local_batch_size
        self.global_batch_size = config.batch.global_batch_size
        self.seq_len = config.batch.seq_len
        self.pad_id = pad_id
        self._per_sample_pad_multiple = config.per_sample_pad_multiple

    def num_tokens_target(self, dp_degree: int) -> int:
        """Token slots in one full batch: ``global_batch_size`` rows * ``seq_len``."""
        return self._resolve_global_batch_size(dp_degree) * self.seq_len

    def trainable_tokens(self, episode: Episode) -> int:
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

    def pack_one_batch(self, episodes: list[Episode], *, dp_degree: int) -> PackedBatch:
        """Pack the front of ``episodes`` into ONE training batch; report how many it consumed.

        Greedy next-fit fills rows up to the global batch's row budget, then stops — the surplus is
        left for the next batch (never truncated). Every packing decision lives here: next-fit
        (``_fill_rows``), the ``[:-1]`` loss-target split (``trainable_tokens`` / ``_pack_row``), and
        per-sample padding.

        Args:
            episodes: Buffered episodes, oldest first. The caller must have at least one full batch
                of fresh tokens buffered (the buffer's readiness gate guarantees this).
            dp_degree: Data-parallel degree, for the row budget and the microbatch grid.

        Returns:
            A `PackedBatch`. Pop ``num_episodes_consumed`` from the front and keep the rest.
        """
        global_batch_size = self._resolve_global_batch_size(dp_degree)
        rows = self._fill_rows(episodes, max_rows=global_batch_size)
        # A single episode longer than seq_len can't pack and would silently under-fill the batch;
        # setup_async's seq-len guard promises this can't happen, so assert rather than drop quietly.
        # Checked over the CONSUMED episodes only (O(batch), not O(whole buffer)).
        assert all(
            self.trainable_tokens(episode) <= self.seq_len
            for row in rows
            for episode in row
        ), "episode longer than seq_len; setup_async's seq-len guard should have prevented this"
        assert len(rows) == global_batch_size, (
            f"expected a full batch of {global_batch_size} rows, packed {len(rows)} "
            "(the buffer's readiness gate should guarantee enough fresh tokens)"
        )
        num_episodes_consumed = sum(len(row) for row in rows)
        packed_rows = [self._pack_row(row) for row in rows]

        gradient_accumulation_steps = global_batch_size // (
            self.local_batch_size * dp_degree
        )
        microbatches: list[list[TrainingBatch]] = []
        for step in range(gradient_accumulation_steps):
            step_batches: list[TrainingBatch] = []
            for rank in range(dp_degree):
                start = (step * dp_degree + rank) * self.local_batch_size
                step_batches.append(
                    self.collate(packed_rows[start : start + self.local_batch_size])
                )
            microbatches.append(step_batches)

        num_global_valid_tokens = sum(
            int(row["loss_mask"].sum().item()) for row in packed_rows
        )
        total_slots = len(packed_rows) * self.seq_len
        non_padded = sum(sum(row["seq_lens"]) for row in packed_rows)
        metrics = [
            m.Metric(
                "batcher/pct_pad_in_batch",
                m.NoReduce(
                    (total_slots - non_padded) / total_slots if total_slots else 0.0
                ),
            )
        ]
        return PackedBatch(
            microbatches=microbatches,
            num_global_valid_tokens=num_global_valid_tokens,
            num_episodes_consumed=num_episodes_consumed,
            metrics=metrics,
        )

    def _resolve_global_batch_size(self, dp_degree: int) -> int:
        if self.global_batch_size == -1:
            return self.local_batch_size * dp_degree
        return self.global_batch_size

    def _fill_rows(
        self, episodes: list[Episode], *, max_rows: int
    ) -> list[list[Episode]]:
        """Next-fit episodes into rows of <= ``seq_len`` tokens, up to ``max_rows`` rows.

        The single source of packing geometry: ``pack_one_batch`` materializes exactly these row
        groups, so the row count and the packed tensors can never drift. Surplus beyond ``max_rows``
        is left for the next batch.

        Example:

            # seq_len=10, episode effective lengths [5, 5, 5], max_rows=2
            _fill_rows([e5, e5, e5], max_rows=2)  # -> [[e5, e5], [e5]]
        """
        rows: list[list[Episode]] = []
        current_row: list[Episode] = []
        current_len = 0
        for episode in episodes:
            length = self.trainable_tokens(episode)
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

    def _pack_row(self, episodes: list[Episode]) -> dict:
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

    # TODO: Make collate configurable (passed as an argument to Batcher),
    # similar to how the pre-trainer accepts a collate_fn for its dataloader.
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
