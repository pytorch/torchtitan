# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Episode, TrainingBatch


def pack(
    samples: Iterable[dict[str, list]],
    max_seq_length: int,
    pad_values: dict[str, int | float | bool],
) -> Iterator[dict[str, torch.Tensor]]:
    """Greedy-pack variable-length samples into [1, max_seq_length] sequences."""
    keys = list(pad_values.keys())
    dtypes: dict[str, torch.dtype] | None = None
    buffer: dict[str, list] = {key: [] for key in keys}
    position_buffer: list[int] = []
    seq_lens_buffer: list[int] = []
    buffer_length = 0

    def _flush() -> dict:
        nonlocal buffer, position_buffer, seq_lens_buffer, buffer_length
        assert dtypes is not None
        pad_length = max_seq_length - buffer_length
        if pad_length > 0:
            for key in keys:
                buffer[key].extend([pad_values[key]] * pad_length)
            position_buffer.extend(range(pad_length))

        result: dict = {
            key: torch.tensor(
                buffer[key],
                dtype=dtypes[key],
            ).unsqueeze(0)
            for key in keys
        }
        result["positions"] = torch.tensor(position_buffer, dtype=torch.long).unsqueeze(
            0
        )
        result["seq_lens"] = list(seq_lens_buffer)

        buffer = {key: [] for key in keys}
        position_buffer = []
        seq_lens_buffer = []
        buffer_length = 0
        return result

    for sample in samples:
        sample_length = len(sample[keys[0]])

        if sample_length > max_seq_length:
            logger.warning(
                "Dropping sample with length %d exceeding max_seq_length %d",
                sample_length,
                max_seq_length,
            )
            continue

        if dtypes is None:
            dtypes = {key: torch.tensor(sample[key]).dtype for key in keys}

        if buffer_length > 0 and buffer_length + sample_length > max_seq_length:
            yield _flush()

        for key in keys:
            buffer[key].extend(sample[key])
        position_buffer.extend(range(sample_length))
        seq_lens_buffer.append(sample_length)
        buffer_length += sample_length

    if buffer_length > 0:
        yield _flush()


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


class Batcher(Configurable):
    """Packs ``list[Episode]`` into ``[grad_accum_steps][dp_degree]``
    microbatches, where each microbatch is a ``TrainingBatch`` of shape
    ``[local_batch_size, seq_len]``.

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
        global_batch_size = self.global_batch_size
        if global_batch_size == -1:
            global_batch_size = self.local_batch_size * dp_degree
        return global_batch_size * self.seq_len

    def batch(
        self,
        episodes: list[Episode],
        *,
        dp_degree: int,
    ) -> tuple[list[list[TrainingBatch]], int, list[m.Metric]]:
        """Pack episodes into ``[B, seq_len]`` microbatches.

        Returns:
            microbatches: shape ``[gradient_accumulation_steps][dp_degree]``,
                each entry is a ``TrainingBatch`` with ``local_batch_size`` rows.
            num_global_valid_tokens: total response tokens across the batch
                (excludes padding). Used to normalize the loss so that
                gradient accumulation matches a single large-batch step.
            packing_metrics: list of Metric objects for logging.
        """
        # TODO: Consider consuming the iterator lazily instead of
        # materializing all rows upfront.
        packed_rows = list(self._pack_episodes(episodes))

        global_batch_size = self.global_batch_size
        if global_batch_size == -1:
            global_batch_size = self.local_batch_size * dp_degree
        num_rows_before_truncate = len(packed_rows)
        if len(packed_rows) > global_batch_size:
            logger.warning(
                "Dropping %d packed rows (%d -> %d) to fit global_batch_size",
                len(packed_rows) - global_batch_size,
                len(packed_rows),
                global_batch_size,
            )
            packed_rows = packed_rows[:global_batch_size]

        gradient_accumulation_steps = global_batch_size // (
            self.local_batch_size * dp_degree
        )

        num_global_valid_tokens = sum(
            int(row["loss_mask"].sum().item()) for row in packed_rows
        )

        microbatches: list[list[TrainingBatch]] = []
        for step in range(gradient_accumulation_steps):
            step_batches: list[TrainingBatch] = []
            for rank in range(dp_degree):
                start = (step * dp_degree + rank) * self.local_batch_size
                end = start + self.local_batch_size
                step_batches.append(self.collate(packed_rows[start:end]))
            microbatches.append(step_batches)

        # TODO: Optimize rollout collection to reduce wasted episodes.
        # Currently the controller estimates token counts without padded
        # tokens, which can overshoot because packing adds prompt tokens
        # and padding. Track packing metrics to monitor waste.
        #
        # Fraction of the batch's token slots that are right-pad (not real content). This is the
        # packing waste only — NOT the trained-token fraction, which is far lower because prompt
        # and env tokens are real content but never trained.
        total_token_slots = len(packed_rows) * self.seq_len
        content_tokens = sum(sum(row["seq_lens"]) for row in packed_rows)
        pct_pad_in_batch = (
            (total_token_slots - content_tokens) / total_token_slots
            if total_token_slots > 0
            else 0.0
        )
        packing_metrics = [
            m.Metric("batcher/pct_pad_in_batch", m.NoReduce(pct_pad_in_batch)),
            m.Metric(
                "batcher/num_packed_rows",
                m.NoReduce(float(len(packed_rows))),
            ),
            m.Metric(
                "batcher/num_rows_wasted",
                m.NoReduce(float(max(0, num_rows_before_truncate - len(packed_rows)))),
            ),
        ]

        return microbatches, num_global_valid_tokens, packing_metrics

    def _pack_episodes(self, episodes: list[Episode]) -> Iterator[dict]:
        """Pack all episodes into [1, seq_len] rows.

        Each episode's raw tokens (length N) are split into
        ``input_ids = raw[:-1]`` and ``labels = raw[1:]`` (both length
        N-1), matching the pre-training dataloader convention.

        When ``_per_sample_pad_multiple`` is non-zero, each sample is padded
        to a multiple of that value so that flex attention block
        boundaries align identically regardless of batch composition.
        """
        pad_values = {
            "input_ids": self.pad_id,
            "labels": self.pad_id,
            "generator_logprobs": 0.0,
            "loss_mask": False,
            "advantages": 0.0,
        }

        def _iterate_samples() -> Iterator[dict]:
            for episode in episodes:
                sample = {
                    "input_ids": episode.token_ids[:-1],
                    "labels": episode.token_ids[1:],
                    "generator_logprobs": episode.logprobs[1:],
                    "loss_mask": episode.loss_mask[1:],
                    "advantages": episode.advantage[1:],
                }
                if self._per_sample_pad_multiple:
                    sample_len = len(sample["input_ids"])
                    align = self._per_sample_pad_multiple
                    padded_len = ((sample_len + align - 1) // align) * align
                    pad_count = padded_len - sample_len
                    if pad_count > 0:
                        for key in sample:
                            sample[key].extend([pad_values[key]] * pad_count)
                yield sample

        yield from pack(
            _iterate_samples(),
            max_seq_length=self.seq_len,
            pad_values=pad_values,
        )

    # TODO: Make collate configurable (passed as an argument to Batcher),
    # similar to how the pre-trainer accepts a collate_fn for its dataloader.
    @staticmethod
    def collate(rows: list[dict]) -> TrainingBatch:
        """Concatenate packed rows into a single [B, L] TrainingBatch."""
        return TrainingBatch(
            token_ids=torch.cat([r["input_ids"] for r in rows]),
            labels=torch.cat([r["labels"] for r in rows]),
            positions=torch.cat([r["positions"] for r in rows]),
            generator_logprobs=torch.cat([r["generator_logprobs"] for r in rows]),
            loss_mask=torch.cat([r["loss_mask"] for r in rows]),
            advantages=torch.cat([r["advantages"] for r in rows]),
        )
