# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

from torchtitan.components.dataloading.utils import pack
from torchtitan.config import BatchConfig, Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Episode, TrainingBatch


class Batcher(Configurable):
    """Packs episodes into ``[B, seq_len]`` batches for the trainer.

    The controller collects rollouts until the total response tokens reach
    ``num_tokens_target`` (= ``global_batch_size * seq_len``), then
    packs all collected episodes into fixed-length rows, truncates to
    ``global_batch_size``, and splits into
    ``[grad_accum_steps][dp_degree]`` microbatches.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch: BatchConfig = field(default_factory=BatchConfig)

    def __init__(self, config: Config, *, pad_id: int):
        self.local_batch_size = config.batch.local_batch_size
        self.global_batch_size = config.batch.global_batch_size
        self.seq_len = config.batch.seq_len
        self.pad_id = pad_id

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
        total_token_slots = len(packed_rows) * self.seq_len
        packing_metrics = [
            m.Metric(
                "batcher/packing_efficiency",
                m.NoReduce(
                    num_global_valid_tokens / total_token_slots
                    if total_token_slots > 0
                    else 0.0
                ),
            ),
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
        """Pack all episodes into [1, seq_len] rows."""

        def _iterate_samples() -> Iterator[dict]:
            for ep in episodes:
                prompt_len = len(ep.prompt_token_ids)
                response_len = len(ep.token_ids)
                yield {
                    "input_ids": ep.prompt_token_ids + ep.token_ids,
                    "generator_logprobs": [0.0] * prompt_len + ep.token_logprobs,
                    "loss_mask": [0.0] * prompt_len + [1.0] * response_len,
                    "advantages": [0.0] * prompt_len + [ep.advantage] * response_len,
                }

        yield from pack(
            _iterate_samples(),
            max_seq_length=self.seq_len,
            pad_values={
                "input_ids": self.pad_id,
                "generator_logprobs": 0.0,
                "loss_mask": 0.0,
                "advantages": 0.0,
            },
        )

    # TODO: Make collate configurable (passed as an argument to Batcher),
    # similar to how the pre-trainer accepts a collate_fn for its dataloader.
    @staticmethod
    def collate(rows: list[dict]) -> TrainingBatch:
        """Concatenate packed rows into a single [B, L] TrainingBatch."""
        return TrainingBatch(
            token_ids=torch.cat([r["input_ids"] for r in rows]),
            positions=torch.cat([r["positions"] for r in rows]),
            generator_logprobs=torch.cat([r["generator_logprobs"] for r in rows]),
            loss_mask=torch.cat([r["loss_mask"] for r in rows]),
            advantages=torch.cat([r["advantages"] for r in rows]),
        )
