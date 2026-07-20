# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain-backed TorchTitan dataloader."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import grain.python as grain
from grain import experimental as grain_experimental

from torchtitan.components.data.collators import Collator, DefaultCollator, TrainerBatch
from torchtitan.components.data.dataset import (
    DatasetBuildContext,
    DatasetConfig,
    DatasetIterationPolicy,
)
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: DatasetConfig
        collator: Collator.Config = field(default_factory=DefaultCollator.Config)
        seed: int = 42
        shuffle: bool = True
        repeat: bool = True
        streaming_shuffle_window_size: int = 1_000
        """Raw stream rows held for shuffling; lower it for large media rows."""
        read_options: grain.ReadOptions = field(default_factory=grain.ReadOptions)
        """Grain map-to-iter reader threads and row buffer."""
        batch_prefetch_buffer_size: int = 8
        """Completed trainer batches prefetched ahead of training."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **_: Any,
    ) -> None:
        if dp_world_size > 1 and not config.repeat:
            raise ValueError(
                "repeat=False with data parallelism can exhaust ranks at different "
                "steps and hang collectives; use repeat=True with a trainer-"
                "controlled step count"
            )
        self._dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            read_options=config.read_options,
        )
        iteration = DatasetIterationPolicy(
            seed=config.seed,
            shuffle=config.shuffle,
            repeat=config.repeat,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            streaming_shuffle_window_size=config.streaming_shuffle_window_size,
        )

        dataset = config.dataset.build(context=context, iteration=iteration)
        collator = config.collator.build(context=context)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=config.read_options)
        dataset = dataset.batch(
            local_batch_size, drop_remainder=True, batch_fn=collator
        )
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset, prefetch_buffer_size=config.batch_prefetch_buffer_size
        )
        self._iterator = iter(dataset)

    def __iter__(self) -> Iterator[TrainerBatch]:
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            self._rank_id: self._iterator.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict["version"] != 1:
            raise ValueError(
                f"unsupported GrainDataLoader state version {state_dict['version']}"
            )
        if state_dict["dp_world_size"] != self._dp_world_size:
            raise ValueError(
                "cannot resume after changing the effective data-parallel degree"
            )
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        self._iterator.set_state(state_dict[self._rank_id])
