# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain-backed TorchTitan dataloader."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import grain.python as grain
import torch
from grain import experimental as grain_experimental
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.data.collators import Collator, DefaultCollator, TrainerBatch
from torchtitan.components.data.dataset import DatasetConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable


# NOTE: This class deliberately inherits from `Exception` and not `StopIteration`.
# According to PEP 479, raising a `StopIteration` or its subclass from within a
# generator will wrap it in a `RuntimeError`. Since this exception is designed
# to be raised from a generator-based dataloader and caught by the training loop,
# inheriting from `StopIteration` would make it uncatchable and would crash the
# program.
# See: https://peps.python.org/pep-0479/
class DataloaderExhaustedError(Exception):
    """An exception that indicates dataloader exhaustion."""

    pass


class BaseDataLoader(Stateful, ABC, Configurable):
    """Base class for all dataloaders.

    This is used to enforce that all dataloaders have the methods defined in
    ``Stateful``, ``state_dict()`` and ``load_state_dict()``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[TrainerBatch]:
        ...

    def close(self) -> None:
        pass


def _configure_data_worker(worker_index: int, worker_count: int) -> None:
    del worker_index, worker_count
    torch.set_num_threads(1)


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: DatasetConfig
        collator: Collator.Config = field(default_factory=DefaultCollator.Config)
        seed: int = 42
        shuffle: bool = True
        repeat: bool = True
        streaming_shuffle_buffer_size: int = 1_000
        """Raw stream rows held for shuffling; lower it for large media rows."""
        num_workers: int = 0
        """One loader-level map-processing pool per rank; zero disables it."""
        num_prefetch_batches: int = 8
        """Complete batches prepared ahead of trainer consumption."""

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
        self._num_workers = config.num_workers
        self._rank_id = f"dp_rank_{dp_rank}"
        read_options = grain.ReadOptions()
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            read_options=read_options,
        )
        dataset_iteration_policy = DatasetIterationPolicy(
            seed=config.seed,
            shuffle=config.shuffle,
            repeat=config.repeat,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            streaming_shuffle_buffer_size=config.streaming_shuffle_buffer_size,
        )

        dataset = config.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        collator = config.collator.build(context=context)
        if config.num_workers and not isinstance(dataset, grain.MapDataset):
            raise ValueError("multiprocessing prefetch requires a map-root dataset")
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=read_options)

        # Run the map-root preprocessing pipeline in one loader-level process
        # pool. Eight workers means eight total workers per rank, not per child.
        if config.num_workers:
            dataset = dataset.mp_prefetch(
                grain.MultiprocessingOptions(
                    num_workers=config.num_workers,
                ),
                worker_init_fn=_configure_data_worker,
            )
        dataset = dataset.batch(
            local_batch_size,
            drop_remainder=config.repeat,
            batch_fn=collator,
        )

        # One background thread runs batching/collation and queues complete
        # batches while the trainer consumes the previous batch.
        # TODO(data-prefetch-benchmark): Benchmark complete-batch thread prefetch
        # with and without process prefetch before changing either default.
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset, prefetch_buffer_size=config.num_prefetch_batches
        )
        self._iterator = iter(dataset)

    def __iter__(self) -> Iterator[TrainerBatch]:
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            "num_workers": self._num_workers,
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
        if state_dict.get("num_workers", 0) != self._num_workers:
            self.close()
            raise ValueError(
                "cannot resume after changing num_workers from "
                f"{state_dict.get('num_workers', 0)} "
                f"to {self._num_workers}"
            )
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        try:
            self._iterator.set_state(state_dict[self._rank_id])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._iterator.close()
