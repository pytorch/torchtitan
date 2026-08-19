# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain-backed TorchTitan dataloader."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any

import grain.python as grain
import tyro
from grain import experimental as grain_experimental
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.data.collators import Collator, TextCollator, TrainerBatch
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
    """Enforces the `Stateful`, `state_dict()`, and `load_state_dict()` contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[TrainerBatch]:
        ...

    def close(self) -> None:
        pass


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: Annotated[DatasetConfig, tyro.conf.Suppress]
        collator: Annotated[Collator.Config, tyro.conf.Suppress] = field(
            default_factory=TextCollator.Config
        )
        seed: int = 42
        shuffle: Annotated[bool, tyro.conf.Suppress] = True
        repeat: Annotated[bool, tyro.conf.Suppress] = True
        streaming_shuffle_buffer_size: Annotated[int, tyro.conf.Suppress] = 1_000
        """Streaming rows retained per rank for approximate shuffling."""
        read_options: Annotated[grain.ReadOptions, tyro.conf.Suppress] = field(
            default_factory=grain.ReadOptions
        )
        """Concurrent indexed reads used when a `MapDataset` becomes an `IterDataset`."""
        num_prefetch_batches: Annotated[int, tyro.conf.Suppress] = 2
        """Collated batches queued per rank for trainer consumption."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
    ) -> None:
        # Validate the run policy.
        # TODO(data-finite-dp): Support finite distributed datasets with a global
        # remainder policy. Simple map datasets can truncate or pad before DP
        # sharding; filtered, mixed, packed, and streaming datasets need coordinated
        # exhaustion so every rank runs the same number of steps.
        if dp_world_size > 1 and not config.repeat:
            raise ValueError(
                "repeat=False with data parallelism can exhaust ranks at different "
                "steps and hang collectives; use repeat=True with a trainer-"
                "controlled step count"
            )
        self._dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"

        # Build the dataset graph and collator.
        read_options = config.read_options
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

        # TODO(data-multiprocessing): CPU-heavy processing should use multiple
        # processes rather than only threads. Grain can divide map-style data among
        # workers, but packing and mixing map data with a stream produce an iterable
        # before the loader sees it. Investigate an earlier boundary where one
        # shared worker pool processes samples, instead of creating a pool per
        # dataset or packing per worker.
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=read_options)

        # Batch and collate samples.
        dataset = dataset.batch(
            local_batch_size,
            drop_remainder=config.repeat,
            batch_fn=collator,
        )

        # Queue completed batches while the trainer consumes the previous batch.
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
        try:
            self._iterator.set_state(state_dict[self._rank_id])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._iterator.close()
