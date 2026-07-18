# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain-backed TorchTitan dataloader."""

import dataclasses
import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import grain.python as grain
from grain import experimental as grain_experimental

from torchtitan.components.data.collators import Collator, TrainerBatch
from torchtitan.components.data.dataset import BuildOptions, DataRuntime, DatasetConfig
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer


def _type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalize_config(value: Any) -> Any:
    """Represent a config tree with type identity for resume compatibility."""
    if dataclasses.is_dataclass(value):
        return {
            "type": _type_name(value),
            "fields": {
                config_field.name: _normalize_config(getattr(value, config_field.name))
                for config_field in dataclasses.fields(value)
                if not config_field.name.startswith("_")
            },
        }
    if isinstance(value, dict):
        return {str(key): _normalize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, partial):
        return {
            "partial": _normalize_config(value.func),
            "args": _normalize_config(value.args),
            "keywords": _normalize_config(value.keywords),
        }
    if inspect.isfunction(value):
        if value.__closure__:
            raise TypeError(
                f"capturing callable {value.__qualname__} cannot be checkpointed; "
                "use a configured processor with explicit fields"
            )
        identity = f"{value.__module__}.{value.__qualname__}"
        if value.__name__ == "<lambda>":
            identity = f"{identity}:{value.__code__.co_firstlineno}"
        return {"callable": identity}
    if callable(value):
        raise TypeError(
            f"callable {_type_name(value)} must be a dataclass config or top-level function"
        )
    return {"type": _type_name(value), "repr": repr(value)}


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: DatasetConfig
        collator: Collator.Config
        seed: int = 42
        shuffle: bool = True
        repeat: bool = True
        read_options: grain.ReadOptions = field(default_factory=grain.ReadOptions)
        batch_prefetch_buffer_size: int = 8

        def to_dict(self) -> dict[str, Any]:
            return _normalize_config(self)

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
        self._dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"
        runtime = DataRuntime(
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            read_options=config.read_options,
        )
        options = BuildOptions(
            seed=config.seed,
            shuffle=config.shuffle,
            repeat=config.repeat,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        self._pipeline = {
            "dataset": _normalize_config(config.dataset),
            "collator": _normalize_config(config.collator),
            "seed": config.seed,
            "shuffle": config.shuffle,
            "repeat": config.repeat,
            "seq_len": seq_len,
            "local_batch_size": local_batch_size,
            "tokenizer": _type_name(tokenizer),
        }

        dataset = config.dataset.build(runtime=runtime, options=options)
        collator = config.collator.build(runtime=runtime)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=config.read_options)
        dataset = dataset.batch(
            local_batch_size,
            drop_remainder=True,
            batch_fn=collator,
        )
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset,
            prefetch_buffer_size=config.batch_prefetch_buffer_size,
        )
        self._iterator = iter(dataset)

    def __iter__(self) -> Iterator[TrainerBatch]:
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            "pipeline": self._pipeline,
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
        if state_dict["pipeline"] != self._pipeline:
            raise ValueError("cannot resume after changing the data pipeline")
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        self._iterator.set_state(state_dict[self._rank_id])
