# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain-backed `BaseDataLoader`.

The dataset config produces trainer-ready batches (`PackedTokenDatasetConfig` maps to
`({"input": [B, L], "positions": [B, L]}, labels)` itself); the loader owns building,
prefetch, and resume. Rank-local Grain state is guarded by data-parallel degree and
the whole-recipe fingerprint.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import grain.python as grain
import torch
from grain import experimental as grain_experimental

from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    DatasetConfig,
    fingerprint_parts,
)
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer


def _jsonable_config_value(value: Any) -> Any:
    """Recursively lower recipe dataclasses and callables for config logging."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable_config_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, dict):
        return {str(key): _jsonable_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}:{qualname}"
    return repr(value)


class GrainDataLoader(BaseDataLoader):
    """Deterministic, resumable dataloader over a grain dataset tree.

    Example (config-registry entry):

        GrainDataLoader.Config(
            dataset_config=PackedTokenDatasetConfig(
                dataset=weighted_interleave(
                    [(math_ds, 2.0), (code_ds, 1.0)]
                ),
            ),
            seed=42,
        )
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset_config: DatasetConfig
        seed: int = 42
        shuffle: bool = True
        infinite: bool = True
        prefetch_buffer_size: int = 8

        def to_dict(self) -> dict[str, Any]:
            # Configurable.Config uses dataclasses.asdict for a nested plain
            # dataclass, which leaves processor functions non-JSON-serializable.
            return {
                field.name: _jsonable_config_value(getattr(self, field.name))
                for field in dataclasses.fields(self)
                if not field.name.startswith("_")
            }

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        # Grain state is exact per batch; checkpoint cadence is owned upstream.
        **_: Any,
    ) -> None:
        self._dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"
        self._pipeline_fingerprint = fingerprint_parts(
            config.dataset_config.fingerprint(),
            f"seed={config.seed}",
            f"shuffle={config.shuffle}",
            f"infinite={config.infinite}",
            f"seq_len={seq_len}",
            f"local_batch_size={local_batch_size}",
        )
        runtime = DataRuntime(
            tokenizer=tokenizer, seq_len=seq_len, local_batch_size=local_batch_size
        )
        options = BuildOptions(
            seed=config.seed,
            shuffle=config.shuffle,
            infinite=config.infinite,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )
        dataset = config.dataset_config.build(runtime=runtime, options=options)
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset()
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset,
            prefetch_buffer_size=config.prefetch_buffer_size,
        )
        self._iterator = iter(dataset)

    def __iter__(self) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            "pipeline_fingerprint": self._pipeline_fingerprint,
            self._rank_id: self._iterator.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Empty state is valid, matching ParallelAwareDataloader (dataloader.py:150-152).
        if not state_dict:
            return
        if state_dict["version"] != 1:
            raise ValueError(
                f"unsupported GrainDataLoader state version {state_dict['version']}"
            )
        if state_dict["dp_world_size"] != self._dp_world_size:
            raise ValueError(
                f"cannot resume: dp_world_size changed "
                f"{state_dict['dp_world_size']} -> {self._dp_world_size}"
            )
        # TODO(data-fingerprint-content): metadata-only today (names + sizes + config
        # identities); content hashing is a follow-up if in-place-edited corpora appear.
        if state_dict["pipeline_fingerprint"] != self._pipeline_fingerprint:
            raise ValueError(
                "cannot resume: data pipeline fingerprint changed; the files, recipe, "
                "loader options, or batch shape differ from the checkpoint"
            )
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        self._iterator.set_state(state_dict[self._rank_id])
