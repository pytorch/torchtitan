# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

import grain.python as grain

import pytest
import torch
import tyro

from torchtitan.components.data.collators import HAS_PIN_MEMORY, TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.load_balancing.loader import (
    LoadBalancingDataLoader,
    ReplicatedInputCoordinator,
)
from torchtitan.components.data.loader import DataloaderExhaustedError, GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import IndexedJsonlSource
from torchtitan.components.data.types import (
    OptimizerStepLayout,
    TokenizedTrainingMicrobatch,
)
from torchtitan.hf_datasets.text_datasets import TextProcessor


class _FakeChild:
    def __init__(self, microbatches: Iterable[TokenizedTrainingMicrobatch]) -> None:
        self._iterator = iter(microbatches)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self) -> None:
        self.closed = True

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        del state_dict


class _WordTokenizer:
    eos_id = 2

    def encode(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        tokens = [int(token) for token in text.split()]
        return ([1] if add_bos else []) + tokens + ([self.eos_id] if add_eos else [])


def _batch(
    segment_lengths: tuple[int, ...], marker: int
) -> TokenizedTrainingMicrobatch:
    positions = torch.cat([torch.arange(length) for length in segment_lengths])
    num_tokens = int(positions.numel())
    input_ids = torch.full((num_tokens,), marker, dtype=torch.int64)
    labels = torch.arange(num_tokens, dtype=torch.int64)
    padding_mask = torch.zeros(num_tokens, dtype=torch.bool)
    if HAS_PIN_MEMORY:
        input_ids = input_ids.pin_memory()
        labels = labels.pin_memory()
        positions = positions.pin_memory()
        padding_mask = padding_mask.pin_memory()
    return TokenizedTrainingMicrobatch(
        input=input_ids,
        labels=labels,
        positions=positions,
        padding_mask=padding_mask,
        num_valid_tokens=num_tokens,
    )


def _child_config(*, num_prefetch_microbatches: int = 3) -> GrainDataLoader.Config:
    return GrainDataLoader.Config(
        dataset=object(),
        collator=TextCollator.Config(),
        shuffle=False,
        repeat=True,
        max_num_documents=8,
        read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=1),
        num_prefetch_microbatches=num_prefetch_microbatches,
    )


def _wrapper_config(
    *, mode: str = "balance", group_size: int = 1
) -> LoadBalancingDataLoader.Config:
    return LoadBalancingDataLoader.Config(
        dataloader=_child_config(),
        mode=mode,
        coordinator=ReplicatedInputCoordinator.Config(group_size=group_size),
    )


def _build_loader(
    config: LoadBalancingDataLoader.Config,
    *,
    dp_world_size: int,
    dp_rank: int,
):
    return config.build(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=_WordTokenizer(),
        max_context_length=8,
        num_tokens_per_microbatch=8,
    )


def _markers(step) -> list[int]:
    return [
        int(microbatch.input[0])
        for group in step.microbatch_groups
        for microbatch in group
    ]


def test_group_size_one_stably_orders_complete_microbatches() -> None:
    streams = {0: [_batch((1,) * 8, 1), _batch((8,), 2)]}

    def build_child(config, **kwargs):
        assert config.num_prefetch_microbatches == 1
        return _FakeChild(streams[kwargs["dp_rank"]])

    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ):
        loader = _build_loader(_wrapper_config(), dp_world_size=1, dp_rank=0)
        step = next(
            loader.iter_optimizer_steps(
                OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)
            )
        )

    assert _markers(step) == [2, 1]
    assert len(step.microbatch_groups) == 2


def test_replicated_group_assigns_exact_candidate_union_across_ranks() -> None:
    streams = {
        0: [_batch((8,), 10), _batch((4, 4), 11)],
        1: [_batch((2, 2, 2, 2), 20), _batch((1,) * 8, 21)],
    }
    build_calls = []

    def build_child(config, **kwargs):
        build_calls.append((config, kwargs))
        return _FakeChild(streams[kwargs["dp_rank"]])

    layout = OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)
    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ):
        rank_0 = _build_loader(
            _wrapper_config(group_size=2), dp_world_size=2, dp_rank=0
        )
        rank_1 = _build_loader(
            _wrapper_config(group_size=2), dp_world_size=2, dp_rank=1
        )
        rank_0_step = next(rank_0.iter_optimizer_steps(layout))
        rank_1_step = next(rank_1.iter_optimizer_steps(layout))

    assert _markers(rank_0_step) == [10, 21]
    assert _markers(rank_1_step) == [11, 20]
    assert sorted(_markers(rank_0_step) + _markers(rank_1_step)) == [10, 11, 20, 21]
    assert [call[1]["dp_rank"] for call in build_calls] == [0, 1, 0, 1]
    assert all(call[1]["dp_world_size"] == 2 for call in build_calls)
    assert all(call[0].num_prefetch_microbatches == 1 for call in build_calls)


def test_independent_grain_replicas_preserve_the_ordinary_candidate_union(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "rows.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps({"text": " ".join([str(index + 3)] * (index % 3 + 1))}) + "\n"
            for index in range(12)
        )
    )
    child_config = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=IndexedJsonlSource.Config(patterns=(str(dataset_path),)),
                processor=TextProcessor.Config(),
            )
        ),
        collator=TextCollator.Config(),
        shuffle=False,
        repeat=True,
        max_num_documents=4,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
        num_prefetch_microbatches=1,
    )
    layout = OptimizerStepLayout(
        num_accumulation_steps=2,
        num_pp_microbatches=1,
    )
    loaders = []
    try:
        for mode in ("balance", "shadow"):
            for dp_rank in range(2):
                loaders.append(
                    LoadBalancingDataLoader.Config(
                        dataloader=child_config,
                        mode=mode,
                        coordinator=ReplicatedInputCoordinator.Config(group_size=2),
                    ).build(
                        dp_world_size=2,
                        dp_rank=dp_rank,
                        tokenizer=_WordTokenizer(),
                        max_context_length=4,
                        num_tokens_per_microbatch=4,
                    )
                )

        steps = [next(loader.iter_optimizer_steps(layout)) for loader in loaders]
    finally:
        for loader in loaders:
            loader.close()

    def contents(step):
        return [
            (
                tuple(microbatch.input.tolist()),
                tuple(microbatch.labels.tolist()),
                tuple(microbatch.positions.tolist()),
                tuple(microbatch.padding_mask.tolist()),
            )
            for group in step.microbatch_groups
            for microbatch in group
        ]

    balanced_union = contents(steps[0]) + contents(steps[1])
    ordinary_union = contents(steps[2]) + contents(steps[3])
    assert sorted(balanced_union) == sorted(ordinary_union)


def test_shadow_mode_returns_the_ordinary_assignment() -> None:
    streams = {
        0: [_batch((8,), 10), _batch((4, 4), 11)],
        1: [_batch((2, 2, 2, 2), 20), _batch((1,) * 8, 21)],
    }

    def build_child(config, **kwargs):
        return _FakeChild(streams[kwargs["dp_rank"]])

    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ):
        loader = _build_loader(
            _wrapper_config(mode="shadow", group_size=2),
            dp_world_size=2,
            dp_rank=0,
        )
        step = next(
            loader.iter_optimizer_steps(
                OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)
            )
        )

    assert _markers(step) == [10, 11]
    metrics = loader.drain_metrics()
    assert metrics["data_load/baseline_predicted_cost"] == 96
    assert metrics["data_load/balanced_predicted_cost"] == 80
    assert loader.drain_metrics() == {}


def test_larger_group_uses_original_dp_world_size_and_logical_ranks() -> None:
    build_calls = []

    def build_child(config, **kwargs):
        build_calls.append(kwargs)
        return _FakeChild([_batch((8,), kwargs["dp_rank"])])

    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ):
        loader = _build_loader(
            _wrapper_config(group_size=4), dp_world_size=8, dp_rank=5
        )
        step = next(
            loader.iter_optimizer_steps(
                OptimizerStepLayout(
                    num_accumulation_steps=1,
                    num_pp_microbatches=1,
                )
            )
        )

    assert [call["dp_rank"] for call in build_calls] == [4, 5, 6, 7]
    assert [call["dp_world_size"] for call in build_calls] == [8, 8, 8, 8]
    assert _markers(step) == [5]
    loader.close()


def test_partial_child_exhaustion_returns_no_step() -> None:
    streams = {0: [_batch((8,), 1)]}

    def build_child(config, **kwargs):
        return _FakeChild(streams[kwargs["dp_rank"]])

    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ):
        loader = _build_loader(_wrapper_config(), dp_world_size=1, dp_rank=0)
        iterator = loader.iter_optimizer_steps(
            OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)
        )

        with pytest.raises(DataloaderExhaustedError):
            next(iterator)


@pytest.mark.parametrize("mode", ["off", "unknown"])
def test_rejects_unknown_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        _wrapper_config(mode=mode)


@pytest.mark.parametrize(
    ("dp_world_size", "dp_rank", "group_size"),
    [(3, 0, 2), (2, 0, 3), (2, 2, 1), (2, 0, 0), (2, 0, -1)],
)
def test_rejects_invalid_group_topology(
    dp_world_size: int, dp_rank: int, group_size: int
) -> None:
    with pytest.raises(ValueError):
        _build_loader(
            _wrapper_config(group_size=group_size),
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )


def test_larger_replicated_group_inherits_grain_repeat_requirement() -> None:
    config = _wrapper_config(group_size=4)
    config.dataloader.repeat = False

    with pytest.raises(ValueError, match="repeat=False"):
        _build_loader(config, dp_world_size=4, dp_rank=0)


def test_closes_earlier_children_when_later_construction_fails() -> None:
    first_child = _FakeChild([])
    num_calls = 0

    def build_child(config, **kwargs):
        nonlocal num_calls
        num_calls += 1
        if num_calls == 2:
            raise RuntimeError("construction failed")
        return first_child

    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ), pytest.raises(RuntimeError, match="construction failed"):
        _build_loader(_wrapper_config(group_size=2), dp_world_size=2, dp_rank=0)

    assert first_child.closed


def test_replicated_coordinator_accepts_user_supplied_dataset_and_tokenizer() -> None:
    config = LoadBalancingDataLoader.Config(
        dataloader=_child_config(),
        coordinator=ReplicatedInputCoordinator.Config(group_size=2),
    )

    with patch.object(
        GrainDataLoader.Config,
        "build",
        autospec=True,
        return_value=_FakeChild([]),
    ) as build_child:
        loader = config.build(
            dp_world_size=2,
            dp_rank=0,
            tokenizer=object(),
            max_context_length=8,
            num_tokens_per_microbatch=8,
        )

    assert build_child.call_count == 2
    loader.close()


def test_wrapper_derives_document_capacity_only_from_child() -> None:
    child_config = _child_config()
    config = LoadBalancingDataLoader.Config(dataloader=child_config)

    assert config.max_num_documents == child_config.max_num_documents
    with pytest.raises(TypeError, match="max_num_documents"):
        LoadBalancingDataLoader.Config(
            dataloader=child_config,
            max_num_documents=3,
        )


def test_all_wrapper_configuration_is_suppressed_from_tyro() -> None:
    hints = get_type_hints(LoadBalancingDataLoader.Config, include_extras=True)

    for config_field in fields(LoadBalancingDataLoader.Config):
        metadata = getattr(hints[config_field.name], "__metadata__", ())
        assert tyro.conf.Suppress in metadata


def test_load_balancing_dataloader_is_publicly_exported() -> None:
    import torchtitan.components.data as data

    assert data.LoadBalancingDataLoader is LoadBalancingDataLoader
