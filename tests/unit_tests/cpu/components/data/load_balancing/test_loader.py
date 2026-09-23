# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_type_hints
from unittest.mock import ANY, MagicMock, patch

import grain.python as grain

import pytest
import torch
import tyro

from torchtitan.components.data.collators import HAS_PIN_MEMORY, TextCollator
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.load_balancing.coordinator import (
    CoordinatedWindow,
    InputCoordinator,
    ReplicatedInputCoordinator,
)
from torchtitan.components.data.load_balancing.loader import LoadBalancingDataLoader
from torchtitan.components.data.load_balancing.planner import (
    BinAssignment,
    PackableItem,
    PackingBin,
)
from torchtitan.components.data.loader import DataloaderExhaustedError, GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import IndexedJsonlSource
from torchtitan.components.data.types import (
    OptimizerStepLayout,
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
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


class _FixedInputCoordinator(InputCoordinator):
    @dataclass(kw_only=True, slots=True)
    class Config(InputCoordinator.Config):
        microbatch: TrainingMicrobatch

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size,
        dp_rank,
        dp_mesh,
        child_loader_factory,
        document_capacity,
    ) -> None:
        del dp_world_size, dp_mesh, child_loader_factory
        self._microbatch = config.microbatch
        self._dp_rank = dp_rank
        self._document_capacity = document_capacity

    def collect(self, *, layout, itemize):
        assert layout.num_microbatches == 1
        stable_id = (self._dp_rank, 0)
        item = itemize(stable_id, self._microbatch)
        return CoordinatedWindow(
            items=(item,),
            bins=(
                PackingBin(
                    stable_id=stable_id,
                    token_capacity=item.num_tokens,
                    document_capacity=self._document_capacity,
                    logical_dp_rank=self._dp_rank,
                    accumulation_index=0,
                    pp_microbatch_index=0,
                ),
            ),
            local_payloads={stable_id: self._microbatch},
            local_bin_ids=(stable_id,),
        )

    def distribute(self, window, assignments):
        item_id = assignments[0].item_ids[0]
        return [window.local_payloads[item_id]]

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        del state_dict

    def close(self):
        pass


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
    optimizer_step_layout: OptimizerStepLayout | None = None,
):
    return config.build(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=_WordTokenizer(),
        max_context_length=8,
        num_tokens_per_microbatch=8,
        optimizer_step_layout=optimizer_step_layout
        or OptimizerStepLayout(num_accumulation_steps=1, num_pp_microbatches=1),
    )


def _markers(microbatches) -> list[int]:
    return [int(microbatch.input[0]) for microbatch in microbatches]


def test_replicated_coordinator_collects_and_distributes_window() -> None:
    streams = {
        0: [_batch((8,), 10), _batch((4, 4), 11)],
        1: [_batch((2, 2, 2, 2), 20), _batch((1,) * 8, 21)],
    }
    built_logical_ranks = []

    def build_child(logical_dp_rank: int):
        built_logical_ranks.append(logical_dp_rank)
        return _FakeChild(streams[logical_dp_rank])

    coordinator = ReplicatedInputCoordinator.Config(group_size=2).build(
        dp_world_size=2,
        dp_rank=0,
        child_loader_factory=build_child,
        document_capacity=8,
    )
    layout = OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)

    def itemize(stable_id, microbatch):
        marker = int(microbatch.input[0])
        cost_by_marker = {10: 64, 11: 32, 20: 16, 21: 8}
        num_documents_by_marker = {10: 1, 11: 2, 20: 4, 21: 8}
        return PackableItem(
            stable_id=stable_id,
            num_tokens=8,
            num_documents=num_documents_by_marker[marker],
            cost=cost_by_marker[marker],
            payload_bytes=0,
            original_bin_id=stable_id,
        )

    window = coordinator.collect(layout=layout, itemize=itemize)
    assignments = (
        BinAssignment(bin_id=(0, 0), item_ids=((0, 0),)),
        BinAssignment(bin_id=(1, 0), item_ids=((0, 1),)),
        BinAssignment(bin_id=(0, 1), item_ids=((1, 1),)),
        BinAssignment(bin_id=(1, 1), item_ids=((1, 0),)),
    )

    output = coordinator.distribute(window, assignments)

    assert built_logical_ranks == [0, 1]
    assert [item.stable_id for item in window.items] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [bin_.stable_id for bin_ in window.bins] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert _markers(output) == [10, 21]


def test_iterates_planned_step_as_a_flat_microbatch_stream() -> None:
    streams = {0: [_batch((1,) * 8, 1), _batch((8,), 2)]}
    layout = OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)

    def build_child(config, **kwargs):
        assert config.num_prefetch_microbatches == 1
        return _FakeChild(streams[kwargs["dp_rank"]])

    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ):
        loader = _build_loader(
            _wrapper_config(),
            dp_world_size=1,
            dp_rank=0,
            optimizer_step_layout=layout,
        )
        iterator = iter(loader)

        assert [int(next(iterator).input[0]) for _ in range(2)] == [2, 1]


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
            _wrapper_config(group_size=2),
            dp_world_size=2,
            dp_rank=0,
            optimizer_step_layout=layout,
        )
        rank_1 = _build_loader(
            _wrapper_config(group_size=2),
            dp_world_size=2,
            dp_rank=1,
            optimizer_step_layout=layout,
        )
        rank_0_iterator = iter(rank_0)
        rank_1_iterator = iter(rank_1)
        rank_0_step = [next(rank_0_iterator) for _ in range(layout.num_microbatches)]
        rank_1_step = [next(rank_1_iterator) for _ in range(layout.num_microbatches)]

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
                        optimizer_step_layout=layout,
                    )
                )

        steps = []
        for loader in loaders:
            iterator = iter(loader)
            steps.append([next(iterator) for _ in range(layout.num_microbatches)])
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
            for microbatch in step
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
            optimizer_step_layout=OptimizerStepLayout(
                num_accumulation_steps=2, num_pp_microbatches=1
            ),
        )
        iterator = iter(loader)
        step = [next(iterator) for _ in range(2)]

    assert _markers(step) == [10, 11]
    metrics = loader.drain_metrics()
    assert metrics["data_load/baseline_predicted_cost"] == 96
    assert metrics["data_load/balanced_predicted_cost"] == 80
    assert "data_load/replicated_read_amplification" not in metrics
    assert loader.drain_metrics() == {}


def test_larger_group_uses_original_dp_world_size_and_logical_ranks() -> None:
    build_calls = []

    def build_child(config, **kwargs):
        build_calls.append(kwargs)
        return _FakeChild([_batch((8,), kwargs["dp_rank"])])

    with patch.object(
        GrainDataLoader.Config, "build", autospec=True, side_effect=build_child
    ):
        layout = OptimizerStepLayout(num_accumulation_steps=1, num_pp_microbatches=1)
        loader = _build_loader(
            _wrapper_config(group_size=4),
            dp_world_size=8,
            dp_rank=5,
            optimizer_step_layout=layout,
        )
        step = [next(iter(loader))]

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
        loader = _build_loader(
            _wrapper_config(),
            dp_world_size=1,
            dp_rank=0,
            optimizer_step_layout=OptimizerStepLayout(
                num_accumulation_steps=2, num_pp_microbatches=1
            ),
        )
        iterator = iter(loader)

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
            optimizer_step_layout=OptimizerStepLayout(
                num_accumulation_steps=1, num_pp_microbatches=1
            ),
        )

    assert build_child.call_count == 2
    loader.close()


def test_loader_passes_dp_mesh_to_input_coordinator() -> None:
    coordinator = MagicMock()
    dp_mesh = object()
    config = _wrapper_config()

    with patch.object(
        ReplicatedInputCoordinator.Config,
        "build",
        return_value=coordinator,
    ) as build_coordinator:
        loader = config.build(
            dp_world_size=2,
            dp_rank=1,
            dp_mesh=dp_mesh,
            tokenizer=_WordTokenizer(),
            max_context_length=8,
            num_tokens_per_microbatch=8,
            optimizer_step_layout=OptimizerStepLayout(
                num_accumulation_steps=1, num_pp_microbatches=1
            ),
        )

    build_coordinator.assert_called_once_with(
        dp_world_size=2,
        dp_rank=1,
        dp_mesh=dp_mesh,
        child_loader_factory=ANY,
        document_capacity=8,
    )
    loader.close()


def test_loader_delegates_input_ownership_and_distribution() -> None:
    config = LoadBalancingDataLoader.Config(
        dataloader=_child_config(),
        coordinator=_FixedInputCoordinator.Config(
            microbatch=_batch((8,), 42),
        ),
    )

    with patch.object(
        GrainDataLoader.Config,
        "build",
        side_effect=AssertionError("the coordinator did not request a child"),
    ):
        loader = _build_loader(config, dp_world_size=1, dp_rank=0)
        output = next(iter(loader))

    assert int(output.input[0]) == 42


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
    import torchtitan.components.data.load_balancing as load_balancing

    assert data.LoadBalancingDataLoader is LoadBalancingDataLoader
    assert load_balancing.CoordinatedWindow is CoordinatedWindow
    assert load_balancing.InputCoordinator is InputCoordinator
