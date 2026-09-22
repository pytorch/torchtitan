# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

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


class _StatefulChild:
    def __init__(
        self,
        microbatches: list[TokenizedTrainingMicrobatch],
        *,
        fail_load: bool = False,
        close_error: Exception | None = None,
    ) -> None:
        self.microbatches = microbatches
        self.index = 0
        self.fail_load = fail_load
        self.close_error = close_error
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.microbatches):
            raise StopIteration
        microbatch = self.microbatches[self.index]
        self.index += 1
        return microbatch

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state_dict):
        if self.fail_load:
            raise RuntimeError("child restore failed")
        self.index = state_dict["index"]

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _WordTokenizer:
    eos_id = 2

    def encode(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        tokens = [int(token) for token in text.split()]
        return ([1] if add_bos else []) + tokens + ([self.eos_id] if add_eos else [])


def _batch(marker: int) -> TokenizedTrainingMicrobatch:
    input_ids = torch.full((4,), marker, dtype=torch.int64)
    labels = torch.arange(4, dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)
    padding_mask = torch.zeros(4, dtype=torch.bool)
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
        num_valid_tokens=4,
    )


def _config(
    *, mode: str = "balance", group_size: int = 1
) -> LoadBalancingDataLoader.Config:
    dataset = (
        ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=IndexedJsonlSource.Config(patterns=("unused.jsonl",)),
                processor=TextProcessor.Config(),
            )
        )
        if group_size == 2
        else object()
    )
    return LoadBalancingDataLoader.Config(
        dataloader=GrainDataLoader.Config(
            dataset=dataset,
            collator=TextCollator.Config(),
            shuffle=False,
            repeat=True,
            max_num_documents=4,
        ),
        mode=mode,
        coordinator=ReplicatedInputCoordinator.Config(group_size=group_size),
    )


def _build(
    config: LoadBalancingDataLoader.Config,
    *,
    dp_world_size: int = 1,
    dp_rank: int = 0,
    tokenizer=None,
) -> LoadBalancingDataLoader:
    return config.build(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=_WordTokenizer() if tokenizer is None else tokenizer,
        max_context_length=4,
        num_tokens_per_microbatch=4,
    )


def _actual_config(dataset_path: str) -> LoadBalancingDataLoader.Config:
    return LoadBalancingDataLoader.Config(
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=IndexedJsonlSource.Config(patterns=(dataset_path,)),
                    processor=TextProcessor.Config(),
                )
            ),
            collator=TextCollator.Config(),
            shuffle=False,
            repeat=True,
            max_num_documents=4,
            num_prefetch_microbatches=1,
        ),
        coordinator=ReplicatedInputCoordinator.Config(group_size=2),
    )


def _distributed_checkpoint_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    dataset_path: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    source = None
    restored = None
    try:
        layout = OptimizerStepLayout(
            num_accumulation_steps=2,
            num_pp_microbatches=1,
        )
        source = _build(
            _actual_config(dataset_path),
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_WordTokenizer(),
        )
        source_iterator = source.iter_optimizer_steps(layout)
        next(source_iterator)
        dcp.save({"dataloader": source}, checkpoint_id=checkpoint_dir)
        expected = next(source_iterator)

        restored = _build(
            _actual_config(dataset_path),
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_WordTokenizer(),
        )
        dcp.load({"dataloader": restored}, checkpoint_id=checkpoint_dir)
        actual = next(restored.iter_optimizer_steps(layout))

        expected_inputs = [
            microbatch.input.tolist()
            for group in expected.microbatch_groups
            for microbatch in group
        ]
        actual_inputs = [
            microbatch.input.tolist()
            for group in actual.microbatch_groups
            for microbatch in group
        ]
        assert actual_inputs == expected_inputs
    finally:
        if source is not None:
            source.close()
        if restored is not None:
            restored.close()
        dist.destroy_process_group()


def _torch_checkpointing_round_trip_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    dataset_path: str,
) -> None:
    from torch_checkpointing.checkpoint_manager import (
        CheckpointManager as BackendCheckpointManager,
    )

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    source = None
    restored = None
    save_manager = None
    load_manager = None
    try:
        layout = OptimizerStepLayout(
            num_accumulation_steps=2,
            num_pp_microbatches=1,
        )
        source = _build(
            _actual_config(dataset_path),
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_WordTokenizer(),
        )
        source_iterator = source.iter_optimizer_steps(layout)
        next(source_iterator)

        save_manager = BackendCheckpointManager.Config.with_sync_save().build()
        save_manager.save(checkpoint_dir, {"dataloader": source.state_dict()})
        expected = next(source_iterator)

        restored = _build(
            _actual_config(dataset_path),
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_WordTokenizer(),
        )
        load_manager = BackendCheckpointManager.Config.with_sync_save().build()
        loaded = load_manager.load(
            checkpoint_dir,
            into={"dataloader": restored.state_dict()},
            strict=True,
        )
        restored.load_state_dict(loaded["dataloader"])
        actual = next(restored.iter_optimizer_steps(layout))

        expected_inputs = [
            microbatch.input.tolist()
            for group in expected.microbatch_groups
            for microbatch in group
        ]
        actual_inputs = [
            microbatch.input.tolist()
            for group in actual.microbatch_groups
            for microbatch in group
        ]
        assert actual_inputs == expected_inputs
    finally:
        if save_manager is not None:
            save_manager.close()
        if load_manager is not None:
            load_manager.close()
        if source is not None:
            source.close()
        if restored is not None:
            restored.close()
        dist.destroy_process_group()


def _markers(step) -> list[int]:
    return [
        int(microbatch.input[0])
        for group in step.microbatch_groups
        for microbatch in group
    ]


def test_state_after_returned_step_points_to_next_window() -> None:
    child = _StatefulChild([_batch(1), _batch(2), _batch(3)])
    layout = OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)

    with patch.object(GrainDataLoader.Config, "build", return_value=child):
        loader = _build(_config())
        next(loader.iter_optimizer_steps(layout))
        state = loader.state_dict()

    rank_state = state["physical_dp_rank_0"]
    assert rank_state["children"]["logical_dp_rank_0"] == {"index": 2}


def test_checkpoint_restore_continues_exactly() -> None:
    layout = OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)
    streams = [_batch(index) for index in range(1, 7)]

    with patch.object(
        GrainDataLoader.Config,
        "build",
        side_effect=lambda **kwargs: _StatefulChild(streams),
    ):
        source = _build(_config())
        source_iterator = source.iter_optimizer_steps(layout)
        next(source_iterator)
        state = source.state_dict()
        expected = next(source_iterator)

        restored = _build(_config())
        restored.load_state_dict(state)
        actual = next(restored.iter_optimizer_steps(layout))

    assert _markers(actual) == _markers(expected)
    assert restored.state_dict()["physical_dp_rank_0"]["children"][
        "logical_dp_rank_0"
    ] == {"index": 4}


def test_state_is_logical_rank_qualified() -> None:
    children = []

    def build_child(**kwargs):
        child = _StatefulChild([_batch(kwargs["dp_rank"])])
        children.append(child)
        return child

    with patch.object(GrainDataLoader.Config, "build", side_effect=build_child):
        loader = _build(_config(group_size=2), dp_world_size=4, dp_rank=2)
        state = loader.state_dict()

    assert state["effective_dp_degree"] == 4
    assert "physical_dp_rank_2" in state
    rank_state = state["physical_dp_rank_2"]
    assert rank_state["balance_group_coordinates"] == [2, 3]
    assert rank_state["physical_logical_dp_rank"] == 2
    assert set(rank_state["children"]) == {
        "logical_dp_rank_2",
        "logical_dp_rank_3",
    }


def test_merged_distributed_checkpoint_selects_physical_rank_state() -> None:
    children = [_StatefulChild([]) for _ in range(6)]
    child_iterator = iter(children)
    with patch.object(
        GrainDataLoader.Config,
        "build",
        side_effect=lambda **kwargs: next(child_iterator),
    ):
        rank_0 = _build(_config(group_size=2), dp_world_size=2, dp_rank=0)
        rank_1 = _build(_config(group_size=2), dp_world_size=2, dp_rank=1)
        merged_state = deepcopy(rank_0.state_dict())
        merged_state["physical_dp_rank_1"] = rank_1.state_dict()["physical_dp_rank_1"]

        restored_rank_1 = _build(_config(group_size=2), dp_world_size=2, dp_rank=1)
        restored_rank_1.load_state_dict(merged_state)

    assert restored_rank_1.state_dict()["physical_dp_rank_1"] == (
        merged_state["physical_dp_rank_1"]
    )


def test_partial_exhaustion_matches_default_loader_cursor_behavior() -> None:
    child = _StatefulChild([_batch(1)])

    with patch.object(GrainDataLoader.Config, "build", return_value=child):
        loader = _build(_config())
        iterator = loader.iter_optimizer_steps(
            OptimizerStepLayout(num_accumulation_steps=2, num_pp_microbatches=1)
        )

        with pytest.raises(DataloaderExhaustedError):
            next(iterator)

    assert child.index == 1
    assert loader.state_dict()["physical_dp_rank_0"]["children"][
        "logical_dp_rank_0"
    ] == {"index": 1}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.update(schema_version=999), "schema version"),
        (
            lambda state: state.update(effective_dp_degree=2),
            "effective DP degree",
        ),
        (
            lambda state: state["physical_dp_rank_0"]["children"].pop(
                "logical_dp_rank_0"
            ),
            "logical child set",
        ),
    ],
)
def test_rejects_incompatible_checkpoint(mutation, message: str) -> None:
    child = _StatefulChild([])
    with patch.object(GrainDataLoader.Config, "build", return_value=child):
        source = _build(_config())
        state = source.state_dict()
        target = _build(_config())

    mutation(state)
    with pytest.raises(ValueError, match=message):
        target.load_state_dict(state)


def test_checkpoint_can_resume_with_a_different_mode() -> None:
    children = []

    def build_child(**kwargs):
        child = _StatefulChild([])
        children.append(child)
        return child

    with patch.object(GrainDataLoader.Config, "build", side_effect=build_child):
        source = _build(_config(mode="balance"))
        state = source.state_dict()
        target = _build(_config(mode="shadow"))

    target.load_state_dict(state)


def test_checkpoint_can_resume_with_a_different_step_layout() -> None:
    source_child = _StatefulChild([])
    target_child = _StatefulChild([_batch(1), _batch(2)])
    children = iter((source_child, target_child))
    with patch.object(
        GrainDataLoader.Config,
        "build",
        side_effect=lambda **kwargs: next(children),
    ):
        source = _build(_config())
        state = source.state_dict()
        target = _build(_config())
        target.load_state_dict(state)

    step = next(
        target.iter_optimizer_steps(
            OptimizerStepLayout(
                num_accumulation_steps=2,
                num_pp_microbatches=1,
            )
        )
    )
    assert _markers(step) == [1, 2]


def test_failed_child_restore_closes_every_child() -> None:
    source_children = [_StatefulChild([]), _StatefulChild([])]
    target_children = [_StatefulChild([]), _StatefulChild([], fail_load=True)]
    children = iter(source_children + target_children)

    with patch.object(
        GrainDataLoader.Config,
        "build",
        side_effect=lambda **kwargs: next(children),
    ):
        source = _build(_config(group_size=2), dp_world_size=2)
        state = source.state_dict()
        target = _build(_config(group_size=2), dp_world_size=2)

    with pytest.raises(RuntimeError, match="child restore failed"):
        target.load_state_dict(state)

    assert [child.close_calls for child in target_children] == [1, 1]


def test_close_is_idempotent_and_attempts_every_child() -> None:
    children = [
        _StatefulChild([], close_error=RuntimeError("first close failed")),
        _StatefulChild([]),
    ]
    child_iterator = iter(children)
    with patch.object(
        GrainDataLoader.Config,
        "build",
        side_effect=lambda **kwargs: next(child_iterator),
    ):
        loader = _build(_config(group_size=2), dp_world_size=2)

    with pytest.raises(RuntimeError, match="first close failed"):
        loader.close()
    loader.close()

    assert [child.close_calls for child in children] == [1, 1]


def test_distributed_checkpoint_round_trip_preserves_each_physical_rank(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "rows.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps({"text": " ".join([str(index + 3)] * (index % 3 + 1))}) + "\n"
            for index in range(20)
        )
    )

    mp.spawn(
        _distributed_checkpoint_worker,
        args=(
            2,
            str(tmp_path / "process_group_init"),
            str(tmp_path / "checkpoint"),
            str(dataset_path),
        ),
        nprocs=2,
        join=True,
    )


def test_torch_checkpointing_round_trip_preserves_each_physical_rank(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch_checkpointing.checkpoint_manager")
    dataset_path = tmp_path / "rows.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps({"text": " ".join([str(index + 3)] * (index % 3 + 1))}) + "\n"
            for index in range(20)
        )
    )

    mp.spawn(
        _torch_checkpointing_round_trip_worker,
        args=(
            2,
            str(tmp_path / "torch_checkpointing_process_group_init"),
            str(tmp_path / "torch_checkpointing_checkpoint"),
            str(dataset_path),
        ),
        nprocs=2,
        join=True,
    )
