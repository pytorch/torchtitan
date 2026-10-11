# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for skipping tasks whose last group had zero reward spread."""

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.rl.rollout.data import (
    IterableRLDataLoader,
    RLDataset,
    ZeroSpreadSkip,
)
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.types import RolloutStatus


class _TaskCycle(RLDataset):
    """Cycles over tasks "t0", "t1", ... in order; each sample is its own key."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        num_tasks: int = 4

    def __init__(self, config: Config) -> None:
        self.num_tasks = config.num_tasks
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self) -> str:
        task = f"t{self.position % self.num_tasks}"
        self.position += 1
        return task

    def sample_key(self, sample: str) -> str:
        return sample

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict) -> None:
        self.position = state_dict["position"]


class _Keyless(_TaskCycle):
    @dataclass(kw_only=True, slots=True)
    class Config(_TaskCycle.Config):
        pass

    sample_key = RLDataset.sample_key


def _loader(num_tasks: int = 4, **skip_kwargs) -> IterableRLDataLoader:
    return IterableRLDataLoader.Config(
        dataset=_TaskCycle.Config(num_tasks=num_tasks),
        zero_spread_skip=ZeroSpreadSkip(**skip_kwargs),
    ).build()


def _draw(loader: IterableRLDataLoader, count: int) -> list[tuple[int, str]]:
    return [next(loader) for _ in range(count)]


def _finish(
    loader: IterableRLDataLoader,
    drawn: list[tuple[int, str]],
    outcomes: dict[str, float | None],
) -> None:
    """Record each drawn task's outcome (tasks not in `outcomes` have none), then acknowledge all."""
    loader.record_group_outcomes(
        {index: outcomes[task] for index, task in drawn if task in outcomes}
    )
    loader.acknowledge(index for index, _ in drawn)


def test_skip_off_iterates_and_saves_like_before() -> None:
    loader = IterableRLDataLoader.Config(dataset=_TaskCycle.Config()).build()
    drawn = _draw(loader, 6)
    loader.record_group_outcomes({0: 1.0, 1: 0.0})
    assert drawn == [(0, "t0"), (1, "t1"), (2, "t2"), (3, "t3"), (4, "t0"), (5, "t1")]
    assert "zero_spread_skip" not in loader.state_dict()
    assert loader.metrics() == []


def test_zero_spread_tasks_are_skipped_then_rechecked() -> None:
    loader = _loader(recheck_after_skips=2, min_active_frac=0.1)
    # t0 all pass, t1 all fail, t2 mixed, t3 no usable outcome (e.g. a rollout errored).
    _finish(loader, _draw(loader, 4), {"t0": 1.0, "t1": 0.0, "t2": None})

    # Epochs 2 and 3 skip t0 and t1; epoch 4 re-checks them.
    assert [task for _, task in _draw(loader, 4)] == ["t2", "t3", "t2", "t3"]
    assert [task for _, task in _draw(loader, 4)] == ["t0", "t1", "t2", "t3"]
    metrics = {metric.key: metric.value.value for metric in loader.metrics()}
    assert metrics["data/zero_spread_skip/skips"] == 4
    assert metrics["data/zero_spread_skip/rechecks"] == 2
    assert metrics["data/zero_spread_skip/marked_all_pass"] == 1
    assert metrics["data/zero_spread_skip/marked_all_fail"] == 1


def test_a_mixed_group_brings_a_task_back() -> None:
    loader = _loader(recheck_after_skips=1, min_active_frac=0.1)
    _finish(loader, _draw(loader, 4), {"t0": 1.0})
    # t0 is skipped once, then re-checked and comes back mixed: it stays in the rotation.
    rechecked = _draw(loader, 4)
    assert [task for _, task in rechecked] == ["t1", "t2", "t3", "t0"]
    _finish(loader, rechecked, {"t0": None})
    assert [task for _, task in _draw(loader, 4)] == ["t1", "t2", "t3", "t0"]


def test_floor_brings_back_the_oldest_skipped_task_first() -> None:
    # 4 tasks seen, min_active_frac 0.5 -> at most 2 skipped.
    loader = _loader(recheck_after_skips=10, min_active_frac=0.5)
    drawn = _draw(loader, 4)
    for index, task in drawn[:3]:
        loader.record_group_outcomes({index: 1.0})
    loader.acknowledge(index for index, _ in drawn)
    assert [task for _, task in _draw(loader, 4)] == ["t0", "t3", "t0", "t3"]
    metrics = {metric.key: metric.value.value for metric in loader.metrics()}
    assert metrics["data/zero_spread_skip/floor_admits"] == 1
    assert metrics["data/zero_spread_skip/skipped_tasks"] == 2


def test_resume_keeps_skips_and_replays_pending_samples_unfiltered() -> None:
    loader = _loader(recheck_after_skips=3, min_active_frac=0.1)
    _finish(loader, _draw(loader, 4), {"t0": 1.0})
    pending = _draw(loader, 2)  # yielded, not acknowledged: replayed after a resume
    state = loader.state_dict()
    expected = [task for _, task in pending] + [task for _, task in _draw(loader, 6)]

    resumed = _loader(recheck_after_skips=3, min_active_frac=0.1)
    resumed.load_state_dict(state)
    assert [task for _, task in _draw(resumed, 8)] == expected


def test_resume_from_a_state_saved_with_the_skip_off_starts_empty() -> None:
    state = IterableRLDataLoader.Config(dataset=_TaskCycle.Config()).build().state_dict()
    loader = _loader()
    loader.load_state_dict(state)
    assert [task for _, task in _draw(loader, 4)] == ["t0", "t1", "t2", "t3"]


def test_skip_needs_a_dataset_with_sample_key() -> None:
    with pytest.raises(ValueError, match="does not implement sample_key"):
        IterableRLDataLoader.Config(
            dataset=_Keyless.Config(), zero_spread_skip=ZeroSpreadSkip()
        ).build()


def _rollout(reward, status=RolloutStatus.COMPLETED, completion=(1,)):
    turn = SimpleNamespace(completion_token_ids=list(completion))
    return SimpleNamespace(status=status, reward=reward, turns=[turn])


def _group(group_id, *rollouts):
    return SimpleNamespace(group_id=group_id, rollouts=list(rollouts))


def test_rollouter_records_outcomes_only_for_fully_completed_groups() -> None:
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._group_outcomes = {}
    loader = _loader(min_active_frac=0.1)
    rollouter._training_dataloader = loader
    rollouter._training_dataloader_lock = threading.Lock()
    drawn = _draw(loader, 4)

    rollouter.record_group_outcome(_group(0, _rollout(1.0), _rollout(1.0)))
    rollouter.record_group_outcome(_group(1, _rollout(0.0), _rollout(1.0)))
    # All-zero, but one rollout errored or truncated: those rewards say nothing about the task.
    rollouter.record_group_outcome(
        _group(2, _rollout(0.0), _rollout(0.0, status=RolloutStatus.ERROR_TIMEOUT))
    )
    rollouter.record_group_outcome(
        _group(3, _rollout(0.0), _rollout(0.0, status=RolloutStatus.TRUNCATED_LENGTH))
    )
    rollouter.record_group_outcome(_group(4))  # a failed group has no rollouts
    rollouter.record_group_outcome(_group(5, _rollout(0.0), _rollout(0.0, completion=())))
    assert rollouter._group_outcomes == {0: 1.0, 1: None}

    rollouter.acknowledge_training_sample_ids(index for index, _ in drawn)
    assert rollouter._group_outcomes == {}
    assert [task for _, task in _draw(loader, 3)] == ["t1", "t2", "t3"]
