# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from torchtitan.rl.evaluator import Evaluator, RolloutEvaluator
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.data import RLDataset, ValidationDataLoader
from torchtitan.rl.rollout.types import RolloutTurnID


class RecordingEvaluator(Evaluator):
    """Records the controller's calls in order; the controller tests use it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | bool]] = []

    def on_step(self, step: int) -> None:
        self.calls.append(("on_step", step))

    async def close(self, *, training_completed: bool) -> None:
        self.calls.append(("close", training_completed))


class _FakeRollouter:
    """Each prompt is its reward; its group runs until the test calls `release_group(reward)`."""

    def __init__(self, rewards: list[int]) -> None:
        self.rewards = rewards
        self.prompts_read: list[int] = []
        self.groups_started: list[int] = []
        self._group_ends = {reward: asyncio.Event() for reward in rewards}

    def iter_validation_samples(self):
        for reward in self.rewards:
            self.prompts_read.append(reward)
            yield reward

    def release_group(self, reward: int) -> None:
        self._group_ends[reward].set()

    async def run_group_rollouts(self, *, sample, group_id, **kwargs) -> RolloutGroup:
        self.groups_started.append(sample)
        await self._group_ends[sample].wait()
        rollout_turn = RolloutTurn(
            rollout_id=RolloutTurnID(group_id=group_id, rollout_id=0, turn_id=0),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.array([1, 2], dtype=np.int32),
            completion_token_ids=np.array([3], dtype=np.int32),
            completion_logprobs=np.array([-0.5], dtype=np.float32),
            min_policy_version=sample % 3,
            max_policy_version=sample % 3 + 1,
        )
        rollout = Rollout(
            group_id=group_id,
            rollout_id=0,
            status=RolloutStatus.COMPLETED,
            turns=[rollout_turn],
            reward=float(sample),
        )
        return RolloutGroup(group_id=group_id, rollouts=[rollout])


class _RecordingMetricsProcessor:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, float]]] = []

    def log(self, step, metrics, *, is_validation) -> None:
        self.logged.append((step, m.MetricsProcessor._aggregate_metrics(metrics)))


def _evaluator(
    rollouter, metrics_processor, *, num_training_steps: int = 10, **config_kwargs
) -> RolloutEvaluator:
    return RolloutEvaluator.Config(**config_kwargs).build(
        num_training_steps=num_training_steps,
        rollouter=rollouter,
        generate_fn=AsyncMock(),
        release_session_fn=AsyncMock(),
        generator_router=SimpleNamespace(
            release_groups=SimpleNamespace(call_one=AsyncMock())
        ),
        sampling=SamplingConfig(),
        rollout_recorder=SimpleNamespace(record=lambda **kwargs: None),
        metrics_processor=metrics_processor,
    )


async def _wait_until(condition) -> None:
    """Let the pass and its `to_thread` reads run until `condition()` holds."""
    for _ in range(1000):
        if condition():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition never held")


def test_every_group_is_counted() -> None:
    async def run_test() -> list[tuple[int, dict[str, float]]]:
        rollouter = _FakeRollouter(list(range(30)))
        metrics_processor = _RecordingMetricsProcessor()
        evaluator = _evaluator(rollouter, metrics_processor, max_prompts_in_flight=8)
        evaluator.on_step(0)
        evaluator.on_step(3)
        # End 8 groups at once, newest first: each folds its metrics while the others wait.
        for num_ended in range(0, 30, 8):
            num_running = min(8, 30 - num_ended)
            await _wait_until(
                lambda: len(rollouter.groups_started) == num_ended + num_running
            )
            for reward in reversed(rollouter.groups_started[num_ended:]):
                rollouter.release_group(reward)
        await _wait_until(lambda: metrics_processor.logged)
        return metrics_processor.logged

    [(step, metrics)] = asyncio.run(run_test())
    assert step == 3  # the last step seen when the pass ended
    assert metrics["validation_reward/_sum"] == sum(range(30))  # 435: all 30 groups
    assert metrics["validation/group_failures/sum"] == 0
    assert metrics["validation/policy_version/min"] == 0
    assert metrics["validation/policy_version/max"] == 3


@pytest.mark.parametrize(
    "config_kwargs, num_training_steps, first_step, pass_length, expected",
    [
        # (due step, step it starts at) for every pass that starts.
        ({"validate_at_start": True}, 4, 0, 0, [(0, 0)]),
        ({"validate_at_start": False}, 4, 0, 0, []),
        ({"validate_at_start": False, "freq": 2}, 6, 0, 0, [(2, 2), (4, 4)]),
        # A resumed run skips the start pass and follows the freq grid.
        ({"validate_at_start": True, "freq": 2}, 8, 3, 0, [(4, 4), (6, 6)]),
        # Due 2 and 4 wait while pass 0 runs; due 6 replaces them, starts at 7. Due 8 waits
        # behind it, and the last step (10) starts nothing.
        ({"validate_at_start": True, "freq": 2}, 10, 0, 7, [(0, 0), (6, 7)]),
        # Pass 0 ends just before the last step: due 2 still never starts.
        ({"validate_at_start": True, "freq": 2}, 4, 0, 4, [(0, 0)]),
    ],
)
def test_which_passes_start(
    config_kwargs, num_training_steps, first_step, pass_length, expected
) -> None:
    """A pass started at step s ends just before on_step(s + pass_length)."""

    async def run_test() -> list[tuple[int, int]]:
        evaluator = _evaluator(
            None,
            _RecordingMetricsProcessor(),
            num_training_steps=num_training_steps,
            **config_kwargs,
        )
        started: list[tuple[int, int]] = []
        pass_ends: dict[int, asyncio.Event] = {}

        async def run_pass(*, due_step: int) -> None:
            started.append((due_step, evaluator._step))
            pass_ends[evaluator._step] = asyncio.Event()
            await pass_ends[evaluator._step].wait()

        evaluator._run_pass = run_pass
        for step in range(first_step, num_training_steps + 1):
            for start_step, pass_end in pass_ends.items():
                if step >= start_step + pass_length:
                    pass_end.set()
            await asyncio.sleep(0)  # an ended pass's task finishes
            evaluator.on_step(step)
            await asyncio.sleep(0)  # a new pass's task starts
        await evaluator.close(training_completed=False)
        return started

    assert asyncio.run(run_test()) == expected


def test_next_prompt_is_read_only_when_a_slot_frees() -> None:
    async def run_test() -> None:
        rollouter = _FakeRollouter(list(range(5)))
        evaluator = _evaluator(
            rollouter, _RecordingMetricsProcessor(), max_prompts_in_flight=2
        )
        evaluator.on_step(0)
        await _wait_until(lambda: rollouter.groups_started == [0, 1])
        await asyncio.sleep(0.05)  # time for a wrong third read to show up
        assert rollouter.prompts_read == [0, 1]

        rollouter.release_group(1)
        await _wait_until(lambda: rollouter.groups_started == [0, 1, 2])
        await asyncio.sleep(0.05)
        assert rollouter.prompts_read == [0, 1, 2]
        await evaluator.close(training_completed=False)

    asyncio.run(run_test())


@pytest.mark.parametrize("training_completed", [False, True])
def test_close_cancels_the_running_pass(training_completed: bool) -> None:
    """A crash (False) only cancels and returns normally, so the crash keeps its own error.
    A completed run (True) cancels the periodic pass, then runs the final pass, logged at N."""

    async def run_test() -> tuple[list[int], list[tuple[int, dict[str, float]]]]:
        rollouter = _FakeRollouter(list(range(3)))
        metrics_processor = _RecordingMetricsProcessor()
        evaluator = _evaluator(
            rollouter,
            metrics_processor,
            num_training_steps=4,
        )
        evaluator.on_step(0)
        await _wait_until(lambda: len(rollouter.groups_started) == 3)
        evaluator.on_step(4)
        # Group ids count down, so the final pass's groups never reuse the cancelled ones'.
        for reward in range(3):
            rollouter.release_group(reward)
        await evaluator.close(training_completed=training_completed)
        return rollouter.groups_started, metrics_processor.logged

    groups_started, logged = asyncio.run(run_test())
    if training_completed:
        assert groups_started == [0, 1, 2, 0, 1, 2]
        assert [step for step, _ in logged] == [4]
        assert logged[0][1]["validation_reward/_sum"] == 3
    else:
        assert groups_started == [0, 1, 2]
        assert logged == []


class _ThreePromptDataset(RLDataset):
    """Yields 0, 1, 2, then ends."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def __iter__(self):
        return iter(range(3))


@pytest.mark.parametrize("raised_by", ["on_step", "close"])
def test_failed_pass_raises_from_the_next_on_step_or_close(raised_by: str) -> None:
    """The dataset ends inside the start pass, with 3 groups running; the error comes out as itself.

    on_step: the next step raises; then a crash's close() returns normally.
    close:   training ends first; a completed run's close() raises (no final pass is asked).
    """

    async def run_test() -> list[tuple[int, dict[str, float]]]:
        rollouter = _FakeRollouter(list(range(3)))
        rollouter.iter_validation_samples = lambda: iter(
            ValidationDataLoader.Config(
                dataset=_ThreePromptDataset.Config(), num_prompts=5
            ).build()
        )
        metrics_processor = _RecordingMetricsProcessor()
        evaluator = _evaluator(
            rollouter, metrics_processor, validate_after_training=False
        )
        evaluator.on_step(0)
        await _wait_until(lambda: evaluator._pass_task.done())
        assert rollouter.groups_started == [0, 1, 2]
        if raised_by == "on_step":
            with pytest.raises(RuntimeError, match="ended after 3 prompts"):
                evaluator.on_step(1)
            await evaluator.close(training_completed=False)
        else:
            with pytest.raises(RuntimeError, match="ended after 3 prompts"):
                await evaluator.close(training_completed=True)
        return metrics_processor.logged

    assert asyncio.run(run_test()) == []
