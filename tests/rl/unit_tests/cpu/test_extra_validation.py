# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extra validation sets: config checks, group ids, metric names, vLLM priority, scheduling."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ExtraValidationConfig,
    ValidationConfig,
)
from torchtitan.rl.generator.types import (
    EXTRA_VALIDATION_GROUP_ID_STRIDE,
    OVERLAPPED_VALIDATION_FIRST_GROUP_ID,
    SamplingConfig,
    vllm_priority,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Rollout, RolloutGroup, RolloutStatus


def _group(group_id: int, rewards: list[float]) -> RolloutGroup:
    return RolloutGroup(
        group_id=group_id,
        rollouts=[
            Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                reward=reward,
            )
            for rollout_id, reward in enumerate(rewards)
        ],
    )


def test_extra_validation_config_checks() -> None:
    ExtraValidationConfig(name="deepswe", freq=20, num_samples_per_prompt=3)
    with pytest.raises(ValueError, match="freq"):
        ExtraValidationConfig(name="deepswe", freq=0)
    with pytest.raises(ValueError, match="name"):
        ExtraValidationConfig(name="a/b", freq=1)
    with pytest.raises(ValueError, match="greedy"):
        ExtraValidationConfig(
            name="deepswe", freq=1, temperature=0.0, num_samples_per_prompt=2
        )
    with pytest.raises(ValueError, match="unique"):
        AsyncLoopConfig(
            extra_validations=[
                ExtraValidationConfig(name="x", freq=1),
                ExtraValidationConfig(name="x", freq=2),
            ]
        )


def test_vllm_priority_puts_extra_validation_after_training() -> None:
    assert vllm_priority(7) == 7
    assert vllm_priority(-3) == -1
    background = vllm_priority(
        OVERLAPPED_VALIDATION_FIRST_GROUP_ID - EXTRA_VALIDATION_GROUP_ID_STRIDE - 5
    )
    assert background > vllm_priority(10**9)


def _controller(extra: ExtraValidationConfig, rewards: list[list[float]]):
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            validation=ValidationConfig(steps=2), extra_validations=[extra]
        )
    )
    controller._sampling = SamplingConfig(temperature=0.8, stop_token_ids=[1])
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
    # Set 0's ids: below the overlapped main-set range.
    base = OVERLAPPED_VALIDATION_FIRST_GROUP_ID - EXTRA_VALIDATION_GROUP_ID_STRIDE
    controller._rollouter = SimpleNamespace(
        get_validation_samples=Mock(return_value=["a", "b"]),
        run_group_rollouts=AsyncMock(
            side_effect=[
                _group(base - i, group_rewards)
                for i, group_rewards in enumerate(rewards)
            ]
        ),
    )
    return controller


def test_extra_validation_pass_uses_its_own_ids_samples_and_metric_names() -> None:
    """2 tasks x 3 samples with 0/1 rewards [[1, 0, 0], [0, 0, 0]]."""
    extra = ExtraValidationConfig(
        name="deepswe", freq=20, num_samples_per_prompt=3, binary_reward=True
    )
    controller = _controller(extra, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    metrics = asyncio.run(controller.validate_extra(extra_index=0, step=5))

    controller._rollouter.get_validation_samples.assert_called_once_with(-1, "deepswe")
    calls = controller._rollouter.run_group_rollouts.call_args_list
    # Set 0's ids: below the overlapped main-set range.
    base = OVERLAPPED_VALIDATION_FIRST_GROUP_ID - EXTRA_VALIDATION_GROUP_ID_STRIDE
    assert [call.kwargs["group_id"] for call in calls] == [base, base - 1]
    assert [call.kwargs["group_size"] for call in calls] == [3, 3]
    assert calls[0].kwargs["sampling"].temperature == 1.0
    controller.generator_router.release_groups.call_one.assert_awaited_once_with(
        [base, base - 1]
    )
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["validation/deepswe_reward/_mean"] == pytest.approx(1 / 6)
    assert aggregated["validation/deepswe/best_of_3/mean"] == 0.5
    assert aggregated["validation/deepswe/pass_at_3/mean"] == 0.5
    assert "timing/validation/deepswe" in aggregated
    assert not any(key.startswith("validation_reward") for key in aggregated)


def test_extra_validation_without_binary_reward_logs_only_best_of() -> None:
    extra = ExtraValidationConfig(name="swe", freq=10, num_samples_per_prompt=2)
    controller = _controller(extra, [[0.5, 0.25], [0.0, 1.0]])
    aggregated = m.MetricsProcessor._aggregate_metrics(
        asyncio.run(controller.validate_extra(extra_index=0, step=1))
    )
    assert aggregated["validation/swe/best_of_2/mean"] == 0.75
    assert not any("pass_at" in key for key in aggregated)


def test_extra_validations_start_every_freq_steps_one_pass_at_a_time() -> None:
    """freq 2: due at steps 2 and 4. The step-2 pass still runs at step 4, so the step-4 pass
    starts at step 5, once the step-2 pass is logged."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            extra_validations=[ExtraValidationConfig(name="deepswe", freq=2)]
        )
    )
    controller._extra_validation_tasks = {}
    controller._extra_validation_requested = set()
    controller.metrics_processor = Mock()
    started: list[int] = []
    ends: dict[int, asyncio.Event] = {}

    async def validate_extra(*, extra_index: int, step: int) -> list[m.Metric]:
        started.append(step)
        ends[step] = asyncio.Event()
        await ends[step].wait()
        return [m.Metric("validation/deepswe_reward", m.Mean(float(step)))]

    controller.validate_extra = validate_extra

    async def run() -> None:
        pull_task = asyncio.create_task(asyncio.sleep(0, result=0.0))
        for step in range(1, 6):
            await controller._step_extra_validations(
                step=step, num_training_steps=10, pull_task=pull_task
            )
            await asyncio.sleep(0)
            if step == 4:
                ends[2].set()
                await asyncio.sleep(0)

    asyncio.run(run())
    assert started == [2, 5]
    logged_steps = [
        call.kwargs["step"] for call in controller.metrics_processor.log.call_args_list
    ]
    assert logged_steps == [5]


def test_verifiers_rollouter_tags_and_routes_extra_validation_samples() -> None:
    from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
    from torchtitan.rl.examples.verifiers.rollouter import VerifiersRollouter

    rollouter = object.__new__(VerifiersRollouter)
    dataset = SimpleNamespace(
        build=lambda: iter(
            [VerifiersTaskSample(verifiers_task_data={"id": i}) for i in range(3)]
        )
    )
    rollouter._verifiers_config = SimpleNamespace(
        extra_validation_sets={"deepswe": SimpleNamespace(dataset=dataset)}
    )
    samples = rollouter.get_validation_samples(2, "deepswe")
    assert [sample.verifiers_task_data["id"] for sample in samples] == [0, 1]
    assert {sample.validation_set for sample in samples} == {"deepswe"}
    with pytest.raises(ValueError, match="unknown validation set"):
        rollouter.get_validation_samples(1, "missing")
