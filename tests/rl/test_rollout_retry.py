# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for rollout retry ownership, budgets, and failure isolation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import RolloutDataLoader
from torchtitan.rl.rollout.rollouter import (
    MessageEnvRolloutWorker,
    Rollouter,
    RolloutInfraBudgetExceeded,
)
from torchtitan.rl.rollout.rubric import RubricOutput
from torchtitan.rl.rollout.token_env import TokenEnvOutput
from torchtitan.rl.rollout.types import (
    Completion,
    Prompt,
    Rollout,
    RolloutGroup,
    RolloutStatus,
)
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder


def _rollouter(**overrides) -> Rollouter:
    loader = MagicMock(spec=RolloutDataLoader)
    loader.__iter__.side_effect = lambda: loader
    config = Rollouter.Config(
        training_dataloader=Mock(build=Mock(return_value=loader)),
        validation_dataset=Mock(),
        worker=Mock(),
        **overrides,
    )
    rollouter = Rollouter(config)
    rollouter._worker_actors = SimpleNamespace(
        run_group=SimpleNamespace(choose=AsyncMock())
    )
    return rollouter


def _group(status: RolloutStatus, *, group_id: int = 7) -> RolloutGroup:
    return RolloutGroup(
        group_id=group_id,
        rollouts=[Rollout(group_id=group_id, rollout_id=0, status=status)],
    )


async def _run(
    rollouter: Rollouter,
    *,
    group_id: int = 7,
    group_size: int = 1,
    generate_fn=None,
    release_session_fn=None
) -> RolloutGroup:
    return await rollouter.run_group_rollouts(
        generate_fn=generate_fn or AsyncMock(),
        release_session_fn=release_session_fn or AsyncMock(),
        sample="prompt",
        group_id=group_id,
        group_size=group_size,
        sampling=SamplingConfig(seed=42),
    )


def _counts(rollouter: Rollouter) -> dict[str, float]:
    return {metric.key: metric.value.value for metric in rollouter.retry_metrics()}


def test_transient_infra_failure_retries_same_prompt_and_succeeds() -> None:
    rollouter = _rollouter()
    choose = rollouter._worker_actors.run_group.choose
    failed = _group(RolloutStatus.ERROR_INFRA)
    completed = _group(RolloutStatus.COMPLETED)
    choose.side_effect = [failed, completed]

    assert asyncio.run(_run(rollouter)) is completed
    assert choose.await_count == 2
    first, second = [call.kwargs for call in choose.await_args_list]
    assert first["sample"] == second["sample"] == "prompt"
    assert first["group_id"] == second["group_id"] == 7
    assert second["previous_rollouts"] is failed.rollouts
    assert second["attempt"] == 1
    assert _counts(rollouter)["rollout_infra_retries_total"] == 1
    assert _counts(rollouter)["rollout_infra_failures_total"] == 1
    rollouter._training_dataloader.commit.assert_not_called()


def test_persistent_infra_failure_exhausts_budget_and_is_not_trainable() -> None:
    rollouter = _rollouter(max_retries_per_prompt=2)
    choose = rollouter._worker_actors.run_group.choose
    choose.side_effect = lambda **kwargs: _group(RolloutStatus.ERROR_INFRA)

    result = asyncio.run(_run(rollouter))
    assert choose.await_count == 3
    assert _counts(rollouter)["rollout_infra_retries_total"] == 2
    assert _counts(rollouter)["rollout_infra_failures_total"] == 3
    assert result.rollouts[0].status is RolloutStatus.ERROR_INFRA
    assert (
        TrainingSampleBuilder.Config()
        .build()
        .build_from_group(rollout_group=result)
        .training_samples
        == []
    )


@pytest.mark.parametrize("status", [RolloutStatus.ERROR_DATA, RolloutStatus.ERROR_ENV])
def test_data_and_environment_failures_are_never_retried(status: RolloutStatus) -> None:
    rollouter = _rollouter()
    choose = rollouter._worker_actors.run_group.choose
    choose.return_value = _group(status)

    result = asyncio.run(_run(rollouter))
    assert result.rollouts[0].status is status
    assert choose.await_count == 1
    assert _counts(rollouter)["rollout_infra_retries_total"] == 0
    assert _counts(rollouter)["rollout_data_failures_total"] == (
        status is RolloutStatus.ERROR_DATA
    )
    assert _counts(rollouter)["rollout_env_failures_total"] == (
        status is RolloutStatus.ERROR_ENV
    )


def test_global_budget_counts_concurrent_prompts_and_aborts_before_another_retry() -> None:
    async def run() -> None:
        rollouter = _rollouter(max_retries_per_prompt=0, max_infra_failures_per_run=1)
        rollouter._worker_actors.run_group.choose.side_effect = lambda **kwargs: _group(
            RolloutStatus.ERROR_INFRA, group_id=kwargs["group_id"]
        )
        results = await asyncio.gather(
            _run(rollouter, group_id=1),
            _run(rollouter, group_id=2),
            return_exceptions=True,
        )
        assert (
            sum(isinstance(result, RolloutInfraBudgetExceeded) for result in results)
            == 1
        )
        assert _counts(rollouter)["rollout_infra_failures_total"] == 2
        with pytest.raises(RolloutInfraBudgetExceeded, match="2 > 1"):
            await _run(rollouter, group_id=3)
        assert rollouter._worker_actors.run_group.choose.await_count == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "error, attempts, status",
    [
        (ConnectionResetError("socket closed"), 3, RolloutStatus.ERROR_INFRA),
        (TimeoutError("RPC timed out"), 3, RolloutStatus.ERROR_INFRA),
        (ValueError("invalid tokens"), 1, RolloutStatus.ERROR_DATA),
        (RuntimeError("environment crashed"), 1, RolloutStatus.ERROR_ENV),
    ],
)
def test_group_rpc_errors_use_their_original_class(error, attempts, status) -> None:
    rollouter = _rollouter()
    wrapper = RuntimeError("remote error")
    rollouter._actor_system = Mock(unwrap_exception=Mock(return_value=error))
    choose = rollouter._worker_actors.run_group.choose
    choose.side_effect = wrapper
    result = asyncio.run(_run(rollouter))
    assert choose.await_count == attempts
    assert result.rollouts[0].status is status


def _worker() -> tuple[MessageEnvRolloutWorker, list[Mock]]:
    worker = object.__new__(MessageEnvRolloutWorker)
    envs = []

    def make_env_group(*, sample, group_size):
        created = []
        for _ in range(group_size):
            env = Mock(
                init=AsyncMock(
                    return_value=TokenEnvOutput(
                        next_prompt=Prompt(token_ids=[1, 2]),
                        status=RolloutStatus.ONGOING,
                    )
                ),
                step=AsyncMock(
                    return_value=TokenEnvOutput(
                        next_prompt=None,
                        status=RolloutStatus.COMPLETED,
                    )
                ),
                close=AsyncMock(),
            )
            created.append(env)
        envs.extend(created)
        return created

    worker.make_env_group = Mock(side_effect=make_env_group)
    worker.score_group = AsyncMock(
        side_effect=lambda rollouts, sample: [
            RubricOutput(reward=float(r.rollout_id)) for r in rollouts
        ]
    )
    worker.advantage_estimator = AdvantageEstimator.Config().build()
    return worker, envs


def _completion(request_id: str) -> Completion:
    return Completion(
        min_policy_version=0,
        max_policy_version=0,
        request_id=request_id,
        token_ids=[3],
        token_logprobs=[-0.5],
        finish_reason="stop",
    )


def test_worker_retries_only_failed_sibling_with_fresh_env_and_session() -> None:
    rollouter = _rollouter()
    worker, envs = _worker()
    rollouter._worker_actors.run_group.choose.side_effect = worker.run_group
    requests = []

    async def generate_fn(**kwargs):
        requests.append(kwargs)
        if kwargs["request_id"] == "group=7/rollout=1/turn=0":
            raise ConnectionError("disconnect")
        return _completion(kwargs["request_id"])

    release = AsyncMock()
    result = asyncio.run(
        _run(
            rollouter, group_size=3, generate_fn=generate_fn, release_session_fn=release
        )
    )
    assert len(requests) == 4
    assert [r["sampling_config"].seed for r in requests] == [42, 43, 44, 43]
    assert requests[-1]["routing_session_id"] == "group=7/rollout=1/attempt=1"
    assert len(envs) == 4
    assert all(env.close.await_count == 1 for env in envs)
    assert release.await_count == 4
    assert [r.status for r in result.rollouts] == [RolloutStatus.COMPLETED] * 3
    assert [r.advantage for r in result.rollouts] == [-1, 0, 1]
    assert _counts(rollouter)["rollout_infra_retries_total"] == 1


@pytest.mark.parametrize(
    "error", [TimeoutError("env timeout"), RuntimeError("env crash")]
)
def test_environment_exception_is_not_misclassified_as_transport(error) -> None:
    worker, _ = _worker()
    env = Mock(init=AsyncMock(side_effect=error))
    release = AsyncMock()
    result = asyncio.run(
        worker._run_single_rollout(
            generate_fn=AsyncMock(),
            release_session_fn=release,
            env=env,
            sampling=SamplingConfig(),
            group_id=1,
            rollout_id=0,
        )
    )
    assert result.status is RolloutStatus.ERROR_ENV
    release.assert_awaited_once()


def test_cancellation_during_backoff_never_dispatches_or_counts_a_retry() -> None:
    async def run() -> None:
        rollouter = _rollouter(retry_backoff_s=1)
        choose = rollouter._worker_actors.run_group.choose
        choose.return_value = _group(RolloutStatus.ERROR_INFRA)
        sleeping = asyncio.Event()

        async def backoff(delay):
            assert delay == 1
            sleeping.set()
            await asyncio.Event().wait()

        with patch(
            "torchtitan.rl.rollout.rollouter.asyncio.sleep", side_effect=backoff
        ):
            task = asyncio.create_task(_run(rollouter))
            await sleeping.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert choose.await_count == 1
        assert _counts(rollouter)["rollout_infra_retries_total"] == 0

    asyncio.run(run())


def test_controller_propagates_budget_failure_without_finalizing_prompt() -> None:
    from torchtitan.rl.controller.controller import Controller

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(num_samples_per_prompt=1)
    )
    controller._sampling = SamplingConfig()
    controller._trainer_policy_version = 0
    controller._rollouter = _rollouter(max_infra_failures_per_run=0)
    controller._rollouter._worker_actors.run_group.choose.return_value = _group(
        RolloutStatus.ERROR_INFRA
    )
    controller.metrics_processor = Mock()
    controller.generator_router = Mock()
    controller.rollout_recorder = Mock()
    buffer = Mock(
        claim_next=AsyncMock(return_value=SimpleNamespace(sample="prompt", group_id=7)),
        finalize_work=AsyncMock(),
    )
    with pytest.raises(RolloutInfraBudgetExceeded):
        asyncio.run(
            controller._rollout_loop(
                group_buffer=buffer,
                generate_fn=AsyncMock(),
                release_session_fn=AsyncMock(),
            )
        )
    buffer.finalize_work.assert_not_awaited()
    controller.metrics_processor.log.assert_called_once()
    controller._rollouter._training_dataloader.commit.assert_not_called()


@pytest.mark.parametrize(
    "setting, value",
    [
        ("max_retries_per_prompt", -1),
        ("max_infra_failures_per_run", -1),
        ("retry_backoff_s", -1),
        ("retry_backoff_s", float("inf")),
        ("retry_backoff_s", float("nan")),
    ],
)
def test_invalid_budgets_fail_at_config_time(setting, value) -> None:
    with pytest.raises(ValueError, match=setting):
        _rollouter(**{setting: value})


def test_environment_construction_timeout_is_not_retried() -> None:
    rollouter = _rollouter()
    worker, _ = _worker()
    worker.make_env_group.side_effect = TimeoutError("environment startup")
    rollouter._worker_actors.run_group.choose.side_effect = worker.run_group
    result = asyncio.run(_run(rollouter))
    assert result.rollouts[0].status is RolloutStatus.ERROR_ENV
    assert _counts(rollouter)["rollout_infra_retries_total"] == 0


def test_failed_siblings_do_not_enter_reward_centering_or_training() -> None:
    rollouter = _rollouter(max_retries_per_prompt=0)
    worker, _ = _worker()
    rollouter._worker_actors.run_group.choose.side_effect = worker.run_group

    async def generate_fn(**kwargs):
        if "rollout=1/" in kwargs["request_id"]:
            raise ConnectionError("disconnect")
        return _completion(kwargs["request_id"])

    group = asyncio.run(_run(rollouter, group_size=3, generate_fn=generate_fn))
    assert [r.reward for r in group.rollouts] == [0.0, None, 2.0]
    assert [r.advantage for r in group.rollouts] == [-1.0, None, 1.0]
    samples = (
        TrainingSampleBuilder.Config().build().build_from_group(rollout_group=group)
    )
    assert [sample.rollout_id.rollout_id for sample in samples.training_samples] == [
        0,
        2,
    ]


def test_validation_does_not_swallow_global_budget_failure() -> None:
    from torchtitan.rl.controller.controller import Controller

    controller = object.__new__(Controller)
    controller.metrics_processor = Mock()
    controller._validation_group_ids = iter([-1])
    controller._make_generate_fn = Mock(return_value=AsyncMock())
    controller._make_release_session_fn = Mock(return_value=AsyncMock())
    controller._rollouter = _rollouter(max_infra_failures_per_run=0)
    controller._rollouter._worker_actors.run_group.choose.return_value = _group(
        RolloutStatus.ERROR_INFRA, group_id=-1
    )
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    with pytest.raises(RolloutInfraBudgetExceeded):
        asyncio.run(
            controller._collect_validation_rollouts(
                samples=["prompt"],
                sampling=SamplingConfig(),
                step=0,
            )
        )
    controller.generator_router.release_groups.call_one.assert_awaited_once_with([-1])
