# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for direct rollout worker execution."""

import asyncio
from types import SimpleNamespace

from renderers import Qwen3RendererConfig

from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.environment.token import TokenEnvOutput
from torchtitan.experiments.rl.rollout import RolloutStatus
from torchtitan.experiments.rl.rollout.rollouter import RolloutWorker
from torchtitan.experiments.rl.rubrics import RubricOutput
from torchtitan.experiments.rl.types import Completion


class _Config:
    def __init__(self, value) -> None:
        self._value = value

    def build(self, **kwargs):
        return self._value


class _MessageEnvConfig:
    def build(self, *, env_input):
        return env_input


class _TokenEnv:
    def __init__(self) -> None:
        self.closed = False

    async def init(self) -> TokenEnvOutput:
        return TokenEnvOutput(
            next_prompt_token_ids=[1, 2],
            next_prompt_messages=[{"role": "user", "content": "prompt"}],
            status=RolloutStatus.ONGOING,
        )

    async def step(self, completion: Completion) -> TokenEnvOutput:
        return TokenEnvOutput(
            next_prompt_token_ids=None,
            status=RolloutStatus.COMPLETED,
            completion_message={"role": "assistant", "content": "answer"},
        )

    async def close(self) -> None:
        self.closed = True


class _TokenEnvConfig:
    def __init__(self) -> None:
        self.envs: list[_TokenEnv] = []
        self.renderers: list[object] = []

    def build(self, *, message_env, renderer) -> _TokenEnv:
        env = _TokenEnv()
        self.envs.append(env)
        self.renderers.append(renderer)
        return env


class _Rubric:
    async def score_group(self, rollouts, env_input):
        return [
            RubricOutput(reward=float(rollout.rollout_id + 1)) for rollout in rollouts
        ]


class _AdvantageEstimator:
    def __call__(self, group):
        return [rollout.reward * 10 for rollout in group.rollouts]


class _CustomWorker(RolloutWorker):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.custom_setting = config.custom_setting

    def make_env_group(self, **kwargs):
        self.make_env_group_called = True
        return super().make_env_group(**kwargs)

    async def score_group(self, rollouts, env_input):
        self.score_group_called = True
        return await super().score_group(rollouts, env_input)


class _GenerateFn:
    def __init__(self) -> None:
        self.calls = []

    async def __call__(self, prompt_token_ids, **kwargs) -> Completion:
        self.calls.append((prompt_token_ids, kwargs))
        return Completion(
            min_policy_version=3,
            max_policy_version=3,
            request_id=kwargs["request_id"],
            token_ids=[4],
            token_logprobs=[-0.5],
            finish_reason="stop",
        )


def test_worker_executes_group_without_actor_mesh() -> None:
    async def run() -> None:
        generate_fn = _GenerateFn()
        token_env_config = _TokenEnvConfig()
        # A RolloutWorker takes its own config, so the worker is usable without a
        # Rollouter or an actor mesh.
        worker_config = SimpleNamespace(
            custom_setting="custom",
            rubric=_Config(_Rubric()),
            message_env=_MessageEnvConfig(),
            token_env=token_env_config,
            advantage=_Config(_AdvantageEstimator()),
        )
        worker = _CustomWorker(worker_config)
        await worker.setup_async(
            renderer_config=Qwen3RendererConfig(enable_thinking=False),
            hf_assets_path="tests/assets/tokenizer",
        )
        group = await worker.run_group(
            generate_fn=generate_fn,
            sample="sample",
            group_id=7,
            group_size=2,
            sampling=SamplingConfig(seed=11),
        )

        assert group.group_id == 7
        assert isinstance(worker, _CustomWorker)
        assert isinstance(worker.rubric, _Rubric)
        assert isinstance(worker.advantage_estimator, _AdvantageEstimator)
        assert worker.custom_setting == "custom"
        assert worker.make_env_group_called
        assert worker.score_group_called
        assert [rollout.reward for rollout in group.rollouts] == [1.0, 2.0]
        assert [rollout.advantage for rollout in group.rollouts] == [10.0, 20.0]
        assert all(env.closed for env in token_env_config.envs)
        assert [type(r).__name__ for r in token_env_config.renderers] == [
            "Qwen3Renderer",
            "Qwen3Renderer",
        ]
        assert [call[1]["request_id"] for call in generate_fn.calls] == [
            "group=7/rollout=0/turn=0",
            "group=7/rollout=1/turn=0",
        ]
        assert [call[1]["sampling_config"].seed for call in generate_fn.calls] == [
            11,
            12,
        ]

    asyncio.run(run())
