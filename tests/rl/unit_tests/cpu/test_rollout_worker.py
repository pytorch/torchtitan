# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for direct rollout worker execution."""

import asyncio
import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from renderers import (
    ParsedResponse,
    PlaceholderRange,
    Qwen3RendererConfig,
    RenderedTokens,
)

from torchtitan.components.renderer import from_renderers

from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout import Prompt, PromptImage, RolloutGroup, RolloutStatus
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.message_env import MessageEnvInitOutput, MessageEnvStepOutput
from torchtitan.rl.rollout.rollouter import MessageEnvRolloutWorker
from torchtitan.rl.rollout.rubric import RewardFn, Rubric, RubricOutput
from torchtitan.rl.rollout.token_env import TokenEnv, TokenEnvOutput
from torchtitan.rl.rollout.types import Completion, Rollout
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder


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
            next_prompt=Prompt(token_ids=[1, 2]),
            next_prompt_messages=[{"role": "user", "content": "prompt"}],
            status=RolloutStatus.ONGOING,
        )

    async def step(self, completion: Completion) -> TokenEnvOutput:
        return TokenEnvOutput(
            next_prompt=None,
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


class _CustomWorker(MessageEnvRolloutWorker):
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

    async def __call__(self, prompt, **kwargs) -> Completion:
        self.calls.append((prompt, kwargs))
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
        # A MessageEnvRolloutWorker takes its own config, so the worker is usable without a
        # Rollouter or an actor mesh.
        worker_config = SimpleNamespace(
            custom_setting="custom",
            rubric=_Config(_Rubric()),
            message_env=_MessageEnvConfig(),
            token_env=token_env_config,
            advantage=_Config(_AdvantageEstimator()),
        )
        worker = _CustomWorker(worker_config)
        released = []

        async def release_session_fn(**kwargs) -> None:
            released.append(kwargs)

        await worker.setup_async(
            tokenizer_config=HuggingFaceTokenizer.Config(),
            renderer_config=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
            hf_assets_path="tests/assets/tokenizer",
        )
        group = await worker.run_group(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
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
        # Each rollout releases its routing session after its last turn.
        assert released == [
            {"routing_session_id": "group=7/rollout=0"},
            {"routing_session_id": "group=7/rollout=1"},
        ]

    asyncio.run(run())


class _RewritingRenderer:
    """Bridges each turn, or rewrites the history and makes TokenEnv re-render it, at random."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.prompts: list[list[int]] = [[1, 2, 3]]

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        return RenderedTokens(token_ids=list(self.prompts[-1]))

    def bridge_to_next_turn(
        self, previous_prompt_ids, previous_completion_ids, new_messages, *, tools=None
    ):
        reply = [self.rng.randrange(100)]
        kind = self.rng.choice(["continue", "strip_thinking", "prefix", "reset"])
        if kind == "continue":
            next_prompt = previous_prompt_ids + previous_completion_ids + reply
        elif kind == "strip_thinking":
            next_prompt = previous_prompt_ids + previous_completion_ids[1:] + reply
        elif kind == "prefix":
            num_kept = self.rng.randint(0, len(previous_prompt_ids))
            next_prompt = previous_prompt_ids[:num_kept]
        else:
            next_prompt = [1] + reply
        self.prompts.append(next_prompt)

        # Only a continuation bridges; a rewrite returns None, as when a renderer can't bridge
        if kind == "continue":
            return RenderedTokens(token_ids=list(next_prompt))
        return None

    def parse_response(self, token_ids, *, tools=None):
        return ParsedResponse(
            content=str(token_ids), reasoning_content=None, tool_calls=[]
        )


class _CountingMessageEnv:
    """Replies to every turn and ends the rollout after `num_turns` turns."""

    def __init__(self, num_turns: int) -> None:
        self.num_turns = num_turns
        self.num_steps = 0

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": "task"}]
        )

    async def step(self, completion_message) -> MessageEnvStepOutput:
        self.num_steps += 1
        return MessageEnvStepOutput(
            env_messages=[{"role": "user", "content": "reply"}],
            done=self.num_steps == self.num_turns,
        )

    async def close(self) -> None:
        pass


class _RewardRolloutId(RewardFn):
    """Rewards rollout_id + 1."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout, env_input) -> float:
        return float(rollout.rollout_id + 1)


class _FirstRolloutFails(MessageEnvRolloutWorker):
    """Rollout 0 fails in `fails_in`: its generate call, its own code, an aborted generation, or
    the rubric."""

    fails_in = ""

    async def _run_single_rollout(self, *, rollout_id: int, **kwargs) -> Rollout:
        if self.fails_in == "rollout code" and rollout_id == 0:
            raise RuntimeError("env bug")
        rollout = await super()._run_single_rollout(rollout_id=rollout_id, **kwargs)
        if self.fails_in == "aborted generation" and rollout_id == 0:
            # What `TokenEnv.step` returns for a completion with finish_reason "abort".
            rollout.status = RolloutStatus.ERROR_ABORT
        return rollout

    async def score_group(self, rollouts, env_input) -> list[RubricOutput]:
        outputs = await super().score_group(rollouts, env_input)
        return [
            RubricOutput(reward=None)
            if self.fails_in == "rubric" and rollout.rollout_id == 0
            else output
            for rollout, output in zip(rollouts, outputs, strict=True)
        ]


@pytest.mark.parametrize(
    "fails_in, status, error",
    [
        ("generate call", RolloutStatus.ERROR_INFRA, "generator unreachable"),
        # Raised past `_run_single_rollout`: `make_scored_group` keeps it.
        ("rollout code", RolloutStatus.ERROR_INFRA, "env bug"),
        ("aborted generation", RolloutStatus.ERROR_ABORT, None),
        ("rubric", RolloutStatus.COMPLETED, None),
    ],
)
def test_a_failed_rollout_is_kept_but_left_out_of_the_advantage(
    fails_in: str, status: RolloutStatus, error: str | None
) -> None:
    class _GenerateFnFailingRolloutZero(_GenerateFn):
        async def __call__(self, prompt, **kwargs) -> Completion:
            if fails_in == "generate call" and kwargs["request_id"].startswith(
                "group=7/rollout=0/"
            ):
                raise ConnectionError("generator unreachable")
            return await super().__call__(prompt, **kwargs)

    async def release_session_fn(**kwargs) -> None:
        pass

    async def run() -> RolloutGroup:
        worker = _FirstRolloutFails(
            SimpleNamespace(
                # The stock rubric would give rollout 0 reward 1, had it scored it.
                rubric=Rubric.Config(reward_fns=[_RewardRolloutId.Config()]),
                message_env=_MessageEnvConfig(),
                token_env=_TokenEnvConfig(),
                advantage=AdvantageEstimator.Config(),
            )
        )
        worker.fails_in = fails_in
        await worker.setup_async(
            tokenizer_config=HuggingFaceTokenizer.Config(),
            renderer_config=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
            hf_assets_path="tests/assets/tokenizer",
        )
        return await worker.run_group(
            generate_fn=_GenerateFnFailingRolloutZero(),
            release_session_fn=release_session_fn,
            sample="sample",
            group_id=7,
            group_size=3,
            sampling=SamplingConfig(),
        )

    failed, *scored = asyncio.run(run()).rollouts
    assert failed.status is status
    if error is not None:
        assert error in failed.logs["error"]
    assert (failed.reward, failed.advantage) == (None, None)
    # Rewards 2 and 3, centered on their mean 2.5.
    assert [rollout.advantage for rollout in scored] == [-0.5, 0.5]


def test_delta_prompts_rebuild_the_full_prompt_samples() -> None:
    # TokenEnv returns each prompt as a delta. Delta prompts must pack into the same samples as full
    # prompts: a turn joins the last sample iff its prompt extends the previous turn's prompt + completion.
    rng = random.Random(0)
    worker = MessageEnvRolloutWorker(
        SimpleNamespace(
            rubric=_Config(None),
            message_env=None,
            token_env=None,
            advantage=_Config(None),
        )
    )
    builder = TrainingSampleBuilder.Config().build()

    async def generate_fn(prompt, **kwargs) -> Completion:
        token_ids = [rng.randrange(100) for _ in range(rng.randint(1, 3))]
        return Completion(
            min_policy_version=1,
            max_policy_version=1,
            request_id=kwargs["request_id"],
            token_ids=token_ids,
            token_logprobs=[-0.5] * len(token_ids),
            finish_reason="stop",
        )

    async def release_session_fn(**kwargs) -> None:
        pass

    async def run() -> None:
        for rollout_id in range(20):
            renderer = _RewritingRenderer(rng)
            env = TokenEnv.Config(max_rollout_tokens=32768).build(
                message_env=_CountingMessageEnv(num_turns=rng.randint(1, 6)),
                renderer=renderer,
            )
            rollout = await worker._run_single_rollout(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                env=env,
                sampling=SamplingConfig(),
                group_id=0,
                rollout_id=rollout_id,
            )
            assert rollout.status == RolloutStatus.COMPLETED
            rollout.advantage = 1.0

            expected, previous = [], []
            for turn_id, (turn, prompt) in enumerate(
                zip(rollout.turns, renderer.prompts, strict=True)
            ):
                completion = turn.completion_token_ids.tolist()
                if expected and prompt[: len(previous)] == previous:
                    expected[-1][1].extend(prompt[len(previous) :] + completion)
                else:
                    expected.append((turn_id, prompt + completion))
                previous = prompt + completion

            samples = builder.rollout_to_training_samples(rollout)
            assert [
                (s.rollout_id.turn_id, s.token_ids.tolist()) for s in samples
            ] == expected

    asyncio.run(run())


def test_rollout_sends_and_records_prompt_images() -> None:
    image = PromptImage(
        hash_value="img-a", source_path="a.png", grid_thw=(1, 4, 4), data=b"img-a"
    )
    placeholder = PlaceholderRange(offset=1, length=4)

    class _ImageTokenEnv(_TokenEnv):
        async def init(self) -> TokenEnvOutput:
            return TokenEnvOutput(
                next_prompt=Prompt(
                    token_ids=[1, 9, 9, 9, 9, 2],
                    images=(image,),
                    image_placeholders=(placeholder,),
                ),
                status=RolloutStatus.ONGOING,
            )

    worker = MessageEnvRolloutWorker(
        SimpleNamespace(
            rubric=_Config(None),
            message_env=None,
            token_env=None,
            advantage=_Config(None),
        )
    )
    generate_fn = _GenerateFn()

    async def release_session_fn(**kwargs) -> None:
        pass

    rollout = asyncio.run(
        worker._run_single_rollout(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            env=_ImageTokenEnv(),
            sampling=SamplingConfig(),
            group_id=0,
            rollout_id=0,
        )
    )
    assert generate_fn.calls[0][0].images == (image,)
    assert rollout.turns[0].prompt_delta_images == (image,)
    assert rollout.turns[0].prompt_delta_image_placeholders == (placeholder,)
