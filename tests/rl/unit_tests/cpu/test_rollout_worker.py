# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for direct rollout worker execution."""

import asyncio
import random
from types import SimpleNamespace

import numpy as np
import pytest
from renderers import ParsedResponse, Qwen3RendererConfig

from torchtitan.components.renderer import from_renderers

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.generator.generation_server import GenerationServer
from torchtitan.rl.generator.http_client import GeneratorClient

from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import RolloutGroup, RolloutStatus
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.message_env import MessageEnvInitOutput, MessageEnvStepOutput
from torchtitan.rl.rollout.rollouter import gather_rollouts, RolloutWorker
from torchtitan.rl.rollout.rubric import RubricOutput
from torchtitan.rl.rollout.token_env import TokenEnv, TokenEnvOutput
from torchtitan.rl.rollout.types import Completion
from torchtitan.rl.routing.inter_generator import GeneratorEndpoint
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
            topk_token_ids=np.array([[4, 9]], dtype=np.int32),
            topk_logprobs=np.array([[-0.5, -1.5]], dtype=np.float32),
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
            rollout_timeout_s=None,
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
        # The generator's top-k rides on the turn, for ScoreCenteringLoss.
        assert group.rollouts[0].turns[0].completion_topk_token_ids.tolist() == [[4, 9]]
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

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        return list(self.prompts[-1])

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
            return SimpleNamespace(token_ids=list(next_prompt))
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


def test_a_rollout_that_raises_is_kept_but_left_out_of_the_advantage() -> None:
    class _FirstRolloutFails(_GenerateFn):
        async def __call__(self, prompt_token_ids, **kwargs) -> Completion:
            if kwargs["request_id"].startswith("group=7/rollout=0/"):
                raise ConnectionError("generator unreachable")
            return await super().__call__(prompt_token_ids, **kwargs)

    async def release_session_fn(**kwargs) -> None:
        pass

    async def run() -> RolloutGroup:
        worker = RolloutWorker(
            SimpleNamespace(
                rubric=_Config(_Rubric()),
                message_env=_MessageEnvConfig(),
                token_env=_TokenEnvConfig(),
                advantage=AdvantageEstimator.Config(),
                rollout_timeout_s=None,
            )
        )
        await worker.setup_async(
            tokenizer_config=HuggingFaceTokenizer.Config(),
            renderer_config=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
            hf_assets_path="tests/assets/tokenizer",
        )
        return await worker.run_group(
            generate_fn=_FirstRolloutFails(),
            release_session_fn=release_session_fn,
            sample="sample",
            group_id=7,
            group_size=3,
            sampling=SamplingConfig(),
        )

    failed, *scored = asyncio.run(run()).rollouts
    assert failed.status is RolloutStatus.ERROR_SYSTEM
    assert "generator unreachable" in failed.logs["error"]
    assert (failed.reward, failed.advantage) == (None, None)
    # `_Rubric` rewards rollout_id + 1, so 2 and 3, centered on their mean 2.5.
    assert [rollout.advantage for rollout in scored] == [-0.5, 0.5]


def test_cancelled_group_waits_for_every_sibling_release() -> None:
    """A cancelled group returns only after each sibling released its session: the controller
    releases the group next, and no sibling's router call may still be in flight."""

    async def run() -> None:
        started = asyncio.Semaphore(0)

        async def generate_fn(prompt_token_ids, **kwargs) -> Completion:
            started.release()
            await asyncio.Event().wait()

        # Router round trips of different lengths: the slow one must finish too.
        delays = iter([0.01, 0.3])
        released = []

        async def release_session_fn(*, routing_session_id: str) -> None:
            await asyncio.sleep(next(delays))
            released.append(routing_session_id)

        worker = _CustomWorker(
            SimpleNamespace(
                custom_setting="custom",
                rubric=_Config(_Rubric()),
                message_env=_MessageEnvConfig(),
                token_env=_TokenEnvConfig(),
                advantage=_Config(_AdvantageEstimator()),
                rollout_timeout_s=None,
            )
        )
        await worker.setup_async(
            tokenizer_config=HuggingFaceTokenizer.Config(),
            renderer_config=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
            hf_assets_path="tests/assets/tokenizer",
        )
        group = asyncio.create_task(
            worker.run_group(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                sample="sample",
                group_id=7,
                group_size=2,
                sampling=SamplingConfig(),
            )
        )
        for _ in range(2):
            await asyncio.wait_for(started.acquire(), timeout=10)
        group.cancel()
        with pytest.raises(asyncio.CancelledError):
            await group
        assert sorted(released) == ["group=7/rollout=0", "group=7/rollout=1"]

    asyncio.run(run())


def test_delta_prompts_rebuild_the_full_prompt_samples() -> None:
    # TokenEnv returns each prompt as a delta. Delta prompts must pack into the same samples as full
    # prompts: a turn joins the last sample iff its prompt extends the previous turn's prompt + completion.
    # Router replay: each turn asks only for the expert rows its sample lacks; a fake row holds its
    # input token, so every sample's rows must equal its trainer inputs.
    rng = random.Random(0)
    worker = RolloutWorker(
        SimpleNamespace(
            rubric=_Config(None),
            message_env=None,
            token_env=None,
            advantage=_Config(None),
            rollout_timeout_s=None,
        )
    )
    builder = TrainingSampleBuilder.Config().build(replay_routed_experts=True)

    async def generate_fn(prompt_token_ids, **kwargs) -> Completion:
        token_ids = [rng.randrange(100) for _ in range(rng.randint(1, 3))]
        start = kwargs["sampling_config"].routed_experts_prompt_start
        # Inside the prompt (the fake renderer may rewrite it to an empty one).
        assert 0 <= start < max(len(prompt_token_ids), 1)
        # Rows of positions start .. P + N - 2: the last token never ran forward.
        inputs = (list(prompt_token_ids) + token_ids)[start:-1]
        return Completion(
            min_policy_version=1,
            max_policy_version=1,
            request_id=kwargs["request_id"],
            token_ids=token_ids,
            token_logprobs=[-0.5] * len(token_ids),
            finish_reason="stop",
            routed_expert_ids=np.asarray(inputs, dtype=np.uint8).reshape(-1, 1, 1),
        )

    async def release_session_fn(**kwargs) -> None:
        pass

    async def run() -> None:
        for rollout_id in range(20):
            renderer = _RewritingRenderer(rng)
            env = TokenEnv.Config().build(
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
            for sample in samples:
                assert (
                    sample.routed_expert_ids.flatten().tolist()
                    == sample.token_ids[:-1].tolist()
                )

    asyncio.run(run())


def _bare_worker(rollout_timeout_s: float | None = None) -> RolloutWorker:
    return RolloutWorker(
        SimpleNamespace(
            rubric=_Config(None),
            message_env=None,
            token_env=None,
            advantage=_Config(None),
            rollout_timeout_s=rollout_timeout_s,
        )
    )


def test_failing_session_release_does_not_fail_the_rollout() -> None:
    async def release_session_fn(**kwargs) -> None:
        raise RuntimeError("router unreachable")

    rollout = asyncio.run(
        _bare_worker()._run_single_rollout(
            generate_fn=_GenerateFn(),
            release_session_fn=release_session_fn,
            env=_TokenEnv(),
            sampling=SamplingConfig(),
            group_id=7,
            rollout_id=0,
        )
    )

    assert rollout.status == RolloutStatus.COMPLETED


async def _hang(prompt_token_ids, **kwargs) -> Completion:
    await asyncio.Event().wait()


async def _connect_timeout(prompt_token_ids, **kwargs) -> Completion:
    raise TimeoutError("connect timed out")


@pytest.mark.parametrize(
    "generate_fn, expected_status",
    [
        # Past rollout_timeout_s: the rollout ends as ERROR_TIMEOUT.
        (_hang, RolloutStatus.ERROR_TIMEOUT),
        # A TimeoutError raised inside the turn (e.g. a connect timeout) is an error of the code,
        # not of the policy.
        (_connect_timeout, RolloutStatus.ERROR_SYSTEM),
    ],
)
def test_rollout_timeout(generate_fn, expected_status) -> None:
    released = []

    async def release_session_fn(**kwargs) -> None:
        released.append(kwargs)

    rollout = asyncio.run(
        asyncio.wait_for(
            _bare_worker(rollout_timeout_s=0.1)._run_single_rollout(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                env=_TokenEnv(),
                sampling=SamplingConfig(),
                group_id=7,
                rollout_id=0,
            ),
            timeout=10,
        )
    )

    assert rollout.status == expected_status
    assert released == [{"routing_session_id": "group=7/rollout=0"}]


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_stopped_rollout_cancels_its_call_and_releases_lease_and_session(stop) -> None:
    """A rollout cancelled from outside, or past `rollout_timeout_s`, mid-call closes its HTTP call
    (the server cancels the handler), gives the lease back and releases its routing session."""

    async def run() -> None:
        started, server_call_cancelled = asyncio.Event(), asyncio.Event()

        async def generate(prompt_token_ids, **kwargs) -> Completion:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                server_call_cancelled.set()
                raise

        server = GenerationServer(generate=generate, max_context_length=64)
        await server.start()
        leases, released = [], []

        async def acquire(session_id, *, group_id):
            leases.append(("acquire", group_id))
            return GeneratorEndpoint(index=0, url=server.url)

        async def release(index, *, group_id):
            leases.append(("release", group_id))

        async def release_session_fn(**kwargs) -> None:
            released.append(kwargs)

        router = SimpleNamespace(
            acquire=SimpleNamespace(call_one=acquire),
            release=SimpleNamespace(call_one=release),
        )
        try:
            rollout = asyncio.create_task(
                _bare_worker(
                    rollout_timeout_s=0.5 if stop == "timeout" else None
                )._run_single_rollout(
                    generate_fn=GeneratorClient(router=router),
                    release_session_fn=release_session_fn,
                    env=_TokenEnv(),
                    sampling=SamplingConfig(stop_token_ids=[99]),
                    group_id=7,
                    rollout_id=0,
                )
            )
            await asyncio.wait_for(started.wait(), timeout=10)
            if stop == "cancel":
                rollout.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await rollout
            else:
                result = await asyncio.wait_for(rollout, timeout=10)
                assert result.status == RolloutStatus.ERROR_TIMEOUT
            await asyncio.wait_for(server_call_cancelled.wait(), timeout=10)
        finally:
            await server.close()

        assert leases == [("acquire", 7), ("release", 7)]
        assert released == [{"routing_session_id": "group=7/rollout=0"}]

    asyncio.run(run())


def test_gather_rollouts_times_the_group_and_its_straggler() -> None:
    async def rollout(seconds: float) -> float:
        await asyncio.sleep(seconds)
        return seconds

    async def fails() -> float:
        raise ValueError("env crashed")

    results, metrics = asyncio.run(
        gather_rollouts([rollout(0.05), rollout(0.01), fails()])
    )

    assert results[:2] == [0.05, 0.01] and isinstance(results[2], ValueError)
    values = m.MetricsProcessor._aggregate_metrics(metrics)
    assert values["rollout/group_wall_s/max"] >= 0.05
    # The last sibling ended ~0.05 s after the first (the one that raised at once).
    assert values["rollout/group_straggler_wait_s/max"] >= 0.04
