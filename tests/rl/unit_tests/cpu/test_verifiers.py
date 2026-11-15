# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf
from aiohttp import ClientSession
from openai import AsyncOpenAI
from renderers.client import (
    _parse_completion_logprobs,
    _resolve_max_prompt_len,
    parse_generate_response,
)

from torchtitan.rl.examples.verifiers import rollouter as verifiers_rollouter
from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    _sampling_config,
    GenerationServer,
    GenerationSession,
)
from torchtitan.rl.examples.verifiers.rollouter import (
    _trainable_token_spans,
    log_failed_rollout,
    RewardFromVerifiers,
    VERIFIERS_REWARD_KEY,
    verifiers_rollout_logs,
    VerifiersRolloutWorker,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.types import Completion, Rollout, RolloutStatus


def test_train_client_config_carries_an_in_tree_renderer() -> None:
    """The renderer config reaches environment-server workers inside Verifiers' client
    config; an in-tree renderer named by `CustomRendererConfig` survives that trip."""
    from renderers import CustomRendererConfig
    from verifiers.v1.configs.client import TrainClientConfig

    from tests.unit_tests.cpu.components.test_renderer import IN_TREE_RENDERER_PATH

    renderer = CustomRendererConfig(
        import_path=IN_TREE_RENDERER_PATH, enable_thinking=False
    )
    sent = TrainClientConfig(
        base_url="http://127.0.0.1:1",
        api_key_var="",
        renderer=renderer,
        renderer_model_name="tests/assets/tokenizer",
    )

    received = TrainClientConfig.model_validate_json(sent.model_dump_json())

    assert received.renderer == renderer


def test_muse_glimmer_renderer_survives_verifiers_worker_config() -> None:
    """An in-tree renderer produces the same tokens before and after transport."""
    from renderers import create_renderer, CustomRendererConfig
    from torchtitan.components.renderer import from_renderers
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.models.muse_glimmer.atem import render_atem_tool_call

    from torchtitan.models.muse_glimmer.renderer import (
        EOM_ID,
        EOM_STR,
        EOT_ID,
        EOT_STR,
        MESSAGE_ID,
        MESSAGE_STR,
        MuseGlimmerRenderer,
        MuseGlimmerRendererConfig,
        START_ID,
        START_STR,
    )
    from verifiers.v1.configs.client import TrainClientConfig

    special_ids = {
        START_STR: START_ID,
        MESSAGE_STR: MESSAGE_ID,
        EOM_STR: EOM_ID,
        EOT_STR: EOT_ID,
    }

    class TokenizerBackend:
        def token_to_id(self, token):
            return special_ids.get(token)

        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return SimpleNamespace(ids=list(text.encode()))

    class WorkerTokenizer:
        bos_token = None
        bos_token_id = None
        name_or_path = "muse-tokenizer"

        def convert_tokens_to_ids(self, token):
            return special_ids.get(token)

        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return list(text.encode())

        def decode(self, token_ids, skip_special_tokens=False):
            assert not skip_special_tokens
            special_tokens = {
                token_id: token for token, token_id in special_ids.items()
            }
            return "".join(
                special_tokens[token_id]
                if token_id in special_tokens
                else chr(token_id)
                for token_id in token_ids
            )

    titan_tokenizer = SimpleNamespace(
        tokenizer=TokenizerBackend(),
        tokenizer_path="muse-tokenizer",
        bos_token=None,
        eos_token=None,
        bos_id=None,
        eos_id=None,
    )
    renderers_config = CustomRendererConfig(
        import_path="torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer",
        reasoning_strength="low",
        knowledge_cutoff="2025-01-01",
        current_date="2026-10-13",
        retain_reasoning_in_history=False,
        answer_from_reasoning_fallback=True,
        thinking_retention="tool_cycle",
    )
    config = from_renderers(renderers_config)
    sent = TrainClientConfig(
        base_url="http://127.0.0.1:1",
        api_key_var="",
        renderer=renderers_config,
        renderer_model_name="muse-tokenizer",
    )
    received = TrainClientConfig.model_validate_json(sent.model_dump_json())

    trainer_renderer = config.build(
        tokenizer=cast(HuggingFaceTokenizer, titan_tokenizer)
    )
    worker_renderer = create_renderer(WorkerTokenizer(), received.renderer)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "reasoning_content": "think", "content": "hello"},
    ]

    assert isinstance(trainer_renderer, MuseGlimmerRenderer)
    assert isinstance(worker_renderer, MuseGlimmerRenderer)
    assert isinstance(worker_renderer._config, MuseGlimmerRendererConfig)
    assert worker_renderer._config == trainer_renderer._config
    assert worker_renderer.is_prefix_stable is False
    assert worker_renderer.render_ids(
        messages, add_generation_prompt=True
    ) == trainer_renderer.render_ids(messages, add_generation_prompt=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Find an answer",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    tool_messages = [
        {"role": "user", "content": "find an answer"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search", "arguments": {"query": "hello"}},
                }
            ],
        },
    ]
    assert worker_renderer.render_ids(tool_messages, tools=tools) == (
        trainer_renderer.render_ids(tool_messages, tools=tools)
    )
    completion_ids = (
        list(" to=search".encode())
        + [MESSAGE_ID]
        + list(render_atem_tool_call("search", {"query": "hello"}).encode())
        + [EOT_ID]
    )
    parsed = worker_renderer.parse_response(
        completion_ids,
        prompt_ids=worker_renderer.render_ids(
            [{"role": "user", "content": "find an answer"}],
            tools=tools,
            add_generation_prompt=True,
        ),
        tools=tools,
    )
    assert parsed.tool_calls[0].name == "search"
    assert parsed.tool_calls[0].arguments == {"query": "hello"}


def test_trainable_token_spans() -> None:
    assert _trainable_token_spans([False, True, True, False, True]) == [
        (1, 3),
        (4, 5),
    ]


def test_verifiers_trace_preserves_generation_metadata() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    node = SimpleNamespace(
        token_ids=[10, 11, 12, 13],
        mask=[False, False, True, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="Answer: $42$"),
    )
    trace = SimpleNamespace(
        nodes=[node],
        branches=[
            SimpleNamespace(
                nodes=[node],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, 0.0, -0.2, -0.3],
            )
        ],
    )
    turns = VerifiersRolloutWorker.trace_to_rollout_turns(
        trace=trace,
        session=GenerationSession(
            group_id=0,
            min_policy_version=3,
            max_policy_version=4,
            metrics=[],
        ),
        group_id=5,
        rollout_id=2,
    )

    assert len(turns) == 1
    assert turns[0].prompt_prefix_len == 0
    assert turns[0].prompt_delta_token_ids.tolist() == [10, 11]
    assert turns[0].completion_token_ids.tolist() == [12, 13]
    assert turns[0].completion_logprobs.tolist() == pytest.approx([-0.2, -0.3])
    assert turns[0].completion_message == {
        "role": "assistant",
        "content": "Answer: $42$",
    }
    assert turns[0].min_policy_version == 3
    assert turns[0].max_policy_version == 4


def test_verifiers_multiturn_trace_matches_titanrl_rollout_structure() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    first_node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="first"),
    )
    second_node = SimpleNamespace(
        token_ids=[12, 13],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="second"),
    )
    trace = SimpleNamespace(
        nodes=[first_node, second_node],
        branches=[
            SimpleNamespace(
                nodes=[first_node, second_node],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, -0.1, 0.0, -0.2],
            )
        ],
    )
    turns = VerifiersRolloutWorker.trace_to_rollout_turns(
        trace=trace,
        session=GenerationSession(
            group_id=0,
            min_policy_version=3,
            max_policy_version=8,
            metrics=[],
        ),
        group_id=5,
        rollout_id=2,
    )

    assert [turn.min_policy_version for turn in turns] == [3, 3]
    assert [turn.max_policy_version for turn in turns] == [8, 8]
    # turn 1's prompt [10, 11, 12] continues turn 0's [10] + [11], so it stores only [12]
    stored = [
        (turn.prompt_prefix_len, turn.prompt_delta_token_ids.tolist()) for turn in turns
    ]
    assert stored == [(0, [10]), (2, [12])]
    assert [turn.completion_token_ids.tolist() for turn in turns] == [[11], [13]]
    assert [turn.completion_logprobs.tolist() for turn in turns] == [
        pytest.approx([-0.1]),
        pytest.approx([-0.2]),
    ]


def test_verifiers_second_branch_stores_its_prompt_as_a_delta_on_the_first() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    def node(token_ids: list[int], content: str) -> SimpleNamespace:
        message = VerifiersAssistantMessage(content=content)
        return SimpleNamespace(
            token_ids=token_ids, mask=[False, True], sampled=True, message=message
        )

    shared = node([10, 11], "a")
    first_leaf = node([12, 13], "b")
    second_leaf = node([14, 15], "c")
    trace = SimpleNamespace(
        nodes=[shared, first_leaf, second_leaf],
        branches=[
            SimpleNamespace(
                nodes=[shared, first_leaf],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, -0.1, 0.0, -0.2],
            ),
            SimpleNamespace(
                nodes=[shared, second_leaf],
                token_ids=[10, 11, 14, 15],
                logprobs=[0.0, -0.1, 0.0, -0.3],
            ),
        ],
    )
    turns = VerifiersRolloutWorker.trace_to_rollout_turns(
        trace=trace,
        session=GenerationSession(
            group_id=0, min_policy_version=1, max_policy_version=1, metrics=[]
        ),
        group_id=5,
        rollout_id=2,
    )

    # The shared node trains once; the second branch's prompt [10, 11, 14] keeps [10, 11] of the
    # previous turn's [10, 11, 12, 13], so it stores (2, [14]) and the builder opens a new sample.
    stored = [
        (turn.prompt_prefix_len, turn.prompt_delta_token_ids.tolist()) for turn in turns
    ]
    assert stored == [(0, [10]), (2, [12]), (2, [14])]
    assert [turn.completion_token_ids.tolist() for turn in turns] == [[11], [13], [15]]


def test_verifiers_trace_attaches_env_replies_to_the_preceding_turn() -> None:
    from verifiers.v1.types import (
        AssistantMessage as VerifiersAssistantMessage,
        UserMessage as VerifiersUserMessage,
    )

    def node(content: str, sampled: bool) -> SimpleNamespace:
        message = (
            VerifiersAssistantMessage(content=content)
            if sampled
            else VerifiersUserMessage(content=content)
        )
        return SimpleNamespace(
            token_ids=[0], mask=[sampled], sampled=sampled, message=message
        )

    # Two branches share the prompt and the first command, then diverge.
    task, first, out, second, other_out, other = (
        node("task", False),
        node("ls", True),
        node("a.txt", False),
        node("cat a.txt", True),
        node("No such file", False),
        node("pwd", True),
    )
    trace = SimpleNamespace(
        nodes=[task, first, out, second, other_out, other],
        branches=[
            SimpleNamespace(
                nodes=branch_nodes,
                token_ids=[0] * len(branch_nodes),
                logprobs=[0.0] * len(branch_nodes),
            )
            for branch_nodes in (
                [task, first, out, second],
                [task, first, other_out, other],
            )
        ],
    )
    turns = VerifiersRolloutWorker.trace_to_rollout_turns(
        trace=trace,
        session=GenerationSession(
            group_id=0, min_policy_version=0, max_policy_version=0, metrics=[]
        ),
        group_id=0,
        rollout_id=0,
    )

    assert [turn.completion_message["content"] for turn in turns] == [
        "ls",
        "cat a.txt",
        "pwd",
    ]
    assert [turn.env_messages for turn in turns] == [
        [
            {"role": "user", "content": "a.txt"},
            {"role": "user", "content": "No such file"},
        ],
        [],
        [],
    ]


def test_verifiers_rollout_logs_keep_the_failure_reason(caplog) -> None:
    from verifiers.v1.trace import Error

    def span(seconds: float) -> SimpleNamespace:
        return SimpleNamespace(duration=seconds)

    trace = SimpleNamespace(
        id="trace-1",
        task=SimpleNamespace(key="task-1"),
        stop_condition="error",
        errors=[
            Error(
                type="HarnessError",
                message="agent timeout",
                traceback="Traceback\n  ...\nTimeoutError",
            )
        ],
        timing=SimpleNamespace(
            setup=span(5.0),
            agent=SimpleNamespace(
                duration=7200.0, model=span(6900.0), harness=span(300.0)
            ),
            scoring=span(0.0),
        ),
        calls=[
            SimpleNamespace(time=span(1801.0), error=None),
            SimpleNamespace(
                time=span(2.0), error=Error(type="APIError", message="502")
            ),
        ],
    )
    logs = verifiers_rollout_logs(SimpleNamespace(errors=[]), trace)

    assert logs["errors"] == [
        {
            "type": "HarnessError",
            "message": "agent timeout",
            "status_code": None,
            "traceback": "Traceback\n  ...\nTimeoutError",
        }
    ]
    assert (logs["agent_sec"], logs["model_sec"], logs["harness_sec"]) == (
        7200.0,
        6900.0,
        300.0,
    )
    assert logs["model_calls"] == 2
    assert logs["failed_model_calls"] == 1
    assert logs["slowest_model_call_sec"] == 1801.0

    with caplog.at_level(logging.WARNING):
        log_failed_rollout(logs, group_id=3, rollout_id=1)
        log_failed_rollout({**logs, "errors": []}, group_id=3, rollout_id=2)
    assert "error=HarnessError: agent timeout" in caplog.text
    assert "error=None: None" in caplog.text


def _completion(request_id: str, **fields) -> Completion:
    return Completion(
        min_policy_version=fields.get("min_policy_version", 0),
        max_policy_version=fields.get("max_policy_version", 0),
        request_id=request_id,
        token_ids=[31, 32],
        token_logprobs=[-0.1, -0.2],
        finish_reason=fields.get("finish_reason", "stop"),
        metrics=fields.get("metrics", []),
    )


_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 2,
    "seed": 4,
    "stop_token_ids": [99],
    "top_k": -1,
    "min_p": 0.0,
    "logprobs": 1,
    "skip_special_tokens": False,
    "routed_experts_prompt_start": 41,
}
# Over aiohttp's default 1 MiB request limit even as compact JSON: 7 bytes per 6-digit id.
_LONG_PROMPT = [151644] * 160_000


def test_generation_server_serves_generate_fn_per_session() -> None:
    """Two calls of one session: what `generate_fn` gets, what the renderer client reads back,
    and what the session keeps (versions from 0 count, metrics folded). The second prompt is
    over 1 MiB of JSON."""
    received = []

    async def generate_fn(
        prompt, *, request_id, group_id, routing_session_id, sampling_config
    ):
        received.append(
            (
                len(prompt.token_ids),
                request_id,
                group_id,
                routing_session_id,
                sampling_config.seed,
            )
        )
        call_index = int(request_id.rsplit("=", 1)[1])
        return _completion(
            request_id,
            min_policy_version=call_index,
            max_policy_version=call_index + 1,
            metrics=[m.Metric("generator/queue_time_ms", m.Mean(float(call_index)))],
        )

    async def run() -> tuple[list[dict], GenerationSession]:
        server = GenerationServer(max_rollout_tokens=4096)
        base_url = await server.start(generate_fn)
        server.open_session("trace-1", group_id=7)
        replies = []
        try:
            async with ClientSession() as client:
                for token_ids in ([10, 11], _LONG_PROMPT):
                    response = await client.post(
                        base_url.removesuffix("/v1") + "/inference/v1/generate",
                        headers={"X-Session-ID": "trace-1"},
                        json={
                            "token_ids": token_ids,
                            "sampling_params": _SAMPLING_PARAMS,
                        },
                    )
                    assert response.status == 200
                    replies.append(parse_generate_response(await response.read()))
            return replies, await server.close_session("trace-1")
        finally:
            await server.close()

    replies, session = asyncio.run(run())

    assert received == [
        (2, "trace-1/call=0", 7, "trace-1", 4),
        (len(_LONG_PROMPT), "trace-1/call=1", 7, "trace-1", 4),
    ]
    [choice] = replies[1]["choices"]
    assert choice["token_ids"] == [31, 32]
    assert _parse_completion_logprobs(choice, [31, 32]) == pytest.approx([-0.1, -0.2])
    assert (
        session.num_calls,
        session.min_policy_version,
        session.max_policy_version,
    ) == (2, 0, 2)
    assert m.MetricsProcessor._aggregate_metrics(session.metrics) == {
        "generator/queue_time_ms/mean": 0.5
    }


@pytest.mark.parametrize(
    "session_id, finish_reason",
    [("trace-1", "abort"), ("trace-1", "error"), ("unknown", "stop")],
)
def test_generation_server_answers_500_without_a_usable_completion(
    session_id: str, finish_reason: str
) -> None:
    """500, not 400: the worker reads a 400 as a prompt past the context."""

    async def generate_fn(prompt, *, request_id, **kwargs):
        return _completion(request_id, finish_reason=finish_reason)

    async def run() -> int:
        server = GenerationServer(max_rollout_tokens=4096)
        base_url = await server.start(generate_fn)
        server.open_session("trace-1", group_id=0)
        try:
            async with ClientSession() as client:
                response = await client.post(
                    base_url.removesuffix("/v1") + "/inference/v1/generate",
                    headers={"X-Session-ID": session_id},
                    json={"token_ids": [10], "sampling_params": _SAMPLING_PARAMS},
                )
                return response.status
        finally:
            await server.close()

    assert asyncio.run(run()) == 500


def test_closing_a_session_waits_for_its_call_in_flight() -> None:
    """The agent hangs up mid-call (as at its deadline), but the generator call behind it still
    runs: the session, and so its routing session, closes only after that call returns."""

    async def run() -> int:
        started, release = asyncio.Event(), asyncio.Event()

        async def generate_fn(prompt, *, request_id, **kwargs):
            started.set()
            await release.wait()
            return _completion(request_id)

        server = GenerationServer(max_rollout_tokens=4096)
        base_url = await server.start(generate_fn)
        server.open_session("trace-1", group_id=0)
        try:
            async with ClientSession() as client:
                request = asyncio.create_task(
                    client.post(
                        base_url.removesuffix("/v1") + "/inference/v1/generate",
                        headers={"X-Session-ID": "trace-1"},
                        json={"token_ids": [10], "sampling_params": _SAMPLING_PARAMS},
                    )
                )
                await started.wait()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            close = asyncio.create_task(server.close_session("trace-1"))
            await asyncio.sleep(0.1)
            assert not close.done()
            release.set()
            return (await close).num_calls
        finally:
            await server.close()

    assert asyncio.run(run()) == 1


def test_renderer_refuses_a_prompt_with_no_room_for_an_output_token() -> None:
    """The renderer client reads the cap from the model card, and refuses a longer prompt
    before sending it. Fails if a renderers upgrade stops reading the card."""

    async def run() -> int | None:
        server = GenerationServer(max_rollout_tokens=4096)
        base_url = await server.start(AsyncMock())
        try:
            client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
            return await _resolve_max_prompt_len(client, GenerationServer.MODEL_NAME)
        finally:
            await server.close()

    # vLLM needs room for one output token: a prompt of 4096 tokens is refused.
    assert asyncio.run(run()) == 4095


@pytest.mark.parametrize(
    "update, error",
    [
        ({}, None),
        ({"top_k": 20}, "top_k=20 is unsupported"),
        # An env that asks for top-k logprobs: the renderer client sends k + 2.
        ({"logprobs": 7}, "logprobs=7 is unsupported"),
        (
            {"repetition_penalty": 1.1},
            r"unsupported sampling parameters: \['repetition_penalty'\]",
        ),
    ],
)
def test_sampling_config_takes_only_what_the_generator_honors(update, error) -> None:
    sampling_params = {**_SAMPLING_PARAMS, **update}
    if error is not None:
        with pytest.raises(ValueError, match=error):
            _sampling_config(sampling_params)
        return
    assert _sampling_config(sampling_params) == SamplingConfig(
        temperature=1.0, top_p=1.0, max_tokens=2, seed=4, stop_token_ids=[99]
    )


def _env_running(*traces, minted=None) -> SimpleNamespace:
    """A fake Verifiers env whose `run_slot` mints `minted` (default: `traces`) and returns an
    ok episode of `traces`."""

    async def run_slot(slot, ctx, *, on_trace):
        for trace in traces if minted is None else minted:
            on_trace(trace)
        return SimpleNamespace(
            ok=True, is_timeout=False, errors=[], traces=list(traces), last_error=None
        )

    return SimpleNamespace(slots=lambda task: [None], run_slot=run_slot)


def test_rollout_whose_first_prompt_does_not_fit_is_flagged(caplog) -> None:
    # The first model call fails for context length, so the rollout has no turns.
    span = SimpleNamespace(duration=0.0)
    trace = SimpleNamespace(
        id="trace",
        task=SimpleNamespace(key="huge-task"),
        agent=SimpleNamespace(trainable=True),
        ok=True,
        is_truncated=False,
        is_timeout=False,
        stop_condition="agent_completed",
        errors=[],
        timing=SimpleNamespace(
            setup=span,
            agent=SimpleNamespace(duration=0.0, model=span, harness=span),
            scoring=span,
        ),
        calls=[
            SimpleNamespace(
                time=span,
                error=SimpleNamespace(
                    message="Prompt length (9) exceeds maximum context length (8)."
                ),
            )
        ],
        nodes=[],
        branches=[],
    )
    worker = object.__new__(VerifiersRolloutWorker)
    worker._generation_server = SimpleNamespace(
        open_session=lambda session_id, group_id: None,
        close_session=AsyncMock(return_value=GenerationSession(group_id=1)),
    )
    worker._client_config = object()
    worker._release_session_fn = AsyncMock()

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            worker._run_episode(
                env=_env_running(trace),
                task=None,
                sampling=SamplingConfig(),
                group_id=1,
                episode_index=2,
            )
        )

    assert rollout.status is RolloutStatus.ERROR
    assert "first prompt of task huge-task does not fit" in caplog.text


@pytest.mark.parametrize(
    "ok, is_timeout, episode_is_timeout, expected",
    [
        (True, True, False, RolloutStatus.ERROR_TIMEOUT),
        (False, True, False, RolloutStatus.ERROR_TIMEOUT),
        # The episode's own deadline flags the episode and leaves it not ok.
        (True, False, True, RolloutStatus.ERROR_TIMEOUT),
        (False, False, False, RolloutStatus.ERROR),
        (True, False, False, RolloutStatus.COMPLETED),
    ],
)
def test_rollout_status_maps_timeouts_to_error_timeout(
    ok: bool, is_timeout: bool, episode_is_timeout: bool, expected: RolloutStatus
) -> None:
    trace = SimpleNamespace(ok=ok, is_timeout=is_timeout, is_truncated=False)
    episode = SimpleNamespace(ok=not episode_is_timeout, is_timeout=episode_is_timeout)

    status = VerifiersRolloutWorker.rollout_status(
        verifiers_episode=episode, trace=trace
    )

    assert status is expected


def test_timed_out_rollout_is_logged_and_keeps_no_reward(caplog) -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    # The agent stage timed out, but Verifiers kept ok=True and still graded the reply.
    node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="answer"),
    )
    span = SimpleNamespace(duration=0.0)
    trace = SimpleNamespace(
        id="trace",
        task=SimpleNamespace(key="slow-task"),
        agent=SimpleNamespace(trainable=True),
        ok=True,
        is_timeout=True,
        is_truncated=False,
        stop_condition="agent_timeout",
        reward=1.0,
        errors=[],
        timing=SimpleNamespace(
            setup=span,
            agent=SimpleNamespace(duration=0.0, model=span, harness=span),
            scoring=span,
        ),
        calls=[],
        nodes=[node],
        branches=[
            SimpleNamespace(nodes=[node], token_ids=[10, 11], logprobs=[0.0, -0.1])
        ],
    )
    worker = object.__new__(VerifiersRolloutWorker)
    worker._generation_server = SimpleNamespace(
        open_session=lambda session_id, group_id: None,
        close_session=AsyncMock(
            return_value=GenerationSession(
                group_id=1, min_policy_version=0, max_policy_version=0
            )
        ),
    )
    worker._client_config = object()
    worker._release_session_fn = AsyncMock()

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            worker._run_episode(
                env=_env_running(trace),
                task=None,
                sampling=SamplingConfig(),
                group_id=1,
                episode_index=2,
            )
        )

    assert rollout.status is RolloutStatus.ERROR_TIMEOUT
    assert "Verifiers rollout failed: trace=trace" in caplog.text
    assert VERIFIERS_REWARD_KEY not in rollout.turns[-1].env_rewards


def test_rollout_releases_every_session_verifiers_minted() -> None:
    """A retried attempt's trace and an untrained agent's trace made model calls too."""
    span = SimpleNamespace(duration=0.0)
    traces = [
        SimpleNamespace(
            id=trace_id,
            task=SimpleNamespace(key="task"),
            agent=SimpleNamespace(trainable=trainable),
            ok=True,
            is_timeout=False,
            is_truncated=False,
            stop_condition=None,
            errors=[],
            timing=SimpleNamespace(
                setup=span,
                agent=SimpleNamespace(duration=0.0, model=span, harness=span),
                scoring=span,
            ),
            calls=[],
            nodes=[],
            branches=[],
        )
        for trace_id, trainable in [
            ("retried", True),
            ("opponent", False),
            ("policy", True),
        ]
    ]
    worker = object.__new__(VerifiersRolloutWorker)
    opened_session_ids, closed_session_ids = [], []

    async def close_session(session_id: str) -> GenerationSession:
        closed_session_ids.append(session_id)
        return GenerationSession(group_id=1)

    worker._generation_server = SimpleNamespace(
        open_session=lambda session_id, group_id: opened_session_ids.append(session_id),
        close_session=close_session,
    )
    worker._client_config = object()
    release_session_fn = AsyncMock()
    worker._release_session_fn = release_session_fn

    # The retried attempt's trace is minted, then left out of the episode.
    rollout = asyncio.run(
        worker._run_episode(
            env=_env_running(*traces[1:], minted=traces),
            task=None,
            sampling=SamplingConfig(),
            group_id=1,
            episode_index=0,
        )
    )

    assert [call.kwargs for call in release_session_fn.call_args_list] == [
        {"routing_session_id": "retried"},
        {"routing_session_id": "opponent"},
        {"routing_session_id": "policy"},
    ]
    assert opened_session_ids == closed_session_ids == ["retried", "opponent", "policy"]
    assert rollout.group_id == 1


def _worker(monkeypatch, load_environment) -> VerifiersRolloutWorker:
    monkeypatch.setattr(verifiers_rollouter.vf, "load_environment", load_environment)
    monkeypatch.setattr(verifiers_rollouter.vf, "resolve_env_config", lambda data: data)
    worker = VerifiersRolloutWorker.Config(
        rubric=Rubric.Config(reward_fns=[RewardFromVerifiers.Config()]),
        max_rollout_tokens=4096,
    ).build()
    worker._release_session_fn = AsyncMock()
    worker._client_config = vf.TrainClientConfig(
        base_url="http://127.0.0.1:1/v1", api_key_var=""
    )
    return worker


def test_worker_serves_one_env_per_env_config_until_close(monkeypatch) -> None:
    events = []

    def load_environment(env_config: dict) -> SimpleNamespace:
        @contextlib.asynccontextmanager
        async def serving():
            events.append(f"serving {env_config['name']}")
            yield
            events.append(f"stopped {env_config['name']}")

        return SimpleNamespace(serving=serving)

    worker = _worker(monkeypatch, load_environment)

    async def run() -> None:
        train, validation = (
            json.dumps({"name": name}) for name in ("train", "validation")
        )
        first = await worker._serving_env(train)
        assert await worker._serving_env(train) is first
        await worker._serving_env(validation)
        await worker.close()

    asyncio.run(run())
    assert events == [
        "serving train",
        "serving validation",
        "stopped validation",
        "stopped train",
    ]


def test_a_second_close_returns_while_the_first_still_stops_an_env(monkeypatch) -> None:
    async def run() -> None:
        env_stopping, release = asyncio.Event(), asyncio.Event()

        def load_environment(env_config: dict) -> SimpleNamespace:
            @contextlib.asynccontextmanager
            async def serving():
                yield
                env_stopping.set()
                await release.wait()

            return SimpleNamespace(serving=serving)

        worker = _worker(monkeypatch, load_environment)
        await worker._serving_env(json.dumps({"name": "train"}))
        first_close = asyncio.create_task(worker.close())
        await env_stopping.wait()
        # Like `__cleanup__` after the Rollouter's close timed out: it must not wait for the first.
        await asyncio.wait_for(worker.close(), timeout=1)
        release.set()
        await first_close

    asyncio.run(run())


class _Task:
    """A Verifiers task class: built from the task's data and the taskset's task config."""

    def __init__(self, data, config) -> None:
        del data, config

    @staticmethod
    def data_type() -> SimpleNamespace:
        return SimpleNamespace(model_validate=lambda data: data)


def test_close_cancels_the_running_rollouts_before_it_stops_the_envs(
    monkeypatch,
) -> None:
    """A rollout's `run_slot` owns its sandbox, so cancelling the rollout stops the sandbox."""
    events = []

    async def run_slot(slot, ctx, *, on_trace):
        try:
            events.append("rollout started")
            await asyncio.Event().wait()
        finally:
            events.append("rollout stopped")

    def load_environment(env_config: dict) -> SimpleNamespace:
        @contextlib.asynccontextmanager
        async def serving():
            yield
            events.append("env stopped")

        return SimpleNamespace(
            serving=serving,
            taskset=SimpleNamespace(
                task_type=lambda: _Task, config=SimpleNamespace(task=None)
            ),
            slots=lambda task: [None],
            run_slot=run_slot,
        )

    worker = _worker(monkeypatch, load_environment)

    async def run() -> None:
        group = asyncio.create_task(
            worker.run_group(
                sample=VerifiersTaskSample(env_config_json="{}", task_data={}),
                group_id=0,
                group_size=2,
                sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=8),
            )
        )
        while events.count("rollout started") < 2:
            await asyncio.sleep(0)
        await worker.close()
        await asyncio.gather(group, return_exceptions=True)

    asyncio.run(run())
    assert events[2:] == ["rollout stopped", "rollout stopped", "env stopped"]


def test_a_failing_rollout_is_error_infra_once_every_sibling_finished(
    monkeypatch,
) -> None:
    """The controller frees the group once `run_group` returns; no sibling may still be generating."""
    finished = []

    async def run_episode(*, episode_index: int, **kwargs) -> Rollout:
        if episode_index == 0:
            raise RuntimeError("rollout 0 failed")
        await asyncio.sleep(0.01)
        finished.append(episode_index)
        return Rollout(
            group_id=1, rollout_id=episode_index, status=RolloutStatus.COMPLETED
        )

    def load_environment(env_config: dict) -> SimpleNamespace:
        return SimpleNamespace(
            serving=contextlib.nullcontext,
            taskset=SimpleNamespace(
                task_type=lambda: _Task, config=SimpleNamespace(task=None)
            ),
        )

    worker = _worker(monkeypatch, load_environment)
    worker._run_episode = run_episode
    group = asyncio.run(
        worker.run_group(
            sample=VerifiersTaskSample(env_config_json="{}", task_data={}),
            group_id=1,
            group_size=2,
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=8),
        )
    )

    assert finished == [1]
    assert [rollout.status for rollout in group.rollouts] == [
        RolloutStatus.ERROR_INFRA,
        RolloutStatus.COMPLETED,
    ]
    # `RewardFromVerifiers` gives a rollout with no turns 0.0.
    assert [rollout.reward for rollout in group.rollouts] == [None, 0.0]
