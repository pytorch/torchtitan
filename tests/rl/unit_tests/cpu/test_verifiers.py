# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("verifiers")

from aiohttp import ClientSession

from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    _parse_sampling_config,
    _validate_token_ids,
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.rollouter import (
    _trainable_token_spans,
    VERIFIERS_REWARD_KEY,
    verifiers_rollout_logs,
    VerifiersRolloutWorker,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import log_failed_rollout
from torchtitan.rl.rollout.types import Completion, Rollout, RolloutStatus
from verifiers.v1.trace import Error as VerifiersError


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


def test_setup_async_builds_the_train_client_and_bypasses_the_proxy(
    monkeypatch,
) -> None:
    from renderers import CustomRendererConfig
    from torchtitan.components.renderer import from_renderers
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    # `setup_async` edits both proxy variables; monkeypatch restores them after the test.
    monkeypatch.setenv("no_proxy", ".corp.internal")
    monkeypatch.setenv("NO_PROXY", "metadata.internal")
    serving = []

    @contextlib.asynccontextmanager
    async def serving_env():
        serving.append("entered")
        yield

    worker = object.__new__(VerifiersRolloutWorker)
    worker._config = SimpleNamespace(renderer_multiplex=8)
    worker._generation_server = SimpleNamespace(
        start=AsyncMock(), base_url="http://127.0.0.1:2/v1"
    )
    worker._verifiers_env = SimpleNamespace(serving=serving_env)
    worker._serving = contextlib.AsyncExitStack()
    renderers_config = CustomRendererConfig(
        import_path="torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer",
        reasoning_strength="low",
    )
    config = from_renderers(renderers_config)

    asyncio.run(
        worker.setup_async(
            tokenizer_config=cast(HuggingFaceTokenizer.Config, None),
            renderer_config=config,
            hf_assets_path="muse-checkpoint",
        )
    )

    sent = worker._train_client_config
    # The worker's agents call its own generation server.
    assert sent.base_url == "http://127.0.0.1:2/v1"
    assert sent.renderer == renderers_config
    assert sent.renderer_model_name == "muse-checkpoint"
    assert serving == ["entered"]
    # Both spellings keep their entries, and Verifiers' calls to 127.0.0.1 skip the proxy.
    expected = ".corp.internal,metadata.internal,127.0.0.1,localhost"
    assert os.environ["no_proxy"] == os.environ["NO_PROXY"] == expected


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
        generation_metadata=VerifiersGenerationMetadata(
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
        generation_metadata=VerifiersGenerationMetadata(
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
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=1, max_policy_version=1, metrics=[]
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
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
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
        for rollout_id, status, rollout_logs in [
            (1, RolloutStatus.ERROR, logs),
            (2, RolloutStatus.FIRST_PROMPT_TOO_LONG, {**logs, "errors": []}),
            (3, RolloutStatus.COMPLETED, logs),  # didn't fail: no log
        ]:
            log_failed_rollout(
                Rollout(
                    group_id=3, rollout_id=rollout_id, status=status, logs=rollout_logs
                )
            )
    assert (
        "rollout group=3/rollout=1 failed: ERROR | HarnessError: agent timeout | "
        "Traceback |   ... | TimeoutError | verifiers_trace_id=" in caplog.text
    )
    assert (
        "rollout group=3/rollout=2 failed: FIRST_PROMPT_TOO_LONG | verifiers_trace_id="
        in caplog.text
    )
    assert "rollout=3" not in caplog.text


# Over aiohttp's default 1 MiB request limit even as compact JSON: 7 bytes per 6-digit id.
_LONG_PROMPT = [151644] * 160_000


def test_generation_server_forwards_token_request() -> None:
    """Two calls of one session; the second prompt is over 1 MiB of JSON."""

    async def run_test() -> None:
        received = []

        async def generate_fn(
            prompt,
            *,
            request_id,
            group_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            received.append(
                {
                    "prompt_token_ids": prompt.token_ids,
                    "request_id": request_id,
                    "group_id": group_id,
                    "routing_session_id": routing_session_id,
                    "sampling_config": sampling_config,
                }
            )
            request_index = int(request_id.rsplit("=", 1)[1])
            return Completion(
                min_policy_version=7 - request_index,
                max_policy_version=8 + request_index,
                request_id=request_id,
                token_ids=[31, 32],
                token_logprobs=[-0.1, -0.2],
                finish_reason="stop",
                metrics=[
                    m.Metric("generator/queue_time_ms", m.Mean(float(request_index))),
                    m.Metric("generator/queue_time_ms", m.Max(float(request_index))),
                ],
            )

        server = GenerationServer.Config(max_rollout_tokens=262144).build()
        server.set_generate_fn(generate_fn)
        server.start_session(session_id="group=1/rollout=2", group_id=1)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.get(f"{server.base_url}/models")
                assert response.status == 200
                assert await response.json() == {
                    "object": "list",
                    "data": [
                        {
                            "id": "torchtitan",
                            "object": "model",
                            "created": 0,
                            "owned_by": "torchtitan",
                            "max_model_len": 262144,
                        }
                    ],
                }
                for token_ids in ([10, 11], _LONG_PROMPT):
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={
                            "token_ids": token_ids,
                            "sampling_params": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": 2,
                                "seed": 4,
                                "logprobs": 1,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    assert response.status == 200
                    payload = await response.json()
            generation_metadata = server.end_session("group=1/rollout=2")
        finally:
            await server.close()

        assert [request["request_id"] for request in received] == [
            "group=1/rollout=2/request=0",
            "group=1/rollout=2/request=1",
        ]
        assert [len(request["prompt_token_ids"]) for request in received] == [
            2,
            len(_LONG_PROMPT),
        ]
        assert all(request["group_id"] == 1 for request in received)
        assert all(
            request["routing_session_id"] == "group=1/rollout=2" for request in received
        )
        assert all(request["sampling_config"].seed == 4 for request in received)
        assert payload == {
            "request_id": "group=1/rollout=2/request=1",
            "choices": [
                {
                    "index": 0,
                    "token_ids": [31, 32],
                    "logprobs": {
                        "content": [
                            {"token": "token_id:31", "logprob": -0.1},
                            {"token": "token_id:32", "logprob": -0.2},
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
            "prompt_logprobs": None,
            "kv_transfer_params": None,
        }
        assert generation_metadata is not None
        assert generation_metadata.min_policy_version == 6
        assert generation_metadata.max_policy_version == 9
        # Both generations folded into one Mean and one Max
        assert len(generation_metadata.metrics) == 2
        assert m.MetricsProcessor._aggregate_metrics(generation_metadata.metrics) == {
            "generator/queue_time_ms/mean": 0.5,
            "generator/queue_time_ms/max": 1.0,
        }

    asyncio.run(run_test())


def test_generation_server_rejects_aborted_generation() -> None:
    async def run_test() -> None:
        async def generate_fn(
            prompt,
            *,
            request_id,
            group_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            return Completion(
                min_policy_version=7,
                max_policy_version=7,
                request_id=request_id,
                token_ids=[],
                token_logprobs=[],
                finish_reason="abort",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        server.start_session(session_id="group=1/rollout=2", group_id=1)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "max_tokens": 2,
                            "stop_token_ids": [99],
                        },
                    },
                )
                assert response.status == 502
                payload = await response.json()
            generation_metadata = server.end_session("group=1/rollout=2")
        finally:
            await server.close()

        assert payload == {
            "error": "generation finished without a usable completion: abort"
        }
        assert generation_metadata is None

    asyncio.run(run_test())


_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 2,
    "stop_token_ids": [99],
}


def test_generation_server_refuses_a_session_with_no_group() -> None:
    """422 for a session `start_session` never opened, and for one `end_session` closed."""

    async def run_test() -> list[int]:
        async def generate_fn(*args, **kwargs):
            raise AssertionError("generate_fn must not run for an unknown session")

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        server.start_session(session_id="ended", group_id=1)
        server.end_session("ended")
        await server.start()
        statuses = []
        try:
            async with ClientSession() as session:
                for session_id in ("never-started", "ended"):
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": session_id},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": _SAMPLING_PARAMS,
                        },
                    )
                    statuses.append(response.status)
                    assert "has no rollout group" in (await response.json())["error"]
        finally:
            await server.close()
        return statuses

    assert asyncio.run(run_test()) == [422, 422]


def test_a_call_that_ends_after_end_session_leaves_nothing_behind() -> None:
    """Verifiers gives up on a call, and the worker ends its session while it still generates."""

    async def run_test() -> tuple[dict, dict]:
        generating = asyncio.Event()
        finish = asyncio.Event()

        async def generate_fn(prompt, *, request_id, group_id, **kwargs):
            generating.set()
            await finish.wait()
            return Completion(
                min_policy_version=0,
                max_policy_version=0,
                request_id=request_id,
                token_ids=[31],
                token_logprobs=[-0.1],
                finish_reason="stop",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        server.start_session(session_id="s1", group_id=7)
        await server.start()
        try:
            async with ClientSession() as session:
                post = asyncio.create_task(
                    session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "s1"},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": _SAMPLING_PARAMS,
                        },
                    )
                )
                await generating.wait()
                server.end_session("s1")
                finish.set()
                await post
            # Read before `close`, which clears both.
            return dict(server.generation_metadata), dict(server.request_counts)
        finally:
            await server.close()

    assert asyncio.run(run_test()) == ({}, {})


def test_concurrent_verifiers_groups_keep_their_session_groups() -> None:
    """Two groups' sessions call one server at once; each call reaches `generate_fn` with its own
    session's group."""

    async def run_test() -> list[tuple[str, int]]:
        both_called = asyncio.Event()
        received = []

        async def generate_fn(prompt, *, request_id, group_id, **kwargs):
            received.append((kwargs["routing_session_id"], group_id))
            if len(received) == 4:
                both_called.set()
            # Hold every call open until all four arrived, so they interleave.
            await asyncio.wait_for(both_called.wait(), timeout=5)
            return Completion(
                min_policy_version=0,
                max_policy_version=0,
                request_id=request_id,
                token_ids=[31],
                token_logprobs=[-0.1],
                finish_reason="stop",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        sessions = {"7-a": 7, "7-b": 7, "8-a": 8, "8-b": 8}
        for session_id, group_id in sessions.items():
            server.start_session(session_id=session_id, group_id=group_id)
        await server.start()
        try:
            async with ClientSession() as session:

                async def post(session_id: str) -> int:
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": session_id},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": _SAMPLING_PARAMS,
                        },
                    )
                    return response.status

                statuses = await asyncio.gather(
                    *(post(session_id) for session_id in sessions)
                )
        finally:
            await server.close()
        assert statuses == [200, 200, 200, 200]
        return sorted(received)

    assert asyncio.run(run_test()) == [("7-a", 7), ("7-b", 7), ("8-a", 8), ("8-b", 8)]


@pytest.mark.parametrize(
    "headers, request_kwargs",
    [
        pytest.param(
            {},
            {"json": {"token_ids": [10, 11], "sampling_params": _SAMPLING_PARAMS}},
            id="no_session_header",
        ),
        pytest.param(
            {"X-Session-ID": "group=1/rollout=2"},
            {"data": b"not json"},
            id="body_not_json",
        ),
        pytest.param(
            {"X-Session-ID": "group=1/rollout=2"},
            {"json": {"token_ids": "10 11", "sampling_params": _SAMPLING_PARAMS}},
            id="token_ids_not_a_list",
        ),
        pytest.param(
            {"X-Session-ID": "group=1/rollout=2"},
            {
                "json": {
                    "token_ids": [10, 11],
                    "sampling_params": {**_SAMPLING_PARAMS, "top_k": 20},
                }
            },
            id="sampling_key_that_filters",
        ),
    ],
)
def test_generation_server_answers_422_to_a_bad_request(
    headers: dict, request_kwargs: dict
) -> None:
    """A bad request is a config or code mistake: 422. Only a prompt too long answers 400, the
    one status the Verifiers worker maps to a prompt-too-long status."""

    async def run_test() -> int:
        async def generate_fn(*args, **kwargs):
            raise AssertionError("generate_fn must not run for a bad request")

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        server.start_session(session_id="group=1/rollout=2", group_id=1)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers=headers,
                    **request_kwargs,
                )
                return response.status
        finally:
            await server.close()

    assert asyncio.run(run_test()) == 422


def test_generation_server_logs_each_request_at_debug_only(caplog) -> None:
    async def run_test() -> None:
        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.get(
                    f"http://{server.host}:{server.port}/healthz"
                )
                assert response.status == 200
        finally:
            await server.close()

    # One DEBUG record: no INFO line, and still logged. Set the level explicitly: imports can
    # change the root level.
    with caplog.at_level(logging.DEBUG, logger="aiohttp.access"):
        asyncio.run(run_test())
    access_records = [r for r in caplog.records if r.name == "aiohttp.access"]
    assert [r.levelno for r in access_records] == [logging.DEBUG]
    assert access_records[0].getMessage().startswith("GET /healthz 200 ")


def test_generation_server_rejects_prompt_without_room_for_output() -> None:
    async def run_test() -> None:
        prompt_lengths = []

        async def generate_fn(prompt, **kwargs):
            prompt_lengths.append(len(prompt.token_ids))
            raise RuntimeError("generator reached")

        server = GenerationServer.Config(max_rollout_tokens=8).build()
        server.set_generate_fn(generate_fn)
        server.start_session(session_id="group=1/rollout=2", group_id=1)
        await server.start()
        try:
            async with ClientSession() as session:
                statuses, errors = [], []
                for num_tokens in (7, 8, 9):
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={
                            "token_ids": [1] * num_tokens,
                            "sampling_params": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": 2,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    statuses.append(response.status)
                    errors.append((await response.json())["error"])
        finally:
            await server.close()

        assert prompt_lengths == [7]
        assert statuses == [500, 400, 400]
        # vLLM's own wording, which Verifiers matches as a context overflow.
        assert "maximum context length" in errors[1]

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "token_ids, accepted",
    [
        ([], True),
        ([0, 151643], True),
        ([1, True], False),
        ([1, 2.0], False),
        ([[1]], False),
        ([1, None], False),
        ((1, 2), False),
    ],
)
def test_validate_token_ids(token_ids, accepted) -> None:
    if accepted:
        assert _validate_token_ids(token_ids, field_name="token_ids") == token_ids
        return
    with pytest.raises(
        ValueError, match="token_ids must be a list of integer token IDs"
    ):
        _validate_token_ids(token_ids, field_name="token_ids")


class _FakeTask:
    """A task that keeps the sample's data as is."""

    def __init__(self, *, data, config) -> None:
        self.data = data

    @staticmethod
    def data_type():
        return SimpleNamespace(model_validate=lambda data: data)


def _worker_with_env(*, run_slot, generation_server=None) -> VerifiersRolloutWorker:
    """A worker whose Verifiers env runs each episode with `run_slot`."""
    worker = object.__new__(VerifiersRolloutWorker)
    worker._verifiers_env = SimpleNamespace(
        taskset=SimpleNamespace(task_type=lambda: _FakeTask),
        config=SimpleNamespace(taskset=SimpleNamespace(task=None)),
        slots=lambda task, n: [SimpleNamespace(task=task) for _ in range(n)],
        run_slot=run_slot,
    )
    worker._generation_server = generation_server or SimpleNamespace(
        model_id="model",
        start_session=lambda *, session_id, group_id: None,
        end_session=lambda session_id: VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
        ),
    )
    worker._train_client_config = object()
    worker._running_rollouts = set()
    return worker


def _trace(*, trace_id="trace", trainable=True, calls=(), **fields) -> SimpleNamespace:
    """A Verifiers trace that finished cleanly, with no turns; `fields` override its own."""
    span = SimpleNamespace(duration=0.0)
    trace = SimpleNamespace(
        id=trace_id,
        task=SimpleNamespace(key="task"),
        agent=SimpleNamespace(trainable=trainable),
        ok=True,
        is_truncated=False,
        is_timeout=False,
        stop_condition="agent_completed",
        reward=0.0,
        rewards={},
        metrics={},
        errors=[],
        last_error=None,
        timing=SimpleNamespace(
            setup=span,
            agent=SimpleNamespace(duration=0.0, model=span, harness=span),
            scoring=span,
        ),
        calls=list(calls),
        nodes=[],
        branches=[],
        num_turns=0,
    )
    for name, value in fields.items():
        setattr(trace, name, value)
    return trace


def _episode(*traces) -> SimpleNamespace:
    return SimpleNamespace(ok=True, is_timeout=False, errors=[], traces=list(traces))


def _run_slot_minting(*traces):
    """A `run_slot` that mints each trace, as Verifiers does, then returns them as the episode."""

    async def run_slot(*, slot, ctx, on_trace):
        for trace in traces:
            on_trace(trace)
        return _episode(*traces)

    return run_slot


def _run_single_rollout(worker, *, release_session_fn=None) -> Rollout:
    return asyncio.run(
        worker._run_single_rollout(
            release_session_fn=release_session_fn or AsyncMock(),
            slot=object(),
            sampling=SamplingConfig(),
            group_id=1,
            rollout_id=2,
        )
    )


_OVERFLOW_CALL = SimpleNamespace(
    time=SimpleNamespace(duration=0.0),
    error=SimpleNamespace(
        message="Prompt length (9) exceeds maximum context length (8).", status_code=400
    ),
)
_SANDBOX_ERROR = VerifiersError(type="SandboxError", message="no capacity")
_HARNESS_ERROR = VerifiersError(type="HarnessError", message="exited with no output")


def _one_turn_fields(*, trainable: bool = True) -> dict:
    """Trace fields for one policy turn: prompt token 10, sampled token 11 (trainable unless
    `trainable=False`)."""
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, trainable],
        sampled=True,
        message=VerifiersAssistantMessage(content="ls"),
    )
    branch = SimpleNamespace(nodes=[node], token_ids=[10, 11], logprobs=[0.0, -0.1])
    return {"nodes": [node], "branches": [branch], "num_turns": 1}


@pytest.mark.parametrize(
    "trace_fields, expected_status, expected_log",
    [
        # The first model call overflows the context, and the harness ends the trace cleanly.
        pytest.param(
            {"calls": [_OVERFLOW_CALL]},
            RolloutStatus.FIRST_PROMPT_TOO_LONG,
            "rollout group=1/rollout=2 failed: FIRST_PROMPT_TOO_LONG | verifiers_trace_id=trace task=huge-task",
            id="first_prompt_too_long",
        ),
        # The sandbox never started.
        pytest.param(
            {"ok": False, "errors": [_SANDBOX_ERROR], "last_error": _SANDBOX_ERROR},
            RolloutStatus.ERROR_INFRA,
            "failed: ERROR_INFRA | SandboxError: no capacity",
            id="sandbox_error",
        ),
        # The harness failed to launch: the policy never answered, so it can't be its failure.
        pytest.param(
            {"ok": False, "errors": [_HARNESS_ERROR], "last_error": _HARNESS_ERROR},
            RolloutStatus.ERROR_INFRA,
            "failed: ERROR_INFRA | HarnessError: exited with no output",
            id="harness_error",
        ),
        # The same HarnessError after one turn: ERROR, and it trains.
        pytest.param(
            {
                "ok": False,
                "errors": [_HARNESS_ERROR],
                "last_error": _HARNESS_ERROR,
                **_one_turn_fields(),
            },
            RolloutStatus.ERROR,
            "failed: ERROR | HarnessError: exited with no output",
            id="harness_error_after_a_turn",
        ),
        # The policy answered, but none of its tokens is trainable, so the rollout has no turn.
        pytest.param(
            _one_turn_fields(trainable=False),
            RolloutStatus.ERROR_INFRA,
            "failed: ERROR_INFRA | no_trainable_token: trace: no model call has a trainable token",
            id="no_trainable_token",
        ),
    ],
)
def test_a_rollout_with_no_turn_never_trains_and_each_failure_is_logged(
    trace_fields: dict, expected_status: RolloutStatus, expected_log: str, caplog
) -> None:
    trace = _trace(task=SimpleNamespace(key="huge-task"), **trace_fields)
    worker = _worker_with_env(run_slot=_run_slot_minting(trace))

    with caplog.at_level(logging.WARNING):
        rollout = _run_single_rollout(worker)

    assert rollout.status is expected_status
    # In these cases, a rollout keeps a turn if and only if it trains.
    assert (len(rollout.turns) > 0) == expected_status.is_trainable()
    assert expected_log in caplog.text


def test_a_rollout_whose_verifiers_episode_raises_is_error_infra_and_logged(
    caplog,
) -> None:
    worker = _worker_with_env(
        run_slot=AsyncMock(side_effect=ConnectionError("sandbox API unreachable"))
    )

    with caplog.at_level(logging.WARNING):
        rollout = _run_single_rollout(worker)

    assert rollout.status is RolloutStatus.ERROR_INFRA
    assert [error["type"] for error in rollout.logs["errors"]] == ["ConnectionError"]
    assert (
        "rollout group=1/rollout=2 failed: ERROR_INFRA | ConnectionError: sandbox API unreachable"
        in caplog.text
    )


def _error(error_type: str, status_code: int | None = None) -> VerifiersError:
    return VerifiersError(type=error_type, message="", status_code=status_code)


# A model call refused with 400: its prompt reached `max_rollout_tokens`.
_REFUSED_CALL = SimpleNamespace(error=_error("ProviderError", 400))


@pytest.mark.parametrize(
    "trace_fields, episode_fields, expected",
    [
        pytest.param(
            {"is_timeout": True, "stop_condition": "setup_timeout", "num_turns": 0},
            {},
            RolloutStatus.ERROR_INFRA,
            id="setup_timeout",
        ),
        pytest.param(
            {"is_timeout": True, "stop_condition": "agent_timeout"},
            {},
            RolloutStatus.ERROR_TIMEOUT,
            id="agent_timeout",
        ),
        # A stage deadline records no error and keeps ok=True.
        pytest.param(
            {"is_timeout": True, "stop_condition": "scoring_timeout"},
            {},
            RolloutStatus.ERROR_TIMEOUT,
            id="scoring_timeout",
        ),
        # The episode's own deadline flags the episode and leaves it not ok.
        pytest.param(
            {},
            {"ok": False, "is_timeout": True},
            RolloutStatus.ERROR_TIMEOUT,
            id="episode_timeout",
        ),
        pytest.param(
            {"ok": False, "last_error": _error("HarnessError")},
            {"ok": False},
            RolloutStatus.ERROR,
            id="harness_error",
        ),
        pytest.param(
            {"ok": False, "last_error": _error("TaskError")},
            {"ok": False},
            RolloutStatus.ERROR,
            id="task_error",
        ),
        # The first call refused: the policy never answered, whatever failed later.
        pytest.param(
            {
                "calls": [_REFUSED_CALL],
                "num_turns": 0,
                "ok": False,
                "last_error": _error("ProviderError", 400),
            },
            {"ok": False},
            RolloutStatus.FIRST_PROMPT_TOO_LONG,
            id="first_prompt_too_long",
        ),
        pytest.param(
            {
                "calls": [_REFUSED_CALL],
                "num_turns": 0,
                "is_timeout": True,
                "stop_condition": "agent_timeout",
            },
            {},
            RolloutStatus.FIRST_PROMPT_TOO_LONG,
            id="first_prompt_too_long_then_deadline",
        ),
        # A later call refused, and the trace ended on it: the harness died on it, or stopped cleanly.
        pytest.param(
            {
                "calls": [_REFUSED_CALL],
                "ok": False,
                "last_error": _error("ProviderError", 400),
            },
            {"ok": False},
            RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
            id="later_prompt_too_long",
        ),
        pytest.param(
            {"calls": [_REFUSED_CALL]},
            {},
            RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
            id="later_prompt_too_long_clean_exit",
        ),
        # A later call refused, then something else ended the trace: that decides.
        pytest.param(
            {
                "calls": [_REFUSED_CALL],
                "is_timeout": True,
                "stop_condition": "agent_timeout",
            },
            {},
            RolloutStatus.ERROR_TIMEOUT,
            id="refused_then_deadline",
        ),
        pytest.param(
            {"calls": [_REFUSED_CALL]},
            {"ok": False},
            RolloutStatus.ERROR_INFRA,
            id="refused_then_episode_failed",
        ),
        pytest.param(
            {"ok": False, "last_error": _error("ProviderError", 422)},
            {"ok": False},
            RolloutStatus.ERROR_INFRA,
            id="malformed_request",
        ),
        pytest.param(
            {"ok": False, "last_error": _error("ProviderError", 502)},
            {"ok": False},
            RolloutStatus.ERROR_INFRA,
            id="bad_completion",
        ),
        pytest.param(
            {"ok": False, "last_error": _error("SandboxError")},
            {"ok": False},
            RolloutStatus.ERROR_INFRA,
            id="sandbox_error",
        ),
        # Another agent's trace, or an env hook, failed; the policy's trace finished.
        pytest.param({}, {"ok": False}, RolloutStatus.ERROR_INFRA, id="episode_failed"),
        pytest.param(
            {"is_truncated": True, "stop_condition": "max_turns"},
            {},
            RolloutStatus.TRUNCATED_MAX_TURNS,
            id="max_turns",
        ),
        pytest.param(
            {
                "is_truncated": True,
                "calls": [SimpleNamespace(error=None, finish_reason="length")],
            },
            {},
            RolloutStatus.TRUNCATED_LENGTH,
            id="length",
        ),
        # Verifiers' token budgets for the whole rollout.
        *(
            pytest.param(
                {"is_truncated": True, "stop_condition": budget},
                {},
                RolloutStatus.TRUNCATED_LENGTH,
                id=budget,
            )
            for budget in ("max_input_tokens", "max_output_tokens", "max_total_tokens")
        ),
        # A truncation this mapping doesn't know doesn't train.
        pytest.param(
            {"is_truncated": True, "stop_condition": "compaction_failed"},
            {},
            RolloutStatus.ERROR_UNKNOWN,
            id="unmapped_truncation",
        ),
        # A retried attempt's error stays in the history; the final attempt finished.
        pytest.param(
            {"last_error": _error("SandboxError")},
            {},
            RolloutStatus.COMPLETED,
            id="ok_after_a_retry",
        ),
    ],
)
def test_rollout_status_says_who_failed(
    trace_fields: dict, episode_fields: dict, expected: RolloutStatus
) -> None:
    trace = _trace(**{"num_turns": 1, **trace_fields})
    episode = SimpleNamespace(**{"ok": True, "is_timeout": False, **episode_fields})

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
    trace = _trace(
        is_timeout=True,
        stop_condition="agent_timeout",
        reward=1.0,
        num_turns=1,
        nodes=[node],
        branches=[
            SimpleNamespace(nodes=[node], token_ids=[10, 11], logprobs=[0.0, -0.1])
        ],
    )
    generation_server = SimpleNamespace(
        model_id="model",
        start_session=lambda *, session_id, group_id: None,
        end_session=lambda session_id: VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
        ),
    )
    worker = _worker_with_env(
        run_slot=_run_slot_minting(trace),
        generation_server=generation_server,
    )

    with caplog.at_level(logging.WARNING):
        rollout = _run_single_rollout(worker)

    assert rollout.status is RolloutStatus.ERROR_TIMEOUT
    assert "rollout group=1/rollout=2 failed: ERROR_TIMEOUT" in caplog.text
    assert VERIFIERS_REWARD_KEY not in rollout.turns[-1].env_rewards


def test_rollout_releases_every_session_verifiers_minted() -> None:
    """A retried attempt and another agent each made their own session; all are released."""
    retried, opponent, policy = (
        _trace(trace_id="9f3c"),
        _trace(trace_id="opponent", trainable=False),
        _trace(trace_id="a71b"),
    )

    async def run_slot(*, slot, ctx, on_trace):
        for trace in (retried, opponent, policy):
            on_trace(trace)
        # The final episode keeps only the last attempt's traces.
        return _episode(opponent, policy)

    events = []
    worker = _worker_with_env(
        run_slot=run_slot,
        generation_server=SimpleNamespace(
            model_id="model",
            start_session=lambda *, session_id, group_id: events.append(
                f"start {session_id} group {group_id}"
            ),
            end_session=lambda session_id: events.append(f"end {session_id}"),
        ),
    )
    release_session_fn = AsyncMock()

    rollout = _run_single_rollout(worker, release_session_fn=release_session_fn)

    assert [call.kwargs for call in release_session_fn.call_args_list] == [
        {"routing_session_id": "9f3c"},
        {"routing_session_id": "opponent"},
        {"routing_session_id": "a71b"},
    ]
    # Each session opens on the group as Verifiers mints its trace, before any model call.
    assert events == [
        "start 9f3c group 1",
        "start opponent group 1",
        "start a71b group 1",
        "end 9f3c",
        "end opponent",
        "end a71b",
    ]
    assert rollout.rollout_id == 2


def test_close_cancels_running_rollouts_then_stops_serving() -> None:
    events = []

    async def run_slot(*, slot, ctx, on_trace):
        on_trace(_trace(trace_id="session"))
        events.append("episode started")
        await asyncio.Event().wait()

    @contextlib.asynccontextmanager
    async def serving_env():
        yield
        events.append("serving stopped")

    async def release_session_fn(*, routing_session_id: str) -> None:
        events.append(f"released {routing_session_id}")

    async def close_generation_server() -> None:
        events.append("generation server stopped")

    async def run() -> None:
        worker = _worker_with_env(
            run_slot=run_slot,
            generation_server=SimpleNamespace(
                model_id="model",
                start_session=lambda *, session_id, group_id: None,
                end_session=lambda session_id: None,
                set_generate_fn=lambda generate_fn: None,
                close=close_generation_server,
            ),
        )
        worker._serving = contextlib.AsyncExitStack()
        await worker._serving.enter_async_context(serving_env())
        group = asyncio.create_task(
            worker.run_group(
                generate_fn=AsyncMock(),
                release_session_fn=release_session_fn,
                sample=VerifiersTaskSample(verifiers_task_data={"name": "task"}),
                group_id=1,
                group_size=1,
                sampling=SamplingConfig(),
            )
        )
        while not events:
            await asyncio.sleep(0)
        await asyncio.wait_for(worker.close(), timeout=5)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(group, timeout=5)

    asyncio.run(run())
    assert events == [
        "episode started",
        "released session",
        "serving stopped",
        "generation server stopped",
    ]


def test_a_rollout_that_raises_fails_its_group_after_its_siblings_stop() -> None:
    """The controller frees the group once `run_group` returns; no sibling may still be generating."""
    stopped = []

    async def run_single_rollout(*, rollout_id: int, **kwargs) -> Rollout:
        if rollout_id == 0:
            await asyncio.sleep(0)  # let the sibling start first
            raise RuntimeError("rollout 0 failed")
        try:
            await asyncio.sleep(10)
        finally:
            stopped.append(rollout_id)
        raise AssertionError("the sibling should have been cancelled")

    worker = _worker_with_env(
        run_slot=None,
        generation_server=SimpleNamespace(set_generate_fn=lambda fn: None),
    )
    worker._run_single_rollout = run_single_rollout

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(
            worker.run_group(
                generate_fn=AsyncMock(),
                release_session_fn=AsyncMock(),
                sample=VerifiersTaskSample(verifiers_task_data={"name": "task"}),
                group_id=1,
                group_size=2,
                sampling=SamplingConfig(),
            )
        )
    assert [str(error) for error in raised.value.exceptions] == ["rollout 0 failed"]
    assert stopped == [1]


def test_parse_sampling_config_requires_temperature_top_p_max_tokens() -> None:
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 5,
        "stop_token_ids": [1],
    }
    for name in ("temperature", "top_p", "max_tokens"):
        with pytest.raises(ValueError, match=rf"must include \['{name}'\]"):
            _parse_sampling_config(
                {key: value for key, value in sampling_params.items() if key != name}
            )


@pytest.mark.parametrize(
    "update, error",
    [
        ({}, None),
        # What Verifiers 0.4.1 sends on every call: values that filter nothing.
        ({"top_k": -1, "min_p": 0.0}, None),
        ({"top_k": 0}, None),
        ({"top_k": 20}, r"must not filter .* got top_k=20, min_p=0.0"),
        ({"min_p": 0.05}, r"must not filter .* got top_k=-1, min_p=0.05"),
    ],
)
def test_parse_sampling_config_accepts_only_full_vocabulary_top_k_and_min_p(
    update, error
) -> None:
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 2,
        "stop_token_ids": [99],
        **update,
    }
    if error is not None:
        with pytest.raises(ValueError, match=error):
            _parse_sampling_config(sampling_params)
        return
    assert _parse_sampling_config(sampling_params) == SamplingConfig(
        temperature=1.0, top_p=1.0, max_tokens=2, stop_token_ids=[99]
    )


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})
