# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import inspect
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("verifiers")

from aiohttp import ClientSession

from torchtitan.rl.actors.system import ActorSystem
from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    _parse_sampling_config,
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.rollouter import (
    _trainable_token_spans,
    log_failed_rollout,
    verifiers_rollout_logs,
    VerifiersRollouter,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.types import Completion, RolloutStatus
from verifiers.v1.errors import model_error, OverlongPromptError


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


def test_verifiers_rollouter_sends_muse_glimmer_renderer_config(monkeypatch) -> None:
    from renderers import CustomRendererConfig
    from torchtitan.components.renderer import from_renderers
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    generation_server = SimpleNamespace(
        start=AsyncMock(),
        close=AsyncMock(),
        base_url="http://127.0.0.1:2",
    )
    env_server = SimpleNamespace(
        start=AsyncMock(return_value="http://127.0.0.1:1"),
        close=AsyncMock(),
    )
    env_client = SimpleNamespace(wait_for_server_startup=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(
        "torchtitan.rl.examples.verifiers.rollouter.VerifiersEnvClient",
        lambda _: env_client,
    )

    rollouter = object.__new__(VerifiersRollouter)
    rollouter._verifiers_config = SimpleNamespace(
        generation_server=SimpleNamespace(build=lambda: generation_server),
        connection_timeout_sec=5.0,
        renderer_multiplex=8,
    )
    rollouter._verifiers_env_server = env_server
    rollouter._generation_server = None
    rollouter._verifiers_env_client = None
    rollouter._verifiers_train_client_config = None
    renderers_config = CustomRendererConfig(
        import_path="torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer",
        reasoning_strength="low",
    )
    config = from_renderers(renderers_config)

    asyncio.run(
        rollouter.setup_async(
            actor_system=cast(ActorSystem, None),
            tokenizer_config=cast(HuggingFaceTokenizer.Config, None),
            renderer_config=config,
            hf_assets_path="muse-checkpoint",
        )
    )

    sent = rollouter._verifiers_train_client_config
    assert sent is not None
    assert sent.renderer == renderers_config
    assert sent.renderer_model_name == "muse-checkpoint"
    asyncio.run(rollouter.close())


def test_setup_async_takes_the_base_rollouter_arguments() -> None:
    """The controller calls `setup_async` with `Rollouter.setup_async`'s arguments."""
    assert inspect.signature(VerifiersRollouter.setup_async).parameters.keys() == (
        inspect.signature(Rollouter.setup_async).parameters.keys()
    )


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
    turns = VerifiersRollouter.trace_to_rollout_turns(
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
    turns = VerifiersRollouter.trace_to_rollout_turns(
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
    turns = VerifiersRollouter.trace_to_rollout_turns(
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
    turns = VerifiersRollouter.trace_to_rollout_turns(
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
        log_failed_rollout(logs, group_id=3, rollout_id=1)
        log_failed_rollout({**logs, "errors": []}, group_id=3, rollout_id=2)
    assert "error=HarnessError: agent timeout" in caplog.text
    assert "error=None: None" in caplog.text


def test_generation_server_forwards_token_request() -> None:
    async def run_test() -> None:
        received = []

        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            group_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            received.append(
                {
                    "prompt_token_ids": prompt_token_ids,
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

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
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
                            "max_model_len": 40960,
                        }
                    ],
                }
                for _ in range(2):
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": 2,
                                "seed": 4,
                                "logprobs": 1,
                                "torchtitan_group_id": 1,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    assert response.status == 200
                    payload = await response.json()
            generation_metadata = server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

        assert [request["request_id"] for request in received] == [
            "group=1/rollout=2/request=0",
            "group=1/rollout=2/request=1",
        ]
        assert all(request["prompt_token_ids"] == [10, 11] for request in received)
        assert all(request["group_id"] == 1 for request in received)
        assert all(
            request["routing_session_id"] == "group=1/rollout=2" for request in received
        )
        assert all(request["sampling_config"].seed == 4 for request in received)
        assert payload["choices"][0]["token_ids"] == [31, 32]
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
            prompt_token_ids,
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
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "torchtitan_group_id": 1,
                            "stop_token_ids": [99],
                        },
                    },
                )
                assert response.status == 502
                payload = await response.json()
            generation_metadata = server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

        assert payload == {
            "error": "generation finished without a usable completion: abort"
        }
        assert generation_metadata is None

    asyncio.run(run_test())


def test_generation_server_requires_group_id() -> None:
    async def run_test() -> None:
        async def generate_fn(*args, **kwargs):
            raise AssertionError("generate_fn must not run without a group id")

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={"token_ids": [10, 11], "sampling_params": {}},
                )
                assert response.status == 400
                payload = await response.json()
        finally:
            await server.close()

        assert "torchtitan_group_id" in payload["error"]

    asyncio.run(run_test())


def test_generation_server_rejects_prompt_without_room_for_output() -> None:
    async def run_test() -> None:
        prompt_lengths = []

        async def generate_fn(prompt_token_ids, **kwargs):
            prompt_lengths.append(len(prompt_token_ids))
            raise RuntimeError("generator reached")

        server = GenerationServer.Config(max_rollout_tokens=8).build()
        server.set_generate_fn(generate_fn)
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
                                "torchtitan_group_id": 1,
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
        # Verifiers ends the rollout as context_length on OverlongPromptError.
        assert isinstance(model_error(errors[1]), OverlongPromptError)

    asyncio.run(run_test())


def test_rollout_whose_first_prompt_does_not_fit_is_flagged(caplog) -> None:
    # Verifiers stops a rollout for context length before its first model call.
    span = SimpleNamespace(duration=0.0)
    trace = SimpleNamespace(
        id="trace",
        task=SimpleNamespace(key="huge-task"),
        agent=SimpleNamespace(trainable=True),
        ok=True,
        is_truncated=True,
        stop_condition="context_length",
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
    rollouter = object.__new__(VerifiersRollouter)
    rollouter._generation_server = SimpleNamespace(
        model_id="model", pop_generation_metadata=lambda trace_id: None
    )
    rollouter._verifiers_env_client = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(ok=True, errors=[], traces=[trace]))
    )
    rollouter._verifiers_train_client_config = object()

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            rollouter._run_single_rollout(
                sample=VerifiersTaskSample(verifiers_task_data={"name": "huge-task"}),
                sampling=SamplingConfig(),
                group_id=1,
                rollout_id=2,
            )
        )

    assert rollout.status is RolloutStatus.ERROR
    assert "first prompt of task huge-task does not fit" in caplog.text


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})
