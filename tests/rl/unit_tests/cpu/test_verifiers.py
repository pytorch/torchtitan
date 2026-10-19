# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import gc
import inspect
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("verifiers")

from aiohttp import ClientSession

from torchtitan.rl.actors.system import ActorSystem
from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.rollouter import (
    _trainable_token_spans,
    log_failed_rollout,
    VERIFIERS_REWARD_KEY,
    verifiers_rollout_logs,
    VerifiersRollouter,
    VerifiersRolloutWorker,
)
from torchtitan.rl.generator.generation_server import (
    _parse_sampling_config,
    _validate_token_ids,
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.generator.http_client import GeneratorClient
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.rubric import RubricOutput
from torchtitan.rl.rollout.types import Completion, Rollout, RolloutStatus
from torchtitan.rl.routing.inter_generator import GeneratorEndpoint


def _recorded_error(error_type: str, status_code: int | None = None) -> SimpleNamespace:
    """An error as Verifiers records it on a trace or an episode (`verifiers.v1.trace.Error`)."""
    fields = {
        "type": error_type,
        "message": f"{error_type} happened",
        "status_code": status_code,
        "traceback": None,
    }
    return SimpleNamespace(**fields, model_dump=lambda mode: fields)


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


def test_verifiers_worker_sends_muse_glimmer_renderer_config(monkeypatch) -> None:
    from renderers import CustomRendererConfig
    from torchtitan.components.renderer import from_renderers
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    env_client = SimpleNamespace(wait_for_server_startup=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(
        "torchtitan.rl.examples.verifiers.rollouter.VerifiersEnvClient",
        lambda _: env_client,
    )

    worker = object.__new__(VerifiersRolloutWorker)
    worker._verifiers_config = SimpleNamespace(
        env_server_address="tcp://127.0.0.1:1",
        extra_env_server_addresses={},
        connection_timeout_sec=5.0,
        renderer_multiplex=8,
    )
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

    sent = worker._verifiers_train_client_config
    assert sent is not None
    assert sent.renderer == renderers_config
    assert sent.renderer_model_name == "muse-checkpoint"


@pytest.mark.parametrize("worker_setup_fails", [False, True])
def test_setup_starts_the_env_server_then_spawns_workers_with_its_address(
    worker_setup_fails,
) -> None:
    from renderers import Qwen3RendererConfig
    from torchtitan.components.renderer import from_renderers

    events = []

    class _Workers:
        def __init__(self) -> None:
            self.setup_async = SimpleNamespace(
                call=AsyncMock(
                    side_effect=TimeoutError("env server not healthy")
                    if worker_setup_fails
                    else None
                )
            )

    class _ActorSystem:
        async def spawn_rollout_workers(self, config, *, num_workers, num_threads):
            events.append(("spawn", config, num_workers, num_threads))
            return _Workers()

        async def stop_rollout_workers(self) -> None:
            events.append(("stop_workers",))

    async def start() -> str:
        events.append(("start_env_server",))
        return "tcp://127.0.0.1:5555"

    async def close_env_server() -> None:
        events.append(("close_env_server",))

    rollouter = object.__new__(VerifiersRollouter)
    rollouter._config = SimpleNamespace(
        worker=None, worker_pool_size=2, num_threads_per_worker=3
    )
    rollouter._verifiers_config = SimpleNamespace(
        rubric="rubric",
        advantage="advantage",
        renderer_multiplex=8,
        connection_timeout_sec=5.0,
    )
    rollouter._verifiers_env_server = SimpleNamespace(
        start=start, close=close_env_server
    )
    rollouter._extra_env_servers = {}
    rollouter._worker_actors = None
    rollouter._actor_system = None
    rollouter._inflight_run_group_calls = set()

    async def run() -> None:
        await rollouter.setup_async(
            actor_system=cast(ActorSystem, _ActorSystem()),
            tokenizer_config="tokenizer_config",
            renderer_config=from_renderers(Qwen3RendererConfig()),
            hf_assets_path="hf_assets_path",
        )
        await rollouter.close()

    if worker_setup_fails:
        # A failed setup stops the workers, then the env server.
        with pytest.raises(TimeoutError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    (start_event, (_, worker_config, num_workers, num_threads), *closes) = events
    assert start_event == ("start_env_server",)
    assert worker_config == VerifiersRolloutWorker.Config(
        rubric="rubric",
        advantage="advantage",
        renderer_multiplex=8,
        connection_timeout_sec=5.0,
        env_server_address="tcp://127.0.0.1:5555",
    )
    assert (num_workers, num_threads) == (2, 3)
    assert closes == [("stop_workers",), ("close_env_server",)]


def test_setup_async_takes_the_base_rollouter_arguments() -> None:
    """The controller calls `setup_async` with `Rollouter.setup_async`'s arguments."""
    base = inspect.signature(Rollouter.setup_async).parameters.keys() - {
        "worker_config"
    }
    assert inspect.signature(VerifiersRollouter.setup_async).parameters.keys() == base


def test_rollouter_rejects_a_worker_config() -> None:
    from dataclasses import replace

    import torchtitan_recipes.rl.verifiers_dapo_math as recipe

    rollouter_config = recipe.rl_dapo_qwen3_4b_verifiers_8k().rollouter
    worker = VerifiersRolloutWorker.Config(
        rubric=rollouter_config.rubric,
        renderer_multiplex=8,
        connection_timeout_sec=5.0,
        env_server_address="tcp://127.0.0.1:5555",
    )
    with pytest.raises(ValueError, match="leave worker unset"):
        replace(rollouter_config, worker=worker)


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
                routed_experts=None,
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
                # Row i holds the experts of position i.
                routed_experts=np.arange(4).reshape(4, 1, 1),
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
    # Like the prompt, a delta: turn 1's rows start at 11, which only its prefill ran.
    assert [turn.routed_expert_ids.flatten().tolist() for turn in turns] == [
        [0],
        [1, 2],
    ]
    assert [turn.completion_logprobs.tolist() for turn in turns] == [
        pytest.approx([-0.1]),
        pytest.approx([-0.2]),
    ]


def _two_turn_trace():
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    first_node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="first"),
    )
    second_node = SimpleNamespace(
        token_ids=[12, 13, 14],
        mask=[False, True, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="second"),
    )
    return SimpleNamespace(
        nodes=[first_node, second_node],
        branches=[
            SimpleNamespace(
                nodes=[first_node, second_node],
                token_ids=[10, 11, 12, 13, 14],
                logprobs=[0.0, -0.1, 0.0, -0.2, -0.3],
                routed_experts=None,
            )
        ],
    )


def _metadata_with_topk(topk_by_generation) -> VerifiersGenerationMetadata:
    return VerifiersGenerationMetadata(
        min_policy_version=3,
        max_policy_version=4,
        metrics=[],
        topk_by_generation=topk_by_generation,
    )


def test_verifiers_trace_attaches_each_generations_topk_rows() -> None:
    # Generations are keyed by (prompt length, completion tokens): prompts [10] and
    # [10, 11, 12] sampled completions [11] and [13, 14].
    second_logprobs = np.array([[-0.2, -1.5], [-0.3, -1.2]], dtype=np.float32)
    topk_by_generation = {
        (1, (11,)): (np.array([[11, 7]]), np.array([[-0.1, -2.0]])),
        (3, (13, 14)): (np.array([[13, 8], [14, 9]]), second_logprobs),
    }
    turns = VerifiersRolloutWorker.trace_to_rollout_turns(
        trace=_two_turn_trace(),
        generation_metadata=_metadata_with_topk(topk_by_generation),
        group_id=5,
        rollout_id=2,
    )

    assert [turn.completion_token_ids.tolist() for turn in turns] == [[11], [13, 14]]
    assert [turn.completion_topk_token_ids.tolist() for turn in turns] == [
        [[11, 7]],
        [[13, 8], [14, 9]],
    ]
    np.testing.assert_array_equal(turns[1].completion_topk_logprobs, second_logprobs)


def test_verifiers_trace_rejects_a_node_without_its_topk_rows() -> None:
    # Top-k was requested, but no generation produced the second node's tokens.
    topk_by_generation = {(1, (11,)): (np.array([[11, 7]]), np.array([[-0.1, -2.0]]))}
    with pytest.raises(ValueError, match="top-k logprobs are unknown"):
        VerifiersRolloutWorker.trace_to_rollout_turns(
            trace=_two_turn_trace(),
            generation_metadata=_metadata_with_topk(topk_by_generation),
            group_id=5,
            rollout_id=2,
        )


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
                routed_experts=None,
            ),
            SimpleNamespace(
                nodes=[shared, second_leaf],
                token_ids=[10, 11, 14, 15],
                logprobs=[0.0, -0.1, 0.0, -0.3],
                routed_experts=None,
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
                routed_experts=None,
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
            metrics_prefix="generator",
        ):
            received.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "request_id": request_id,
                    "group_id": group_id,
                    "routing_session_id": routing_session_id,
                    "sampling_config": sampling_config,
                    "metrics_prefix": metrics_prefix,
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

        server = GenerationServer(generate=generate_fn, max_context_length=40960)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.get(f"{server.url}/v1/models")
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
                for metrics_prefix in ({}, {"torchtitan_metrics_prefix": "validation"}):
                    response = await session.post(
                        f"{server.url}/inference/v1/generate",
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
                                **metrics_prefix,
                            },
                        },
                    )
                    assert response.status == 200
                    payload = await response.json()
            generation_metadata = server.pop_generation_metadata(
                ["group=1/rollout=2"]
            ).get("group=1/rollout=2")
        finally:
            await server.close()

        assert [request["request_id"] for request in received] == [
            "group=1/rollout=2/http=0",
            "group=1/rollout=2/http=1",
        ]
        assert all(request["prompt_token_ids"] == [10, 11] for request in received)
        assert all(request["group_id"] == 1 for request in received)
        assert all(
            request["routing_session_id"] == "group=1/rollout=2" for request in received
        )
        assert all(request["sampling_config"].seed == 4 for request in received)
        assert [request["metrics_prefix"] for request in received] == [
            "generator",
            "validation",
        ]
        assert payload == {
            "request_id": "group=1/rollout=2/http=1",
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
        # Both generations folded into one Mean and one Max, plus the server's GC Sum and Max
        assert len(generation_metadata.metrics) == 4
        metrics = m.MetricsProcessor._aggregate_metrics(generation_metadata.metrics)
        assert {
            key: value for key, value in metrics.items() if key.startswith("generator/")
        } == {
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
            metrics_prefix="generator",
        ):
            return Completion(
                min_policy_version=7,
                max_policy_version=7,
                request_id=request_id,
                token_ids=[],
                token_logprobs=[],
                finish_reason="abort",
            )

        server = GenerationServer(generate=generate_fn, max_context_length=40960)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"{server.url}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "max_tokens": 2,
                            "torchtitan_group_id": 1,
                            "stop_token_ids": [99],
                        },
                    },
                )
                assert response.status == 502
                payload = await response.json()
            generation_metadata = server.pop_generation_metadata(
                ["group=1/rollout=2"]
            ).get("group=1/rollout=2")
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

        server = GenerationServer(generate=generate_fn, max_context_length=40960)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"{server.url}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={"token_ids": [10, 11], "sampling_params": {}},
                )
                assert response.status == 400
                payload = await response.json()
        finally:
            await server.close()

        assert "torchtitan_group_id" in payload["error"]

    asyncio.run(run_test())


def test_generation_server_does_not_log_each_request(caplog) -> None:
    async def run_test() -> None:
        server = GenerationServer(generate=AsyncMock(), max_context_length=40960)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.get(f"{server.url}/healthz")
                assert response.status == 200
        finally:
            await server.close()

    # aiohttp skips the access log above INFO, and imports can change the root level.
    with caplog.at_level(logging.INFO, logger="aiohttp.access"):
        asyncio.run(run_test())
    assert "aiohttp.access" not in [record.name for record in caplog.records]


def test_generation_server_rejects_prompt_without_room_for_output() -> None:
    async def run_test() -> None:
        prompt_lengths = []

        async def generate_fn(prompt_token_ids, **kwargs):
            prompt_lengths.append(len(prompt_token_ids))
            raise RuntimeError("generator reached")

        server = GenerationServer(generate=generate_fn, max_context_length=8)
        await server.start()
        try:
            async with ClientSession() as session:
                statuses, errors = [], []
                for num_tokens in (7, 8, 9):
                    response = await session.post(
                        f"{server.url}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={
                            "token_ids": [1] * num_tokens,
                            "sampling_params": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": 2,
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


@pytest.mark.parametrize(
    "trace_ok, trace_errors, expected",
    [
        # The prompt is past the context: the data's problem, flagged and kept as ERROR.
        (True, [], RolloutStatus.ERROR),
        # The generator unreachable: not the policy's doing, so its group still trains.
        (False, [_recorded_error("ProviderError", 503)], RolloutStatus.ERROR_SYSTEM),
    ],
)
def test_rollout_whose_first_model_call_failed_has_no_turns(
    caplog, trace_ok, trace_errors, expected
) -> None:
    span = SimpleNamespace(duration=0.0)
    trace = SimpleNamespace(
        id="trace",
        task=SimpleNamespace(key="huge-task"),
        agent=SimpleNamespace(trainable=True),
        ok=trace_ok,
        is_truncated=False,
        is_timeout=False,
        stop_condition="agent_completed",
        errors=trace_errors,
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
    worker._verifiers_env_client = SimpleNamespace(
        run=AsyncMock(
            return_value=SimpleNamespace(
                ok=True, is_timeout=False, errors=[], traces=[trace]
            )
        )
    )
    worker._verifiers_train_client_config = SimpleNamespace(
        model_copy=lambda update: update
    )
    client = SimpleNamespace(
        metrics_prefix="generator",
        acquire_rollout=AsyncMock(
            return_value=GeneratorEndpoint(index=0, url="http://localhost:1")
        ),
        # Its one model call failed, so its generator folded nothing for it.
        release_rollout=AsyncMock(return_value={}),
    )

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            worker._run_single_rollout(
                generator_client=cast(GeneratorClient, client),
                sample=VerifiersTaskSample(verifiers_task_data={"name": "huge-task"}),
                sampling=SamplingConfig(),
                group_id=1,
                rollout_id=2,
            )
        )

    assert rollout.status is expected and rollout.turns == []
    is_flagged = "first prompt of task huge-task does not fit" in caplog.text
    assert is_flagged == (expected is RolloutStatus.ERROR)


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
    trace = SimpleNamespace(ok=ok, is_timeout=is_timeout, is_truncated=False, errors=[])
    episode = SimpleNamespace(
        ok=not episode_is_timeout, is_timeout=episode_is_timeout, errors=[]
    )

    status = VerifiersRolloutWorker.rollout_status(
        verifiers_episode=episode, trace=trace
    )

    assert status is expected


@pytest.mark.parametrize(
    "trace_errors, episode_errors, expected",
    [
        # The generator unreachable (Verifiers surfaces a failed connection as 503).
        ([_recorded_error("ProviderError", 503)], [], RolloutStatus.ERROR_SYSTEM),
        ([_recorded_error("SandboxError")], [], RolloutStatus.ERROR_SYSTEM),
        # An untyped error is a bug in the code around the policy.
        ([_recorded_error("RuntimeError")], [], RolloutStatus.ERROR_SYSTEM),
        # The env's own `run()` raised: recorded on the episode, not the trace.
        ([], [_recorded_error("EnvError")], RolloutStatus.ERROR_SYSTEM),
        # A prompt past the context, or the agent's own exit: the policy's doing.
        ([_recorded_error("ProviderError", 400)], [], RolloutStatus.ERROR),
        ([_recorded_error("HarnessError")], [], RolloutStatus.ERROR),
        # The first error stopped the rollout; a later teardown error doesn't change it.
        (
            [_recorded_error("ProviderError", 400), _recorded_error("SandboxError")],
            [],
            RolloutStatus.ERROR,
        ),
    ],
)
def test_rollout_status_maps_recorded_infra_errors_to_error_system(
    trace_errors, episode_errors, expected: RolloutStatus
) -> None:
    trace = SimpleNamespace(
        ok=False, is_timeout=False, is_truncated=False, errors=trace_errors
    )
    episode = SimpleNamespace(ok=False, is_timeout=False, errors=episode_errors)

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
            SimpleNamespace(
                nodes=[node],
                token_ids=[10, 11],
                logprobs=[0.0, -0.1],
                routed_experts=None,
            )
        ],
    )
    worker = object.__new__(VerifiersRolloutWorker)
    worker._verifiers_env_client = SimpleNamespace(
        run=AsyncMock(
            return_value=SimpleNamespace(
                ok=True, is_timeout=False, errors=[], traces=[trace]
            )
        )
    )
    worker._verifiers_train_client_config = SimpleNamespace(
        model_copy=lambda update: update
    )
    client = SimpleNamespace(
        metrics_prefix="generator",
        acquire_rollout=AsyncMock(
            return_value=GeneratorEndpoint(index=0, url="http://localhost:1")
        ),
        release_rollout=AsyncMock(
            return_value={
                "trace": VerifiersGenerationMetadata(
                    min_policy_version=0, max_policy_version=0, metrics=[]
                )
            }
        ),
    )

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            worker._run_single_rollout(
                generator_client=cast(GeneratorClient, client),
                sample=VerifiersTaskSample(verifiers_task_data={"name": "slow-task"}),
                sampling=SamplingConfig(),
                group_id=1,
                rollout_id=2,
            )
        )

    assert rollout.status is RolloutStatus.ERROR_TIMEOUT
    assert "Verifiers rollout failed: trace=trace" in caplog.text
    assert VERIFIERS_REWARD_KEY not in rollout.turns[-1].env_rewards


def test_rollout_releases_every_trace_session_in_one_call() -> None:
    """Each trace generated under its own id; another agent's trace has one too."""
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
        for trace_id, trainable in [("opponent", False), ("policy", True)]
    ]
    worker = object.__new__(VerifiersRolloutWorker)
    worker._verifiers_env_client = SimpleNamespace(
        run=AsyncMock(
            return_value=SimpleNamespace(
                ok=True, is_timeout=False, errors=[], traces=traces
            )
        )
    )
    worker._verifiers_train_client_config = SimpleNamespace(
        model_copy=lambda update: update
    )
    client = SimpleNamespace(
        metrics_prefix="generator",
        acquire_rollout=AsyncMock(
            return_value=GeneratorEndpoint(index=0, url="http://localhost:1")
        ),
        release_rollout=AsyncMock(return_value={}),
    )

    rollout = asyncio.run(
        worker._run_single_rollout(
            generator_client=cast(GeneratorClient, client),
            sample=VerifiersTaskSample(verifiers_task_data={"name": "task"}),
            sampling=SamplingConfig(),
            group_id=1,
            rollout_id=0,
        )
    )

    client.release_rollout.assert_awaited_once()
    assert client.release_rollout.await_args.kwargs["session_ids"] == [
        "opponent",
        "policy",
    ]
    assert rollout.group_id == 1


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


def test_parse_sampling_config_reads_num_topk_logprobs() -> None:
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 5,
        "stop_token_ids": [1],
    }
    assert _parse_sampling_config(sampling_params).num_topk_logprobs == 0
    sampling_params["num_topk_logprobs"] = 32
    assert _parse_sampling_config(sampling_params).num_topk_logprobs == 32


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})


@pytest.mark.parametrize(
    "num_raised, advantages",
    [
        # Centered on the mean of the two scored rollouts (rewards 1 and 2).
        (1, [-0.5, 0.5]),
        (3, []),
    ],
)
def test_a_rollout_that_raises_is_kept_but_left_out_of_the_advantage(
    monkeypatch, num_raised, advantages
) -> None:
    async def run_single_rollout(self, *, group_id, rollout_id, **kwargs):
        if rollout_id < num_raised:
            raise RuntimeError("env server request failed")
        return Rollout(
            group_id=group_id, rollout_id=rollout_id, status=RolloutStatus.COMPLETED
        )

    async def score_group(self, rollouts, sample):
        return [
            RubricOutput(reward=float(rollout.rollout_id), reward_breakdown={})
            for rollout in rollouts
        ]

    monkeypatch.setattr(
        VerifiersRolloutWorker, "_run_single_rollout", run_single_rollout
    )
    monkeypatch.setattr(VerifiersRolloutWorker, "score_group", score_group)
    worker = object.__new__(VerifiersRolloutWorker)
    worker.advantage_estimator = AdvantageEstimator.Config().build()

    group = asyncio.run(
        worker.run_group(
            generate_fn=GeneratorClient(router=None),
            release_session_fn=None,
            sample=None,
            group_id=1,
            group_size=3,
            sampling=SamplingConfig(),
        )
    )

    failed, scored = group.rollouts[:num_raised], group.rollouts[num_raised:]
    for rollout in failed:
        assert rollout.status is RolloutStatus.ERROR_SYSTEM
        assert "env server request failed" in rollout.logs["error"]
        assert (rollout.reward, rollout.advantage) == (None, None)
    assert [rollout.advantage for rollout in scored] == advantages


def test_cancelled_run_group_waits_for_every_sibling_release() -> None:
    """A cancelled group returns only after each sibling gave its lease back with
    `release_rollout(session_ids=[])`: no sibling's lease outlives `cancel_group`."""

    async def run() -> None:
        started = asyncio.Semaphore(0)

        async def env_client_run(**kwargs):
            started.release()
            await asyncio.Event().wait()

        # Router round trips of different lengths: the slow one must finish too.
        delays = iter([0.01, 0.3])

        async def release_rollout(endpoint, *, group_id, session_ids):
            await asyncio.sleep(next(delays))
            released.append(session_ids)
            return {}

        released = []
        worker = object.__new__(VerifiersRolloutWorker)
        worker._verifiers_env_client = SimpleNamespace(run=env_client_run)
        worker._verifiers_train_client_config = SimpleNamespace(
            model_copy=lambda update: update
        )
        client = GeneratorClient(router=None)
        object.__setattr__(
            client,
            "acquire_rollout",
            AsyncMock(
                return_value=GeneratorEndpoint(index=0, url="http://localhost:1")
            ),
        )
        object.__setattr__(client, "release_rollout", release_rollout)
        group = asyncio.create_task(
            worker.run_group(
                generate_fn=client,
                release_session_fn=None,
                sample=VerifiersTaskSample(verifiers_task_data={"name": "task"}),
                group_id=1,
                group_size=2,
                sampling=SamplingConfig(),
            )
        )
        for _ in range(2):
            await started.acquire()
        group.cancel()
        with pytest.raises(asyncio.CancelledError):
            await group
        assert released == [[], []]

    asyncio.run(run())


def test_env_server_worker_logs_gc_time(capsys) -> None:
    from torchtitan.rl.examples.verifiers import worker_setup

    callbacks = list(gc.callbacks)
    worker_setup._log_gc_time(pause_s=0.0, summary_every_s=0.0)
    try:
        gc.collect(2)
    finally:
        gc.callbacks[:] = callbacks
    stderr = capsys.readouterr().err
    assert "GC gen 2 paused" in stderr
    assert "collections over the last" in stderr
