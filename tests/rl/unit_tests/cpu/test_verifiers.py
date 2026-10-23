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
    _fork_counts,
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
from torchtitan.rl.rollout.types import (
    Completion,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)
from torchtitan.rl.routing.inter_generator import GeneratorEndpoint
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder


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
            training_sample_builder="training_sample_builder",
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
        training_sample_builder="training_sample_builder",
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
        training_sample_builder=TrainingSampleBuilder.Config(),
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
        # Both generations folded into one Mean and one Max, plus the server's GC and
        # event-loop lag (a Sum and a Max each)
        assert len(generation_metadata.metrics) == 6
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
                generator_sessions={},
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
    "trace_errors, episode_errors, num_turns, expected",
    [
        # The generator unreachable (Verifiers surfaces a failed connection as 503).
        ([_recorded_error("ProviderError", 503)], [], 1, RolloutStatus.ERROR_SYSTEM),
        ([_recorded_error("SandboxError")], [], 1, RolloutStatus.ERROR_SYSTEM),
        # An untyped error is a bug in the code around the policy.
        ([_recorded_error("RuntimeError")], [], 1, RolloutStatus.ERROR_SYSTEM),
        # The env's own `run()` raised: recorded on the episode, not the trace.
        ([], [_recorded_error("EnvError")], 1, RolloutStatus.ERROR_SYSTEM),
        # A prompt past the context, or the agent's own exit: the policy's doing.
        ([_recorded_error("ProviderError", 400)], [], 1, RolloutStatus.ERROR),
        ([_recorded_error("HarnessError")], [], 1, RolloutStatus.ERROR),
        # A generator 4xx stays ERROR even before the first turn.
        ([_recorded_error("ProviderError", 400)], [], 0, RolloutStatus.ERROR),
        # The harness failed before the policy's first turn: the policy never ran.
        ([_recorded_error("HarnessError")], [], 0, RolloutStatus.ERROR_SYSTEM),
        # The first error stopped the rollout; a later teardown error doesn't change it.
        (
            [_recorded_error("ProviderError", 400), _recorded_error("SandboxError")],
            [],
            1,
            RolloutStatus.ERROR,
        ),
    ],
)
def test_rollout_status_maps_recorded_infra_errors_to_error_system(
    trace_errors, episode_errors, num_turns: int, expected: RolloutStatus
) -> None:
    trace = SimpleNamespace(
        ok=False,
        is_timeout=False,
        is_truncated=False,
        errors=trace_errors,
        num_turns=num_turns,
    )
    episode = SimpleNamespace(ok=False, is_timeout=False, errors=episode_errors)

    status = VerifiersRolloutWorker.rollout_status(
        verifiers_episode=episode, trace=trace
    )

    assert status is expected


def test_a_harness_that_fails_before_the_first_turn_is_error_system() -> None:
    from verifiers.v1.episode import WireEpisode
    from verifiers.v1.trace import WireTrace

    task = {"type": "swe", "data": {}}
    # As Verifiers records it: the error at the harness boundary, no model call, no node.
    trace = WireTrace.model_validate(
        {
            "task": task,
            "agent": {"config": {}},
            "stop_condition": "harness_error",
            "errors": [
                {"type": "HarnessError", "message": "opening harness session timed out"}
            ],
        }
    )
    episode = WireEpisode.model_validate({"task": task, "ok": False, "traces": [trace]})

    status = VerifiersRolloutWorker.rollout_status(
        verifiers_episode=episode, trace=trace
    )

    assert status is RolloutStatus.ERROR_SYSTEM


def test_timed_out_rollout_is_logged_and_keeps_no_reward(caplog) -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    # The agent stage timed out, but Verifiers kept ok=True and still graded the reply.
    node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="answer"),
        parent=None,
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
                generator_sessions={},
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

    generator_sessions = {}
    rollout = asyncio.run(
        worker._run_single_rollout(
            generator_client=cast(GeneratorClient, client),
            sample=VerifiersTaskSample(verifiers_task_data={"name": "task"}),
            sampling=SamplingConfig(),
            group_id=1,
            rollout_id=0,
            generator_sessions=generator_sessions,
        )
    )

    client.release_rollout.assert_awaited_once()
    assert client.release_rollout.await_args.kwargs["session_ids"] == [
        "opponent",
        "policy",
    ]
    assert rollout.group_id == 1
    # Its router-replay rows are fetched from the trainable trace's session.
    assert generator_sessions == {0: ("http://localhost:1", "policy")}


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


def test_cut_stragglers_cuts_the_unfinished_rollout_and_the_rest_train(
    monkeypatch,
) -> None:
    released = []

    async def run_single_rollout(self, *, group_id, rollout_id, **kwargs):
        try:
            if rollout_id == 3:
                await asyncio.Event().wait()
            return Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                logs={"stop_condition": "agent_completed"},
            )
        finally:
            released.append(rollout_id)

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

    async def run():
        cut_stragglers = asyncio.get_running_loop().create_future()
        group = asyncio.create_task(
            worker.run_group(
                generate_fn=GeneratorClient(router=None),
                release_session_fn=None,
                sample=None,
                group_id=1,
                group_size=4,
                sampling=SamplingConfig(),
                cut_stragglers=cut_stragglers,
            )
        )
        while len(released) < 3:
            await asyncio.sleep(0)
        cut_stragglers.set_result(1)
        return await group

    group = asyncio.run(asyncio.wait_for(run(), timeout=30))

    *trained, cut = group.rollouts
    assert cut.status is RolloutStatus.ERROR_SYSTEM and cut.logs["straggler_cut"]
    assert (cut.reward, cut.advantage) == (None, None)
    assert [rollout.advantage for rollout in trained] == [-1.0, 0.0, 1.0]
    assert sorted(released) == [0, 1, 2, 3]
    values = m.MetricsProcessor._aggregate_metrics(group.metrics)
    assert values["rollout/stragglers_cut/sum"] == 1
    # The cut rollout has no stop condition: it counts as "other"
    assert values["rollout/stop/other/mean"] == 0.25


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


def test_rollout_logs_why_its_trace_forked() -> None:
    """One compaction and one unmatched re-sent assistant message."""
    from verifiers.v1.types import (
        AssistantMessage as VerifiersAssistantMessage,
        UserMessage as VerifiersUserMessage,
    )

    def node(parent, message, sampled):
        return SimpleNamespace(parent=parent, message=message, sampled=sampled)

    user = VerifiersUserMessage(content="fix it")
    answer = VerifiersAssistantMessage(content="answer")
    trace = SimpleNamespace(
        nodes=[
            node(None, user, False),
            node(0, answer, True),
            node(0, answer, False),  # the agent re-sent the answer re-encoded
            # Compaction: the summary request starts a new root and re-carries the history.
            node(None, VerifiersUserMessage(content="summarize"), False),
            node(3, answer, False),
            node(4, VerifiersUserMessage(content="now summarize"), False),
            node(5, VerifiersAssistantMessage(content="## Objective"), True),
            # The conversation resumes from the summary under the first root.
            node(0, VerifiersUserMessage(content="summary: ..."), False),
            node(7, answer, True),
        ]
    )

    assert _fork_counts(trace) == {
        "num_compactions": 1,
        "num_resent_assistant_messages": 1,
    }


def _trace_of(*nodes):
    return SimpleNamespace(
        nodes=[
            SimpleNamespace(parent=parent, message=message, sampled=sampled)
            for parent, message, sampled in nodes
        ]
    )


def _system(day: str):
    from verifiers.v1.types import SystemMessage as VerifiersSystemMessage

    return VerifiersSystemMessage(
        content=f"You are opencode.\n<env>\n  Today's date: {day} Oct 2026\n</env>"
    )


def test_each_summary_request_counts_as_one_compaction() -> None:
    """Two summary requests share one root (same system message), each a new chain from it."""
    from verifiers.v1.types import (
        AssistantMessage as VerifiersAssistantMessage,
        SystemMessage as VerifiersSystemMessage,
        UserMessage as VerifiersUserMessage,
    )

    answer = VerifiersAssistantMessage(content="answer")
    summarize = VerifiersSystemMessage(content="Summarize the conversation.")
    trace = _trace_of(
        (None, _system("Tue 20"), False),
        (0, VerifiersUserMessage(content="fix it"), False),
        (1, answer, True),
        (None, summarize, False),  # 3: the summary root
        (3, VerifiersUserMessage(content="history 1"), False),
        (4, VerifiersAssistantMessage(content="## Summary 1"), True),
        (0, VerifiersUserMessage(content="summary 1, go on"), False),
        (6, answer, True),
        (3, VerifiersUserMessage(content="history 2"), False),
        (8, VerifiersAssistantMessage(content="## Summary 2"), True),
        (0, VerifiersUserMessage(content="summary 2, go on"), False),
        (10, answer, True),
        (11, VerifiersUserMessage(content="tool output"), False),
        (12, answer, True),
    )

    assert _fork_counts(trace)["num_compactions"] == 2


def test_a_midnight_fork_is_not_a_compaction() -> None:
    """The main conversation re-sent under a new date starts a new root; its turns don't count."""
    from verifiers.v1.types import (
        AssistantMessage as VerifiersAssistantMessage,
        UserMessage as VerifiersUserMessage,
    )

    answer = VerifiersAssistantMessage(content="answer")
    trace = _trace_of(
        (None, _system("Tue 20"), False),
        (0, VerifiersUserMessage(content="fix it"), False),
        (1, answer, True),
        (None, _system("Wed 21"), False),  # 3: the same history re-sent after 00:00 UTC
        (3, VerifiersUserMessage(content="fix it"), False),
        (4, answer, False),
        (5, VerifiersUserMessage(content="tool output"), False),
        (6, answer, True),
        (7, VerifiersUserMessage(content="tool output"), False),
        (8, answer, True),
    )

    assert _fork_counts(trace)["num_compactions"] == 0


def test_a_summary_after_a_midnight_fork_counts_once() -> None:
    from verifiers.v1.types import (
        AssistantMessage as VerifiersAssistantMessage,
        SystemMessage as VerifiersSystemMessage,
        UserMessage as VerifiersUserMessage,
    )

    answer = VerifiersAssistantMessage(content="answer")
    trace = _trace_of(
        (None, _system("Tue 20"), False),
        (0, VerifiersUserMessage(content="fix it"), False),
        (1, answer, True),
        (None, _system("Wed 21"), False),  # 3: midnight fork
        (3, VerifiersUserMessage(content="fix it"), False),
        (4, answer, False),
        (5, VerifiersUserMessage(content="tool output"), False),
        (6, answer, True),
        (None, VerifiersSystemMessage(content="Summarize the conversation."), False),
        (8, VerifiersUserMessage(content="history"), False),
        (9, VerifiersAssistantMessage(content="## Summary"), True),
        # The resume carries the new date: a new chain under the re-dated root.
        (3, VerifiersUserMessage(content="summary, go on"), False),
        (11, answer, True),
    )

    assert _fork_counts(trace)["num_compactions"] == 1


def test_group_logs_the_share_of_each_stop_condition(monkeypatch) -> None:
    stops = ["agent_completed", "agent_timeout", "agent_timeout", "image_pull_error"]

    async def run_single_rollout(self, *, group_id, rollout_id, **kwargs):
        if rollout_id == len(stops):
            raise RuntimeError("env server request failed")  # no trace: "other"
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=RolloutStatus.COMPLETED,
            logs={"stop_condition": stops[rollout_id]},
        )

    async def score_group(self, rollouts, sample):
        return [RubricOutput(reward=0.0, reward_breakdown={}) for _ in rollouts]

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
            group_size=len(stops) + 1,
            sampling=SamplingConfig(),
        )
    )

    shares = {
        key: value
        for key, value in m.MetricsProcessor._aggregate_metrics(group.metrics).items()
        if key.startswith("rollout/stop/")
    }
    # Every bucket is logged, zeros included, so each averages to its share across groups.
    assert shares == {
        "rollout/stop/agent_completed/mean": 0.2,
        "rollout/stop/agent_timeout/mean": 0.4,
        "rollout/stop/max_turns/mean": 0.0,
        "rollout/stop/max_input_tokens/mean": 0.0,
        "rollout/stop/sandbox_error/mean": 0.0,
        "rollout/stop/harness_error/mean": 0.0,
        "rollout/stop/agentstalled_error/mean": 0.0,
        "rollout/stop/other/mean": 0.4,
    }


def test_rows_fetched_by_handle_are_todays_routed_expert_ids_bitwise() -> None:
    """Commit one rollout's turns twice through Verifiers' graph and delta stream: once with the
    rows the generation server used to send, once with the handles it sends now. Fetching rows by
    handle gives the same bytes per node, per branch and per training turn.

    Two branches share the system node, like a compaction. Bridged turns repair the previous
    turn's placeholder row over the stream; unbridged ones also return the prefix's rows.
    """
    import base64

    import verifiers.v1 as vf
    from renderers.client import parse_generate_response
    from verifiers.v1 import graph
    from verifiers.v1.episode import EnvInfo, Episode, WireEpisode
    from verifiers.v1.serve.delta import (
        DeltaStreamer,
        dump,
        EpisodeAssembly,
        pack,
        TraceSummary,
        unpack,
    )
    from verifiers.v1.types import TurnTokens

    system = vf.SystemMessage(content="s")
    summary_request = vf.UserMessage(content="sum")
    users = [vf.UserMessage(content=f"u{i}") for i in range(4)]
    replies = [vf.AssistantMessage(content=f"a{i}") for i in range(5)]
    # Turn i samples replies[i]. A bridged prompt is the previous prompt + completion + new tokens.
    # (prompt messages, prompt ids, message spans, completion ids, bridged)
    turns = [
        ([system, users[0]], [1, 2, 3, 4, 5, 6], [(0, 3), (3, 5)], [7, 8], False),
        (
            [system, users[0], replies[0], users[1]],
            [*range(1, 9), 9, 10, 11],
            [(0, 3), (3, 5), None, (8, 10)],
            [12, 13],
            True,
        ),
        (
            [system, users[0], replies[0], users[1], replies[1], users[2]],
            [*range(1, 14), 14, 15, 16, 17],
            [(0, 3), (3, 5), None, (8, 10), None, (13, 16)],
            [18, 19, 20],
            True,
        ),
        (
            [system, summary_request],
            [1, 2, 3, 21, 22, 23],
            [(0, 3), (3, 5)],
            [24, 25],
            False,
        ),
        (
            [system, summary_request, replies[3], users[3]],
            [1, 2, 3, 21, 22, 23, 24, 25, 26, 27],
            [(0, 3), (3, 5), None, (8, 9)],
            [28],
            True,
        ),
    ]
    rng = np.random.default_rng(0)
    completions = iter(completion_ids for *_, completion_ids, _ in turns)
    sent_rows: list[np.ndarray] = []

    async def generate(prompt_token_ids, **kwargs):
        completion_ids = next(completions)
        start = kwargs["sampling_config"].routed_experts_prompt_start
        # The engine forwards every position but the last sampled token.
        num_rows = len(prompt_token_ids) + len(completion_ids) - 1 - start
        sent_rows.append(rng.integers(0, 256, (num_rows, 3, 2), dtype=np.uint8))
        return Completion(
            min_policy_version=1,
            max_policy_version=1,
            request_id=kwargs["request_id"],
            token_ids=completion_ids,
            token_logprobs=[-0.5] * len(completion_ids),
            finish_reason="stop",
            routed_expert_ids=sent_rows[-1],
        )

    async def run() -> tuple[dict[str, Rollout], list[m.Metric]]:
        traces = {
            name: vf.Trace(
                agent=vf.AgentInfo(config=vf.AgentConfig()),
                task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
            )
            for name in ("rows", "handles")
        }
        frames: dict[str, list[bytes]] = {name: [] for name in traces}
        streamers = {}
        for name, trace in traces.items():

            async def send(delta: dict, frames=frames[name]) -> None:
                frames.append(pack(delta))

            streamers[name] = DeltaStreamer(lambda trace=trace: [trace], send)
        session_id = traces["handles"].id
        server = GenerationServer(generate=generate, max_context_length=64)
        await server.start()
        try:
            for turn_idx, turn in enumerate(turns):
                messages, prompt_ids, spans, completion_ids, bridged = turn
                path_len = graph.prepare_turn(traces["rows"], messages).path_len
                start = path_len - 1 if bridged else 0
                sampling_params = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 4,
                    "stop_token_ids": [99],
                    "torchtitan_group_id": 7,
                    "routed_experts_prompt_start": start,
                }
                async with ClientSession() as session:
                    async with session.post(
                        f"{server.url}/inference/v1/generate",
                        json={
                            "token_ids": prompt_ids,
                            "sampling_params": sampling_params,
                        },
                        headers={"X-Session-ID": session_id},
                    ) as response:
                        raw = await response.read()
                payloads = {
                    # What the server sent before it kept the rows.
                    "rows": {
                        "data": base64.b64encode(sent_rows[-1]),
                        "shape": list(sent_rows[-1].shape),
                        "start": start,
                        "dtype": "uint8",
                    },
                    "handles": parse_generate_response(raw)["choices"][0][
                        "routed_experts"
                    ],
                }
                for name, trace in traces.items():
                    graph.prepare_turn(trace, messages).commit(
                        vf.Response(
                            id="",
                            created=0,
                            model="t",
                            message=replies[turn_idx],
                            finish_reason="stop",
                            # As Verifiers' train client builds it: no validation.
                            tokens=TurnTokens.model_construct(
                                prompt_ids=prompt_ids,
                                completion_ids=completion_ids,
                                completion_logprobs=[-0.5] * len(completion_ids),
                                message_spans=spans,
                                bridged=bridged,
                                routed_experts=payloads[name],
                            ),
                        )
                    )
                    await streamers[name].flush()

            # The client side: assemble each stream and validate it, as Verifiers' client does.
            wire_traces = {}
            for name, trace in traces.items():
                assembly = EpisodeAssembly()
                for frame in frames[name]:
                    assembly.apply(unpack(frame))
                record = assembly.finish(
                    dump(Episode(env=EnvInfo(), task=trace.task), exclude={"traces"}),
                    [TraceSummary(id=trace.id, nodes=len(trace.nodes), calls=0)],
                )
                wire_traces[name] = WireEpisode.model_validate(record).traces[0]
            assert any(b"routing_repairs" in frame for frame in frames["handles"])

            store = server.session_stores[session_id]
            for rows_part, handles_part in [
                *zip(wire_traces["rows"].nodes, wire_traces["handles"].nodes),
                *zip(wire_traces["rows"].branches, wire_traces["handles"].branches),
            ]:
                row_ids = handles_part.routed_experts.view("<u4").ravel()
                assert (
                    store.rows[row_ids].tobytes() == rows_part.routed_experts.tobytes()
                )

            # The training turns: today's slices of the rows vs the worker's fetch by handle.
            metadata = VerifiersGenerationMetadata(
                min_policy_version=1, max_policy_version=1, metrics=[]
            )
            rollouts = {
                name: Rollout(
                    group_id=7,
                    rollout_id=0,
                    status=RolloutStatus.COMPLETED,
                    reward=1.0,
                    turns=VerifiersRolloutWorker.trace_to_rollout_turns(
                        trace=trace,
                        generation_metadata=metadata,
                        group_id=7,
                        rollout_id=0,
                    ),
                )
                for name, trace in wire_traces.items()
            }
            worker = object.__new__(VerifiersRolloutWorker)
            worker._training_sample_builder = TrainingSampleBuilder.Config().build()
            metrics = await worker._fetch_routed_expert_ids(
                RolloutGroup(group_id=7, rollouts=[rollouts["handles"]]),
                generator_client=GeneratorClient(router=None),
                generator_sessions={0: (server.url, session_id)},
            )
        finally:
            await server.close()
        return rollouts, metrics

    rollouts, metrics = asyncio.run(run())

    assert len(rollouts["rows"].turns) == len(turns)
    for rows_turn, fetched_turn in zip(
        rollouts["rows"].turns, rollouts["handles"].turns, strict=True
    ):
        rows, fetched = rows_turn.routed_expert_ids, fetched_turn.routed_expert_ids
        assert (fetched.dtype, fetched.shape) == (rows.dtype, rows.shape)
        assert fetched.tobytes() == rows.tobytes()
    assert m.MetricsProcessor._aggregate_metrics(metrics) == {
        "rollout/routed_experts/fetches/sum": 1.0,
        "rollout/routed_experts/fetch_retries/sum": 0.0,
        "rollout/routed_experts/fetch_failures/sum": 0.0,
    }


@pytest.mark.parametrize(
    "rewards, failed_attempts, fetched, fetch_metrics",
    [
        # Rollout 1's fetch fails twice: it gets None, so the batcher drops the group, counted.
        ([0.0, 1.0], [0, 2], [True, False], (2, 1, 1)),
        ([0.0, 1.0], [0, 1], [True, True], (2, 1, 0)),
        # Zero reward spread: the batcher trains neither rollout, so neither fetches.
        ([1.0, 1.0], [0, 0], [False, False], None),
    ],
)
def test_only_trained_rollouts_swap_their_handles_for_rows(
    monkeypatch, rewards, failed_attempts, fetched, fetch_metrics
) -> None:
    """Rollouts 0 and 1 carry handles to rows 0-2 and 3-4 of their sessions; rollout 2 raised.
    No rollout leaves the worker with handles."""
    # Distinct per row and per session.
    rows = {
        f"trace-{i}": (100 * i + np.arange(20, dtype=np.uint8)).reshape(5, 2, 2)
        for i in range(2)
    }

    async def run_single_rollout(
        self, *, group_id, rollout_id, generator_sessions, **kwargs
    ):
        if rollout_id == 2:
            raise RuntimeError("env server request failed")
        generator_sessions[rollout_id] = ("http://generator", f"trace-{rollout_id}")
        handles = np.arange(5, dtype="<u4").view(np.uint8).reshape(5, 4)
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=RolloutStatus.COMPLETED,
            turns=[
                RolloutTurn(
                    rollout_id=RolloutTurnID(
                        group_id=group_id, rollout_id=rollout_id, turn_id=turn_id
                    ),
                    prompt_prefix_len=0,
                    prompt_delta_token_ids=np.zeros(1, np.int32),
                    completion_token_ids=np.ones(1, np.int32),
                    completion_logprobs=np.zeros(1, np.float32),
                    routed_expert_ids=turn_handles,
                )
                for turn_id, turn_handles in enumerate([handles[:3], handles[3:]])
            ],
        )

    async def score_group(self, rollouts, sample):
        return [
            RubricOutput(reward=rewards[rollout.rollout_id], reward_breakdown={})
            for rollout in rollouts
        ]

    attempts = {f"trace-{rollout_id}": 0 for rollout_id in range(2)}

    async def fetch_routed_expert_ids(url, *, session_id, handles):
        attempts[session_id] += 1
        rollout_id = int(session_id.removeprefix("trace-"))
        if attempts[session_id] <= failed_attempts[rollout_id]:
            raise RuntimeError("connection reset")
        return rows[session_id][handles.view("<u4").ravel()]

    monkeypatch.setattr(
        VerifiersRolloutWorker, "_run_single_rollout", run_single_rollout
    )
    monkeypatch.setattr(VerifiersRolloutWorker, "score_group", score_group)
    worker = object.__new__(VerifiersRolloutWorker)
    worker.advantage_estimator = AdvantageEstimator.Config().build()
    worker._training_sample_builder = TrainingSampleBuilder.Config().build()
    client = GeneratorClient(router=None)
    object.__setattr__(client, "fetch_routed_expert_ids", fetch_routed_expert_ids)

    group = asyncio.run(
        worker.run_group(
            generate_fn=client,
            release_session_fn=None,
            sample=None,
            group_id=1,
            group_size=3,
            sampling=SamplingConfig(),
        )
    )

    for rollout_id, is_fetched in enumerate(fetched):
        turn_rows = [
            turn.routed_expert_ids for turn in group.rollouts[rollout_id].turns
        ]
        if is_fetched:
            np.testing.assert_array_equal(turn_rows[0], rows[f"trace-{rollout_id}"][:3])
            np.testing.assert_array_equal(turn_rows[1], rows[f"trace-{rollout_id}"][3:])
        else:
            assert turn_rows == [None, None]
    assert group.rollouts[2].status is RolloutStatus.ERROR_SYSTEM
    metrics = {
        key: value
        for key, value in m.MetricsProcessor._aggregate_metrics(group.metrics).items()
        if key.startswith("rollout/routed_experts/")
    }
    if fetch_metrics is None:
        assert metrics == {}
    else:
        assert metrics == {
            f"rollout/routed_experts/{key}/sum": float(value)
            for key, value in zip(
                ("fetches", "fetch_retries", "fetch_failures"), fetch_metrics
            )
        }
