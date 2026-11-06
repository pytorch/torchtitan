# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("verifiers")

from aiohttp import ClientSession

from torchtitan.rl.examples.verifiers import rollouter as verifiers_rollouter

from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    _parse_sampling_config,
    _validate_token_ids,
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.rollouter import (
    _trainable_token_spans,
    log_failed_rollout,
    VERIFIERS_REWARD_KEY,
    verifiers_rollout_logs,
    VerifiersRolloutWorker,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.rubric import RubricOutput
from torchtitan.rl.rollout.types import Completion, Rollout, RolloutStatus
from verifiers.v1.utils.scope import run_scope


# vLLM's wording, which the generation server keeps.
_CONTEXT_OVERFLOW_MESSAGE = "This model's maximum context length is 8 tokens"
# Verifiers' wording when the agent process exits non-zero.
_AGENT_EXIT_MESSAGE = "harness 'terminus' exited 1: Traceback ..."


def _recorded_error(
    error_type: str, status_code: int | None = None, message: str | None = None
) -> SimpleNamespace:
    """An error as Verifiers records it on a trace or an episode (`verifiers.v1.trace.Error`)."""
    fields = {
        "type": error_type,
        "message": message or f"{error_type} happened",
        "status_code": status_code,
        "traceback": None,
    }
    return SimpleNamespace(**fields, model_dump=lambda mode: fields)


def test_train_client_config_carries_an_in_tree_renderer() -> None:
    """Verifiers keys its clients and renderers by the client config's JSON; an in-tree
    renderer named by `CustomRendererConfig` survives that round trip."""
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


def test_container_watchdog_removes_a_killed_workers_containers(tmp_path) -> None:
    """A worker killed by SIGKILL runs no cleanup; its watchdog still removes its containers."""
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    # The second `ps` lists a container whose `docker run` was in flight at the first one.
    fake_docker.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> {docker_log}\n'
        'if [ "$1" = ps ]; then\n'
        f"  if [ -e {tmp_path}/swept ]; then echo container-3; exit; fi\n"
        f"  touch {tmp_path}/swept; echo container-1; echo container-2\n"
        "fi\n"
    )
    fake_docker.chmod(0o755)
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time\n"
            "from torchtitan.rl.examples.verifiers import rollouter\n"
            "rollouter._SECOND_SWEEP_DELAY_SEC = 1\n"
            "rollouter._start_container_watchdog('run-1')\n"
            "print('armed', flush=True)\n"
            "time.sleep(600)\n",
        ],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert worker.stdout.readline() == "armed\n"
    time.sleep(1)
    assert not docker_log.exists()  # it waits for the worker to exit
    # Kill the worker's whole process group: the watchdog, in its own session, survives.
    os.killpg(worker.pid, signal.SIGKILL)
    worker.wait()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (
        docker_log.exists() and "container-3" in docker_log.read_text()
    ):
        time.sleep(0.1)
    assert docker_log.read_text().splitlines() == [
        "ps -aq --filter label=verifiers.run=run-1",
        "rm -f container-1 container-2",
        "ps -aq --filter label=verifiers.run=run-1",
        "rm -f container-3",
    ]


def test_verifiers_worker_serves_its_env_with_its_renderer_until_close(
    monkeypatch, tmp_path
) -> None:
    import contextlib

    from renderers import CustomRendererConfig
    from torchtitan.components.renderer import from_renderers
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    events = []

    @contextlib.asynccontextmanager
    async def serving():
        events.append("serving")
        yield
        events.append("stopped serving")

    async def record(event: str) -> None:
        events.append(event)

    environment = object()
    monkeypatch.setattr(
        "torchtitan.rl.examples.verifiers.rollouter.load_environment",
        lambda config: SimpleNamespace(serving=serving),
    )
    # A fake `docker` on PATH logs the watchdog's calls.
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$@" >> {docker_log}\n')
    fake_docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(verifiers_rollouter, "_SECOND_SWEEP_DELAY_SEC", 1)
    monkeypatch.setenv("VF_RUN_ID", "launcher-run")
    monkeypatch.setattr(
        "torchtitan.rl.examples.verifiers.rollouter.resolve_env_config",
        lambda config: config,
    )
    worker = object.__new__(VerifiersRolloutWorker)
    worker._verifiers_config = SimpleNamespace(
        environment=environment, renderer_multiplex=8
    )
    worker._generation_server = SimpleNamespace(
        start=lambda: record("generation server started"),
        close=lambda: record("generation server closed"),
        base_url="http://127.0.0.1:2/v1",
    )
    worker._verifiers_env_serving = contextlib.AsyncExitStack()
    renderers_config = CustomRendererConfig(
        import_path="torchtitan.models.muse_glimmer.renderer.MuseGlimmerRenderer",
        reasoning_strength="low",
    )

    async def run() -> None:
        await worker.setup_async(
            tokenizer_config=cast(HuggingFaceTokenizer.Config, None),
            renderer_config=from_renderers(renderers_config),
            hf_assets_path="muse-checkpoint",
        )
        assert events == ["generation server started", "serving"]
        await worker.close()

    asyncio.run(run())
    assert events == [
        "generation server started",
        "serving",
        "stopped serving",
        "generation server closed",
    ]
    sent = worker._verifiers_train_client_config
    # The worker's agents call its own generation server.
    assert sent.base_url == "http://127.0.0.1:2/v1"
    assert sent.renderer == renderers_config
    assert sent.renderer_model_name == "muse-checkpoint"
    # After a clean close the watchdog sweeps once, for the label Verifiers stamps: this
    # worker's own id, under the launcher's. No second sweep follows.
    time.sleep(2)
    assert docker_log.read_text().splitlines() == [
        f"ps -aq --filter label=verifiers.run={run_scope()}"
    ]
    assert run_scope().startswith("launcher-run-")


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
                token_ids=np.array([10, 11, 12, 13], dtype=np.int32),
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
                token_ids=np.array([10, 11, 12, 13], dtype=np.int32),
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
                token_ids=np.array([10, 11, 12, 13], dtype=np.int32),
                logprobs=[0.0, -0.1, 0.0, -0.2],
            ),
            SimpleNamespace(
                nodes=[shared, second_leaf],
                token_ids=np.array([10, 11, 14, 15], dtype=np.int32),
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
                token_ids=np.array([0] * len(branch_nodes), dtype=np.int32),
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


def _env_whose_rollouts_end_as(episode: object) -> SimpleNamespace:
    """A Verifiers env stand-in: every `run_slot` returns `episode`."""
    return SimpleNamespace(run_slot=AsyncMock(return_value=episode))


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
    worker._generation_server = SimpleNamespace(
        model_id="model", pop_generation_metadata=lambda trace_id: None
    )
    worker._verifiers_env = _env_whose_rollouts_end_as(
        SimpleNamespace(ok=True, is_timeout=False, errors=[], traces=[trace])
    )
    worker._verifiers_train_client_config = object()

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            worker._run_verifiers_rollout(
                release_session_fn=AsyncMock(),
                slot=SimpleNamespace(
                    task=SimpleNamespace(data=SimpleNamespace(name="huge-task"))
                ),
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
        (
            [_recorded_error("ProviderError", 400, _CONTEXT_OVERFLOW_MESSAGE)],
            [],
            1,
            RolloutStatus.ERROR,
        ),
        (
            [_recorded_error("HarnessError", message=_AGENT_EXIT_MESSAGE)],
            [],
            1,
            RolloutStatus.ERROR,
        ),
        # A prompt past the context stays ERROR even before the first turn.
        (
            [_recorded_error("ProviderError", 400, _CONTEXT_OVERFLOW_MESSAGE)],
            [],
            0,
            RolloutStatus.ERROR,
        ),
        # An error in the harness's own code, even after the first turn, is not the agent's exit.
        (
            [
                _recorded_error(
                    "HarnessError", message="harness 'terminus' metric: KeyError: 'x'"
                )
            ],
            [],
            1,
            RolloutStatus.ERROR_SYSTEM,
        ),
        (
            [
                _recorded_error(
                    "HarnessError",
                    message="harness 'terminus': RuntimeError: tmux exited 1: reset",
                )
            ],
            [],
            1,
            RolloutStatus.ERROR_SYSTEM,
        ),
        # Any other generator 4xx is a bug in the code around the policy.
        (
            [_recorded_error("ProviderError", 400, "missing X-Session-ID header")],
            [],
            1,
            RolloutStatus.ERROR_SYSTEM,
        ),
        # The harness failed before the policy's first turn: the policy never ran.
        (
            [_recorded_error("HarnessError", message=_AGENT_EXIT_MESSAGE)],
            [],
            0,
            RolloutStatus.ERROR_SYSTEM,
        ),
        # The first error stopped the rollout; a later teardown error doesn't change it.
        (
            [
                _recorded_error("ProviderError", 400, _CONTEXT_OVERFLOW_MESSAGE),
                _recorded_error("SandboxError"),
            ],
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
                token_ids=np.array([10, 11], dtype=np.int32),
                logprobs=[0.0, -0.1],
            )
        ],
    )
    worker = object.__new__(VerifiersRolloutWorker)
    worker._generation_server = SimpleNamespace(
        model_id="model",
        pop_generation_metadata=lambda trace_id: VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
        ),
    )
    worker._verifiers_env = _env_whose_rollouts_end_as(
        SimpleNamespace(ok=True, is_timeout=False, errors=[], traces=[trace])
    )
    worker._verifiers_train_client_config = object()

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            worker._run_verifiers_rollout(
                release_session_fn=AsyncMock(),
                slot=SimpleNamespace(),
                sampling=SamplingConfig(),
                group_id=1,
                rollout_id=2,
            )
        )

    assert rollout.status is RolloutStatus.ERROR_TIMEOUT
    assert "Verifiers rollout failed: trace=trace" in caplog.text
    assert VERIFIERS_REWARD_KEY not in rollout.turns[-1].env_rewards


# TODO: share one fake trace and rollouter between this test,
# test_rollout_whose_first_model_call_failed_has_no_turns and
# test_timed_out_rollout_is_logged_and_keeps_no_reward.
def test_rollout_releases_every_trace_session_even_if_one_release_fails(caplog) -> None:
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
    popped_trace_ids = []
    worker._generation_server = SimpleNamespace(
        model_id="model", pop_generation_metadata=popped_trace_ids.append
    )
    worker._verifiers_env = _env_whose_rollouts_end_as(
        SimpleNamespace(ok=True, is_timeout=False, errors=[], traces=traces)
    )
    worker._verifiers_train_client_config = object()
    release_session_fn = AsyncMock(side_effect=[RuntimeError("router down"), None])

    with caplog.at_level(logging.ERROR):
        rollout = asyncio.run(
            worker._run_verifiers_rollout(
                release_session_fn=release_session_fn,
                slot=SimpleNamespace(),
                sampling=SamplingConfig(),
                group_id=1,
                rollout_id=0,
            )
        )

    assert [call.kwargs for call in release_session_fn.call_args_list] == [
        {"routing_session_id": "opponent"},
        {"routing_session_id": "policy"},
    ]
    assert "releasing routing session opponent failed" in caplog.text
    assert popped_trace_ids == ["opponent", "policy"]
    assert rollout.group_id == 1


@pytest.mark.parametrize(
    "num_raised, advantages",
    [
        # Centered on the mean of the two scored rollouts (rewards 1 and 2).
        (1, [-0.5, 0.5]),
        (3, []),
    ],
)
def test_a_rollout_that_raises_is_kept_but_left_out_of_the_advantage(
    num_raised, advantages
) -> None:
    async def run_single_rollout(*, group_id, rollout_id, **kwargs):
        if rollout_id < num_raised:
            raise RuntimeError("Verifiers episode failed")
        return Rollout(
            group_id=group_id, rollout_id=rollout_id, status=RolloutStatus.COMPLETED
        )

    async def score_group(rollouts, sample):
        return [
            RubricOutput(reward=float(rollout.rollout_id), reward_breakdown={})
            for rollout in rollouts
        ]

    class _Task:
        """A Verifiers task type: built from its validated data and the taskset's task config."""

        def __init__(self, data, task_config) -> None:
            self.data = data

        @staticmethod
        def data_type():
            return SimpleNamespace(model_validate=lambda data: data)

    worker = object.__new__(VerifiersRolloutWorker)
    worker._generation_server = SimpleNamespace(set_generate_fn=lambda fn: None)
    worker._verifiers_env = SimpleNamespace(
        taskset=SimpleNamespace(
            task_type=lambda: _Task, config=SimpleNamespace(task=None)
        ),
        slots=lambda task, n: [task] * n,
    )
    worker._run_verifiers_rollout = run_single_rollout
    worker.score_group = score_group
    worker.advantage_estimator = AdvantageEstimator.Config().build()

    group = asyncio.run(
        worker.run_group(
            generate_fn=AsyncMock(),
            release_session_fn=AsyncMock(),
            sample=VerifiersTaskSample(verifiers_task_data={}),
            group_id=1,
            group_size=3,
            sampling=SamplingConfig(),
        )
    )

    failed, scored = group.rollouts[:num_raised], group.rollouts[num_raised:]
    for rollout in failed:
        assert rollout.status is RolloutStatus.ERROR_SYSTEM
        assert "Verifiers episode failed" in rollout.logs["error"]
        assert (rollout.reward, rollout.advantage) == (None, None)
    assert [rollout.advantage for rollout in scored] == advantages


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


def test_parse_sampling_config_accepts_top_k_and_min_p_only_switched_off() -> None:
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 5,
        "stop_token_ids": [1],
    }
    # What Verifiers' train client sends with every request.
    config = _parse_sampling_config({**sampling_params, "top_k": -1, "min_p": 0.0})
    assert config.max_tokens == 5
    with pytest.raises(ValueError, match=r"sampling_params\.top_k=20 is unsupported"):
        _parse_sampling_config({**sampling_params, "top_k": 20, "min_p": 0.0})
    with pytest.raises(
        ValueError, match=r"sampling_params\.min_p=0\.05 is unsupported"
    ):
        _parse_sampling_config({**sampling_params, "min_p": 0.05})


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})
