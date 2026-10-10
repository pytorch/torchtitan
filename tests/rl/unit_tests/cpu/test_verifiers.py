# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("verifiers")

from aiohttp import ClientSession

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
from torchtitan.rl.generator import SamplingConfig
from torchtitan.rl.rollout.types import Rollout, RolloutStatus
from torchtitan.rl.types import Completion


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
    assert turns[0].prompt_token_ids == [10, 11]
    assert turns[0].completion_token_ids == [12, 13]
    assert turns[0].completion_logprobs == [-0.2, -0.3]
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
    assert [turn.prompt_token_ids for turn in turns] == [[10], [10, 11, 12]]
    assert [turn.completion_token_ids for turn in turns] == [[11], [13]]
    assert [turn.completion_logprobs for turn in turns] == [[-0.1], [-0.2]]


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
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
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
        server.generate_fns[1] = generate_fn
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
        server.generate_fns[1] = generate_fn
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


def test_generation_server_uses_each_groups_generate_fn() -> None:
    """Each request goes to its own group's `GenerateFn`, even after another group registers
    one: a validation group (-1) and a training group (7) run at once."""

    async def run_test() -> None:
        received: list[tuple[str, int]] = []

        def make_generate_fn(name: str):
            async def generate_fn(prompt_token_ids, *, request_id, group_id, **kwargs):
                received.append((name, group_id))
                return Completion(
                    min_policy_version=0,
                    max_policy_version=0,
                    request_id=request_id,
                    token_ids=[31],
                    token_logprobs=[-0.1],
                    finish_reason="stop",
                )

            return generate_fn

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        await server.start()
        try:
            async with ClientSession() as session:

                async def generate(group_id: int) -> int:
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": f"group={group_id}/rollout=0"},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": {
                                "torchtitan_group_id": group_id,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    return response.status

                server.generate_fns[-1] = make_generate_fn("validation")
                assert await generate(-1) == 200
                server.generate_fns[7] = make_generate_fn("training")
                assert await generate(-1) == 200
                assert await generate(7) == 200
                # A group with no registered function gets an error, not another group's.
                assert await generate(3) == 503
        finally:
            await server.close()

        assert received == [("validation", -1), ("validation", -1), ("training", 7)]

    asyncio.run(run_test())


def test_rollouter_registers_the_groups_generate_fn_while_it_runs() -> None:
    """`run_group_rollouts` keeps its group's `GenerateFn` registered for the group's rollouts,
    while another group starts, and removes it after."""

    async def run_test() -> None:
        rollouter = object.__new__(VerifiersRollouter)
        rollouter._generation_server = GenerationServer.Config(
            max_rollout_tokens=40960
        ).build()
        rollouter._rubric = SimpleNamespace(
            score_group=AsyncMock(
                return_value=[SimpleNamespace(reward=1.0, reward_breakdown={})]
            )
        )
        rollouter._advantage_estimator = lambda group: [0.0]
        validation_rollout_can_end = asyncio.Event()
        seen: list[tuple[int, str]] = []

        async def run_single_rollout(*, sample, sampling, group_id, rollout_id):
            seen.append((group_id, rollouter._generation_server.generate_fns[group_id]))
            if group_id == -1:
                await validation_rollout_can_end.wait()
                seen.append(
                    (group_id, rollouter._generation_server.generate_fns[group_id])
                )
            return Rollout(
                group_id=group_id, rollout_id=rollout_id, status=RolloutStatus.COMPLETED
            )

        rollouter._run_single_rollout = run_single_rollout

        def run_group(group_id: int, generate_fn: str):
            return rollouter.run_group_rollouts(
                generate_fn=generate_fn,
                sample=None,
                group_id=group_id,
                group_size=1,
                sampling=SamplingConfig(),
            )

        validation_group = asyncio.create_task(run_group(-1, "validation_fn"))
        await asyncio.sleep(0)
        await run_group(7, "training_fn")
        validation_rollout_can_end.set()
        await validation_group

        assert seen == [
            (-1, "validation_fn"),
            (7, "training_fn"),
            (-1, "validation_fn"),
        ]
        assert rollouter._generation_server.generate_fns == {}

    asyncio.run(run_test())


def test_rollouter_removes_the_groups_generate_fn_when_a_rollout_raises() -> None:
    rollouter = object.__new__(VerifiersRollouter)
    rollouter._generation_server = GenerationServer.Config(
        max_rollout_tokens=40960
    ).build()

    async def run_single_rollout(*, sample, sampling, group_id, rollout_id):
        raise RuntimeError("sandbox lost")

    rollouter._run_single_rollout = run_single_rollout

    with pytest.raises(RuntimeError, match="sandbox lost"):
        asyncio.run(
            rollouter.run_group_rollouts(
                generate_fn="training_fn",
                sample=None,
                group_id=7,
                group_size=2,
                sampling=SamplingConfig(),
            )
        )
    assert rollouter._generation_server.generate_fns == {}


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})
