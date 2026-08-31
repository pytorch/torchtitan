# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from aiohttp import ClientSession

from torchtitan.experiments.rl.rollout.verifiers.model_adapter import (
    GenerationMetadata,
    GeneratorModelAdapter,
)
from torchtitan.experiments.rl.rollout.verifiers.rollouter import (
    _trainable_token_spans,
    VerifiersRollouter,
)
from torchtitan.experiments.rl.types import Completion


def test_trainable_token_spans() -> None:
    assert _trainable_token_spans([False, True, True, False, True]) == [
        (1, 3),
        (4, 5),
    ]


def test_verifiers_trace_preserves_generation_metadata() -> None:
    node = SimpleNamespace(
        token_ids=[10, 11, 12, 13],
        mask=[False, False, True, True],
        sampled=True,
    )
    trace = SimpleNamespace(
        calls=[SimpleNamespace(node=0)],
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
        generation_metadata=[
            GenerationMetadata(
                min_policy_version=3,
                max_policy_version=4,
                metrics=[],
            )
        ],
        group_id=5,
        rollout_id=2,
    )

    assert len(turns) == 1
    assert turns[0].prompt_token_ids == [10, 11]
    assert turns[0].completion_token_ids == [12, 13]
    assert turns[0].completion_logprobs == [-0.2, -0.3]
    assert turns[0].min_policy_version == 3
    assert turns[0].max_policy_version == 4


def test_model_adapter_forwards_token_request() -> None:
    async def run_test() -> None:
        received = {}

        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            received.update(
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
            )
            return Completion(
                min_policy_version=7,
                max_policy_version=8,
                request_id=request_id,
                token_ids=[31, 32],
                token_logprobs=[-0.1, -0.2],
                finish_reason="stop",
            )

        adapter = GeneratorModelAdapter(
            host="127.0.0.1",
            port=0,
            model="test-model",
            max_model_len=128,
        )
        adapter.set_generate_fn(generate_fn)
        await adapter.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://127.0.0.1:{adapter.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "temperature": 1.0,
                            "top_p": 0.9,
                            "max_tokens": 2,
                            "seed": 4,
                            "logprobs": 1,
                        },
                    },
                )
                assert response.status == 200
                payload = await response.json()
            generation_metadata = adapter.pop_generation_metadata("group=1/rollout=2")
        finally:
            await adapter.close()

        assert received["prompt_token_ids"] == [10, 11]
        assert received["request_id"] == "group=1/rollout=2/turn=0"
        assert received["routing_session_id"] == "group=1/rollout=2"
        assert received["sampling_config"].seed == 4
        assert payload["choices"][0]["token_ids"] == [31, 32]
        assert [
            (item.min_policy_version, item.max_policy_version)
            for item in generation_metadata
        ] == [(7, 8)]

    asyncio.run(run_test())


def test_model_adapter_rejects_aborted_generation() -> None:
    async def run_test() -> None:
        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
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

        adapter = GeneratorModelAdapter(
            host="127.0.0.1",
            port=0,
            model="test-model",
            max_model_len=128,
        )
        adapter.set_generate_fn(generate_fn)
        await adapter.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://127.0.0.1:{adapter.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={"token_ids": [10, 11], "sampling_params": {}},
                )
                assert response.status == 502
                payload = await response.json()
            generation_metadata = adapter.pop_generation_metadata("group=1/rollout=2")
        finally:
            await adapter.close()

        assert payload == {
            "error": "generation finished without a usable completion: abort"
        }
        assert generation_metadata == []

    asyncio.run(run_test())
