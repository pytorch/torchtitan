# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the generator's HTTP endpoint (`GenerationServer`) and its client (`GeneratorClient`)."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from aiohttp import ClientSession, ClientTimeout

from torchtitan.rl.generator.generation_server import (
    _decode_metrics,
    _encode_metrics,
    GenerationServer,
    response_to_completion,
)
from torchtitan.rl.generator.http_client import GeneratorClient
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Completion
from torchtitan.rl.routing.inter_generator import GeneratorEndpoint

_SAMPLING = SamplingConfig(temperature=1.0, max_tokens=4, seed=3, stop_token_ids=[99])


def _completion(request_id: str, **overrides) -> Completion:
    completion = Completion(
        min_policy_version=3,
        max_policy_version=4,
        request_id=request_id,
        token_ids=[31, 32, 33],
        token_logprobs=[-0.125, -1 / 3, -2.5],
        finish_reason="stop",
        metrics=[
            m.Metric("generator/queue_time_ms", m.Mean(5.0, 2.0)),
            m.Metric("generator/queue_time_ms", m.Max(4.0)),
        ],
    )
    return replace(completion, **overrides)


def _metric_fields(metrics: list[m.Metric]) -> list[tuple]:
    """Metric values define no equality, so compare their key, type and fields."""
    return [(metric.key, type(metric.value), vars(metric.value)) for metric in metrics]


class _Server:
    """A started `GenerationServer` around a fake `generate` that records its calls."""

    def __init__(self, generate=None):
        self.calls: list[dict] = []

        async def record(prompt_token_ids, **kwargs):
            self.calls.append({"prompt_token_ids": prompt_token_ids, **kwargs})
            return _completion(kwargs["request_id"])

        self.server = GenerationServer(
            generate=generate or record, max_context_length=64
        )

    async def __aenter__(self) -> GenerationServer:
        await self.server.start()
        return self.server

    async def __aexit__(self, *exc) -> None:
        await self.server.close()


async def _post(server: GenerationServer, body: dict, session_id: str = "s0"):
    async with ClientSession() as session:
        async with session.post(
            f"{server.url}/inference/v1/generate",
            json=body,
            headers={"X-Session-ID": session_id},
        ) as response:
            return response.status, await response.json()


def _body(*, group_id: int = 7, **extra) -> dict:
    return {
        "token_ids": [1, 2],
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 4,
            "stop_token_ids": [99],
            "torchtitan_group_id": group_id,
        },
        **extra,
    }


def test_torchtitan_response_round_trips_every_completion_field() -> None:
    expected = [
        _completion("r/http=0", token_logprobs=[-0.125, -1 / 3, float("-inf")]),
        _completion("r/http=1", token_ids=[], token_logprobs=[], finish_reason="abort"),
        _completion("r/http=2", token_logprobs=[float("nan"), 0.0, -2.5]),
        _completion(
            "r/http=3",
            routed_expert_ids=np.arange(12, dtype=np.uint8).reshape(3, 2, 2),
        ),
    ]

    async def run() -> list[Completion]:
        replies = iter(expected)

        async def generate(prompt_token_ids, **kwargs):
            return next(replies)

        received = []
        async with _Server(generate) as server:
            for _ in expected:
                status, payload = await _post(
                    server, _body(response_format="torchtitan")
                )
                # Aborts come back as 200 here: the caller maps the finish reason itself.
                assert status == 200
                received.append(response_to_completion(payload))
        return received

    for got, want in zip(asyncio.run(run()), expected, strict=True):
        # Bit for bit, so -inf, NaN and every rounding digit survive the wire.
        assert (
            np.asarray(got.token_logprobs, dtype=np.float64).tobytes()
            == np.asarray(want.token_logprobs, dtype=np.float64).tobytes()
        )
        assert _metric_fields(got.metrics) == _metric_fields(want.metrics)
        if want.routed_expert_ids is None:
            assert got.routed_expert_ids is None
        else:
            assert got.routed_expert_ids.dtype == want.routed_expert_ids.dtype
            np.testing.assert_array_equal(got.routed_expert_ids, want.routed_expert_ids)
        assert replace(
            got, token_logprobs=[], metrics=[], routed_expert_ids=None
        ) == replace(want, token_logprobs=[], metrics=[], routed_expert_ids=None)


def test_verifiers_response_returns_routed_experts_from_the_prompt_start() -> None:
    # Prompt [1, 2, 3] + completion [31, 32, 33]; the previous turn already returned position 0,
    # so the generator returns the rows of positions 1..4.
    rows = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)

    async def run() -> tuple[list[dict], int, bytes]:
        calls = []

        async def generate(prompt_token_ids, **kwargs):
            calls.append(kwargs)
            return _completion(kwargs["request_id"], routed_expert_ids=rows)

        async with _Server(generate) as server:
            body = _body(token_ids=[1, 2, 3])
            body["sampling_params"]["routed_experts_prompt_start"] = 1
            out_of_range = _body(token_ids=[1, 2, 3])
            out_of_range["sampling_params"]["routed_experts_prompt_start"] = 3
            async with ClientSession() as session:
                async with session.post(
                    f"{server.url}/inference/v1/generate",
                    json=body,
                    headers={"X-Session-ID": "s0"},
                ) as response:
                    raw = await response.read()
            status, _ = await _post(server, out_of_range)
        return calls, status, raw

    calls, out_of_range_status, raw = asyncio.run(run())
    assert [call["sampling_config"].routed_experts_prompt_start for call in calls] == [
        1
    ]
    # A start outside the prompt would assert inside vLLM's engine step.
    assert out_of_range_status == 400
    # Compact JSON: Verifiers' client cuts the base64 out at this exact prefix.
    assert b'"routed_experts":{"data":"' in raw
    payload = json.loads(raw)["choices"][0]["routed_experts"]
    assert (payload["shape"], payload["start"], payload["dtype"]) == (
        [4, 2, 2],
        1,
        "uint8",
    )
    data = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
    np.testing.assert_array_equal(data.reshape(payload["shape"]), rows)


def test_topk_logprobs_reach_both_response_formats() -> None:
    topk_token_ids = np.array([[31, 5], [32, 6], [33, 7]], dtype=np.int32)
    topk_logprobs = np.array([[-0.1, -2.0], [-0.2, -3.0], [-0.3, -1.5]], np.float32)

    async def run() -> None:
        async def generate(prompt_token_ids, **kwargs):
            return _completion(
                kwargs["request_id"],
                topk_token_ids=topk_token_ids,
                topk_logprobs=topk_logprobs,
            )

        async with _Server(generate) as server:
            # Native rollouts: the top-k rides in the torchtitan response.
            _, payload = await _post(server, _body(response_format="torchtitan"))
            completion = response_to_completion(payload)
            np.testing.assert_array_equal(completion.topk_token_ids, topk_token_ids)
            np.testing.assert_array_equal(completion.topk_logprobs, topk_logprobs)
            # Verifiers drops extra response fields, so the server folds the top-k per
            # generation, keyed by (prompt length, completion tokens).
            status, _ = await _post(server, _body(), session_id="trace-a")
            assert status == 200
            [metadata] = server.pop_generation_metadata(["trace-a"]).values()
            ids, logprobs = metadata.topk_by_generation[(2, (31, 32, 33))]
            np.testing.assert_array_equal(ids, topk_token_ids)
            np.testing.assert_array_equal(logprobs, topk_logprobs)

    asyncio.run(run())


def test_server_mints_unique_request_ids_even_for_a_repeated_client_id() -> None:
    async def run() -> list[dict]:
        recorder = _Server()
        async with recorder as server:
            await asyncio.gather(
                *(
                    _post(
                        server,
                        _body(request_id="group=7/rollout=0/turn=0"),
                        session_id="group=7/rollout=0",
                    )
                    for _ in range(8)
                ),
                _post(server, _body(), session_id="trace-a"),
            )
        return recorder.calls

    request_ids = [call["request_id"] for call in asyncio.run(run())]
    assert len(set(request_ids)) == len(request_ids) == 9
    prefixes = [request_id.rsplit("/http=", 1)[0] for request_id in request_ids]
    assert prefixes.count("group=7/rollout=0/turn=0") == 8
    # Without a client id, the session id prefixes it.
    assert prefixes.count("trace-a") == 1


def test_only_the_default_format_folds_session_metadata() -> None:
    async def run() -> None:
        async with _Server() as server:
            for session_id, group_id, response_format in [
                ("native", 1, "torchtitan"),
                ("trace-a", 1, None),
                ("trace-a", 1, None),
                ("trace-b", 2, None),
                ("retried", 2, None),
                ("validation", -1, None),
            ]:
                extra = (
                    {}
                    if response_format is None
                    else {"response_format": response_format}
                )
                status, _ = await _post(
                    server, _body(group_id=group_id, **extra), session_id=session_id
                )
                assert status == 200

            assert set(server.generation_metadata) == {
                "trace-a",
                "trace-b",
                "retried",
                "validation",
            }
            # Validation logs at the training step under the same keys: no GC time on it.
            validation = server.pop_generation_metadata(["validation"])["validation"]
            assert not [x for x in validation.metrics if x.key.startswith("perf/")]
            popped = server.pop_generation_metadata(["trace-a", "native"])
            assert set(popped) == {"trace-a"}
            assert popped["trace-a"].min_policy_version == 3
            metrics = m.MetricsProcessor._aggregate_metrics(popped["trace-a"].metrics)
            # The server adds its process's GC time to every training completion.
            assert metrics.keys() >= {
                "perf/generator/gc_seconds/sum",
                "perf/generator/gc_seconds/max",
            }
            assert {
                key: value
                for key, value in metrics.items()
                if key.startswith("generator/")
            } == {
                "generator/queue_time_ms/mean": 2.5,
                "generator/queue_time_ms/max": 4.0,
            }
            # A retried trace nobody popped goes when its group is released.
            server.drop_generation_metadata([2])
            assert server.generation_metadata == {}

    asyncio.run(run())


def test_a_handler_whose_client_left_does_not_fold() -> None:
    async def run() -> None:
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def generate(prompt_token_ids, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async with _Server(generate) as server:
            async with ClientSession(timeout=ClientTimeout(total=5)) as session:
                post = asyncio.create_task(
                    session.post(
                        f"{server.url}/inference/v1/generate",
                        json=_body(),
                        headers={"X-Session-ID": "trace-a"},
                    )
                )
                await asyncio.wait_for(started.wait(), 5)
                post.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await post
            await asyncio.wait_for(cancelled.wait(), 5)
            assert server.generation_metadata == {}

    asyncio.run(run())


def test_metrics_codec_round_trips_the_fold_types_and_rejects_others() -> None:
    metrics = [
        m.Metric("a", m.Mean(3.0, 2.0)),
        m.Metric("a", m.Max(2.0)),
        m.Metric("b", m.Min(-1.0)),
        m.Metric("c", m.Sum(4.0)),
    ]
    decoded = _decode_metrics(json.loads(json.dumps(_encode_metrics(metrics))))
    assert _metric_fields(decoded) == _metric_fields(metrics)
    with pytest.raises(TypeError, match="Std"):
        _encode_metrics([m.Metric("d", m.Std(1.0))])


class _Router:
    """Fake router actor: hands out one endpoint and records leases."""

    def __init__(self, url: str):
        self.leases: list[tuple[str, int, int]] = []

        async def acquire(session_id, *, group_id):
            self.leases.append(("acquire", session_id, group_id))
            return GeneratorEndpoint(index=0, url=url)

        async def release(index, *, group_id):
            self.leases.append(("release", index, group_id))

        self.acquire = SimpleNamespace(call_one=acquire)
        self.release = SimpleNamespace(call_one=release)


def test_client_generates_through_a_lease_and_always_releases_it() -> None:
    async def run() -> None:
        recorder = _Server()
        async with recorder as server:
            router = _Router(server.url)
            client = GeneratorClient(
                router=router, metrics_prefix="validation_generator"
            )
            completion = await client(
                [1, 2],
                request_id="group=7/rollout=3/turn=0",
                group_id=7,
                routing_session_id="group=7/rollout=3",
                sampling_config=_SAMPLING,
            )
            assert completion.token_ids == [31, 32, 33]
            assert completion.request_id == "group=7/rollout=3/turn=0/http=0"
            [call] = recorder.calls
            assert call["sampling_config"] == _SAMPLING
            assert call["metrics_prefix"] == "validation_generator"
            assert router.leases == [
                ("acquire", "group=7/rollout=3", 7),
                ("release", 0, 7),
            ]

            # A prompt with no room for an output token is a 400; the lease still goes back.
            with pytest.raises(RuntimeError, match="400"):
                await client(
                    [1] * 64,
                    request_id="group=7/rollout=3/turn=1",
                    group_id=7,
                    routing_session_id="group=7/rollout=3",
                    sampling_config=_SAMPLING,
                )
            assert router.leases[-1] == ("release", 0, 7)

    asyncio.run(run())
