# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``VLLMGenerator.generate``.

Exercises the endpoint in isolation by swapping in a fake vLLM engine —
no Monarch, no GPU, no real model. Covers the token-in / token-out
contract, the metric payload (timing math, edge cases, prefix override),
and the completion/metrics separation.
"""

import asyncio
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.observability import metrics as m


class _FakeRenderer:
    """Stub for vLLM's Renderer.render_cmpl.

    Mirrors the real shape: takes a list of ``{"prompt_token_ids": ids}``
    dicts and returns a list of typed ``TokensInput`` EngineInputs with
    ``type="token"`` plus a stamped ``arrival_time``.
    """

    def render_cmpl(self, prompts):
        return [
            {
                "type": "token",
                "prompt_token_ids": p["prompt_token_ids"],
                "arrival_time": 0.0,
            }
            for p in prompts
        ]


class _FakeEngine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.add_requests = []
        self._stepped = False
        self.renderer = _FakeRenderer()

    def add_request(self, *args, **kwargs):
        self.add_requests.append((args, kwargs))

    def has_unfinished_requests(self):
        return not self._stepped

    def step(self):
        self._stepped = True
        return self.outputs


def _sample(*, index=0, token_ids=(10, 11), finish_reason="stop"):
    return SimpleNamespace(
        index=index,
        text="ok",
        token_ids=list(token_ids),
        logprobs=[{tok: SimpleNamespace(logprob=-0.1)} for tok in token_ids],
        finish_reason=finish_reason,
    )


def _request_output(
    *,
    request_id="0",
    prompt_token_ids=(1, 2),
    outputs=None,
    num_generation_tokens=4,
):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        num_cached_tokens=0,
        metrics=SimpleNamespace(
            first_token_latency=0.012,
            queued_ts=1.0,
            scheduled_ts=1.005,
            first_token_ts=1.017,
            last_token_ts=1.047,
            num_generation_tokens=num_generation_tokens,
        ),
        outputs=list(outputs or [_sample()]),
    )


def _generator(outputs):
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._engine = _FakeEngine(outputs)
    generator.policy_version = 7
    generator.config = SimpleNamespace(
        sampling=SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4),
        debug=SimpleNamespace(seed=None),
    )
    return generator


def _run_generate(generator, tokenized_prompts, **kwargs):
    return asyncio.run(
        VLLMGenerator.generate._method(
            generator, tokenized_prompts, **kwargs
        )  # noqa: SLF001
    )


def test_generate_passes_token_prompt_to_vllm():
    generator = _generator([_request_output(prompt_token_ids=[1, 2, 3])])

    _run_generate(generator, [[1, 2, 3]])

    # add_request is invoked entirely with kwargs. The ``prompt`` kwarg
    # carries the renderer's output: a typed EngineInput (``type="token"``)
    # with the prompt token IDs and a stamped ``arrival_time`` — keeping us
    # off vLLM's deprecated raw-prompt path.
    (_args, kwargs) = generator._engine.add_requests[0]
    assert kwargs["request_id"] == "0"
    assert kwargs["prompt"]["type"] == "token"
    assert kwargs["prompt"]["prompt_token_ids"] == [1, 2, 3]
    assert "arrival_time" in kwargs["prompt"]


def test_generate_carries_finish_reason_and_metrics():
    output = _request_output(
        outputs=[
            _sample(index=0, token_ids=(10, 11), finish_reason="length"),
            _sample(index=1, token_ids=(12,), finish_reason="stop"),
        ]
    )
    generator = _generator([output])

    completions, generation_metrics = _run_generate(generator, [[1, 2]])

    assert [c.finish_reason for c in completions] == ["length", "stop"]
    assert not hasattr(completions[0], "metrics")
    aggregate = m.MetricsProcessor._aggregate_metrics(generation_metrics)
    assert aggregate["generator/output_tokens/sum"] == 3
    assert aggregate["generator/num_cached_tokens/mean"] == 0
    assert aggregate["generator/num_cached_tokens/max"] == 0
    assert aggregate["generator/time_to_first_token_ms/mean"] == 12
    assert aggregate["generator/time_to_first_token_ms/max"] == 12
    assert aggregate["generator/queue_time_ms/mean"] == pytest.approx(5)
    assert aggregate["generator/queue_time_ms/max"] == pytest.approx(5)
    assert aggregate["generator/prefill_time_ms/mean"] == pytest.approx(12)
    assert aggregate["generator/prefill_time_ms/max"] == pytest.approx(12)
    assert aggregate["generator/decode_time_ms/mean"] == pytest.approx(30)
    assert aggregate["generator/decode_time_ms/max"] == pytest.approx(30)
    assert aggregate["generator/inter_token_latency_ms/mean"] == pytest.approx(10)
    assert aggregate["generator/inter_token_latency_ms/max"] == pytest.approx(10)
    assert "generator/e2e_latency_ms/mean" not in aggregate


def test_generate_metrics_prefix_override_namespaces_keys():
    output = _request_output(
        outputs=[_sample(index=0, token_ids=(10, 11))],
    )
    generator = _generator([output])

    _, generation_metrics = _run_generate(
        generator, [[1, 2]], metrics_prefix="validation/generator"
    )

    metric_keys = {metric.key for metric in generation_metrics}
    assert "validation/generator/output_tokens" in metric_keys
    assert "validation/generator/queue_time_ms" in metric_keys
    assert all(key.startswith("validation/generator/") for key in metric_keys)


def test_decode_metrics_are_absent_for_single_generated_token():
    generator = _generator(
        [
            _request_output(
                outputs=[_sample(index=0, token_ids=(10,))],
                num_generation_tokens=1,
            )
        ]
    )

    _, generation_metrics = _run_generate(generator, [[1, 2]])

    metric_keys = {metric.key for metric in generation_metrics}
    assert "generator/prefill_time_ms" in metric_keys
    assert "generator/decode_time_ms" not in metric_keys
    assert "generator/inter_token_latency_ms" not in metric_keys
