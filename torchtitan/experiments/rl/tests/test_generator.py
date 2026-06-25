# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the `VLLMGenerator` continuous-batching mechanics.

Exercises the per-request pieces in isolation with a fake vLLM engine — no Monarch,
no GPU, no real model, and no broadcast (the engine loop's broadcast/step is a TP collective,
not unit-tested here; `test_engine_loop.py` covers the decision logic in `_decide_next_action`).
Covers completion (token-out + the metrics that ride with it),
the SamplingParams contract, and the vLLM metric timing math.
"""

import asyncio
from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind

from torchtitan.config import DebugConfig
from torchtitan.experiments.rl.actors.generator import (
    _prepare_generation_request_metrics,
    GenerationFuture,
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability import metrics as m


class _FakeRenderer:
    """Stub for vLLM's Renderer.render_cmpl: token-id dicts in, typed EngineInputs out."""

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
    def __init__(self):
        self.add_requests = []
        self.renderer = _FakeRenderer()

    def add_request(self, *args, **kwargs):
        self.add_requests.append((args, kwargs))


def _sample(*, token_ids=(10, 11), finish_reason="stop"):
    return SimpleNamespace(
        token_ids=list(token_ids),
        logprobs=[{tok: SimpleNamespace(logprob=-0.1)} for tok in token_ids],
        finish_reason=finish_reason,
    )


def _request_output(*, request_id="r0", outputs=None, num_generation_tokens=4):
    return SimpleNamespace(
        request_id=request_id,
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


def _generator():
    """A bare generator (no __init__ / engine build) with just the CB state set."""
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._engine = _FakeEngine()
    generator._rank = 0
    generator.policy_version = 7
    generator._generation_futures = {}
    generator.config = SimpleNamespace(
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=4),
        debug=SimpleNamespace(seed=None),
    )
    return generator


# --- completion (token-out) ---


def test_process_finished_requests_resolves_future_with_completion():
    async def main():
        generator = _generator()
        future = asyncio.get_running_loop().create_future()
        # Admitted (sampled) under v7 (the min); a weight pull then advanced the live version to 8 (the max).
        generator.policy_version = 8
        generation_future = GenerationFuture(future=future, metrics_prefix="generator")
        generation_future.min_policy_version = 7
        generator._generation_futures = {"r0": generation_future}

        generator._process_finished_requests(
            [
                _request_output(
                    outputs=[_sample(token_ids=(10, 11), finish_reason="length")]
                )
            ]
        )

        completion = await future
        assert completion.request_id == "r0"
        assert completion.token_ids == [10, 11]
        assert completion.token_logprobs == [-0.1, -0.1]
        assert completion.finish_reason == "length"
        assert completion.min_policy_version == 7  # min = version it was admitted under
        assert completion.max_policy_version == 8  # max = live version at finish
        # The request is popped from the in-flight map.
        assert generator._generation_futures == {}
        # The per-generation metrics ride on the completion.
        assert (
            m.MetricsProcessor._aggregate_metrics(completion.metrics)[
                "generator/inflight_requests_at_completion/max"
            ]
            == 1
        )

    asyncio.run(main())


def test_process_finished_requests_noop_on_followers():
    # Followers hold no futures, so completion is a no-op (it returns before touching outputs).
    generator = _generator()
    generator._rank = 1
    generator._process_finished_requests([_request_output(request_id="r0")])
    assert generator._generation_futures == {}


# --- SamplingParams contract (must match the batched path exactly) ---


def test_build_sampling_params_matches_contract():
    # seed and stop_token_ids are carried on the SamplingConfig (the rollouter
    # offsets the seed per sample); _build_sampling_params just reads them.
    generator = _generator()
    params = generator._build_sampling_params(
        SamplingConfig(
            temperature=0.3,
            top_p=0.9,
            max_tokens=64,
            seed=44,
            stop_token_ids=[99],
        )
    )
    assert params.temperature == 0.3 and params.top_p == 0.9
    assert params.max_tokens == 64
    assert params.n == 1
    assert params.logprobs == 0
    assert params.output_kind == RequestOutputKind.FINAL_ONLY
    assert params.stop_token_ids == [99]
    assert params.seed == 44


def test_build_sampling_params_seed_and_stop_default_to_none():
    generator = _generator()
    params = generator._build_sampling_params(
        SamplingConfig(temperature=0.8, top_p=0.95, max_tokens=8)
    )
    assert params.seed is None
    assert not params.stop_token_ids  # vLLM normalizes None -> []


# --- vLLM metric timing math (the `_prepare_generation_request_metrics` helper) ---


def test_metric_timing_math_and_prefix_override():
    metrics = _prepare_generation_request_metrics(
        _request_output(), prefix="validation_generator"
    )
    aggregate = m.MetricsProcessor._aggregate_metrics(metrics)
    assert all(key.startswith("validation_generator/") for key in aggregate)
    assert aggregate["validation_generator/queue_time_ms/mean"] == pytest.approx(5)
    assert aggregate["validation_generator/time_to_first_token_ms/mean"] == 12
    assert aggregate["validation_generator/prefill_time_ms/mean"] == pytest.approx(12)
    assert aggregate["validation_generator/decode_time_ms/mean"] == pytest.approx(30)
    assert aggregate[
        "validation_generator/inter_token_latency_ms/mean"
    ] == pytest.approx(10)


def test_decode_metrics_absent_for_single_generated_token():
    metrics = _prepare_generation_request_metrics(
        _request_output(num_generation_tokens=1), prefix="generator"
    )
    keys = {metric.key for metric in metrics}
    assert "generator/prefill_time_ms" in keys
    assert "generator/decode_time_ms" not in keys
    assert "generator/inter_token_latency_ms" not in keys


# --- config guards (weight-sync invariants) ---

# A valid inference parallelism; the weight-sync guards run after it is accepted.
_PARALLELISM = InferenceParallelismConfig()


def test_batch_invariant_requires_prefix_cache_reset():
    with pytest.raises(ValueError, match="reset_prefix_cache_on_weight_sync"):
        VLLMGenerator.Config(
            parallelism=_PARALLELISM,
            debug=DebugConfig(batch_invariant=True),
            reset_prefix_cache_on_weight_sync=False,
        )


def test_reset_running_requests_requires_prefix_cache_reset():
    with pytest.raises(ValueError, match="reset_prefix_cache_on_weight_sync"):
        VLLMGenerator.Config(
            parallelism=_PARALLELISM,
            reset_running_requests_on_weight_sync=True,
            reset_prefix_cache_on_weight_sync=False,
        )


def test_trainer_requires_prefix_cache_reset_when_hotswap_off():
    # Strict drain (hot_swap=False) needs the prefix cache reset so post-pull requests don't reuse old-weight KV.
    import dataclasses

    from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
        rl_grpo_qwen3_0_6b_varlen,
    )

    config = rl_grpo_qwen3_0_6b_varlen()
    # hot_swap defaults True; the guard fires only in drain mode (hot_swap=False) with reset also off.
    with pytest.raises(ValueError, match="reset_prefix_cache_on_weight_sync"):
        dataclasses.replace(
            config,
            generator_router=dataclasses.replace(
                config.generator_router, hot_swap=False
            ),
            generator=dataclasses.replace(
                config.generator, reset_prefix_cache_on_weight_sync=False
            ),
        )


# --- CUDA graph config (VLLMCudagraphConfig.get_vllm_compilation_config) ---


def test_cudagraph_disabled_returns_none():
    assert (
        VLLMCudagraphConfig(enable=False).get_vllm_compilation_config(max_num_seqs=256)
        is None
    )


def test_cudagraph_default_mode_is_full_decode_only():
    # Default mode; decode-only graphs avoid the mixed-batch corruption (#3668),
    # with no inductor compile (CompilationMode.NONE == 0).
    cfg = VLLMCudagraphConfig(enable=True).get_vllm_compilation_config(max_num_seqs=256)
    assert cfg.cudagraph_mode.name == "FULL_DECODE_ONLY"
    assert int(cfg.mode) == 0


def test_cudagraph_full_mode_no_compile():
    # FULL captures the whole forward (incl. attention) with no inductor compile.
    cfg = VLLMCudagraphConfig(enable=True, mode="FULL").get_vllm_compilation_config(
        max_num_seqs=256
    )
    assert cfg.cudagraph_mode.name == "FULL"
    assert int(cfg.mode) == 0


def test_cudagraph_decode_only_capture_sizes_cover_max_num_seqs():
    # FULL_DECODE_ONLY only graphs decode, so capture up to max_num_seqs (plus
    # max_num_seqs itself when not a power of 2).
    cfg = VLLMCudagraphConfig(enable=True).get_vllm_compilation_config(max_num_seqs=500)
    assert cfg.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]


def test_cudagraph_full_mode_extends_capture_sizes_to_chunk():
    # FULL also graphs prefill, so sizes extend to the chunked-prefill chunk
    # (max_num_batched_tokens, 2048) on top of max_num_seqs.
    cfg = VLLMCudagraphConfig(enable=True, mode="FULL").get_vllm_compilation_config(
        max_num_seqs=500
    )
    assert cfg.cudagraph_capture_sizes[-1] == 2048
    assert 500 in cfg.cudagraph_capture_sizes  # decode batch captured exactly


def test_cudagraph_rejects_nonpositive_max_num_seqs():
    with pytest.raises(ValueError, match="max_num_seqs must be positive"):
        VLLMCudagraphConfig(enable=True).get_vllm_compilation_config(max_num_seqs=0)
