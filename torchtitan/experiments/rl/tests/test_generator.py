# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the `VLLMGenerator` continuous-batching mechanics.

Exercises the per-request pieces in isolation with a fake vLLM engine — no Monarch,
no GPU, no real model, and no broadcast (the engine loop's broadcast/step is a TP collective,
not unit-tested here; `test_engine_loop.py` covers the decision logic in `_decide_next_action`).
Covers completion (token-out + the metrics that ride with it),
the SamplingParams contract, and the vLLM metric timing math.

The uneven-decode integration test requires four GPUs and a torchrun launcher.
"""

import asyncio
import gc
import math
import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import CommConfig, DebugConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.rl.actors.generator import (
    _extract_request_metrics_inputs,
    _prepare_generation_request_metrics,
    GenerationFuture,
    RequestDispatcher,
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.models.vllm_registry import (
    InferenceParallelismConfig,
    register_to_vllm,
)
from torchtitan.experiments.rl.models.vllm_worker import TorchTitanGPUModelRunner
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.routing.intra_generator_router import (
    IntraGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import LeastLoadedRoutingStrategy


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
    """A bare generator (no __init__ / engine build) with just the state the
    per-request helpers (`_build_sampling_params`) read."""
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._engine = _FakeEngine()
    generator._rank = 0
    generator.policy_version = 7
    generator.config = SimpleNamespace(
        sampling=SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=4),
        debug=SimpleNamespace(seed=None),
    )
    return generator


def _dispatcher(*, rank=0, dp_degree=1, tp_degree=1, dp_routing_strategy=None):
    """A bare RequestDispatcher; broadcast_group is unused unless ``setup`` runs."""
    return RequestDispatcher(
        rank=rank,
        dp_rank=rank // tp_degree,
        tp_rank=rank % tp_degree,
        dp_degree=dp_degree,
        broadcast_group=None,
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=dp_routing_strategy or LeastLoadedRoutingStrategy.Config()
        ),
    )


# --- completion (token-out) ---


def test_process_finished_requests_resolves_future_with_completion():
    async def main():
        # DP=1: rank 0 is the single replica's leader, so it builds and resolves locally.
        dispatcher = _dispatcher()
        future = asyncio.get_running_loop().create_future()
        # Admitted (sampled) under v7 (the min); a weight pull then advanced the live version to 8 (the max).
        generation_future = GenerationFuture(future=future, metrics_prefix="generator")
        generation_future.min_policy_version = 7
        dispatcher._rank0_generation_futures = {"r0": generation_future}

        dispatcher.process_finished_requests(
            [
                _request_output(
                    outputs=[_sample(token_ids=(10, 11), finish_reason="length")]
                )
            ],
            policy_version=8,
        )

        completion = await future
        assert completion.request_id == "r0"
        assert completion.token_ids == [10, 11]
        assert completion.token_logprobs == [-0.1, -0.1]
        assert completion.finish_reason == "length"
        assert completion.min_policy_version == 7  # min = version it was admitted under
        assert completion.max_policy_version == 8  # max = live version at finish
        # The request is popped from the in-flight map.
        assert dispatcher._rank0_generation_futures == {}
        # The per-generation metrics ride on the completion (built on rank 0).
        assert (
            m.MetricsProcessor._aggregate_metrics(completion.metrics)[
                "generator/inflight_requests_at_completion/max"
            ]
            == 1
        )

    asyncio.run(main())


def test_process_finished_requests_noop_on_nonzero_tp_rank():
    # tp_rank != 0 hold no finished outputs, so processing returns before building or sending.
    dispatcher = _dispatcher(rank=1, dp_degree=1, tp_degree=2)
    assert dispatcher._tp_rank != 0
    dispatcher.process_finished_requests(
        [_request_output(request_id="r0")], policy_version=7
    )
    assert dispatcher._rank0_generation_futures == {}


def test_process_finished_requests_releases_dp_router_load():
    async def main():
        dispatcher = _dispatcher(dp_degree=2)
        assert dispatcher._rank0_dp_router is not None
        future = asyncio.get_running_loop().create_future()
        generation_future = GenerationFuture(future=future, metrics_prefix="generator")
        generation_future.min_policy_version = 7
        dispatcher._rank0_generation_futures = {"r0": generation_future}
        dispatcher._rank0_dp_router.reserve("r0", routing_session_id=None)
        # The reservation is recorded (least-loaded picks DP rank 0) and loads it.
        assert dispatcher._rank0_dp_router._reservations == {"r0": 0}
        assert [h.reserved_load for h in dispatcher._rank0_dp_router._handles] == [1, 0]

        dispatcher.process_finished_requests(
            [_request_output(request_id="r0")], policy_version=7
        )

        await future
        # Resolving the completion releases the reservation and its load.
        assert dispatcher._rank0_dp_router._reservations == {}
        assert [h.reserved_load for h in dispatcher._rank0_dp_router._handles] == [0, 0]

    asyncio.run(main())


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
        _extract_request_metrics_inputs(_request_output()),
        prefix="validation_generator",
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
        _extract_request_metrics_inputs(_request_output(num_generation_tokens=1)),
        prefix="generator",
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


def test_qwen36_27b_config_applies_offset_rmsnorm_to_both_actors():
    from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
        rl_grpo_qwen3_6_27b_varlen_perf,
    )

    config = rl_grpo_qwen3_6_27b_varlen_perf()
    override_import = "torchtitan.overrides.offset_rmsnorm.triton_offset_rmsnorm"

    assert config.hf_assets_path.endswith("Qwen3.6-27B")
    assert config.trainer.override.imports == [override_import]
    assert config.generator.override.imports == [override_import]
    assert config.trainer.parallelism.data_parallel_shard_degree == 2
    assert config.trainer.parallelism.tensor_parallel_degree == 2
    assert config.generator.parallelism.tensor_parallel_degree == 4
    assert config.trainer.optimizer.implementation == "fused_opt_states_bf16"
    assert isinstance(config.trainer.ac_config, FullAC.Config)
    assert config.generator.cudagraph.enable


# --- CUDA graph config (VLLMCudagraphConfig.get_vllm_compilation_config) ---


def test_cudagraph_disabled_preserves_sequence_parallel_config():
    compilation_config = VLLMCudagraphConfig(enable=False).get_vllm_compilation_config(
        max_num_seqs=256,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=True,
    )

    assert compilation_config.cudagraph_mode.name == "NONE"
    assert compilation_config.pass_config.enable_sp
    assert compilation_config.pass_config.sp_min_token_num == 1


def test_expert_sequence_parallel_padding_filters_cudagraph_sizes():
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=4,
        expert_parallel_degree=4,
    )
    cfg = VLLMCudagraphConfig(
        enable=True,
        capture_sizes=[1, 4, 5, 8],
    ).get_vllm_compilation_config(
        max_num_seqs=8,
        expert_sequence_parallel_size=parallelism.expert_sequence_parallel_size,
        enable_sequence_parallel=False,
    )

    assert cfg.cudagraph_capture_sizes == [4, 8]


def test_expert_sequence_parallel_padding_keeps_small_cudagraph_batches():
    cfg = VLLMCudagraphConfig(enable=True).get_vllm_compilation_config(
        max_num_seqs=1,
        expert_sequence_parallel_size=8,
        enable_sequence_parallel=False,
    )

    assert cfg.cudagraph_capture_sizes[0] == 8
    assert all(size % 8 == 0 for size in cfg.cudagraph_capture_sizes)


def test_expert_sequence_parallel_padding_rejects_no_valid_cudagraph_sizes():
    with pytest.raises(ValueError, match="No CUDA graph capture sizes"):
        VLLMCudagraphConfig(
            enable=True,
            capture_sizes=[1, 2, 3],
        ).get_vllm_compilation_config(
            max_num_seqs=3,
            expert_sequence_parallel_size=4,
            enable_sequence_parallel=False,
        )


def test_expert_sequence_parallel_padding_disabled_without_ep():
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=4,
        expert_parallel_degree=1,
    )

    assert parallelism.expert_sequence_parallel_size == 1


@pytest.mark.parametrize(
    ("enable_dense_sp", "enable_expert_sp", "expected_num_tokens"),
    [
        (False, True, 8),
        (True, False, 8),
        (False, False, 5),
    ],
)
def test_sequence_parallel_padding_rounds_runner_tokens(
    enable_dense_sp, enable_expert_sp, expected_num_tokens
):
    model_runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_dense_sp)
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_expert_parallel=enable_expert_sp,
                tensor_parallel_size=4,
            ),
        ),
    )

    assert (
        TorchTitanGPUModelRunner._pad_for_sequence_parallelism(model_runner, 5)
        == expected_num_tokens
    )


def test_cudagraph_default_mode_is_full_decode_only():
    # Default mode; decode-only graphs avoid the mixed-batch corruption (#3668),
    # with no inductor compile (CompilationMode.NONE == 0).
    cfg = VLLMCudagraphConfig(enable=True).get_vllm_compilation_config(
        max_num_seqs=256,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_mode.name == "FULL"
    assert int(cfg.mode) == 0


def test_cudagraph_full_mode_no_compile():
    # FULL captures the whole forward (incl. attention) with no inductor compile.
    cfg = VLLMCudagraphConfig(enable=True, mode="FULL").get_vllm_compilation_config(
        max_num_seqs=256,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_mode.name == "FULL"
    assert int(cfg.mode) == 0


def test_cudagraph_decode_only_capture_sizes_cover_max_num_seqs():
    # FULL_DECODE_ONLY only graphs decode, so capture up to max_num_seqs (plus
    # max_num_seqs itself when not a power of 2).
    cfg = VLLMCudagraphConfig(
        enable=True, mode="FULL_DECODE_ONLY"
    ).get_vllm_compilation_config(
        max_num_seqs=500,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]


def test_cudagraph_full_mode_extends_capture_sizes_to_chunk():
    # FULL also graphs prefill, so sizes extend to the chunked-prefill chunk
    # (max_num_batched_tokens, 2048) on top of max_num_seqs.
    cfg = VLLMCudagraphConfig(enable=True, mode="FULL").get_vllm_compilation_config(
        max_num_seqs=500,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_capture_sizes[-1] == 2048
    assert 500 in cfg.cudagraph_capture_sizes  # decode batch captured exactly


def test_cudagraph_rejects_nonpositive_max_num_seqs():
    with pytest.raises(ValueError, match="max_num_seqs must be positive"):
        VLLMCudagraphConfig(enable=True).get_vllm_compilation_config(
            max_num_seqs=0,
            expert_sequence_parallel_size=1,
            enable_sequence_parallel=False,
        )


def test_inference_parallelism_propagates_dense_sequence_parallelism():
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=4,
        enable_sequence_parallel=True,
    )

    assert parallelism.to_training().enable_sequence_parallel

    compilation_config = VLLMCudagraphConfig(
        capture_sizes=[1, 4, 5, 8]
    ).get_vllm_compilation_config(
        max_num_seqs=8,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=parallelism.enable_sequence_parallel,
    )
    assert compilation_config.cudagraph_capture_sizes == [1, 4, 5, 8]
    assert compilation_config.pass_config.enable_sp
    assert compilation_config.pass_config.sp_min_token_num == 1


def test_inference_parallelism_disables_dense_sequence_parallelism():
    parallelism = InferenceParallelismConfig(tensor_parallel_degree=4)

    assert not parallelism.to_training().enable_sequence_parallel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vllm_uneven_decode_tp_padding():
    """Three decode tokens run through EP-internal TP sequence sharding."""
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size != 4:
        pytest.skip(f"requires exactly 4 GPUs, got {world_size}")

    from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
        rl_grpo_qwen3_moe_debug_varlen,
    )
    from torchtitan.experiments.rl.tests.test_bitwise_parity import (
        _make_prompt_tokens,
        _run_engine,
        build_inference_engine,
    )

    config = rl_grpo_qwen3_moe_debug_varlen()
    config.generator.parallelism.data_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 4
    config.generator.gpu_memory_limit = 0.5

    temporary_dump_folder = None
    if not dist.is_initialized():
        temporary_dump_folder = tempfile.mkdtemp(prefix="rl_generator_moe_")
        dist_utils.init_distributed(
            CommConfig(),
            base_folder=temporary_dump_folder,
        )

    register_to_vllm(
        config.model_spec,
        parallelism=config.generator.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(enable=False),
        override=config.generator.override,
    )

    engine = build_inference_engine(config)
    try:
        prompt_ids = _make_prompt_tokens(3, 100, engine.get_tokenizer())
        outputs = _run_engine(
            engine,
            "uneven_decode",
            prompt_ids,
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=2,
                ignore_eos=True,
                logprobs=1,
                output_kind=RequestOutputKind.FINAL_ONLY,
            ),
        )

        for output in outputs:
            sample = output.outputs[0]
            assert len(sample.token_ids) == 2
            assert len(sample.logprobs) == 2
            assert all(
                math.isfinite(list(logprobs.values())[0].logprob)
                for logprobs in sample.logprobs
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
        renderer = getattr(engine, "renderer", None)
        if renderer is not None:
            renderer.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        if temporary_dump_folder is not None:
            shutil.rmtree(temporary_dump_folder, ignore_errors=True)
