# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the generator's per-request pieces: `VLLMEngine` (with a fake vLLM
`LLMEngine`), `RequestDispatcher`, and the `Generator` config and endpoints.

Exercises them in isolation -- no Monarch, no GPU, no real model, and no broadcast
(`test_engine_loop.py` covers the engine loop). Covers completion (token-out + the
metrics that ride with it), the SamplingParams contract, and the vLLM metric timing math.

The uneven-decode integration test requires four GPUs and a torchrun launcher.
"""

import asyncio
import concurrent.futures
import gc
import math
import os
import shutil
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
import torch.distributed as dist

import torchtitan.rl.generator.generator as generator_module
import torchtitan.rl.generator.vllm.engine as vllm_engine_module
from torchstore import RankRole
from torchtitan.components.optim import AdamW
from torchtitan.config import CommConfig, DebugConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.rl.generator.dispatcher import (
    _prepare_generation_request_metrics,
    _RejectedRequests,
    OutstandingGeneration,
    RequestDispatcher,
)
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.engine import (
    _build_sampling_params,
    _extract_request_metrics_inputs,
    _finished_request,
    add_extra_vllm_engine_args,
    VLLMEngine,
)
from torchtitan.rl.generator.vllm.registry import register_to_vllm
from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
from torchtitan.rl.generator.vllm.worker import (
    TorchTitanGPUModelRunner,
    TorchTitanGPUWorker,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.routing.strategies import LeastLoadedRoutingStrategy
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.sampling_params import RequestOutputKind


class _FakeRenderer:
    """Stub for vLLM's Renderer. render_cmpl: token-id dicts in, typed EngineInputs out."""

    def render_cmpl(self, prompts):
        return [
            {
                "type": "token",
                "prompt_token_ids": p["prompt_token_ids"],
                "cache_salt": p.get("cache_salt"),
                "arrival_time": 0.0,
            }
            for p in prompts
        ]

    def shutdown(self):
        pass


class _FakeLLMEngine:
    def __init__(self):
        self.add_requests = []
        self.reset_prefix_cache_calls = []
        self.renderer = _FakeRenderer()

    def add_request(self, *args, **kwargs):
        self.add_requests.append((args, kwargs))

    def has_unfinished_requests(self):
        return False

    def reset_prefix_cache(self, *args, **kwargs):
        self.reset_prefix_cache_calls.append((args, kwargs))


def _sample(*, token_ids=(10, 11), finish_reason="stop"):
    # What vLLM returns with flat_logprobs=True and logprobs=0: one logprob (-0.1) per generated token.
    logprobs = FlatLogprobs()
    for tok in token_ids:
        logprobs.append({tok: Logprob(logprob=-0.1)})
    return SimpleNamespace(
        token_ids=list(token_ids),
        logprobs=logprobs,
        finish_reason=finish_reason,
    )


def _request_output(
    *, request_id="r0", outputs=None, num_generation_tokens=4, num_preemptions=0
):
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
            num_preemptions=num_preemptions,
        ),
        outputs=list(outputs or [_sample()]),
    )


def _vllm_engine(*, reset_kv_cache_on_model_state_dict_load=False, model=None):
    """A `VLLMEngine` around a fake `LLMEngine` (no engine build)."""
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._engine = _FakeLLMEngine()
    engine._reset_kv_cache_on_model_state_dict_load = (
        reset_kv_cache_on_model_state_dict_load
    )
    engine._get_model = lambda: model
    engine._session_kv_holder = None
    return engine


def test_initialize_torchstore_client_uses_requester_index():
    async def main():
        generator = Generator.__new__(Generator)
        client = AsyncMock()
        with patch.object(generator_module.ts, "client", client):
            await generator.initialize_torchstore_client(requester_index=3)

        client.assert_awaited_once_with(role=RankRole.REQUESTER, group=3)

    asyncio.run(main())


def test_prefetch_model_state_dict_updates_staging_buffers_in_place():
    async def main():
        staging_state_dict = {"weight": "old"}
        generator = Generator.__new__(Generator)
        generator._prefetched_model_state_dict = staging_state_dict

        def fill_state_dict(*args, **kwargs):
            assert kwargs["user_state_dict"] is staging_state_dict
            staging_state_dict["weight"] = "fetched"
            return {"weight": "fetched"}

        load_state_dict = Mock()
        engine = _vllm_engine(
            model=SimpleNamespace(
                load_state_dict=load_state_dict,
                _prefetched_model_state_dict=staging_state_dict,
            )
        )

        get_state_dict = AsyncMock(side_effect=fill_state_dict)
        with patch.object(generator_module.ts, "get_state_dict", get_state_dict):
            await generator.prefetch_model_state_dict()
        engine.load_prefetched_model_state_dict()

        assert generator._prefetched_model_state_dict is staging_state_dict
        get_state_dict.assert_awaited_once_with(
            "model_state_dict",
            user_state_dict=staging_state_dict,
            strict=False,
            direct_rdma=False,
        )
        load_state_dict.assert_called_once_with({"weight": "fetched"})

    asyncio.run(main())


def _dispatcher(*, rank=0, dp_degree=1, tp_degree=1, dp_routing_strategy=None):
    """A bare RequestDispatcher; broadcast_group is unused unless ``setup`` runs."""
    return RequestDispatcher(
        rank=rank,
        dp_rank=rank // tp_degree,
        tp_rank=rank % tp_degree,
        dp_degree=dp_degree,
        broadcast_group=None,
        open_result_channel=None,
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=dp_routing_strategy or LeastLoadedRoutingStrategy.Config()
        ),
    )


# --- completion (token-out) ---


def test_process_finished_requests_resolves_reply_with_completion():
    async def main():
        # DP=1: rank 0 is the single replica's leader, so it builds and resolves locally.
        dispatcher = _dispatcher()
        reply = concurrent.futures.Future()
        reply.set_running_or_notify_cancel()  # admitted, as the engine loop leaves it
        # Admitted (sampled) under v7 (the min); a model state dict load then advanced the live version to 8 (the max).
        generation = OutstandingGeneration(reply=reply, metrics_prefix="generator")
        generation.min_policy_version = 7
        dispatcher._rank0_outstanding_generations = {"r0": generation}

        dispatcher.process_finished_requests(
            [
                _finished_request(
                    _request_output(
                        outputs=[_sample(token_ids=(10, 11), finish_reason="length")]
                    )
                )
            ],
            policy_version=8,
        )

        completion = await asyncio.wrap_future(reply)
        assert completion.request_id == "r0"
        assert completion.token_ids == [10, 11]
        assert completion.token_logprobs == [-0.1, -0.1]
        assert completion.finish_reason == "length"
        assert completion.min_policy_version == 7  # min = version it was admitted under
        assert completion.max_policy_version == 8  # max = live version at finish
        # The request is popped from the in-flight map.
        assert dispatcher._rank0_outstanding_generations == {}
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
        [_finished_request(_request_output(request_id="r0"))], policy_version=7
    )
    assert dispatcher._rank0_outstanding_generations == {}


def test_process_finished_requests_releases_dp_router_load():
    async def main():
        dispatcher = _dispatcher(dp_degree=2)
        assert dispatcher._rank0_dp_router is not None
        reply = concurrent.futures.Future()
        reply.set_running_or_notify_cancel()  # admitted, as the engine loop leaves it
        generation = OutstandingGeneration(reply=reply, metrics_prefix="generator")
        generation.min_policy_version = 7
        dispatcher._rank0_outstanding_generations = {"r0": generation}
        dispatcher._rank0_dp_router.reserve("r0", routing_session_id=None)
        # The reservation is recorded (least-loaded picks DP rank 0) and loads it.
        assert dispatcher._rank0_dp_router._reservations == {"r0": 0}
        assert [h.reserved_load for h in dispatcher._rank0_dp_router._handles] == [1, 0]

        dispatcher.process_finished_requests(
            [_finished_request(_request_output(request_id="r0"))], policy_version=7
        )

        await asyncio.wrap_future(reply)
        # Resolving the completion releases the reservation and its load.
        assert dispatcher._rank0_dp_router._reservations == {}
        assert [h.reserved_load for h in dispatcher._rank0_dp_router._handles] == [0, 0]

    asyncio.run(main())


# --- SamplingParams contract (must match the batched path exactly) ---


def test_build_sampling_params_matches_contract():
    # seed and stop_token_ids are carried on the SamplingConfig (the rollouter
    # offsets the seed per sample); _build_sampling_params just reads them.
    params = _build_sampling_params(
        SamplingConfig(
            temperature=0.3,
            max_tokens=64,
            seed=44,
            stop_token_ids=[99],
        )
    )
    assert params.temperature == 0.3 and params.top_p == 1.0
    assert params.max_tokens == 64
    assert params.n == 1
    assert params.logprobs == 0
    assert params.flat_logprobs and not params.detokenize
    assert params.output_kind == RequestOutputKind.FINAL_ONLY
    assert params.stop_token_ids == [99]
    assert params.ignore_eos
    assert params.seed == 44


def test_build_sampling_params_seed_defaults_to_none():
    params = _build_sampling_params(
        SamplingConfig(temperature=0.8, max_tokens=8, stop_token_ids=[99])
    )
    assert params.seed is None


def _engine_request(request_id: str, *, min_policy_version: int, group_id: int = 3):
    request = EngineRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling=SamplingConfig(),
        group_id=group_id,
        routing_session_id="group=3/rollout=0",
        arrival_time=1.5,
    )
    request.min_policy_version = min_policy_version
    return request


def test_admission_passes_group_id_as_priority_and_arrival_time():
    # Priority is the group id (3), not the pinned version (6); every validation id shares -1.
    engine = _vllm_engine()
    engine.add_requests(
        [
            _engine_request("train", min_policy_version=6, group_id=3),
            _engine_request("val_1", min_policy_version=6, group_id=-1),
            _engine_request("val_5", min_policy_version=6, group_id=-5),
        ]
    )

    llm_engine = cast(_FakeLLMEngine, engine._engine)
    added_kwargs = [kwargs for _, kwargs in llm_engine.add_requests]
    assert [kwargs["priority"] for kwargs in added_kwargs] == [3, -1, -1]
    assert [kwargs["arrival_time"] for kwargs in added_kwargs] == [1.5] * 3


def test_session_kv_holder_tracks_admitted_requests_and_releases_sessions():
    engine = _vllm_engine()
    holder = Mock()
    engine._session_kv_holder = holder
    engine._engine.add_request = Mock(return_value="r0-internal")
    engine._engine.step = lambda: []

    assert engine.holds_session_kv
    engine.add_requests([_engine_request("r0", min_policy_version=6)])
    engine.step()
    engine.release_sessions(["group=3/rollout=0"], [3])

    # The holder tracks vLLM's internal request id, which carries a random suffix.
    holder.track.assert_called_once_with("r0-internal", "group=3/rollout=0", 3)
    holder.after_step.assert_called_once_with()
    holder.release.assert_called_once_with(["group=3/rollout=0"], [3])


def test_admission_salts_prompt_with_min_policy_version():
    # The request's pinned version, not the installed one, salts the prefix cache.
    engine = _vllm_engine()
    request = _engine_request("r0", min_policy_version=6)

    assert engine.add_requests([request]) == []

    _, kwargs = cast(_FakeLLMEngine, engine._engine).add_requests[0]
    assert kwargs["prompt"]["cache_salt"] == "6"


def test_admission_with_kv_reset_does_not_salt_prompt():
    engine = _vllm_engine(reset_kv_cache_on_model_state_dict_load=True)

    engine.add_requests([_engine_request("r0", min_policy_version=6)])

    _, kwargs = cast(_FakeLLMEngine, engine._engine).add_requests[0]
    assert kwargs["prompt"]["cache_salt"] is None


def test_admission_rejects_only_the_invalid_request():
    engine = _vllm_engine()
    invalid = _engine_request("invalid", min_policy_version=6)
    invalid.sampling = SamplingConfig(max_tokens=0)
    valid = _engine_request("valid", min_policy_version=6)

    rejected_requests = engine.add_requests([invalid, valid])

    [(request_id, exc)] = rejected_requests
    assert request_id == "invalid"
    assert isinstance(exc, VLLMValidationError)
    assert "max_tokens must be at least 1" in str(exc)
    llm_engine = cast(_FakeLLMEngine, engine._engine)
    assert [kwargs["request_id"] for _, kwargs in llm_engine.add_requests] == ["valid"]


def test_process_rejected_requests_fails_only_their_replies():
    async def main():
        dispatcher = _dispatcher(dp_degree=2)
        replies = {}
        for request_id in ("r0", "r1"):
            replies[request_id] = concurrent.futures.Future()
            replies[request_id].set_running_or_notify_cancel()
            dispatcher.rank0_register_generation(
                request_id, "generator", replies[request_id]
            )
            dispatcher._rank0_dp_router.reserve(request_id, routing_session_id=None)
        error = ValueError("prompt too long")

        dispatcher.process_rejected_requests([("r0", error)])

        with pytest.raises(ValueError, match="prompt too long"):
            await asyncio.wrap_future(replies["r0"])
        assert not replies["r1"].done()
        assert list(dispatcher._rank0_outstanding_generations) == ["r1"]
        # The rejected request's DP rank load is freed.
        assert list(dispatcher._rank0_dp_router._reservations) == ["r1"]

    asyncio.run(main())


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_peer_dp_replica_sends_rejected_requests_to_rank0(tp_rank):
    # DP=2, TP=2: global rank 2 is DP rank 1's tp_rank=0; rank 3 is its tp_rank=1.
    dispatcher = _dispatcher(rank=2 + tp_rank, dp_degree=2, tp_degree=2)
    dispatcher._result_port = Mock()
    error = ValueError("prompt too long")

    dispatcher.process_rejected_requests([("r0", error)])

    if tp_rank == 0:
        dispatcher._result_port.send.assert_called_once_with(
            _RejectedRequests(requests=[("r0", error)])
        )
    else:
        dispatcher._result_port.send.assert_not_called()


def test_stamp_sets_outstanding_generation_min_policy_version():
    # A request may reuse KV cached under its pinned version, so that version bounds
    # the completion's staleness.
    dispatcher = _dispatcher()
    dispatcher._rank0_outstanding_generations = {
        "r0": SimpleNamespace(min_policy_version=None),
        "r1": SimpleNamespace(min_policy_version=None),
    }

    dispatcher.rank0_stamp_min_policy_version(
        [
            [
                _engine_request("r0", min_policy_version=4),
                _engine_request("r1", min_policy_version=6),
            ]
        ]
    )

    assert dispatcher._rank0_outstanding_generations["r0"].min_policy_version == 4
    assert dispatcher._rank0_outstanding_generations["r1"].min_policy_version == 6


def test_step_returns_finished_requests_without_grad():
    engine = _vllm_engine()
    grad_enabled = []

    def step():
        grad_enabled.append(torch.is_grad_enabled())
        return [_request_output(outputs=[_sample(token_ids=(10, 11))])]

    engine._engine.step = step

    [finished] = engine.step()

    assert grad_enabled == [False]
    assert finished.request_id == "r0"
    assert finished.token_ids == [10, 11]
    assert finished.token_logprobs == [-0.1, -0.1]
    assert finished.finish_reason == "stop"
    assert finished.metrics_inputs.num_generation_tokens == 4


def test_step_rejects_more_than_one_sample_per_request():
    engine = _vllm_engine()
    engine._engine.step = lambda: [_request_output(outputs=[_sample(), _sample()])]

    with pytest.raises(ValueError, match="expected n=1"):
        engine.step()


def test_shutdown_closes_the_renderer():
    engine = _vllm_engine()
    engine._engine.renderer.shutdown = Mock()

    engine.shutdown()

    engine._engine.renderer.shutdown.assert_called_once_with()


def test_prefetched_model_state_dict_is_the_buffers_the_load_reads():
    # The generator prefetches into this dict, so it must be the one the load copies from.
    buffers = {"weight": torch.zeros(1)}
    loaded = []
    engine = _vllm_engine(
        model=SimpleNamespace(
            load_state_dict=loaded.append, _prefetched_model_state_dict=buffers
        )
    )

    assert engine.prefetched_model_state_dict is buffers
    engine.load_prefetched_model_state_dict()
    assert loaded == [buffers]


@pytest.mark.parametrize("reset_kv_cache", [False, True])
def test_reset_kv_cache_on_model_state_dict_load_controls_cache_reset(
    reset_kv_cache: bool,
):
    model = SimpleNamespace(
        load_state_dict=lambda state_dict: None, _prefetched_model_state_dict={}
    )
    engine = _vllm_engine(
        reset_kv_cache_on_model_state_dict_load=reset_kv_cache, model=model
    )

    engine.load_prefetched_model_state_dict()

    expected = [((), {"reset_running_requests": True})] if reset_kv_cache else []
    assert cast(_FakeLLMEngine, engine._engine).reset_prefix_cache_calls == expected


def test_sampling_config_rejects_top_p_below_one():
    with pytest.raises(ValueError, match="top_p must be 1.0, got 0.95"):
        SamplingConfig(top_p=0.95)


# --- vLLM metric timing math (the `_prepare_generation_request_metrics` helper) ---


def test_metric_timing_math_and_prefix_override():
    metrics = _prepare_generation_request_metrics(
        _extract_request_metrics_inputs(_request_output(num_preemptions=2)),
        prefix="validation_generator",
    )
    aggregate = m.MetricsProcessor._aggregate_metrics(metrics)
    assert all(key.startswith("validation_generator/") for key in aggregate)
    assert aggregate["validation_generator/num_preemptions/mean"] == 2
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


# --- config guards (model state dict load invariants) ---

# A valid inference parallelism; the model state dict load guards run after it is accepted.
_PARALLELISM = InferenceParallelismConfig()
# The fields of a model config that VLLMEngine reads when it builds EngineArgs.
_MODEL_CONFIG = SimpleNamespace(
    first_full_attention=SimpleNamespace(inner_attention=VarlenInnerAttention.Config()),
    max_context_length=1024,
)


def test_generator_dp_requires_expert_parallelism():
    with pytest.raises(ValueError, match="Controller.Config.num_generators"):
        Generator.Config(
            model=_MODEL_CONFIG,
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=1,
            ),
        )


def test_generator_dp_can_supply_expert_parallelism():
    Generator.Config(
        model=_MODEL_CONFIG,
        parallelism=InferenceParallelismConfig(
            data_parallel_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        ),
    )


def test_batch_invariant_requires_kv_cache_reset():
    with pytest.raises(ValueError, match="reset_kv_cache_on_model_state_dict_load"):
        Generator.Config(
            model=_MODEL_CONFIG,
            parallelism=_PARALLELISM,
            debug=DebugConfig(batch_invariant=True),
        )


def test_session_kv_holder_rejects_kv_cache_reset_on_model_state_dict_load(tmp_path):
    with pytest.raises(
        ValueError, match="needs reset_kv_cache_on_model_state_dict_load=False"
    ):
        VLLMEngine.Config(session_kv_holder=VLLMSessionKVHolder.Config()).build(
            model=_MODEL_CONFIG,
            parallelism=_PARALLELISM,
            model_path="unused",
            max_num_seqs=8,
            model_dtype="bfloat16",
            gpu_memory_limit=0.9,
            max_num_batched_tokens=None,
            checkpointer=None,
            dist_moe_runtime=None,
            debug=DebugConfig(),
            reset_kv_cache_on_model_state_dict_load=True,
            rank=0,
            generator_name="generator",
            output_dir=str(tmp_path),
        )


def test_qwen36_27b_perf_config():
    from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_6_27b_varlen_perf

    config = rl_grpo_qwen3_6_27b_varlen_perf()

    assert config.hf_assets_path.endswith("Qwen3.6-27B")
    assert "offset_rmsnorm" in config.trainer.model.local_compile_regions
    assert "offset_rmsnorm" in config.generator.model.local_compile_regions
    assert config.trainer.parallelism.data_parallel_shard_degree == 2
    assert config.trainer.parallelism.tensor_parallel_degree == 2
    assert config.generator.parallelism.tensor_parallel_degree == 4
    optimizer = config.trainer.optim.optimizer.optimizers[0]
    assert isinstance(optimizer, AdamW.Config)
    assert optimizer.moment_dtype == "bfloat16"
    assert isinstance(config.trainer.activation_remat, FullActivationRematConfig)
    assert config.generator.engine.cuda_graph.mode == "FULL"


# --- vLLM engine kwargs (what VLLMEngine passes to EngineArgs) ---


def _build_engine_kwargs(monkeypatch, tmp_path, config: Generator.Config) -> dict:
    """Build ``config.engine`` as the generator does, with vLLM's ``EngineArgs``,
    ``LLMEngine`` and model registration mocked; return the ``EngineArgs`` kwargs."""
    engine_args = Mock()
    monkeypatch.setattr(vllm_engine_module, "EngineArgs", engine_args)
    monkeypatch.setattr(vllm_engine_module, "LLMEngine", Mock())
    monkeypatch.setattr(vllm_engine_module, "register_to_vllm", Mock())
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    config.engine.build(
        model=config.model,
        parallelism=config.parallelism,
        model_path="unused",
        max_num_seqs=8,
        model_dtype=config.model_dtype,
        gpu_memory_limit=config.gpu_memory_limit,
        max_num_batched_tokens=config.max_num_batched_tokens,
        checkpointer=None,
        dist_moe_runtime=None,
        debug=config.debug,
        reset_kv_cache_on_model_state_dict_load=False,
        rank=0,
        generator_name="generator",
        output_dir=str(tmp_path),
    )
    return engine_args.call_args.kwargs


def test_generator_defaults_to_the_vllm_engine():
    assert isinstance(Generator.Config(model=_MODEL_CONFIG).engine, VLLMEngine.Config)


def test_extra_engine_args_reach_engine_args(monkeypatch, tmp_path):
    engine_kwargs = _build_engine_kwargs(
        monkeypatch,
        tmp_path,
        Generator.Config(
            model=_MODEL_CONFIG,
            engine=VLLMEngine.Config(extra_engine_args={"watermark": 0.03}),
        ),
    )
    assert engine_kwargs["watermark"] == 0.03
    assert engine_kwargs["scheduling_policy"] == "priority"


@pytest.mark.parametrize(
    "key, match",
    [
        # Always set by TorchTitan.
        ("scheduling_policy", r"controls: scheduling_policy$"),
        # Set only when the config field is set, so absent from engine_kwargs here.
        (
            "max_num_batched_tokens",
            r"use Generator\.Config\.max_num_batched_tokens",
        ),
        ("seed", r"use Generator\.Config\.debug\.seed"),
    ],
)
def test_extra_engine_args_rejects_torchtitan_owned_keys(key, match):
    engine_kwargs = {"scheduling_policy": "fcfs"}
    with pytest.raises(ValueError, match=match):
        add_extra_vllm_engine_args(engine_kwargs, {key: 1})
    assert engine_kwargs == {"scheduling_policy": "fcfs"}


# --- CUDA graph config (VLLMCudaGraphConfig.get_vllm_compilation_config) ---


def test_cuda_graph_none_mode_preserves_sequence_parallel_config():
    compilation_config = VLLMCudaGraphConfig(mode="NONE").get_vllm_compilation_config(
        max_num_seqs=256,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=True,
    )

    assert compilation_config.cudagraph_mode.name == "NONE"
    assert compilation_config.pass_config.enable_sp
    assert compilation_config.pass_config.sp_min_token_num == 1


def test_expert_sequence_parallel_padding_filters_cuda_graph_sizes():
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=4,
        expert_parallel_degree=4,
    )
    cfg = VLLMCudaGraphConfig(
        mode="FULL", capture_sizes=[1, 4, 5, 8]
    ).get_vllm_compilation_config(
        max_num_seqs=8,
        expert_sequence_parallel_size=parallelism.expert_sequence_parallel_size,
        enable_sequence_parallel=False,
    )

    assert cfg.cudagraph_capture_sizes == [4, 8]


def test_expert_sequence_parallel_padding_keeps_small_cuda_graph_batches():
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
        max_num_seqs=1,
        expert_sequence_parallel_size=8,
        enable_sequence_parallel=False,
    )

    assert cfg.cudagraph_capture_sizes[0] == 8
    assert all(size % 8 == 0 for size in cfg.cudagraph_capture_sizes)


def test_expert_sequence_parallel_padding_rejects_no_valid_cuda_graph_sizes():
    with pytest.raises(ValueError, match="No CUDA graph capture sizes"):
        VLLMCudaGraphConfig(
            mode="FULL", capture_sizes=[1, 2, 3]
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


def test_only_weights_use_cumem_allocator(monkeypatch):
    base_worker_cls = TorchTitanGPUWorker.__mro__[1]
    monkeypatch.setattr(
        base_worker_cls,
        "_maybe_get_memory_pool_context",
        lambda self, tag: nullcontext(tag),
    )
    worker = object.__new__(TorchTitanGPUWorker)

    with worker._maybe_get_memory_pool_context("kv_cache") as value:
        assert value is None
    with worker._maybe_get_memory_pool_context("weights") as value:
        assert value == "weights"
    with worker._maybe_get_memory_pool_context("other") as value:
        assert value is None


def test_cuda_graph_default_mode_is_full():
    cfg = VLLMCudaGraphConfig().get_vllm_compilation_config(
        max_num_seqs=256,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_mode.name == "FULL"
    assert int(cfg.mode) == 0


def test_cuda_graph_full_mode_no_compile():
    # FULL captures the whole forward (incl. attention) with no inductor compile.
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
        max_num_seqs=256,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_mode.name == "FULL"
    assert int(cfg.mode) == 0


def test_cuda_graph_decode_only_capture_sizes_cover_max_num_seqs():
    # FULL_DECODE_ONLY only graphs decode, so capture up to max_num_seqs (plus
    # max_num_seqs itself when not a power of 2).
    cfg = VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY").get_vllm_compilation_config(
        max_num_seqs=500,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]


def test_cuda_graph_full_mode_extends_capture_sizes_to_chunk():
    # FULL also graphs prefill, so sizes extend to the chunked-prefill chunk
    # (max_num_batched_tokens, 2048) on top of max_num_seqs.
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
        max_num_seqs=500,
        expert_sequence_parallel_size=1,
        enable_sequence_parallel=False,
    )
    assert cfg.cudagraph_capture_sizes[-1] == 2048
    assert 500 in cfg.cudagraph_capture_sizes  # decode batch captured exactly


def test_cuda_graph_rejects_nonpositive_max_num_seqs():
    with pytest.raises(ValueError, match="max_num_seqs must be positive"):
        VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
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

    compilation_config = VLLMCudaGraphConfig(
        mode="FULL", capture_sizes=[1, 4, 5, 8]
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

    from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_moe_debug_varlen

    from tests.rl.unit_tests.gpu.test_bitwise_parity import (
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
        config.generator.model,
        parallelism=config.generator.parallelism,
        checkpointer_config=None,
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
