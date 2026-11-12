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
"""

import asyncio
import concurrent.futures
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

import torchtitan.rl.generator.generator as generator_module
import torchtitan.rl.generator.vllm.engine as vllm_engine_module
import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils
from renderers import PlaceholderRange
from torchstore import RankRole
from torchtitan.components.optim import AdamW
from torchtitan.config import DebugConfig, PrecisionConfig
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
from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
from torchtitan.rl.generator.vllm.worker import (
    TorchTitanCudaGraphManager,
    TorchTitanGPUWorker,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Prompt, PromptImage
from torchtitan.rl.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.routing.strategies import LeastLoadedRoutingStrategy
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    DeviceConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.sampling_params import RequestOutputKind
from vllm.utils.cache import LRUCache


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
        self.reset_encoder_cache_calls = 0
        self.renderer = _FakeRenderer()

    def add_request(self, *args, **kwargs):
        self.add_requests.append((args, kwargs))

    def has_unfinished_requests(self):
        return False

    def reset_prefix_cache(self, *args, **kwargs):
        self.reset_prefix_cache_calls.append((args, kwargs))

    def reset_encoder_cache(self):
        self.reset_encoder_cache_calls += 1


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
    engine._vision_processor = None
    engine._pixel_values_lru_cache = LRUCache(
        1 << 20, getsizeof=lambda pixel_values: pixel_values.nbytes
    )
    return engine


class _FakeVisionProcessor:
    """Stands in for `VisionProcessor`: every image file becomes ``pixel_values``; counts
    the files it preprocesses."""

    def __init__(self, pixel_values: torch.Tensor | None = None):
        self.pixel_values = (
            pixel_values if pixel_values is not None else torch.zeros(16, 8)
        )
        self.num_preprocessed_files = 0

    def resize_and_normalize(self, data: bytes):
        self.num_preprocessed_files += 1
        return torch.zeros(1, 32, 32, 3), torch.tensor([1, 2, 2])

    def patchify(self, pixels: torch.Tensor):
        return self.pixel_values, torch.tensor([1, 2, 2])


def test_initialize_torchstore_client_uses_requester_index():
    async def main():
        generator = Generator.__new__(Generator)
        client = AsyncMock()
        with patch.object(generator_module.ts, "client", client):
            await generator.initialize_torchstore_client(requester_index=3)

        client.assert_awaited_once_with(role=RankRole.REQUESTER, group=3)
        assert generator._torchstore_loop is asyncio.get_running_loop()

    asyncio.run(main())


def test_prefetch_model_state_dict_updates_staging_buffers_in_place():
    async def main():
        staging_state_dict = {"weight": torch.zeros(2)}
        generator = Generator.__new__(Generator)
        generator.config = SimpleNamespace(model_state_dict_load_mode="cpu_prefetch")
        generator._prefetched_model_state_dict = staging_state_dict

        def fill_state_dict(*args, **kwargs):
            assert kwargs["user_state_dict"] is staging_state_dict
            staging_state_dict["weight"].fill_(1)
            return staging_state_dict

        get_state_dict = AsyncMock(side_effect=fill_state_dict)
        with patch.object(generator_module.ts, "get_state_dict", get_state_dict):
            await generator.prefetch_model_state_dict()
        weights = {"weight": torch.zeros(2)}
        assert generator._copy_prefetched_model_state_dict(weights) == weights.keys()

        assert generator._prefetched_model_state_dict is staging_state_dict
        get_state_dict.assert_awaited_once_with(
            "model_state_dict",
            user_state_dict=staging_state_dict,
            strict=False,
            direct_rdma=False,
        )
        assert torch.equal(weights["weight"], torch.ones(2))

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
        generation = OutstandingGeneration(reply=reply)
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
        generation = OutstandingGeneration(reply=reply)
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
        prompt=Prompt(token_ids=[1, 2]),
        sampling=SamplingConfig(),
        group_id=group_id,
        routing_session_id="group=3/rollout=0",
        arrival_time=1.5,
    )
    request.min_policy_version = min_policy_version
    return request


def _image_prompt() -> Prompt:
    """Text, a (1, 4, 4)-patch image as 4 placeholders, text."""
    image = PromptImage(
        hash_value="k", source_path="k.png", grid_thw=(1, 4, 4), data=b"k"
    )
    return Prompt(
        token_ids=[1, 7, 7, 7, 7, 2],
        images=(image,),
        image_placeholders=(PlaceholderRange(offset=1, length=4),),
    )


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
    engine._vision_processor = _FakeVisionProcessor()
    image_request = _engine_request("r1", min_policy_version=6)
    image_request.prompt = _image_prompt()

    engine.add_requests([_engine_request("r0", min_policy_version=6), image_request])

    [(_, text_kwargs), (_, image_kwargs)] = cast(
        _FakeLLMEngine, engine._engine
    ).add_requests
    assert text_kwargs["prompt"]["cache_salt"] is None
    assert image_kwargs["prompt"].get("cache_salt") is None
    # The engine resets its encoder cache on every load instead.
    assert image_kwargs["prompt"]["mm_hashes"] == {"image": ["k"]}


def test_admission_submits_image_prompts_as_multimodal_inputs():
    engine = _vllm_engine()
    pixel_values = torch.zeros(16, 8)
    engine._vision_processor = _FakeVisionProcessor(pixel_values)
    request = _engine_request("r0", min_policy_version=6)
    request.prompt = _image_prompt()

    assert (
        engine.add_requests([request, _engine_request("r1", min_policy_version=6)])
        == []
    )

    [(_, image_kwargs), (_, text_kwargs)] = cast(
        _FakeLLMEngine, engine._engine
    ).add_requests
    prompt = image_kwargs["prompt"]
    assert prompt["type"] == "multimodal"
    assert prompt["prompt_token_ids"] == [1, 7, 7, 7, 7, 2]
    assert prompt["cache_salt"] == "6"
    # The image hash carries the pinned version: encoder outputs follow the vision weights.
    assert prompt["mm_hashes"] == {"image": ["k@6"]}
    assert prompt["mm_kwargs"]["image"][0]["pixel_values"].data is pixel_values
    assert text_kwargs["prompt"]["type"] == "token"


def test_admission_preprocesses_each_image_file_once_while_cached():
    # The samples of a group, and later turns, carry the same image file.
    engine = _vllm_engine()
    vision_processor = _FakeVisionProcessor()
    engine._vision_processor = vision_processor
    requests = [
        _engine_request(f"r{index}", min_policy_version=6) for index in range(3)
    ]
    for request in requests:
        request.prompt = _image_prompt()

    assert engine.add_requests(requests) == []

    assert vision_processor.num_preprocessed_files == 1


def test_admission_rejects_image_prompts_without_an_image_source():
    engine = _vllm_engine()
    request = _engine_request("r0", min_policy_version=6)
    request.prompt = _image_prompt()

    [(request_id, error)] = engine.add_requests([request])

    assert request_id == "r0" and "Controller.Config.vision_processor" in str(error)
    assert cast(_FakeLLMEngine, engine._engine).add_requests == []


def test_admission_rejects_only_the_request_whose_image_does_not_decode():
    # A truncated dataset file fails its rollout, not the generator.
    engine = _vllm_engine()

    class _TruncatedFileVisionProcessor(_FakeVisionProcessor):
        def resize_and_normalize(self, data: bytes):
            raise RuntimeError("decode_png: truncated file")  # what torchvision raises

    engine._vision_processor = _TruncatedFileVisionProcessor()
    corrupt = _engine_request("corrupt", min_policy_version=6)
    corrupt.prompt = _image_prompt()

    [(request_id, error)] = engine.add_requests(
        [corrupt, _engine_request("text", min_policy_version=6)]
    )

    assert request_id == "corrupt" and isinstance(error, ValueError)
    assert "k.png" in str(error) and "truncated" in str(error)
    llm_engine = cast(_FakeLLMEngine, engine._engine)
    assert [kwargs["request_id"] for _, kwargs in llm_engine.add_requests] == ["text"]


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
            dispatcher.rank0_register_generation(request_id, replies[request_id])
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


@pytest.mark.parametrize("serves_images", [False, True])
@pytest.mark.parametrize("reset_kv_cache", [False, True])
def test_reset_kv_cache_on_model_state_dict_load_controls_cache_reset(
    reset_kv_cache: bool, serves_images: bool
):
    model = SimpleNamespace(
        model_state_dict_loader=SimpleNamespace(load=lambda fetch: None)
    )
    engine = _vllm_engine(
        reset_kv_cache_on_model_state_dict_load=reset_kv_cache, model=model
    )
    if serves_images:
        engine._vision_processor = _FakeVisionProcessor()

    engine.load_model_state_dict(lambda state_dict: state_dict.keys())

    fake = cast(_FakeLLMEngine, engine._engine)
    expected = [((), {"reset_running_requests": True})] if reset_kv_cache else []
    assert fake.reset_prefix_cache_calls == expected
    # Unsalted image hashes would let new requests reuse old encoder outputs.
    assert fake.reset_encoder_cache_calls == int(reset_kv_cache and serves_images)


def test_sampling_config_rejects_top_p_below_one():
    with pytest.raises(ValueError, match="top_p must be 1.0, got 0.95"):
        SamplingConfig(top_p=0.95)


# --- vLLM metric timing math (the `_prepare_generation_request_metrics` helper) ---


def test_metric_timing_math():
    metrics = _prepare_generation_request_metrics(
        _extract_request_metrics_inputs(_request_output(num_preemptions=2))
    )
    aggregate = m.MetricsProcessor._aggregate_metrics(metrics)
    assert all(key.startswith("generator/") for key in aggregate)
    assert aggregate["generator/num_preemptions/mean"] == 2
    assert aggregate["generator/queue_time_ms/mean"] == pytest.approx(5)
    assert aggregate["generator/time_to_first_token_ms/mean"] == 12
    assert aggregate["generator/prefill_time_ms/mean"] == pytest.approx(12)
    assert aggregate["generator/decode_time_ms/mean"] == pytest.approx(30)
    assert aggregate["generator/inter_token_latency_ms/mean"] == pytest.approx(10)


def test_decode_metrics_absent_for_single_generated_token():
    metrics = _prepare_generation_request_metrics(
        _extract_request_metrics_inputs(_request_output(num_generation_tokens=1))
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
            precision=PrecisionConfig(),
            gpu_memory_limit=0.9,
            max_num_batched_tokens=None,
            checkpointer=None,
            dist_moe_runtime=None,
            debug=DebugConfig(),
            reset_kv_cache_on_model_state_dict_load=True,
            model_state_dict_load_mode="cpu_prefetch",
            vision_processor=None,
            rank=0,
            generator_name="generator",
            output_dir=str(tmp_path),
        )


def test_qwen36_27b_perf_config():
    from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_6_27b_varlen_perf

    config = rl_grpo_qwen3_6_27b_varlen_perf()

    assert config.hf_assets_path.endswith("Qwen3.6-27B")
    assert "offset_rmsnorm" in config.trainer.model.local_compile.regions
    assert "offset_rmsnorm" in config.generator.model.local_compile.regions
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
    config.engine.build(
        model=config.model,
        parallelism=config.parallelism,
        model_path="unused",
        max_num_seqs=8,
        precision=PrecisionConfig(),
        gpu_memory_limit=config.gpu_memory_limit,
        max_num_batched_tokens=config.max_num_batched_tokens,
        checkpointer=None,
        dist_moe_runtime=None,
        debug=config.debug,
        reset_kv_cache_on_model_state_dict_load=False,
        model_state_dict_load_mode="cpu_prefetch",
        vision_processor=None,
        rank=0,
        generator_name="generator",
        output_dir=str(tmp_path),
    )
    return engine_args.call_args.kwargs


def test_engine_config_reaches_engine_args(monkeypatch, tmp_path):
    engine_kwargs = _build_engine_kwargs(
        monkeypatch,
        tmp_path,
        Generator.Config(
            model=_MODEL_CONFIG,
            engine=VLLMEngine.Config(
                watermark=0.03,
                extra_engine_args={"long_prefill_token_threshold": 4096},
            ),
        ),
    )
    assert engine_kwargs["watermark"] == 0.03
    assert engine_kwargs["long_prefill_token_threshold"] == 4096
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
        ("watermark", r"use VLLMEngine\.Config\.watermark"),
    ],
)
def test_extra_engine_args_rejects_torchtitan_owned_keys(key, match):
    engine_kwargs = {"scheduling_policy": "fcfs"}
    with pytest.raises(ValueError, match=match):
        add_extra_vllm_engine_args(engine_kwargs, {key: 1})
    assert engine_kwargs == {"scheduling_policy": "fcfs"}


# --- CUDA graph config (VLLMCudaGraphConfig.get_vllm_compilation_config) ---


def test_expert_sequence_parallel_padding_filters_cuda_graph_sizes():
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=4,
        expert_parallel_degree=4,
    )
    cfg = VLLMCudaGraphConfig(
        mode="FULL", capture_sizes=[1, 4, 5, 8]
    ).get_vllm_compilation_config(
        max_num_seqs=8,
        max_model_len=4096,
        expert_sequence_parallel_size=parallelism.expert_sequence_parallel_size,
    )

    assert cfg.prefill_cudagraph_capture_token_sizes == [4, 8]
    assert cfg.decode_cudagraph_capture_request_sizes == [4, 8]


def test_expert_sequence_parallel_padding_keeps_small_cuda_graph_batches():
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
        max_num_seqs=1,
        max_model_len=4096,
        expert_sequence_parallel_size=8,
    )

    assert cfg.prefill_cudagraph_capture_token_sizes[0] == 8
    assert all(size % 8 == 0 for size in cfg.prefill_cudagraph_capture_token_sizes)
    # max_num_seqs is 1, not a multiple of 8: no decode graph.
    assert cfg.decode_cudagraph_capture_request_sizes == []


def test_expert_sequence_parallel_padding_rejects_no_valid_cuda_graph_sizes():
    with pytest.raises(ValueError, match="No CUDA graph capture sizes"):
        VLLMCudaGraphConfig(
            mode="FULL", capture_sizes=[1, 2, 3]
        ).get_vllm_compilation_config(
            max_num_seqs=3,
            max_model_len=4096,
            expert_sequence_parallel_size=4,
        )


def test_expert_sequence_parallel_padding_disabled_without_ep():
    parallelism = InferenceParallelismConfig(
        tensor_parallel_degree=4,
        expert_parallel_degree=1,
    )

    assert parallelism.expert_sequence_parallel_size == 1


def _full_mode_graph_manager(monkeypatch) -> TorchTitanCudaGraphManager:
    """A FULL-mode manager on CPU: mixed graphs of 1, 2, 8, 16 tokens, decode graphs of
    1 and 2 requests, marked captured so ``dispatch`` returns graphs."""
    # vLLM's manager reads its pipeline stage; CPU tests set up no process groups.
    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    # An explicit device: CPU CI has no platform for vLLM to infer one from.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL,
        prefill_cudagraph_capture_token_sizes=[1, 2, 8, 16],
        decode_cudagraph_capture_request_sizes=[1, 2],
        max_cudagraph_capture_size=16,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=64,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        is_encoder_decoder=False,
    )
    manager = TorchTitanCudaGraphManager(
        config, torch.device("cpu"), CUDAGraphMode.FULL, decode_query_lens={1: 4}
    )
    manager._graphs_captured = True
    return manager


def test_full_mode_captures_decode_graphs_next_to_mixed_graphs(monkeypatch):
    manager = _full_mode_graph_manager(monkeypatch)
    captured = [
        (descriptor.num_tokens, descriptor.uniform_token_count)
        for descriptor in manager._capture_descs[CUDAGraphMode.FULL]
    ]
    # Widest first. uniform_token_count 1 marks a decode graph.
    assert captured == [(16, None), (8, None), (2, 1), (2, None), (1, 1), (1, None)]


@pytest.mark.parametrize(
    "num_reqs,num_tokens,uniform_token_count,batch_invariant,expected",
    [
        (2, 2, 1, False, (CUDAGraphMode.FULL, 2, 1)),
        # 3 decodes are past the largest decode graph (2): the 8-token mixed graph.
        (3, 3, 1, False, (CUDAGraphMode.FULL, 8, None)),
        # Batch-invariant runs keep them eager on the decode kernels.
        (3, 3, 1, True, (CUDAGraphMode.NONE, 3, None)),
        # DP pads 1 decode to a peer's 2: the 2-request decode graph.
        (1, 2, 1, False, (CUDAGraphMode.FULL, 2, 1)),
        # A prefill takes a mixed graph, batch-invariant or not.
        (1, 5, None, True, (CUDAGraphMode.FULL, 8, None)),
    ],
)
def test_full_mode_decode_dispatch(
    monkeypatch, num_reqs, num_tokens, uniform_token_count, batch_invariant, expected
):
    monkeypatch.setattr(
        "torchtitan.rl.generator.vllm.worker.is_in_batch_invariant_mode",
        lambda: batch_invariant,
    )
    manager = _full_mode_graph_manager(monkeypatch)
    # With DP, vLLM dispatches again after the ranks agree on a size, so the
    # model state must read the last dispatch: run a different one first.
    manager.dispatch(1, 1, 1, num_active_loras=0)

    descriptor = manager.dispatch(
        num_reqs, num_tokens, uniform_token_count, num_active_loras=0
    )
    assert (
        descriptor.cg_mode,
        descriptor.num_tokens,
        descriptor.uniform_token_count,
    ) == expected
    assert manager.last_dispatched_descriptor is descriptor


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
        max_model_len=4096,
        expert_sequence_parallel_size=1,
    )
    assert cfg.cudagraph_mode.name == "FULL"
    assert int(cfg.mode) == 0


def test_cuda_graph_full_mode_no_compile():
    # FULL captures the whole forward (incl. attention) with no inductor compile.
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
        max_num_seqs=256,
        max_model_len=4096,
        expert_sequence_parallel_size=1,
    )
    assert cfg.cudagraph_mode.name == "FULL"
    assert int(cfg.mode) == 0


def test_cuda_graph_decode_only_capture_sizes_cover_max_num_seqs():
    # FULL_DECODE_ONLY only graphs decode, so capture up to max_num_seqs itself.
    cfg = VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY").get_vllm_compilation_config(
        max_num_seqs=500,
        max_model_len=4096,
        expert_sequence_parallel_size=1,
    )
    sizes = cfg.decode_cudagraph_capture_request_sizes
    assert 40 in sizes  # 34 sequences pad to 40, not 64
    assert sizes[-1] == 500  # a full decode batch has its own graph


def test_cuda_graph_full_mode_extends_capture_sizes_to_chunk():
    # FULL also graphs prefill, so sizes extend to the chunked-prefill chunk
    # (max_num_batched_tokens) on a coarser ladder past 1024.
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
        max_num_seqs=500,
        max_model_len=16384,
        expert_sequence_parallel_size=1,
        max_num_batched_tokens=16384,
    )
    sizes = cfg.prefill_cudagraph_capture_token_sizes
    assert 40 in sizes  # 34 sequences pad to 40, not 64
    assert sizes[-1] == 16384  # the largest prefill chunk is graphed
    # Decode graphs stop at a full decode batch, which has its own graph.
    assert cfg.decode_cudagraph_capture_request_sizes[-1] == 500


def test_cuda_graph_full_mode_caps_sizes_at_the_largest_step():
    # A step packs at most max_num_seqs requests of at most max_model_len tokens each.
    kwargs = dict(
        max_num_seqs=8,
        max_model_len=1000,
        expert_sequence_parallel_size=1,
    )
    # Packed prefills reach the 2048-token budget, so it is graphed.
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(**kwargs)
    assert cfg.prefill_cudagraph_capture_token_sizes[-2:] == [1536, 2048]
    one_request = {**kwargs, "max_num_seqs": 1}
    cfg = VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(**one_request)
    assert cfg.prefill_cudagraph_capture_token_sizes[-2:] == [992, 1000]
    with pytest.raises(ValueError, match="no step reaches"):
        VLLMCudaGraphConfig(
            mode="FULL", capture_sizes=[512, 2048]
        ).get_vllm_compilation_config(**one_request)
    cfg = VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY").get_vllm_compilation_config(
        **{**kwargs, "max_num_seqs": 1500}
    )
    assert cfg.decode_cudagraph_capture_request_sizes[-1] == 1500


def test_cuda_graph_rejects_nonpositive_max_num_seqs():
    with pytest.raises(ValueError, match="max_num_seqs must be positive"):
        VLLMCudaGraphConfig(mode="FULL").get_vllm_compilation_config(
            max_num_seqs=0,
            max_model_len=4096,
            expert_sequence_parallel_size=1,
        )
