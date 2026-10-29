# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM generator at DP > 1 with FULL CUDA graphs (multi-GPU).

One torchrun launch per test: like test_bitwise_parity's classes, each test builds a
vLLM engine that frees its GPU memory only at process exit.
"""

import contextlib
import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.qwen3 import build_model_config
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_model_config
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.registry import register_to_vllm
from torchtitan_recipes.tests.rl.alphabet_sort import (
    rl_grpo_qwen3_5_debug_varlen,
    rl_grpo_qwen3_moe_debug_varlen_batch_invariant,
)
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.sampling_params import RequestOutputKind

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    _make_prompt_tokens,
    build_inference_engine,
)

# Runs under a torchrun launch with two or more GPUs.
pytestmark = pytest.mark.multi_gpu

# Rank 0's long prompt; the other ranks are padded to its prefill chunk.
LONG_PROMPT_TOKENS = 1500

# The MoE debug models' torch._grouped_mm copies to the host under CUDA graph capture
# before SM90, so FULL graphs can't be captured there.
requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FULL graphs of the MoE debug models need SM90+",
)


def _run(engine, prefix, prompts, params, *, long_prompt=None):
    """Run ``prompts`` to completion; ``long_prompt`` joins at the third step."""
    for index, prompt in enumerate(prompts):
        engine.add_request(f"{prefix}{index}", {"prompt_token_ids": prompt}, params)
    outputs = {}
    step = 0
    while engine.has_unfinished_requests():
        if step == 2 and long_prompt is not None:
            engine.add_request(
                f"{prefix}long",
                {"prompt_token_ids": long_prompt},
                SamplingParams(max_tokens=1, output_kind=RequestOutputKind.FINAL_ONLY),
            )
        for output in engine.step():
            if output.request_id == f"{prefix}long":
                continue
            sample = output.outputs[0]
            outputs[output.request_id.removeprefix(prefix)] = (
                list(sample.token_ids),
                [next(iter(logprobs.values())).logprob for logprobs in sample.logprobs],
            )
        step += 1
    return outputs


def _world_size() -> int:
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size < 2:
        pytest.skip(f"requires at least 2 GPUs, got {world_size}")
    return world_size


@contextlib.contextmanager
def _dp_engine(config):
    """Build the generator's engine; yield it, its dispatch records and the largest decode graph."""
    generator = config.generator
    temporary_dump_folder = None
    if not dist.is_initialized():
        temporary_dump_folder = tempfile.mkdtemp(prefix="rl_generator_dp_")
        dist_utils.init_distributed(CommConfig(), base_folder=temporary_dump_folder)
    register_to_vllm(
        generator.model, parallelism=generator.parallelism, checkpointer_config=None
    )
    engine = build_inference_engine(config)
    records = []

    def instrument(worker):
        runner = worker.model_runner
        original = runner._determine_batch_execution_and_padding

        def record(num_tokens, *args, **kwargs):
            result = original(num_tokens, *args, **kwargs)
            mode, descriptor = result[:2]
            records.append((mode, descriptor.num_tokens, descriptor.uniform))
            return result

        runner._determine_batch_execution_and_padding = record
        return runner.cudagraph_dispatcher.max_decode_capture_size

    try:
        (max_decode_capture_size,) = engine.collective_rpc(instrument)
        yield engine, records, max_decode_capture_size
    finally:
        if dist.is_initialized():
            dist.barrier()
        renderer = getattr(engine, "renderer", None)
        if renderer is not None:
            renderer.shutdown()
        if temporary_dump_folder is not None:
            shutil.rmtree(temporary_dump_folder, ignore_errors=True)


def _greedy_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        max_tokens=16,
        ignore_eos=True,
        logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


@requires_sm90
def test_vllm_dp_decode_graphs_with_prefilling_peer():
    """A decoding DP rank replays decode graphs, and the mixed graph when padded to a peer's prefill.

    Batch-invariant mode makes each rank's completions independent of the other
    ranks' load, so they must not change when rank 0 prefills a long prompt
    while the others decode.
    """
    world_size = _world_size()
    config = rl_grpo_qwen3_moe_debug_varlen_batch_invariant()
    generator = config.generator
    # Every DP rank holds every expert: the local MoE path captures in FULL graphs.
    generator.parallelism = InferenceParallelismConfig(data_parallel_degree=world_size)
    generator.model = build_model_config(
        "debugmodel_moe",
        seq_len=2048,
        attn_backend="varlen",
        parallelism=generator.parallelism.to_training(),
    )
    # Local compile regions do not support batch-invariant mode.
    generator.model.local_compile.regions = []
    generator.engine.cuda_graph = VLLMCudaGraphConfig(mode="FULL")
    generator.gpu_memory_limit = 0.5

    with _dp_engine(config) as (engine, records, max_decode_capture_size):
        prompts = _make_prompt_tokens(3, 64, engine.get_tokenizer())
        alone = _run(engine, "alone", prompts, _greedy_params())
        alone_records = records[:]
        records.clear()
        engine.reset_prefix_cache()
        long_prompt = None
        if dist.get_rank() == 0:
            long_prompt = _make_prompt_tokens(
                1, LONG_PROMPT_TOKENS, engine.get_tokenizer()
            )[0]
        beside_prefill = _run(
            engine, "beside", prompts, _greedy_params(), long_prompt=long_prompt
        )

        assert beside_prefill == alone
        assert any(
            mode == CUDAGraphMode.FULL and uniform for mode, _, uniform in alone_records
        ), alone_records
        if dist.get_rank() != 0:
            # Padded to rank 0's long-prompt chunk, past the largest decode graph: the
            # mixed graph. (A rank's own short prefill also replays a mixed graph.)
            assert max_decode_capture_size < LONG_PROMPT_TOKENS
            assert any(
                mode == CUDAGraphMode.FULL
                and num_tokens >= LONG_PROMPT_TOKENS
                and not uniform
                for mode, num_tokens, uniform in records
            ), records


@requires_sm90
def test_vllm_dp_linear_attention_full_graphs():
    """At DP>1, GDN steps with prefill tokens replay a FULL graph instead of running eager.

    Rank 0 prefills a long prompt; every other rank, decoding or idle (dummy batch),
    is padded to it and replays the same mixed graph.
    """
    world_size = _world_size()
    config = rl_grpo_qwen3_5_debug_varlen()
    generator = config.generator
    # Every DP rank holds every expert: the local MoE path captures in FULL graphs.
    generator.parallelism = InferenceParallelismConfig(data_parallel_degree=world_size)
    generator.model = build_qwen3_5_model_config(
        "debugmodel_moe",
        seq_len=2048,
        attn_backend="varlen",
        parallelism=generator.parallelism.to_training(),
    )
    # Local compile is not under test.
    generator.model.local_compile.regions = []
    generator.engine.cuda_graph = VLLMCudaGraphConfig(mode="FULL")
    generator.gpu_memory_limit = 0.5

    with _dp_engine(config) as (engine, records, max_decode_capture_size):
        tokenizer = engine.get_tokenizer()
        prompts = _make_prompt_tokens(3, 64, tokenizer)
        long_prompt = None
        if dist.get_rank() == 0:
            long_prompt = _make_prompt_tokens(1, LONG_PROMPT_TOKENS, tokenizer)[0]
        _run(engine, "busy", prompts, _greedy_params(), long_prompt=long_prompt)
        busy_records = records[:]
        records.clear()
        engine.reset_prefix_cache()
        _run(
            engine,
            "idle",
            prompts if dist.get_rank() == 0 else [],
            _greedy_params(),
            long_prompt=long_prompt,
        )

        # Every rank replays the mixed graph for rank 0's long-prompt chunk: its own
        # prefill, or padded to it while decoding (busy) or idle (dummy batch).
        assert max_decode_capture_size < LONG_PROMPT_TOKENS
        for phase_records in (busy_records, records):
            assert any(
                mode == CUDAGraphMode.FULL
                and num_tokens >= LONG_PROMPT_TOKENS
                and not uniform
                for mode, num_tokens, uniform in phase_records
            ), phase_records
