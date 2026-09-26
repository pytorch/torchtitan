# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Four-GPU vLLM generator integration test."""

import gc
import math
import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.rl.examples.alphabet_sort.config_registry import (
    rl_grpo_qwen3_moe_debug_varlen,
)
from torchtitan.rl.model.vllm_registry import register_to_vllm
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from tests.rl.integration_tests.test_bitwise_parity import (
    _make_prompt_tokens,
    _run_engine,
    build_inference_engine,
)


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
        config.model,
        parallelism=config.generator.parallelism,
        compile_config=config.compile,
        checkpointer_config=None,
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
