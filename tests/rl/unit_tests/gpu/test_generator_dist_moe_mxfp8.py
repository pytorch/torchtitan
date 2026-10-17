# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-GPU vLLM generator test for MXFP8 Mega Dist-MoE inference weights."""

import gc
import math
import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist

pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE integration tests require the optional dist_moe package",
)

from torchtitan.config import CommConfig
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.common.dist_moe import (
    _dist_moe as dist_moe,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.qwen3 import build_model_config
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm.registry import register_to_vllm
from torchtitan.tools.utils import has_cuda_capability
from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_moe_debug_varlen
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    _make_prompt_tokens,
    _run_engine,
    build_inference_engine,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_capability(10, 0),
    reason="Dist-MoE requires an SM100-or-newer GPU",
)
def test_vllm_generator_enables_mxfp8_mega_inference_weights():
    """vLLM's model wrapper prepares fused-SwiGLU MXFP8 weights.

    The wrapper enables inference on each MXFP8 expert before it first unshards
    the weights, so every MXFP8 Mega expert must already have inference
    operands (no DGRAD scales) once the engine is built.
    """
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size != 1:
        pytest.skip(f"requires exactly 1 GPU, got {world_size}")

    config = rl_grpo_qwen3_moe_debug_varlen()
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
        expert_parallel_degree=1,
    )
    _, generator_model = build_rl_model_configs(
        build_model_config,
        "debugmodel_moe",
        seq_len=config.generator.model.max_context_length,
        attn_backend="varlen",
        trainer_parallelism=config.trainer.parallelism,
        generator_parallelism=generator_parallelism,
    )
    config.generator.model = generator_model
    config.generator.parallelism = generator_parallelism
    config.generator.gpu_memory_limit = 0.1
    config.generator.dist_moe_runtime = DistMoeRuntime.Config(inference=True)
    config.generator = apply_transforms(
        config.generator,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(pipeline="mega"),
            )
        ],
    )

    temporary_dump_folder = None
    if not dist.is_initialized():
        temporary_dump_folder = tempfile.mkdtemp(prefix="rl_generator_mxfp8_")
        dist_utils.init_distributed(
            CommConfig(),
            base_folder=temporary_dump_folder,
        )

    register_to_vllm(
        config.generator.model,
        parallelism=config.generator.parallelism,
        checkpointer_config=None,
        dist_moe_runtime=config.generator.dist_moe_runtime,
    )

    engine = build_inference_engine(config)
    try:
        wrapper = engine.model_executor.driver_worker.get_model()
        experts = [
            module
            for module in wrapper.model.modules()
            if isinstance(module, MXFP8DistMoeRoutedExperts)
        ]
        assert experts
        for module in experts:
            assert module.inference
            for weight in (module.w13.weight, module.w2.weight):
                # The wrapper already unsharded the weights, so their operands
                # come from the first preparation
                assert isinstance(weight, _UnshardedFSDPTensor)
                assert weight.operands.dgrad_scale.numel() == 0

        prompt_ids = _make_prompt_tokens(2, 32, engine.get_tokenizer())
        outputs = _run_engine(
            engine,
            "mxfp8_mega_inference",
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
