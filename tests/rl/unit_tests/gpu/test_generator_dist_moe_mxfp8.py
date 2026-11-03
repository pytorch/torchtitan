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
from types import SimpleNamespace

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
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm.registry import register_to_vllm, VLLM_MODEL_NAME
from torchtitan.tools.utils import has_cuda_capability
from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_moe_debug_varlen
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.model_executor.model_loader import utils as model_loader_utils
from vllm.sampling_params import RequestOutputKind

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    _make_prompt_tokens,
    _run_engine,
    _sync_trainer_weights_to_vllm,
    build_inference_engine,
)


def _mxfp8_generator_config() -> Controller.Config:
    """Build a one-GPU debug MoE generator with MXFP8 Mega inference weights."""
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
    # Batch-invariant mode is process-wide and cannot be turned off, so an
    # earlier test in this process (e.g. Kimi K3 parity) may have enabled it.
    # Local compile regions reject that mode and this test does not need them.
    generator_model.local_compile.regions = []
    config.generator.model = generator_model
    config.generator.parallelism = generator_parallelism
    config.generator.gpu_memory_limit = 0.1
    config.generator.engine.cuda_graph.mode = "FULL"
    config.generator.dist_moe_runtime = DistMoeRuntime.Config(inference=True)
    config.generator = apply_transforms(
        config.generator,
        [
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(pipeline="mega"),
                mxfp8_mega_fused_inference=True,
            )
        ],
    )
    return config


def _clear_cached_model_class() -> None:
    """Drop vLLM's cached model class, so the next engine builds the model
    registered next, not one an earlier test (e.g. Kimi K3) registered.
    """
    # TODO: Fix this in register_to_vllm (or vLLM) instead of the test.
    for key, (_, arch) in list(model_loader_utils._MODEL_ARCH_BY_HASH.items()):
        if arch == VLLM_MODEL_NAME:
            del model_loader_utils._MODEL_ARCH_BY_HASH[key]


def _state_dict_with_random_experts(wrapper, experts, *, seed: int) -> dict:
    """Build a full state dict to load into the vLLM model, with only the
    expert weights changed.

    - Expert W13/W2 weights: new random values from ``seed``
    - All other weights: copies of the model's current values, so loading
      them changes nothing (FSDP keeps their unsharded storage)
    """
    expert_names = {f"{name}.{p}.weight" for name in experts for p in ("w13", "w2")}
    live = {
        **dict(wrapper.model.named_parameters()),
        **dict(wrapper.model.named_buffers()),
    }
    rng = torch.Generator().manual_seed(seed)
    return {
        name: (
            0.02 * torch.randn(buffer.shape, generator=rng).to(buffer.dtype)
            if name in expert_names
            else live[name].detach().cpu()
        )
        for name, buffer in wrapper._prefetched_model_state_dict.items()
    }


def _operands(experts) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return each expert weight's FSDP-managed qdata and FPROP scale."""
    return [
        (weight.operands.qdata, weight.operands.fprop_scale)
        for module in experts.values()
        for weight in (module.w13.weight, module.w2.weight)
    ]


def _generate(engine, prompt_ids, request_prefix: str) -> list[list[float]]:
    """Greedily generate two tokens per prompt and return their logprobs."""
    outputs = _run_engine(
        engine,
        request_prefix,
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
    logprobs = [
        [list(step.values())[0].logprob for step in output.outputs[0].logprobs]
        for output in outputs
    ]
    assert all(len(sample) == 2 for sample in logprobs)
    assert all(math.isfinite(logprob) for sample in logprobs for logprob in sample)
    return logprobs


def _check_inference_operands(engine) -> None:
    """Check the experts' inference operands across FULL-graph weight loads.

    - The engine replays FULL CUDA graphs rather than running eagerly
    - Every MXFP8 expert uses the Mega fused inference weights (no DGRAD
      scales) from the first unshard
    - Each weight load refills the same operand tensors at the same addresses
    - The operand contents and generated logprobs follow the loaded weights,
      and reloading the same weights reproduces the same logprobs
    """
    compilation_config = engine.vllm_config.compilation_config
    assert compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert not engine.vllm_config.model_config.enforce_eager

    wrapper = engine.model_executor.driver_worker.get_model()
    experts = {
        name: module
        for name, module in wrapper.model.named_modules()
        if isinstance(module, MXFP8DistMoeRoutedExperts)
    }
    assert experts
    for module in experts.values():
        assert module.mega_fused_inference
        for weight in (module.w13.weight, module.w2.weight):
            # The wrapper already unsharded the weights, so their operands come
            # from the first preparation
            assert isinstance(weight, _UnshardedFSDPTensor)
            assert weight.operands.dgrad_scale.numel() == 0

    weights_a = _state_dict_with_random_experts(wrapper, experts, seed=0)
    weights_b = _state_dict_with_random_experts(wrapper, experts, seed=1)
    prompt_ids = _make_prompt_tokens(2, 32, engine.get_tokenizer())

    # The CUDA graphs captured at engine startup read these exact tensors and
    # addresses, so every load must refill them in place.
    operands = _operands(experts)
    data_ptrs = [(q.data_ptr(), s.data_ptr()) for q, s in operands]

    # Reuse the trainer weight sync, which loads through the generator's
    # weight-load path.
    _sync_trainer_weights_to_vllm(SimpleNamespace(state_dict=lambda: weights_a), engine)
    contents_a = [(q.clone(), s.clone()) for q, s in _operands(experts)]
    logprobs_a = _generate(engine, prompt_ids, "weights_a")

    _sync_trainer_weights_to_vllm(SimpleNamespace(state_dict=lambda: weights_b), engine)
    for (q, s), (q_a, s_a) in zip(_operands(experts), contents_a, strict=True):
        assert not torch.equal(q, q_a)
        assert not torch.equal(s, s_a)
    logprobs_b = _generate(engine, prompt_ids, "weights_b")

    _sync_trainer_weights_to_vllm(SimpleNamespace(state_dict=lambda: weights_a), engine)
    logprobs_a_again = _generate(engine, prompt_ids, "weights_a_again")

    for (q, s), (q_0, s_0), ptrs in zip(
        _operands(experts), operands, data_ptrs, strict=True
    ):
        assert q is q_0 and s is s_0
        assert (q.data_ptr(), s.data_ptr()) == ptrs
    assert logprobs_b != logprobs_a
    assert logprobs_a_again == logprobs_a


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_capability(10, 0),
    reason="Dist-MoE requires an SM100-or-newer GPU",
)
def test_vllm_generator_enables_mxfp8_mega_inference_weights():
    """The generator serves MXFP8 Mega experts with Mega fused inference weights.

    With ``mxfp8_mega_fused_inference``, every MXFP8 expert's operands
    have no DGRAD scales from the first unshard. Under FULL CUDA graphs, weight
    loads refill those retained operands in place: their objects stay fixed
    while their contents and the generated logprobs follow the loaded weights.
    """
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size != 1:
        pytest.skip(f"requires exactly 1 GPU, got {world_size}")

    config = _mxfp8_generator_config()
    owns_process_group = not dist.is_initialized()
    temporary_dump_folder = None
    engine = None
    try:
        if owns_process_group:
            temporary_dump_folder = tempfile.mkdtemp(prefix="rl_generator_mxfp8_")
            dist_utils.init_distributed(CommConfig(), base_folder=temporary_dump_folder)
        _clear_cached_model_class()
        register_to_vllm(
            config.generator.model,
            parallelism=config.generator.parallelism,
            checkpointer_config=None,
            dist_moe_runtime=config.generator.dist_moe_runtime,
        )
        engine = build_inference_engine(config)
        _check_inference_operands(engine)
    finally:
        if engine is not None:
            renderer = getattr(engine, "renderer", None)
            if renderer is not None:
                renderer.shutdown()
            del engine
        _clear_cached_model_class()
        gc.collect()
        torch.cuda.empty_cache()
        # A process group provided by an earlier test or fixture stays alive.
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        if temporary_dump_folder is not None:
            shutil.rmtree(temporary_dump_folder, ignore_errors=True)
