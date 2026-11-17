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
from torchtitan.rl.quantized_operands import pack_quantized_operands
from torchtitan.tools.utils import has_cuda_capability
from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_moe_debug_varlen
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.model_executor.model_loader import utils as model_loader_utils
from vllm.sampling_params import RequestOutputKind

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    _run_engine,
    _sync_trainer_weights_to_vllm,
    build_inference_engine,
    make_prompt_tokens,
)


def _mxfp8_generator_config(num_gpus: int) -> Controller.Config:
    """Build a debug MoE generator with MXFP8 Mega inference weights, EP over the GPUs."""
    config = rl_grpo_qwen3_moe_debug_varlen()
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=num_gpus,
        tensor_parallel_degree=1,
        expert_parallel_degree=num_gpus,
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
    # Any rank may receive every routed token.
    config.generator.dist_moe_runtime = DistMoeRuntime.Config(
        scratch_capacity_factor=float(num_gpus), inference=True
    )
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
        for name, buffer in (
            wrapper.model_state_dict_loader.prefetched_model_state_dict.items()
        )
    }


def _trainer_model(wrapper, state_dict: dict[str, torch.Tensor]) -> SimpleNamespace:
    """Stand in for the trainer model: ``state_dict`` under the generator model's
    buffer and ``preserve_dtype_params`` names, which match the trainer's."""
    return SimpleNamespace(
        state_dict=lambda: state_dict,
        named_buffers=wrapper.model.named_buffers,
        # Quantized operands, as bytes, keep their dtype too.
        named_preserve_dtype_params=lambda: [
            *wrapper.model.named_preserve_dtype_params(),
            *((k, v) for k, v in state_dict.items() if v.dtype == torch.uint8),
        ],
    )


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
    prompt_ids = make_prompt_tokens(2, 32, engine.get_tokenizer())

    # The CUDA graphs captured at engine startup read these exact tensors and
    # addresses, so every load must refill them in place.
    operands = _operands(experts)
    data_ptrs = [(q.data_ptr(), s.data_ptr()) for q, s in operands]

    # Reuse the trainer weight sync, which loads through the generator's
    # weight-load path.
    _sync_trainer_weights_to_vllm(_trainer_model(wrapper, weights_a), engine)
    contents_a = [(q.clone(), s.clone()) for q, s in _operands(experts)]
    logprobs_a = _generate(engine, prompt_ids, "weights_a")

    _sync_trainer_weights_to_vllm(_trainer_model(wrapper, weights_b), engine)
    for (q, s), (q_a, s_a) in zip(_operands(experts), contents_a, strict=True):
        assert not torch.equal(q, q_a)
        assert not torch.equal(s, s_a)
    logprobs_b = _generate(engine, prompt_ids, "weights_b")

    _sync_trainer_weights_to_vllm(_trainer_model(wrapper, weights_a), engine)
    logprobs_a_again = _generate(engine, prompt_ids, "weights_a_again")

    for (q, s), (q_0, s_0), ptrs in zip(
        _operands(experts), operands, data_ptrs, strict=True
    ):
        assert q is q_0 and s is s_0
        assert (q.data_ptr(), s.data_ptr()) == ptrs
    assert logprobs_b != logprobs_a
    assert logprobs_a_again == logprobs_a


def _check_quantized_operand_loads(engine) -> None:
    """After a BF16 load, loads of the trainer's quantized operands (each rank gets
    its experts' rows of quantizing all of them) write the operands in place, and
    reloading the same weights gives bitwise the BF16 load's operands and logprobs."""
    wrapper = engine.model_executor.driver_worker.get_model()
    experts = {
        name: module
        for name, module in wrapper.model.named_modules()
        if isinstance(module, MXFP8DistMoeRoutedExperts)
    }
    weights_a = _state_dict_with_random_experts(wrapper, experts, seed=0)
    weights_b = _state_dict_with_random_experts(wrapper, experts, seed=1)
    prompt_ids = make_prompt_tokens(2, 32, engine.get_tokenizer())
    _sync_trainer_weights_to_vllm(_trainer_model(wrapper, weights_a), engine)
    bf16 = [
        (q.view(torch.uint8).clone(), s.view(torch.uint8).clone())
        for q, s in _operands(experts)
    ]
    bf16_logprobs = _generate(engine, prompt_ids, "bf16")
    data_ptrs = [(q.data_ptr(), s.data_ptr()) for q, s in _operands(experts)]

    formats = wrapper.model_state_dict_loader.receive_quantized_operands()
    for weights in (weights_b, weights_a):
        on_gpu = {k: v.cuda() if k in formats else v for k, v in weights.items()}
        packed = pack_quantized_operands(on_gpu, formats, dtype=None)
        _sync_trainer_weights_to_vllm(_trainer_model(wrapper, packed), engine)
        # B's operands differ, so the comparison with A's below is not vacuous.
        q = _operands(experts)[0][0].view(torch.uint8)
        assert q.equal(bf16[0][0]) == (weights is weights_a)

    assert [(q.data_ptr(), s.data_ptr()) for q, s in _operands(experts)] == data_ptrs
    for (q, s), (q_bf16, s_bf16) in zip(_operands(experts), bf16, strict=True):
        assert q.view(torch.uint8).equal(q_bf16) and s.view(torch.uint8).equal(s_bf16)
    assert _generate(engine, prompt_ids, "quantized_operands") == bf16_logprobs


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_capability(10, 0),
    reason="Dist-MoE requires an SM100-or-newer GPU",
)
@pytest.mark.parametrize("num_gpus", [1, 2])
def test_vllm_generator_enables_mxfp8_mega_inference_weights(num_gpus: int):
    """The generator serves MXFP8 Mega experts with Mega fused inference weights.

    With ``mxfp8_mega_fused_inference``, every MXFP8 expert's operands
    have no DGRAD scales from the first unshard. Under FULL CUDA graphs, weight
    loads refill those retained operands in place: their objects stay fixed
    while their contents and the generated logprobs follow the loaded weights.
    On 2 GPUs (EP 2), loads of quantized operands do (receive_quantized_operands).
    """
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size != num_gpus:
        pytest.skip(f"requires exactly {num_gpus} GPUs, got {world_size}")

    config = _mxfp8_generator_config(num_gpus)
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
            precision=config.precision,
            checkpointer_config=None,
            dist_moe_runtime=config.generator.dist_moe_runtime,
            model_state_dict_load_mode="cpu_prefetch",
        )
        engine = build_inference_engine(config)
        if num_gpus == 1:
            _check_inference_operands(engine)
        else:
            _check_quantized_operand_loads(engine)
    finally:
        if engine is not None:
            dist.barrier()
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
