#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test that vLLM engine and TorchTitan Module forward produce bitwise identical
log-probs for the same prompt + generated token sequence.

Flow:
  1. Generate tokens from a single prompt via the vLLM engine (greedy decoding).
  2. Feed [prompt + generated tokens] through a standalone TorchTitan model
     forward and compute per-token log-probs.
  3. Compare the two sets of log-probs bitwise.

Run:
    torchrun --nproc_per_node=2 \
        torchtitan/experiments/rl/tests/test_attn_numerics.py
"""

import logging
import os

# Must set spawn method before any CUDA operations or vLLM imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind

from torchtitan.config import CommConfig, TORCH_DTYPE_MAP
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.actors.generator import (
    GeneratorCompileConfig,
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.actors.utils import (
    compute_token_log_probs,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.simple_grpo_sum_digits import GRPOLoss, RLTrainer
from torchtitan.models.qwen3 import model_registry
from torchtitan.tools import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_attention_module(model, label):
    """Log the inner_attention class name from layer 0 (rank 0 only)."""
    if dist.get_rank() != 0:
        return
    layer_0 = next(iter(model.layers.values()))
    attn_cls = type(layer_0.attention.inner_attention).__name__
    logger.info(f"{label} attention module: {attn_cls}")


# ---------------------------------------------------------------------------
# Step 1: vLLM engine generation
# ---------------------------------------------------------------------------


def create_vllm_engine(config: RLTrainer.Config) -> LLMEngine:
    """Create a vLLM LLMEngine from the RL config."""
    gen_config = config.generator
    model_path = config.hf_assets_path

    # Enable batch-invariant mode for deterministic ops (Triton mm, log_softmax,
    # mean, TF32 disable, deterministic algorithms).
    from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant_mode

    enable_batch_invariant_mode()

    assert gen_config.attention_backend in "CUSTOM"
    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=gen_config.compile.is_eager,
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_backend="CUSTOM",
    )
    vllm_compilation_config = gen_config.compile.get_vllm_compilation_config()
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.seed is not None:
        engine_kwargs["seed"] = gen_config.seed
    engine_args = EngineArgs(**engine_kwargs)

    if dist.get_rank() == 0:
        logger.info("Creating vLLM LLMEngine ...")
    engine = LLMEngine.from_engine_args(engine_args)
    if dist.get_rank() == 0:
        logger.info("vLLM LLMEngine ready.")
    return engine


def generate_with_vllm(
    engine: LLMEngine, prompt: str, gen_config: VLLMGenerator.Config
) -> tuple[list[int], list[int], list[float]]:
    """Generate tokens from *prompt*, return IDs + log-probs."""
    sampling = gen_config.sampling
    sampling_params = SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        logprobs=1,
        prompt_logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    engine.add_request("0", prompt, sampling_params)

    outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        outputs.extend(step_outputs)

    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    output = outputs[0]

    logger.info('vLLM text output: "%s"', output.outputs[0].text)
    prompt_token_ids = list(output.prompt_token_ids)
    sample = output.outputs[0]
    generated_token_ids = list(sample.token_ids)
    vllm_log_probs = [list(lp_dict.values())[0].logprob for lp_dict in sample.logprobs]

    if dist.get_rank() == 0:
        logger.info(
            f"vLLM generated {len(generated_token_ids)} tokens "
            f"(prompt length={len(prompt_token_ids)})"
        )
    return prompt_token_ids, generated_token_ids, vllm_log_probs


# ---------------------------------------------------------------------------
# Step 2: Build trainer-side TorchTitan model
# ---------------------------------------------------------------------------


def build_trainer_model(config):
    """Build, parallelize, and load weights for the trainer model.

    Mirrors PolicyTrainer._build_model() but without the Monarch actor
    framework.
    """
    from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant_mode

    # Enable batch-invariant Triton kernels (log_softmax, mean) and
    # deterministic cuBLAS before building the model.
    enable_batch_invariant_mode()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import dataclasses

    model_spec = config.model_spec
    hf_assets_path = config.hf_assets_path

    # Override attention backend to varlen for batch-invariant mode,
    # matching PolicyTrainer._build_model() behavior.
    if config.batch_invariant_mode:
        attn_cfg = model_spec.model.layer.attention
        new_attn = dataclasses.replace(attn_cfg, attn_backend="varlen")
        new_layer = dataclasses.replace(model_spec.model.layer, attention=new_attn)
        model_config = dataclasses.replace(model_spec.model, layer=new_layer)
    else:
        model_config = model_spec.model

    device_module, device_type = utils.device_module, utils.device_type
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)

    world_size = dist.get_world_size()
    parallelism = config.trainer.parallelism

    parallel_dims = ParallelDims(
        dp_shard=parallelism.data_parallel_shard_degree,
        dp_replicate=parallelism.data_parallel_replicate_degree,
        cp=parallelism.context_parallel_degree,
        tp=parallelism.tensor_parallel_degree,
        pp=parallelism.pipeline_parallel_degree,
        ep=parallelism.expert_parallel_degree,
        etp=parallelism.expert_tensor_parallel_degree,
        world_size=world_size,
    )

    # Build model on meta device
    with torch.device("meta"):
        with utils.set_default_dtype(TORCH_DTYPE_MAP[config.trainer.training.dtype]):
            model = model_config.build()

    # Disable torch.compile on VarlenAttentionWrapper to match the
    # generator's uncompiled varlen_attn_out path.
    if config.batch_invariant_mode:
        from torch.nn.attention.varlen import varlen_attn

        from torchtitan.models.common.attention import VarlenAttentionWrapper

        VarlenAttentionWrapper._compiled_varlen_attn = varlen_attn

    # Parallelize (trainer path: has_position_id=False)
    model = parallelize_qwen3(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
    )

    # Materialize on device
    model.to_empty(device=device_type)
    with torch.no_grad():
        model.init_states(buffer_device=None)

    # Load HF checkpoint (same logic as PolicyTrainer._load_initial_hf_weights)
    if model_spec.state_dict_adapter is not None:
        sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
        storage_reader = sd_adapter.get_hf_storage_reader(hf_assets_path)
        hf_state_dict = sd_adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = sd_adapter.from_hf(hf_state_dict)
        set_model_state_dict(
            model=model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=False),
        )
        if dist.get_rank() == 0:
            logger.info(
                f"Loaded HF weights from {hf_assets_path} "
                f"({len(torchtitan_state_dict)} params)"
            )

    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _test_config() -> RLTrainer.Config:
    """Test-specific config: greedy sampling, fewer tokens, single sample."""
    model_spec = model_registry("0.6B_varlen")
    return RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        batch_invariant_mode=True,
        trainer=PolicyTrainer.Config(
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
            ),
            compile=CompileConfig(enable=True, backend="aot_eager"),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=0.5,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
            ),
            compile=GeneratorCompileConfig(backend="eager", cudagraph_mode="piecewise"),
            num_samples_per_prompt=1,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=200,
            ),
        ),
    )


def main():
    # ---- Config setup ----
    config = _test_config()
    config.model_spec.parallelize_fn = parallelize_qwen3

    # Activate FA3 so varlen attention uses the same FA3 kernel as the
    # generator's CUSTOM backend.
    from torch.nn.attention import (
        activate_flash_attention_impl,
        current_flash_attention_impl,
    )

    if current_flash_attention_impl() != "FA3":
        activate_flash_attention_impl("FA3")

    # Register model with vLLM
    register_model_to_vllm_model_registry(config.model_spec)

    # Initialize distributed (needed for both vLLM and trainer model)
    dist_utils.init_distributed(CommConfig())

    # ---- Step 1: Generate via vLLM ----
    prompt = (
        "You are a helpful assistant that solves math problems step by step. "
        "Show your reasoning clearly and carefully before giving the final answer."
    )

    engine = create_vllm_engine(config)
    vllm_model = engine.model_executor.driver_worker.get_model().model
    _log_attention_module(vllm_model, "vLLM")
    prompt_token_ids, generated_token_ids, vllm_log_probs = generate_with_vllm(
        engine, prompt, config.generator
    )

    if dist.get_rank() == 0:
        logger.info(f"Prompt token IDs: {prompt_token_ids}")
        logger.info(
            f"Generated token IDs ({len(generated_token_ids)}): {generated_token_ids}"
        )

    # Free vLLM engine GPU memory before building the trainer model
    del vllm_model, engine
    torch.cuda.empty_cache()

    # ---- Step 2: Build trainer model and compute log-probs ----
    trainer_model, device = build_trainer_model(config)
    _log_attention_module(trainer_model, "Trainer")

    with torch.no_grad():
        trainer_token_log_probs = compute_token_log_probs(
            trainer_model,
            prompt_token_ids,
            generated_token_ids,
            device,
        )

    if dist.get_rank() == 0:
        logger.info(
            f"Trainer computed {trainer_token_log_probs.shape[0]} token log-probs"
        )
        n_preview = min(5, len(vllm_log_probs))
        logger.info(f"  vLLM   log-probs[:{n_preview}]: {vllm_log_probs[:n_preview]}")
        logger.info(
            f"  Trainer log-probs[:{n_preview}]: "
            f"{trainer_token_log_probs[:n_preview].tolist()}"
        )

    # ---- Step 3: Compare (rank 0 only) ----
    if dist.get_rank() == 0:
        result = verify_logprob_identity(
            [vllm_log_probs],
            [trainer_token_log_probs],
        )

        logger.info("=" * 60)
        logger.info("LOGPROB COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Bitwise identical : {result['logprob_bitwise_identical']}")
        logger.info(f"  Tokens checked    : {result['total_tokens_checked']}")
        logger.info(f"  Tokens different  : {result['num_tokens_different']}")
        logger.info(f"  Max delta         : {result['logprob_max_delta']:.6e}")
        logger.info(f"  Avg delta         : {result['avg_delta']:.6e}")
        logger.info(f"  Diff mean         : {result['logprob_diff_mean']:.6e}")
        logger.info(f"  Diff max          : {result['logprob_diff_max']:.6e}")
        logger.info("=" * 60)

        assert result["logprob_bitwise_identical"], (
            f"FAIL: vLLM and trainer log-probs are NOT bitwise identical.\n"
            f"  max_delta={result['logprob_max_delta']:.6e}, "
            f"  diff_max={result['logprob_diff_max']:.6e}"
        )
        logger.info("PASS: vLLM and trainer log-probs are bitwise identical.")


if __name__ == "__main__":
    main()
