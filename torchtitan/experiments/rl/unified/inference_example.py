#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example inference script using TorchTitan models with vLLM LLMEngine.

This script uses the RL unified config_registry to configure both
the vLLM engine and sampling parameters.

Run: torchrun --nproc_per_node=<world_size> \
      torchtitan/experiments/rl/unified/infer.py
"""
import os

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from torchtitan.experiments.rl.unified.config_registry import rl_grpo_qwen3_0_6b

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.logger import init_logger


logger = init_logger(__name__)


def generate():

    config = rl_grpo_qwen3_0_6b()
    gen_config = config.generator
    model_path = config.trainer.hf_assets_path

    # Register TorchTitan model with vLLM before engine creation
    from torchtitan.experiments.rl.unified.plugin import (
        register_model_to_vllm_model_registry,
        VLLM_MODEL_NAME,
    )

    register_model_to_vllm_model_registry(config.model_spec)
    logger.info("Registered TorchTitan model with vLLM")

    logger.debug("Initializing vLLM LLMEngine with TorchTitan model")
    logger.debug(f"Model: {model_path}")
    logger.debug(
        f"Tensor Parallel Size: {gen_config.parallelism.tensor_parallel_degree}"
    )

    # Create EngineArgs from config
    engine_args = EngineArgs(
        # Model configuration
        model=model_path,
        trust_remote_code=True,
        dtype=gen_config.model_dtype,
        # Parallelism configuration
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        # Use external_launcher only when launched via torchrun (multi-GPU);
        # for single-GPU, let vLLM pick the default executor.
        distributed_executor_backend=("external_launcher"),
        # Memory and performance
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=gen_config.enforce_eager,
        # Seed
        seed=gen_config.seed,
        # HuggingFace overrides
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    )

    logger.debug("Initializing LLMEngine from EngineArgs...")
    engine = LLMEngine.from_engine_args(engine_args)

    logger.debug("vLLM LLMEngine initialized successfully")

    # Create sampling parameters from config
    sampling = gen_config.sampling
    sampling_params = SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
    )

    logger.debug(
        f"Sampling params: temperature={sampling.temperature}, "
        f"top_p={sampling.top_p}, max_tokens={sampling.max_tokens}"
    )

    # Example prompt
    prompt = "Hello, my name is"
    logger.debug(f"Prompt: {prompt}")

    # Add request to engine
    logger.debug("Adding request to engine...")
    request_id = "0"
    engine.add_request(request_id, prompt, sampling_params)

    # Generate text by stepping through engine
    logger.debug("Generating text...")
    while engine.has_unfinished_requests():
        request_outputs = engine.step()

        # Process finished requests
        for request_output in request_outputs:
            if request_output.finished:
                prompt = request_output.prompt
                generated_text = request_output.outputs[0].text

                # Print results
                logger.debug("Generation complete")
                print(f"\nPrompt: {prompt}")
                print(f"Generated text: {generated_text!r}\n")


if __name__ == "__main__":
    generate()
