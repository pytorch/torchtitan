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

Run: torchrun --nproc_per_node=2 \
      torchtitan/experiments/rl/inference_example.py
"""
import os

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.logger import init_logger

from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b


logger = init_logger(__name__)


def generate():

    config = rl_grpo_qwen3_0_6b()
    gen_config = config.generator
    model_path = config.hf_assets_path

    # Patch model_spec to use the RL-specific parallelize function.
    # TODO: Switch to canonical Qwen3 parallel plan
    from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3

    config.model_spec.parallelize_fn = parallelize_qwen3

    # Register TorchTitan model with vLLM before engine creation
    from torchtitan.experiments.rl.plugin import (
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
    engine_kwargs = dict(
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
        enforce_eager=gen_config.compile.is_eager,
        # HuggingFace overrides
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_backend="CUSTOM",
    )
    vllm_compilation_config = gen_config.compile.get_vllm_compilation_config()
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.seed is not None:
        engine_kwargs["seed"] = gen_config.seed
    engine_args = EngineArgs(**engine_kwargs)

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
