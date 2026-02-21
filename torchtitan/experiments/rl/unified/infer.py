#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example inference script using TorchTitan models with vLLM LLMEngine.

This script uses JobConfig loaded from a TOML file to configure both
the vLLM engine and sampling parameters.

Run: torchrun --nproc_per_node=2 \
      torchtitan/experiments/rl/unified/infer.py \
      --job.config_file torchtitan/experiments/rl/unified/run_configs/qwen3_0.6b.toml
"""
import os

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from torchtitan.config import ConfigManager
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.logger import init_logger


logger = init_logger(__name__)


def generate():

    config_manager = ConfigManager()
    job_config = config_manager.parse_args()

    # Load TorchTitan plugin at runtime (like native_sampler does)
    from torchtitan.experiments.rl.unified.plugin import register

    register(model_flavor="0.6B")
    logger.info("Loaded TorchTitan vLLM plugin")

    logger.info("Initializing vLLM LLMEngine with TorchTitan model")
    logger.info(f"Model: {job_config.checkpoint.initial_load_path}")
    logger.info(
        f"Tensor Parallel Size: {job_config.generation.parallelism.tensor_parallel_degree}"
    )

    # Create EngineArgs from JobConfig
    # Map TorchTitan parallelism to vLLM parallelism
    generation = job_config.generation

    engine_args = EngineArgs(
        # Model configuration
        model=job_config.checkpoint.initial_load_path,
        trust_remote_code=True,
        dtype=generation.dtype,
        # Parallelism configuration
        tensor_parallel_size=generation.parallelism.tensor_parallel_degree,
        distributed_executor_backend="external_launcher",
        # Memory and performance
        gpu_memory_utilization=generation.gpu_memory_utilization,
        enforce_eager=generation.enforce_eager,
        # Seed
        seed=job_config.debug.seed,
        # HuggingFace overrides
        hf_overrides={"architectures": ["Qwen3TorchTitanForCausalLM"]},
    )

    logger.info("Initializing LLMEngine from EngineArgs...")
    engine = LLMEngine.from_engine_args(engine_args)

    logger.info("vLLM LLMEngine initialized successfully")

    # Create sampling parameters from JobConfig
    sampling = job_config.generation.sampling
    sampling_params = SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
    )

    logger.info(
        f"Sampling params: temperature={sampling.temperature}, "
        f"top_p={sampling.top_p}, max_tokens={sampling.max_tokens}"
    )

    # Example prompt
    prompt = "Hello, my name is"
    logger.info(f"Prompt: {prompt}")

    # Add request to engine
    logger.info("Adding request to engine...")
    request_id = "0"
    engine.add_request(request_id, prompt, sampling_params)

    # Generate text by stepping through engine
    logger.info("Generating text...")
    while engine.has_unfinished_requests():
        request_outputs = engine.step()

        # Process finished requests
        for request_output in request_outputs:
            if request_output.finished:
                prompt = request_output.prompt
                generated_text = request_output.outputs[0].text

                # Print results
                logger.info("Generation complete")
                print(f"\nPrompt: {prompt}")
                print(f"Generated text: {generated_text!r}\n")


if __name__ == "__main__":
    generate()
