#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse

# Import unified module - this automatically registers TorchTitan models with vLLM
from torchtitan.experiments.rl import unified  # noqa: F401
from vllm import LLM, SamplingParams
from vllm.logger import init_logger


logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TorchTitan model inference with vLLM Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-ckpt-path",
        type=str,
        default="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B/",
        help="Path to TorchTitan checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1 for single GPU)",
    )
    return parser.parse_args()


def infer():
    args = parse_args()

    logger.info("Initializing vLLM with TorchTitan model")
    logger.info(f"Model: {args.model_ckpt_path}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    # Initialize vLLM with custom TorchTitan model
    # The LLM initialization will internally:
    # 1. Load ModelSpec for Qwen3 (from models/__init__.py register())
    # 2. Create TorchTitanVLLMModel instance
    # 3. Create JobConfig and ParallelDims from vLLM config
    # 4. Apply parallelization using parallelize_qwen3
    # 5. Load model weights and prepare for inference
    # The tensor_parallel_size will be used by vLLM to configure parallelization
    # and will be available in vllm_config in worker processes
    logger.info("Creating vLLM LLM engine...")

    llm = LLM(
        model=args.model_ckpt_path,  # Model checkpoint path
        hf_overrides={
            # Override architectures to use our registered TorchTitan model class
            "architectures": ["Qwen3TorchTitanForCausalLM"],
        },
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,  # Use eager mode
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.5,
    )

    logger.info("vLLM engine initialized successfully")
    logger.info(f"Prompt: {args.prompt}")

    # Prepare prompt and sampling parameters
    prompts = [args.prompt]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # Generate text
    logger.info("Generating text...")
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Print results
    logger.info("Generation complete")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {generated_text!r}\n")


if __name__ == "__main__":
    infer()
