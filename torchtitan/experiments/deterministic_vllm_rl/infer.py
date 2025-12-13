#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from vllm import LLM, SamplingParams
from vllm.logger import init_logger

# Import models module - this automatically registers TorchTitan models with vLLM
from torchtitan.experiments.deterministic_vllm_rl import models  # noqa: F401


logger = init_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TorchTitan model inference with vLLM Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        default="torchtitan/experiments/deterministic_vllm_rl/example_checkpoint/qwen3-0.6B",
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


def main():
    args = parse_args()

    logger.info("Initializing vLLM with TorchTitan model")
    logger.info(f"Model: {args.model_ckpt_path}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    # Initialize vLLM with custom TorchTitan model
    # The LLM initialization will internally:
    # 1. Load TrainSpec for Qwen3 (from models/__init__.py register())
    # 2. Create TorchTitanVLLMModel instance
    # 3. Create JobConfig and ParallelDims from vLLM config
    # 4. Apply parallelization using parallelize_qwen3
    # 5. Load model weights and prepare for inference
    logger.info("Creating vLLM LLM engine...")

    llm = LLM(
        model=args.model_ckpt_path,  # Model checkpoint path
        hf_overrides={
            "checkpoint_dir": args.model_ckpt_path,
        },
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,  # Use eager mode
        tensor_parallel_size=args.tensor_parallel_size,
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
    main()
