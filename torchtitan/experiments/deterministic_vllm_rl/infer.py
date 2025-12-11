#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from vllm import LLM, SamplingParams

# Import models module - this automatically registers TorchTitan models with vLLM
from torchtitan.experiments.deterministic_vllm_rl import models  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TorchTitan Qwen3 model inference with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
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

    print("=" * 80)
    print("INITIALIZING vLLM WITH TORCHTITAN QWEN3 MODEL ")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print()

    # Build hf_overrides with checkpoint path
    hf_overrides = {
        "checkpoint_dir": args.model,
    }

    # Initialize vLLM with custom TorchTitan Qwen3 model
    # The LLM initialization will internally:
    # 1. Load TrainSpec for Qwen3 (from register())
    # 2. Create TorchTitanVLLMModel instance
    # 3. Process parallelism settings via process_parallelism_settings()
    # 4. Build device mesh and apply parallelization via build_device_mesh_and_parallelize()
    # 5. Load model weights and prepare for inference
    print("Initializing vLLM engine...")
    llm = LLM(
        model=args.model,  # Model checkpoint path
        hf_overrides=hf_overrides,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,  # Use eager mode
        tensor_parallel_size=args.tensor_parallel_size,  # Multi-GPU support
    )

    print("=" * 80)
    print("vLLM ENGINE INITIALIZED - CONFIGURATION DETAILS")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print()

    # Prepare prompt and sampling parameters
    prompts = [args.prompt]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # Generate text
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text!r}")
        print()


if __name__ == "__main__":
    main()
