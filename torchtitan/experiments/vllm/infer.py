#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example CLI to run TorchTitan Qwen3 model inference with vLLM:

# Run inference
python torchtitan/experiments/vllm/infer.py
"""

import argparse

# Import and register the TorchTitan vLLM plugin
from torchtitan.experiments.vllm.register import register
from vllm import LLM, SamplingParams

# Register TorchTitan models with vLLM.
# NOTE(jianiw): We could use plug-in system instead: https://docs.vllm.ai/en/latest/design/plugin_system/
register()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TorchTitan Qwen3 model inference with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="torchtitan/experiments/vllm/example_checkpoint/qwen3-0.6B",
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
    print("INITIALIZING vLLM WITH TORCHTITAN QWEN3 MODEL")
    print("=" * 80)

    # Build hf_overrides with checkpoint path
    hf_overrides = {
        "checkpoint_dir": args.model,
    }

    # Initialize vLLM with custom TorchTitan Qwen3 model
    llm = LLM(
        model=args.model,  # Use temporary directory with config.json
        hf_overrides=hf_overrides,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,  # Use eager mode for debugging
        # Disable kv cache, required for now
        enable_prefix_caching=False,
        tensor_parallel_size=args.tensor_parallel_size,  # Multi-GPU support
    )

    print("=" * 80)
    print("vLLM ENGINE INITIALIZED - STARTING GENERATION")
    print("=" * 80)

    # Prepare prompt
    prompts = [args.prompt]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # Generate
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
