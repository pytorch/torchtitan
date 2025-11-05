# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple inference script for TorchTitan-trained Qwen3 model using vLLM.

This script demonstrates how to:
1. Register a custom TorchTitan Qwen3 model with vLLM
2. Load a TorchTitan checkpoint into vLLM
3. Run inference using vLLM's optimized engine

Usage:
    python infer.py --model-path /path/to/torchtitan/checkpoint --prompt "Hello, world!"
"""

import argparse
import logging
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.model_executor.models import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def register_torchtitan_qwen3_model():
    """Register the TorchTitan Qwen3 model with vLLM's model registry."""
    from torchtitan.experiments.vllm.model.qwen3 import TorchTitanQwen3ForCausalLM

    logger.info("Registering TorchTitan Qwen3 model with vLLM")

    # Register the model using Qwen3's architecture but with custom weight loading
    ModelRegistry.register_model(
        "TorchTitanQwen3ForCausalLM",
        TorchTitanQwen3ForCausalLM,
    )

    print("Successfully registered TorchTitanQwen3ForCausalLM")


def run_inference(
    model: str,
    prompts: list[str],
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
):
    """
    Run inference using vLLM with a TorchTitan-trained Qwen3 model.

    Args:
        model: Model name
        prompts: List of prompts to generate from
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        tensor_parallel_size: Number of GPUs for tensor parallelism
    """
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # entry point:
    try:
        llm = LLM(
            model=model,
            model_impl="vllm",
            skip_tokenizer_init=True,
        )
    except Exception as e:
        logger.error(
            "Failed to initialize vLLM engine with TorchTitanQwen3ForCausalLM model\n"
        )
        raise

    logger.info("Model loaded successfully, starting generation...")

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logger.info("-" * 80)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated_text}")

    logger.info("-" * 80)
    logger.info(f"Generated {len(outputs)} outputs successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with TorchTitan Qwen3 model using vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="torchtitan/experiments/vllm/checkpoint/",
        help="Path to the TorchTitan checkpoint or HuggingFace model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Single prompt to generate from",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing prompts (one per line)",
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
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )

    args = parser.parse_args()

    # Register the custom model
    register_torchtitan_qwen3_model()

    # Prepare prompts
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
        prompts = prompts_path.read_text().strip().split("\n")
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")
    else:
        prompts = [args.prompt]

    # Run inference
    run_inference(
        model=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
