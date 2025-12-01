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

import torch.nn as nn
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_context import ParallelContext


def build_qwen3_torchtitan(vllm_config, parallel_context: ParallelContext) -> nn.Module:
    """
    Factory function to build Qwen3 with TorchTitan + vLLM.

    This is registered with vLLM's ModelRegistry to enable:
        LLM(model="Qwen/Qwen3-0.6B", ...)

    Args:
        vllm_config: vLLM configuration object
        parallel_context: Parallelism context with TP/PP info

    Returns:
        TorchTitanQwen3ForCausalLM instance
    """
    from torchtitan.experiments.vllm.model.qwen3 import TorchTitanQwen3ForCausalLM

    # Create model
    model = TorchTitanQwen3ForCausalLM(
        vllm_config=vllm_config, parallel_context=parallel_context
    )

    # Apply tensor parallelism if TP > 1
    # This must happen AFTER model creation and attention replacement
    # but BEFORE dtype conversion (to avoid dtype issues with DTensors)
    if parallel_context is not None:
        tp_size = parallel_context.get_tensor_parallel_world_size()
        if tp_size > 1:
            from torch.distributed.device_mesh import init_device_mesh
            from torchtitan.models.qwen3.infra.parallelize import apply_non_moe_tp

            print(f"🔧 Applying Tensor Parallelism (TP={tp_size})...")

            # Create DeviceMesh for TorchTitan
            tp_mesh = init_device_mesh(
                "cuda",
                (tp_size,),
                mesh_dim_names=("tp",),
            )

            # Apply TorchTitan's tensor parallelism to shard weights
            apply_non_moe_tp(
                model.model,
                tp_mesh=tp_mesh,
                loss_parallel=False,  # Don't shard the output for loss computation
                enable_float8_tensorwise_tp=False,
                enable_async_tp=False,
            )

            print(f"✅ Applied Tensor Parallelism (TP={tp_size})")

    # Convert to dtype if specified (happens after TP)
    if hasattr(vllm_config, "model_config") and hasattr(
        vllm_config.model_config, "dtype"
    ):
        model = model.to(dtype=vllm_config.model_config.dtype)

    return model


def register_torchtitan_model():
    """
    Register the TorchTitan Qwen3 custom model with vLLM using factory function pattern.

    This registers a factory function that vLLM will call to create the model,
    allowing us to apply tensor parallelism and other transformations.
    """
    try:
        from vllm import ModelRegistry

        # Register the factory function with vLLM
        # vLLM will call build_qwen3_torchtitan(vllm_config, parallel_context)
        ModelRegistry.register_model(
            "Qwen3TorchTitanForCausalLM", build_qwen3_torchtitan
        )

        print("✅ Successfully registered TorchTitan Qwen3 custom model with vLLM")
        return True

    except Exception as e:
        print(f"⚠️  Warning: Failed to register custom model: {e}")
        return False


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
    print("REGISTERING TORCHTITAN QWEN3 CUSTOM MODEL")
    print("=" * 80)

    # Register the custom model with vLLM
    register_torchtitan_model()

    # Create a temporary directory with minimal config.json for vLLM

    print(f"Using checkpoint and config.json from: {args.model}")

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
