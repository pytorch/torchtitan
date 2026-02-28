#!/usr/bin/env python3
"""Diagnostic: Load NVFP4 model in vLLM and check for NaN in forward pass."""

import torch
import sys


def main():
    from vllm import LLM, SamplingParams

    model_path = (
        sys.argv[1] if len(sys.argv) > 1 else "/home/w/torchtitan/outputs/nvfp4_export"
    )

    # Load with enforce_eager to avoid CUDA graph issues
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=256,
    )

    # Try a simple generation
    print("\n--- Generating ---")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10, logprobs=3)
    try:
        outputs = llm.generate(["The capital of France is"], sampling_params)
        for output in outputs:
            print(f"Output text: {repr(output.outputs[0].text)}")
            if output.outputs[0].token_ids:
                print(f"Token IDs: {list(output.outputs[0].token_ids)[:10]}")
            if output.outputs[0].logprobs:
                for i, lp in enumerate(output.outputs[0].logprobs[:3]):
                    print(f"  Token {i}: {lp}")
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
