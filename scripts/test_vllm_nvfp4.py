#!/usr/bin/env python3
"""Test NVFP4 GPT-OSS model inference with vLLM."""

import sys
import json


def main():
    model_path = (
        sys.argv[1] if len(sys.argv) > 1 else "/home/w/torchtitan/outputs/nvfp4_export"
    )

    print(f"Loading NVFP4 model from {model_path}...")

    with open(f"{model_path}/config.json") as f:
        config = json.load(f)
    print(f"  model_type: {config.get('model_type')}")
    print(f"  hidden_size: {config.get('hidden_size')}")
    if "quantization_config" in config:
        qc = config["quantization_config"]
        print(f"  quant_method: {qc.get('quant_method')}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_model_len=256,
    )

    prompts = [
        "The capital of France is",
        "In a distant galaxy, a lone explorer",
        "The key principles of quantum mechanics include",
    ]

    params = SamplingParams(temperature=0.0, max_tokens=32)

    print(f"\n--- Generating ({len(prompts)} prompts) ---")
    outputs = llm.generate(prompts, params)

    all_ok = True
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        print(f"\nPrompt {i}: {prompts[i]!r}")
        print(f"Output:  {text!r}")

        if not text.strip():
            print("  WARNING: Empty output!")
            all_ok = False
        elif len(set(text.strip())) <= 2:
            print("  WARNING: Degenerate output!")
            all_ok = False
        else:
            print("  OK")

    separator = "=" * 60
    print(f"\n{separator}")
    if all_ok:
        print("SUCCESS: All outputs look reasonable!")
    else:
        print("ISSUES: Some outputs had problems.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
