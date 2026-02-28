#!/usr/bin/env python3
"""Test NVFP4 GPT-OSS model inference with TRT-LLM."""

import sys
import os


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/models/nvfp4_export"

    print(f"Loading NVFP4 model from {model_path}...")
    print(f"Config check:")

    import json

    with open(f"{model_path}/config.json") as f:
        config = json.load(f)
    print(f"  model_type: {config.get('model_type')}")
    print(f"  hidden_size: {config.get('hidden_size')}")
    print(
        f"  num_experts: {config.get('num_local_experts', config.get('num_experts'))}"
    )
    if "quantization_config" in config:
        qc = config["quantization_config"]
        print(f"  quant_method: {qc.get('quant_method')}")
        print(f"  ignore: {qc.get('ignore', [])}")

    # Import TRT-LLM
    from tensorrt_llm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_batch_size=1,
        max_num_tokens=256,
    )

    prompts = [
        "The capital of France is",
        "In a distant galaxy, a lone explorer",
        "The key principles of quantum mechanics include",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    print(f"\n--- Generating ({len(prompts)} prompts) ---")
    outputs = llm.generate(prompts, sampling_params)

    all_ok = True
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        tokens = (
            list(output.outputs[0].token_ids) if output.outputs[0].token_ids else []
        )
        print(f"\nPrompt {i}: {prompts[i]!r}")
        print(f"Output:  {text!r}")
        print(f"Tokens:  {tokens[:10]}")

        # Basic quality checks
        if not text.strip():
            print("  WARNING: Empty output!")
            all_ok = False
        elif text.strip() == "!" * len(text.strip()):
            print("  WARNING: Garbage output (repeated punctuation)!")
            all_ok = False
        elif len(set(text.strip())) <= 2:
            print("  WARNING: Degenerate output (very low entropy)!")
            all_ok = False
        else:
            print("  OK: Output looks reasonable")

    print(f"\n{'=' * 60}")
    if all_ok:
        print("SUCCESS: All outputs look reasonable!")
    else:
        print("ISSUES: Some outputs had problems. Check above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
