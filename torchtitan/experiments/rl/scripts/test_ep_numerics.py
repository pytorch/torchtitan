#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Numerics parity test: TorchTitan vLLM wrapper (with EP) vs vLLM native.

Runs greedy decoding on the same prompts with both backends and compares
output tokens. Both should produce identical results.

Usage:
    # Step 1: Run vLLM native baseline (TP=2, no EP)
    torchrun --nproc_per_node=2 \
        torchtitan/experiments/rl/scripts/test_ep_numerics.py \
        --model_path /data/users/jianiw/model/Qwen3-30B-A3B \
        --mode native --tp 2

    # Step 2: Run TorchTitan wrapper with EP (TP=2, EP=2)
    torchrun --nproc_per_node=2 \
        torchtitan/experiments/rl/scripts/test_ep_numerics.py \
        --model_path /data/users/jianiw/model/Qwen3-30B-A3B \
        --mode torchtitan_ep --tp 2

    # Step 3: Run TorchTitan wrapper without EP (TP=2, EP=1)
    torchrun --nproc_per_node=2 \
        torchtitan/experiments/rl/scripts/test_ep_numerics.py \
        --model_path /data/users/jianiw/model/Qwen3-30B-A3B \
        --mode torchtitan --tp 2

    Compare the printed output tokens across runs.
"""

import argparse
import json
import os
import sys

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch.distributed as dist
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)

PROMPTS = [
    "The capital of France is",
    "Write a short poem about the ocean:",
    "What is 2 + 2? The answer is",
    "In the year 2050, artificial intelligence will",
]

MAX_TOKENS = 50


def run_native(model_path: str, tp_size: int):
    """Run vLLM native inference (baseline)."""
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=20,
    )

    for i, prompt in enumerate(PROMPTS):
        engine.add_request(str(i), prompt, sampling_params)

    results = {}
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                sample = output.outputs[0]
                token_ids = list(sample.token_ids)
                logprobs_list = []
                if sample.logprobs:
                    for lp_dict in sample.logprobs:
                        top_entries = {
                            str(tid): lp.logprob for tid, lp in lp_dict.items()
                        }
                        logprobs_list.append(top_entries)

                results[int(output.request_id)] = {
                    "prompt": output.prompt,
                    "text": sample.text,
                    "token_ids": token_ids,
                    "logprobs": logprobs_list,
                }

    return results


def run_torchtitan(model_path: str, tp_size: int, enable_ep: bool):
    """Run TorchTitan wrapper inference (with or without EP)."""
    from torchtitan.experiments.rl.models.vllm_registry import (
        register_model_to_vllm_model_registry,
        VLLM_MODEL_NAME,
    )
    from torchtitan.models.qwen3 import model_registry

    moe_comm_backend = "standard" if enable_ep else None
    model_spec = model_registry(
        "30B-A3B",
        attn_backend="varlen",
        moe_comm_backend=moe_comm_backend,
    )

    register_model_to_vllm_model_registry(model_spec)

    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_ep,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_backend="CUSTOM",
    )
    engine_args = EngineArgs(**engine_kwargs)
    engine = LLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=20,
    )

    for i, prompt in enumerate(PROMPTS):
        engine.add_request(str(i), prompt, sampling_params)

    results = {}
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                sample = output.outputs[0]
                token_ids = list(sample.token_ids)
                logprobs_list = []
                if sample.logprobs:
                    for lp_dict in sample.logprobs:
                        top_entries = {
                            str(tid): lp.logprob for tid, lp in lp_dict.items()
                        }
                        logprobs_list.append(top_entries)

                results[int(output.request_id)] = {
                    "prompt": output.prompt,
                    "text": sample.text,
                    "token_ids": token_ids,
                    "logprobs": logprobs_list,
                }

    return results


def print_results(results: dict, mode: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    print(f"\n{'=' * 60}")
    print(f"Mode: {mode}")
    print(f"{'=' * 60}")
    for i in sorted(results.keys()):
        r = results[i]
        print(f"\nPrompt {i}: {r['prompt']!r}")
        print(f"Output:   {r['text']!r}")
        print(f"Tokens:   {r['token_ids']}")
    print(f"{'=' * 60}\n")


def save_results(results: dict, mode: str, output_dir: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{mode}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")


def compare_results(file_a: str, file_b: str):
    """Compare two saved result files."""
    with open(file_a) as f:
        results_a = json.load(f)
    with open(file_b) as f:
        results_b = json.load(f)

    all_match = True
    for key in sorted(results_a.keys()):
        if key not in results_b:
            print(f"Prompt {key}: MISSING in {file_b}")
            all_match = False
            continue
        tokens_a = results_a[key]["token_ids"]
        tokens_b = results_b[key]["token_ids"]
        if tokens_a == tokens_b:
            print(f"Prompt {key}: MATCH ({len(tokens_a)} tokens)")
        else:
            print(f"Prompt {key}: MISMATCH")
            print(f"  {file_a}: {tokens_a}")
            print(f"  {file_b}: {tokens_b}")
            # Find first divergence
            for j, (a, b) in enumerate(zip(tokens_a, tokens_b)):
                if a != b:
                    print(f"  First divergence at position {j}: {a} vs {b}")

                    lp_a = results_a[key].get("logprobs", [])
                    lp_b = results_b[key].get("logprobs", [])
                    if j < len(lp_a) and j < len(lp_b):
                        print(f"  Logprobs A: {lp_a[j]}")
                        print(f"  Logprobs B: {lp_b[j]}")
                    break
            all_match = False

    if all_match:
        print("\nAll prompts match!")
    else:
        print("\nSome prompts diverged.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=["native", "torchtitan", "torchtitan_ep", "compare"],
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/ep_numerics_test",
    )
    parser.add_argument("--file_a", type=str, help="First file for compare mode")
    parser.add_argument("--file_b", type=str, help="Second file for compare mode")
    args = parser.parse_args()

    if args.mode == "compare":
        compare_results(args.file_a, args.file_b)
        return

    if args.mode == "native":
        results = run_native(args.model_path, args.tp)
    elif args.mode == "torchtitan":
        results = run_torchtitan(args.model_path, args.tp, enable_ep=False)
    elif args.mode == "torchtitan_ep":
        results = run_torchtitan(args.model_path, args.tp, enable_ep=True)

    print_results(results, args.mode)
    save_results(results, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
