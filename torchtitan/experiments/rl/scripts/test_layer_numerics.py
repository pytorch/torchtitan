#!/usr/bin/env python3
"""
Compare vLLM native Qwen3MoE vs TorchTitan wrapper on a single prefill pass.

Both models load from the same HF checkpoint and process the same input tokens.
We compare the output logits (before sampling) to verify numerical parity.

This test avoids KV cache / decode issues by only doing prefill.

Usage:
    torchrun --nproc_per_node=2 \
        torchtitan/experiments/rl/scripts/test_layer_numerics.py \
        --model_path /data/users/jianiw/model/Qwen3-30B-A3B
"""

import argparse
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)

PROMPT = "The capital of France is"


def run_native(model_path: str, tp_size: int):
    """Run vLLM native model, return prefill logits."""
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.45,
        enforce_eager=True,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    # Use prompt_logprobs to get logits at every prompt position
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=5,
        logprobs=5,
    )

    engine.add_request("0", PROMPT, sampling_params)

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                sample = output.outputs[0]
                generated_token = sample.token_ids[0]
                generated_logprobs = sample.logprobs[0] if sample.logprobs else {}

                # prompt_logprobs: list of dicts, one per prompt token
                prompt_lps = output.prompt_logprobs

                return {
                    "generated_token": generated_token,
                    "generated_top5": {
                        str(tid): lp.logprob
                        for tid, lp in generated_logprobs.items()
                    },
                    "prompt_logprobs": [
                        {str(tid): lp.logprob for tid, lp in d.items()}
                        if d is not None
                        else None
                        for d in prompt_lps
                    ],
                }


def run_torchtitan_ep(model_path: str, tp_size: int):
    """Run TorchTitan wrapper with EP, return prefill logits."""
    from dataclasses import replace

    from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
    from torchtitan.experiments.rl.plugin import (
        register_model_to_vllm_model_registry,
        VLLM_MODEL_NAME,
    )
    from torchtitan.models.qwen3 import model_registry

    model_spec = model_registry(
        "30B-A3B",
        attn_backend="varlen",
        moe_comm_backend="standard",
    )
    model_spec = replace(model_spec, parallelize_fn=parallelize_qwen3)
    register_model_to_vllm_model_registry(model_spec)

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_backend="CUSTOM",
    )
    engine = LLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=5,
        logprobs=5,
    )

    engine.add_request("0", PROMPT, sampling_params)

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                sample = output.outputs[0]
                generated_token = sample.token_ids[0]
                generated_logprobs = sample.logprobs[0] if sample.logprobs else {}

                prompt_lps = output.prompt_logprobs

                return {
                    "generated_token": generated_token,
                    "generated_top5": {
                        str(tid): lp.logprob
                        for tid, lp in generated_logprobs.items()
                    },
                    "prompt_logprobs": [
                        {str(tid): lp.logprob for tid, lp in d.items()}
                        if d is not None
                        else None
                        for d in prompt_lps
                    ],
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=["native", "torchtitan_ep", "both"],
        default="both",
    )
    args = parser.parse_args()

    rank = dist.get_rank() if dist.is_initialized() else 0

    if args.mode in ("native", "both"):
        native_result = run_native(args.model_path, args.tp)
        if rank == 0:
            print(f"\n=== Native vLLM ===")
            print(f"Generated token: {native_result['generated_token']}")
            print(f"Top5: {native_result['generated_top5']}")

    if args.mode in ("torchtitan_ep", "both"):
        ep_result = run_torchtitan_ep(args.model_path, args.tp)
        if rank == 0:
            print(f"\n=== TorchTitan EP ===")
            print(f"Generated token: {ep_result['generated_token']}")
            print(f"Top5: {ep_result['generated_top5']}")

    if args.mode == "both" and rank == 0:
        print(f"\n=== Comparison ===")
        if native_result["generated_token"] == ep_result["generated_token"]:
            print(f"Generated token: MATCH ({native_result['generated_token']})")
        else:
            print(
                f"Generated token: MISMATCH "
                f"(native={native_result['generated_token']}, "
                f"ep={ep_result['generated_token']})"
            )

        # Compare prompt logprobs
        for i, (nlp, elp) in enumerate(
            zip(
                native_result["prompt_logprobs"],
                ep_result["prompt_logprobs"],
            )
        ):
            if nlp is None or elp is None:
                continue
            n_top = max(nlp, key=lambda k: nlp[k])
            e_top = max(elp, key=lambda k: elp[k])
            match = "OK" if n_top == e_top else "DIFF"
            print(
                f"  pos {i}: native_top={n_top}({nlp[n_top]:.4f}) "
                f"ep_top={e_top}({elp[e_top]:.4f}) [{match}]"
            )


if __name__ == "__main__":
    main()
