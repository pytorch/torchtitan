"""Compare prompt logprobs: native vLLM EP vs TorchTitan EP."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import json, torch.distributed as dist
from dataclasses import replace
from vllm import EngineArgs, LLMEngine, SamplingParams

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
PROMPTS = [
    "The capital of France is",
    "Write a short poem about the ocean:",
    "What is 2 + 2? The answer is",
    "In the year 2050, artificial intelligence will",
]

def run_engine(mode, tp_size=2):
    if mode == "native_ep":
        engine_args = EngineArgs(
            model=model_path, trust_remote_code=True, dtype="bfloat16",
            tensor_parallel_size=tp_size, enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=0.40, enforce_eager=True,
        )
    elif mode == "torchtitan_ep":
        from torchtitan.models.qwen3 import model_registry
        from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
        from torchtitan.experiments.rl.plugin import register_model_to_vllm_model_registry, VLLM_MODEL_NAME
        model_spec = model_registry("30B-A3B", attn_backend="varlen", moe_comm_backend="standard")
        model_spec = replace(model_spec, parallelize_fn=parallelize_qwen3)
        register_model_to_vllm_model_registry(model_spec)
        engine_args = EngineArgs(
            model=model_path, trust_remote_code=True, dtype="bfloat16",
            tensor_parallel_size=tp_size, enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=0.40, enforce_eager=True,
            hf_overrides={"architectures": [VLLM_MODEL_NAME]},
            attention_backend="CUSTOM",
        )

    engine = LLMEngine.from_engine_args(engine_args)
    engine.add_request("0", PROMPTS[0], SamplingParams(
        temperature=0.0, max_tokens=5, prompt_logprobs=5, logprobs=5,
    ))

    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                return o

# Run both
import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "native_ep"
result = run_engine(mode)

rank = dist.get_rank() if dist.is_initialized() else 0
if rank == 0:
    print(f"\n=== {mode} ===", flush=True)
    print(f"Generated: {result.outputs[0].text!r}", flush=True)
    print(f"Generated tokens: {list(result.outputs[0].token_ids)}", flush=True)

    # Print prompt logprobs
    if result.prompt_logprobs:
        for i, lp in enumerate(result.prompt_logprobs):
            if lp is None:
                continue
            top = sorted(lp.items(), key=lambda x: x[1].logprob, reverse=True)[:3]
            top_str = ", ".join(f"{tid}:{v.logprob:.4f}" for tid, v in top)
            print(f"  pos {i}: top3=[{top_str}]", flush=True)

    # Print generation logprobs
    if result.outputs[0].logprobs:
        for i, lp in enumerate(result.outputs[0].logprobs):
            top = sorted(lp.items(), key=lambda x: x[1].logprob, reverse=True)[:3]
            top_str = ", ".join(f"{tid}:{v.logprob:.4f}" for tid, v in top)
            print(f"  gen {i}: top3=[{top_str}]", flush=True)

if dist.is_initialized():
    dist.barrier()
