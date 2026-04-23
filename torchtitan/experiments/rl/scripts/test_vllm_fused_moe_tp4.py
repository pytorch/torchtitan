"""
Test: Replace TorchTitan's MoE with vLLM's FusedMoE for TP=4 inference.

Instead of creating new FusedMoE blocks (which need vllm_config context),
we use vLLM's native model for the SAME checkpoint and just check if
TorchTitan's attention + vLLM's MoE works.

Approach: Load model normally, then replace each MoE layer's forward
with a simple SwiGLU implementation using plain torch ops (no grouped_mm,
no DTensor). This isolates whether grouped_mm or DTensor MoE is the issue.

torchrun --nproc_per_node=4 test_vllm_fused_moe_tp4.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Partial, Replicate, Shard
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry, VLLM_MODEL_NAME,
)
from vllm import EngineArgs, LLMEngine, SamplingParams

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"
model_spec = model_registry("30B-A3B", attn_backend="varlen")
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=4, distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.95, max_model_len=256, enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank()
model = engine.model_executor.driver_worker.model_runner.model.model


# Replace GroupedExperts.forward with a plain implementation that does
# per-expert for-loop (no grouped_mm, bypasses any grouped_mm issues)
from torchtitan.models.common.moe import GroupedExperts

original_experts_forward = GroupedExperts._experts_forward

def plain_experts_forward(self, x, num_tokens_per_expert):
    """For-loop expert forward — no grouped_mm, uses plain matmul."""
    if isinstance(self.w1, DTensor):
        w1 = self.w1.to_local()
        w2 = self.w2.to_local()
        w3 = self.w3.to_local()
    else:
        w1, w2, w3 = self.w1, self.w2, self.w3

    # Use for-loop instead of grouped_mm
    from torchtitan.models.common.moe import _run_experts_for_loop
    return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

GroupedExperts._experts_forward = plain_experts_forward

if rank == 0:
    print("[SWAP] Replaced grouped_mm with for-loop expert forward", flush=True)

# Run inference
prompts = ["The capital of France is", "In the year 2050, artificial intelligence will"]
for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=20))

while engine.has_unfinished_requests():
    for o in engine.step():
        if o.finished and rank == 0:
            print(f"Prompt: {o.prompt!r}", flush=True)
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(flush=True)

dist.barrier()
