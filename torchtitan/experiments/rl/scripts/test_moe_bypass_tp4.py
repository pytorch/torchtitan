"""
Test: Bypass MoE entirely (zero output) — test attention-only path.
If this produces diverse output, attention is fine and MoE is the issue.

torchrun --nproc_per_node=4 test_moe_bypass_tp4.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
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

# Replace MoE forward with zero output
from torchtitan.models.common.moe import MoE

original_moe_forward = MoE.forward

def zero_moe_forward(self, x):
    """Return zeros — bypass MoE entirely."""
    if isinstance(x, DTensor):
        return DTensor.from_local(
            torch.zeros_like(x._local_tensor),
            x.device_mesh, x.placements,
        )
    return torch.zeros_like(x)

MoE.forward = zero_moe_forward

if rank == 0:
    print("[BYPASS] MoE output zeroed", flush=True)

prompts = ["The capital of France is"]
for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=20))

while engine.has_unfinished_requests():
    for o in engine.step():
        if o.finished and rank == 0:
            tokens = list(o.outputs[0].token_ids)
            unique = len(set(tokens))
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(f"Unique: {unique}/{len(tokens)}", flush=True)

dist.barrier()
