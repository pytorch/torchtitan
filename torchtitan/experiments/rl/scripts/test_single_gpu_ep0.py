"""Test: TorchTitan wrapper on single GPU (TP=1, EP=0) with Qwen3-30B-A3B."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch.distributed as dist
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from vllm import EngineArgs, LLMEngine, SamplingParams

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"

# No EP, no moe_comm_backend override (uses LocalTokenDispatcher)
model_spec = model_registry("30B-A3B", attn_backend="varlen")
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=1,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.95,
    enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank() if dist.is_initialized() else 0
engine.add_request("0", "The capital of France is", SamplingParams(temperature=0.0, max_tokens=20))

while engine.has_unfinished_requests():
    for o in engine.step():
        if o.finished and rank == 0:
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(f"Tokens: {list(o.outputs[0].token_ids)}", flush=True)

if dist.is_initialized():
    dist.barrier()
