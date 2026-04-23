"""Test: TorchTitan wrapper TP=4, no EP, reduced max_model_len."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch.distributed as dist
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
    tensor_parallel_size=4,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.95,
    max_model_len=256,
    enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank()
prompts = [
    "The capital of France is",
    "In the year 2050, artificial intelligence will",
]
for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=30))

while engine.has_unfinished_requests():
    for o in engine.step():
        if o.finished and rank == 0:
            print(f"Prompt: {o.prompt!r}", flush=True)
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(flush=True)

dist.barrier()
