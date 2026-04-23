"""
Test: Run Qwen3-30B-A3B through vLLM wrapper but with only layer 0.
If output is correct, the issue accumulates across layers.

torchrun --nproc_per_node=2 /tmp/test_single_layer.py
"""
import os, torch, torch.distributed as dist
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

from dataclasses import replace
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from vllm import EngineArgs, LLMEngine, SamplingParams

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"

# Build model spec but override to only 1 layer
spec = model_registry("30B-A3B", attn_backend="varlen", moe_comm_backend="standard")
# Hack: modify the config to only keep first layer
orig_config = spec.model
# Try with N layers
import sys
n_layers = int(os.environ.get("N_LAYERS", "1"))
single_layer_config = replace(orig_config, layers=orig_config.layers[:n_layers])
model_spec = replace(spec, model=single_layer_config, parallelize_fn=parallelize_qwen3)
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path,
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.40,
    enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank()
prompts = ["The capital of France is"]
for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=20))

while engine.has_unfinished_requests():
    outputs = engine.step()
    for o in outputs:
        if o.finished and rank == 0:
            tokens = list(o.outputs[0].token_ids)
            unique = len(set(tokens))
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(f"Tokens: {tokens}", flush=True)
            print(f"Unique: {unique}/{len(tokens)} {'(diverse)' if unique > 5 else '(repetitive)'}", flush=True)

dist.barrier()
