"""
Test debugmodel_moe with TorchTitan vLLM wrapper: TP=2, EP=2.

torchrun --nproc_per_node=2 /tmp/test_debug_moe_ep.py
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch.distributed as dist
from dataclasses import replace
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)

ckpt_dir = "/tmp/debugmodel_moe_ckpt"

# EP needs AllToAllTokenDispatcher
model_spec = model_registry("debugmodel_moe", attn_backend="varlen", moe_comm_backend="standard")
model_spec = replace(model_spec, parallelize_fn=parallelize_qwen3)
register_model_to_vllm_model_registry(model_spec)

from vllm import EngineArgs, LLMEngine, SamplingParams

engine = LLMEngine.from_engine_args(EngineArgs(
    model=ckpt_dir,
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.30,
    enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

prompts = ["1 2 3 4 5"]
for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=20))

while engine.has_unfinished_requests():
    outputs = engine.step()
    for o in outputs:
        if o.finished and dist.get_rank() == 0:
            print(f"Prompt: {o.prompt!r}", flush=True)
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            tokens = list(o.outputs[0].token_ids)
            print(f"Tokens: {tokens}", flush=True)
            unique = len(set(tokens))
            print(f"Unique tokens: {unique}/{len(tokens)} {'(looks repetitive)' if unique <= 3 else '(diverse)'}", flush=True)
