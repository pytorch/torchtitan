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
from vllm import EngineArgs, LLMEngine, SamplingParams

model_spec = model_registry("0.6B", attn_backend="varlen")
model_spec = replace(model_spec, parallelize_fn=parallelize_qwen3)
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=2,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.40,
    enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

prompts = [
    "The capital of France is",
    "What is 2 + 2? The answer is",
]

for i, p in enumerate(prompts):
    engine.add_request(str(i), p, SamplingParams(temperature=0.0, max_tokens=30))

while engine.has_unfinished_requests():
    outputs = engine.step()
    for o in outputs:
        if o.finished and dist.get_rank() == 0:
            print(f"Prompt: {o.prompt!r}")
            print(f"Output: {o.outputs[0].text!r}")
            print(f"Tokens: {list(o.outputs[0].token_ids)}")
            print()
