"""
Test debugmodel_moe with TorchTitan vLLM wrapper: TP=2, no EP.
Verifies if MoE + PrepareModuleInputOutput works in the vLLM path.

torchrun --nproc_per_node=2 /tmp/test_debug_moe.py
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
from torchtitan.experiments.rl.scripts.generate_debug_checkpoint import main as gen_ckpt

# Step 1: Generate debug checkpoint if it doesn't exist
ckpt_dir = "/tmp/debugmodel_moe_ckpt"
if not os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
    import sys
    sys.argv = [
        "gen", "--model_name", "debugmodel_moe",
        "--output_dir", ckpt_dir,
    ]
    gen_ckpt()
    print(f"Generated checkpoint at {ckpt_dir}", flush=True)

# Step 2: Run with TorchTitan wrapper, TP=2, NO EP
model_spec = model_registry("debugmodel_moe", attn_backend="varlen")
model_spec = replace(model_spec, parallelize_fn=parallelize_qwen3)
register_model_to_vllm_model_registry(model_spec)

from vllm import EngineArgs, LLMEngine, SamplingParams

engine = LLMEngine.from_engine_args(EngineArgs(
    model=ckpt_dir,
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=2,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.30,
    enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

# Use simple numeric prompts since the debug model has tiny vocab
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
            # Check if output is repetitive garbage
            unique = len(set(tokens))
            print(f"Unique tokens: {unique}/{len(tokens)} {'(looks repetitive)' if unique <= 3 else '(diverse)'}", flush=True)
