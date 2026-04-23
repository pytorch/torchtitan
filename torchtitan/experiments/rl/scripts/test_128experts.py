"""Test debug model but with 128 experts (matching real model expert count)."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

import torch.distributed as dist
from dataclasses import replace as dc_replace
from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import register_model_to_vllm_model_registry, VLLM_MODEL_NAME

# Generate checkpoint with 128 experts
from torchtitan.experiments.rl.scripts.generate_debug_checkpoint import _build_hf_config
import torch, json, os, shutil
from safetensors.torch import save_file

ckpt_dir = "/tmp/debugmodel_moe128_ckpt"
if not os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
    # Build debug model with 128 experts
    spec = model_registry("debugmodel_moe")
    orig_cfg = spec.model
    # Modify all MoE layers to have 128 experts
    new_layers = []
    for l in orig_cfg.layers:
        if l.moe is not None:
            new_experts = dc_replace(l.moe.experts, num_experts=128)
            new_router = dc_replace(l.moe.router, num_experts=128)
            new_td = dc_replace(l.moe.experts.token_dispatcher, num_experts=128)
            new_experts = dc_replace(new_experts, token_dispatcher=new_td)
            new_moe = dc_replace(l.moe, num_experts=128, experts=new_experts, router=new_router)
            # Also update the gate Linear to match new num_experts
            from torchtitan.models.common.linear import Linear
            gate_cfg = dc_replace(new_router.gate, out_features=128)
            new_router = dc_replace(new_router, gate=gate_cfg)
            new_moe = dc_replace(new_moe, router=new_router)
            new_layers.append(dc_replace(l, moe=new_moe))
        else:
            new_layers.append(l)
    new_cfg = dc_replace(orig_cfg, layers=new_layers)
    new_spec = dc_replace(spec, model=new_cfg)

    model = new_cfg.build()
    # Random init
    for p in model.parameters():
        p.data.normal_(std=0.02)

    # Save as safetensors
    sd = model.state_dict()
    adapter = new_spec.state_dict_adapter(new_cfg, None)
    hf_sd = adapter.to_hf(sd)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_file({k: v for k, v in hf_sd.items()}, os.path.join(ckpt_dir, "model.safetensors"))

    # Save minimal config.json
    hf_config = _build_hf_config("debugmodel_moe", new_spec)
    hf_config["num_experts"] = 128
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    # Copy tokenizer
    tok_path = "./tests/assets/tokenizer"
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tok_path, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ckpt_dir, fn))

    print(f"Generated 128-expert checkpoint at {ckpt_dir}", flush=True)

# Run with EP
model_spec = model_registry("debugmodel_moe", attn_backend="varlen", moe_comm_backend="standard")
# Override to 128 experts
orig_cfg = model_spec.model
new_layers = []
for l in orig_cfg.layers:
    if l.moe is not None:
        from torchtitan.models.common.linear import Linear
        new_experts = dc_replace(l.moe.experts, num_experts=128)
        new_td = dc_replace(l.moe.experts.token_dispatcher, num_experts=128)
        new_experts = dc_replace(new_experts, token_dispatcher=new_td)
        gate_cfg = dc_replace(l.moe.router.gate, out_features=128)
        new_router = dc_replace(l.moe.router, num_experts=128, gate=gate_cfg)
        new_moe = dc_replace(l.moe, num_experts=128, experts=new_experts, router=new_router)
        new_layers.append(dc_replace(l, moe=new_moe))
    else:
        new_layers.append(l)
new_cfg = dc_replace(orig_cfg, layers=new_layers)
model_spec = dc_replace(model_spec, model=new_cfg, parallelize_fn=parallelize_qwen3)
register_model_to_vllm_model_registry(model_spec)

from vllm import EngineArgs, LLMEngine, SamplingParams

engine = LLMEngine.from_engine_args(EngineArgs(
    model=ckpt_dir,
    trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=2,
    enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.30, enforce_eager=True,
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
            tokens = list(o.outputs[0].token_ids)
            unique = len(set(tokens))
            print(f"Output: {o.outputs[0].text!r}", flush=True)
            print(f"Unique: {unique}/{len(tokens)} {'(diverse)' if unique > 5 else '(repetitive)'}", flush=True)

dist.barrier()
