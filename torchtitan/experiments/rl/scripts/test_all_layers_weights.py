"""Check loaded parameter norms for all layers."""
import os, torch, torch.distributed as dist
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

from dataclasses import replace
from torch.distributed._tensor import DTensor

from torchtitan.models.qwen3 import model_registry
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import register_model_to_vllm_model_registry, VLLM_MODEL_NAME
from vllm import EngineArgs, LLMEngine

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"

model_spec = model_registry("30B-A3B", attn_backend="varlen", moe_comm_backend="standard")
model_spec = replace(model_spec, parallelize_fn=parallelize_qwen3)
register_model_to_vllm_model_registry(model_spec)

engine = LLMEngine.from_engine_args(EngineArgs(
    model=model_path, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=2, enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=0.40, enforce_eager=True,
    hf_overrides={"architectures": [VLLM_MODEL_NAME]},
    attention_backend="CUSTOM",
))

rank = dist.get_rank()
if rank != 0:
    dist.barrier()
    exit()

model = engine.model_executor.driver_worker.model_runner.model.model

# Check every layer's gate norm and expert w1 norm
for i, layer in enumerate(model.layers.values()):
    if not layer.moe_enabled:
        continue
    gate = layer.moe.router.gate.weight
    gate_local = gate._local_tensor if isinstance(gate, DTensor) else gate
    gate_norm = gate_local.float().norm().item()

    w1 = layer.moe.experts.w1
    w1_local = w1._local_tensor if isinstance(w1, DTensor) else w1
    w1_norm = w1_local.float().norm().item()

    # Check if any are zero or NaN
    zero = gate_norm == 0 or w1_norm == 0
    nan = torch.isnan(gate_local).any().item() or torch.isnan(w1_local).any().item()

    if zero or nan or i in [0, 23, 47]:
        print(
            f"layer {i}: gate_norm={gate_norm:.4f} w1_norm={w1_norm:.4f}"
            f"{' ZERO!' if zero else ''}{' NAN!' if nan else ''}",
            flush=True,
        )

print(f"Total MoE layers checked: {sum(1 for l in model.layers.values() if l.moe_enabled)}", flush=True)

dist.barrier()
