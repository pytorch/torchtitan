"""Verify TP shard assignment for Q projection weights."""
import os, json, torch, torch.distributed as dist
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_MOE_BACKEND"] = "triton"

from dataclasses import replace
from torch.distributed._tensor import DTensor
from safetensors.torch import load_file

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
model = engine.model_executor.driver_worker.model_runner.model.model

# Load HF weights directly
with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
    index = json.load(f)

# Check Q projection for layer 0
hf_key = "model.layers.0.self_attn.q_proj.weight"
fname = index["weight_map"][hf_key]
shard = load_file(os.path.join(model_path, fname))
hf_q = shard[hf_key].to(dtype=torch.bfloat16, device=f"cuda:{rank}")

# Get loaded weight
layer0 = list(model.layers.values())[0]
wq = layer0.attention.qkv_linear.wq.weight
wq_local = wq._local_tensor if isinstance(wq, DTensor) else wq

# HF q_proj is [4096, 2048]. ColwiseParallel shards output dim (dim 0).
# Rank 0 should have [0:2048, :], rank 1 should have [2048:4096, :]
half = hf_q.shape[0] // 2
expected = hf_q[rank * half:(rank + 1) * half, :]

diff = (wq_local - expected).float().abs().max().item()
print(
    f"[Q CHECK] rank={rank}, wq_local={wq_local.shape}, expected={expected.shape}, "
    f"diff={diff:.6f}, "
    f"wq[0,:3]={wq_local[0,:3].tolist()}, "
    f"exp[0,:3]={expected[0,:3].tolist()}",
    flush=True,
)

# Check output layer (ColwiseParallel too)
hf_lm_key = "lm_head.weight"
if hf_lm_key in index["weight_map"]:
    fname2 = index["weight_map"][hf_lm_key]
    shard2 = load_file(os.path.join(model_path, fname2))
    hf_lm = shard2[hf_lm_key].to(dtype=torch.bfloat16, device=f"cuda:{rank}")

    lm_w = model.output.weight
    lm_local = lm_w._local_tensor if isinstance(lm_w, DTensor) else lm_w

    half_lm = hf_lm.shape[0] // 2
    expected_lm = hf_lm[rank * half_lm:(rank + 1) * half_lm, :]
    diff_lm = (lm_local - expected_lm).float().abs().max().item()
    print(
        f"[LM CHECK] rank={rank}, diff={diff_lm:.6f}, "
        f"lm[0,:3]={lm_local[0,:3].tolist()}, "
        f"exp[0,:3]={expected_lm[0,:3].tolist()}",
        flush=True,
    )

# Check wo (RowwiseParallel, shards along input dim = dim 1)
hf_wo_key = "model.layers.0.self_attn.o_proj.weight"
fname3 = index["weight_map"][hf_wo_key]
shard3 = load_file(os.path.join(model_path, fname3))
hf_wo = shard3[hf_wo_key].to(dtype=torch.bfloat16, device=f"cuda:{rank}")

wo = layer0.attention.wo.weight
wo_local = wo._local_tensor if isinstance(wo, DTensor) else wo

# RowwiseParallel shards input dim (dim 1 for weight matrix)
half_wo = hf_wo.shape[1] // 2
expected_wo = hf_wo[:, rank * half_wo:(rank + 1) * half_wo]
diff_wo = (wo_local - expected_wo).float().abs().max().item()
print(
    f"[WO CHECK] rank={rank}, diff={diff_wo:.6f}, "
    f"wo_shape={wo_local.shape}, exp_shape={expected_wo.shape}, "
    f"wo[0,:3]={wo_local[0,:3].tolist()}, "
    f"exp[0,:3]={expected_wo[0,:3].tolist()}",
    flush=True,
)

dist.barrier()
