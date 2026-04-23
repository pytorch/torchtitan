"""
Test: does MoE.forward (with DTensor weights) match raw dispatcher + expert_fwd?

torchrun --nproc_per_node=2 /tmp/test_moe_layer.py
"""
import os, json, torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor
from safetensors.torch import load_file
from torch.distributed.tensor.parallel import parallelize_module
from dataclasses import replace

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
from torchtitan.distributed.expert_parallel import ExpertParallel

model_path = "/data/users/jianiw/model/Qwen3-30B-A3B"

# Load layer 0 weights
with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
    index = json.load(f)
layer0_hf_keys = [k for k in index["weight_map"]
                  if "layers.0.mlp.experts" in k or "layers.0.mlp.gate" in k]
layer0_files = set(index["weight_map"][k] for k in layer0_hf_keys)
hf_weights = {}
for fname in layer0_files:
    shard = load_file(os.path.join(model_path, fname))
    for k in layer0_hf_keys:
        if k in shard:
            hf_weights[k] = shard[k].to(dtype=torch.bfloat16, device="cuda")

num_experts = 128

def stack_experts(prefix):
    return torch.stack([hf_weights[f"model.layers.0.mlp.experts.{i}.{prefix}.weight"] for i in range(num_experts)])

w1 = stack_experts("gate_proj")
w2 = stack_experts("down_proj")
w3 = stack_experts("up_proj")
gate_w = hf_weights["model.layers.0.mlp.gate.weight"]

torch.manual_seed(42)
dim = 2048
x = torch.randn(1, 8, dim, device="cuda", dtype=torch.bfloat16)

ep_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
local_n = num_experts // 2
s = rank * local_n

# Build MoE module with AllToAllDispatcher + EP
spec = model_registry("30B-A3B", moe_comm_backend="standard")
moe_cfg = spec.model.layers[0].moe

with torch.device("meta"):
    moe = moe_cfg.build()
parallelize_module(moe.experts, ep_mesh, ExpertParallel())
moe.to_empty(device="cuda")
moe = moe.to(dtype=torch.bfloat16)

# Load weights
moe.experts.w1._local_tensor.copy_(w1[s:s+local_n])
moe.experts.w2._local_tensor.copy_(w2[s:s+local_n])
moe.experts.w3._local_tensor.copy_(w3[s:s+local_n])
moe.router.gate.weight.data.copy_(gate_w)
moe.eval()

# Verify to_local() returns correct weights
w1_local = moe.experts.w1.to_local()
expected = w1[s:s+local_n]
w1_diff = (w1_local - expected).float().abs().max().item()
print(f"[WEIGHTS] rank={rank}, to_local() w1 diff: {w1_diff:.6f}", flush=True)
print(f"[WEIGHTS] rank={rank}, w1_local[0,0,:3]={w1_local[0,0,:3].tolist()}, expected={expected[0,0,:3].tolist()}", flush=True)

# Run through MoE.forward
with torch.inference_mode():
    out_moe = moe(x)

print(f"[MOE] rank={rank}, norm={out_moe.float().norm():.4f}, [0,0,:5]={out_moe[0,0,:5].tolist()}", flush=True)

# Compare with reference
dist.barrier()
if rank == 0:
    from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
    ref_spec = model_registry("30B-A3B")
    ref_cfg = ref_spec.model.layers[0].moe
    moe_ref = ref_cfg.build().to("cuda", dtype=torch.bfloat16)
    moe_ref.experts.w1.data.copy_(w1)
    moe_ref.experts.w2.data.copy_(w2)
    moe_ref.experts.w3.data.copy_(w3)
    moe_ref.router.gate.weight.data.copy_(gate_w)
    moe_ref.eval()
    with torch.inference_mode():
        out_ref = moe_ref(x)
    print(f"[REF] norm={out_ref.float().norm():.4f}", flush=True)
    diff = (out_ref - out_moe).float().abs().max().item()
    print(f"[COMPARE MoE.forward] max_diff={diff:.6f} {'PASS' if diff < 0.01 else 'FAIL'}", flush=True)

dist.barrier()
