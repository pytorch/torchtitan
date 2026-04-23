"""
Test full MoE module: EP=1 (LocalDispatcher) vs EP=2 (AllToAllDispatcher).
Uses actual Qwen3 model config to match real model setup.

torchrun --nproc_per_node=2 /tmp/test_dispatcher.py
"""
import os, torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

from torchtitan.models.qwen3 import model_registry
from torchtitan.distributed.expert_parallel import ExpertParallel
from torch.distributed.tensor.parallel import parallelize_module

num_tokens = 8

# Get MoE configs for both dispatcher types
spec_local = model_registry("30B-A3B")  # LocalTokenDispatcher
spec_a2a = model_registry("30B-A3B", moe_comm_backend="standard")  # AllToAllTokenDispatcher
moe_cfg_local = spec_local.model.layers[0].moe
moe_cfg_a2a = spec_a2a.model.layers[0].moe
dim = spec_local.model.dim

# Deterministic weights
torch.manual_seed(123)

# ===== Reference: MoE with LocalTokenDispatcher (EP=1) =====
if rank == 0:
    moe_ref = moe_cfg_local.build().to("cuda", dtype=torch.bfloat16)
    # Randomize weights
    for p in moe_ref.parameters():
        p.data.normal_(std=0.02)
    # Save weights for EP test
    torch.save(moe_ref.state_dict(), "/tmp/_moe_ref_sd.pt")
    moe_ref.eval()

    torch.manual_seed(42)
    x = torch.randn(1, num_tokens, dim, device="cuda", dtype=torch.bfloat16)
    torch.save(x.cpu(), "/tmp/_moe_x.pt")

    with torch.no_grad():
        out_ref = moe_ref(x)
    print(f"[REF] norm={out_ref.float().norm():.4f}, [0,0,:5]={out_ref[0,0,:5].tolist()}", flush=True)

dist.barrier()

# Load reference state dict and input on all ranks
ref_sd = torch.load("/tmp/_moe_ref_sd.pt", map_location="cuda", weights_only=True)
x = torch.load("/tmp/_moe_x.pt", weights_only=True).to("cuda")

# ===== EP=2: MoE with AllToAllTokenDispatcher =====
ep_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))

# Build on meta, apply EP, materialize
with torch.device("meta"):
    moe_ep = moe_cfg_a2a.build()

# Apply EP sharding
parallelize_module(moe_ep.experts, ep_mesh, ExpertParallel())

# Materialize
moe_ep.to_empty(device="cuda")
moe_ep = moe_ep.to(dtype=torch.bfloat16)

# Load reference weights
# Non-expert params (router.gate) — copy directly
param_dict = dict(moe_ep.named_parameters())
for k, v in ref_sd.items():
    if "experts" not in k and k in param_dict:
        param = param_dict[k]
        if isinstance(param, DTensor):
            param._local_tensor.copy_(v)
        else:
            param.data.copy_(v)

# Expert params — shard and copy
local_n = moe_cfg_a2a.num_experts // 2
s = rank * local_n
for k in ["experts.w1", "experts.w2", "experts.w3"]:
    param = dict(moe_ep.named_parameters())[k]
    assert isinstance(param, DTensor)
    param._local_tensor.copy_(ref_sd[k][s:s+local_n])

# Also copy buffers (tokens_per_expert, expert_bias)
for k, v in ref_sd.items():
    if k in dict(moe_ep.named_buffers()):
        buf = dict(moe_ep.named_buffers())[k]
        buf.copy_(v)

moe_ep.eval()

with torch.no_grad():
    out_ep = moe_ep(x)

print(f"[EP] rank={rank}, norm={out_ep.float().norm():.4f}, [0,0,:5]={out_ep[0,0,:5].tolist()}", flush=True)

dist.barrier()

if rank == 0:
    diff = (out_ref - out_ep).float().abs()
    print(f"\n[COMPARE] max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}", flush=True)
    if diff.max() < 0.01:
        print("[COMPARE] PASS", flush=True)
    else:
        print("[COMPARE] FAIL", flush=True)

dist.barrier()
