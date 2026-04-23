import torch, torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)
mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('tp',))

torch.manual_seed(42)
t = torch.randn(1, 8, 2048, device='cuda', dtype=torch.bfloat16)

# Step 1: Shard(1) - each rank has 4 tokens
x_shard = DTensor.from_local(t[:, rank*4:(rank+1)*4, :], mesh, (Shard(1),))

# Step 2: all-gather to Replicate
x_rep = x_shard.redistribute(placements=(Replicate(),))

# Step 3: extract local (full) tensor, simulate MoE
x_local = x_rep.to_local()
moe_out = x_local * 0.1

# Step 4: wrap as Replicate, shard back to Shard(1)
out_rep = DTensor.from_local(moe_out, mesh, (Replicate(),), run_check=False)
out_shard = out_rep.redistribute(placements=(Shard(1),))

# Step 5: residual
result = x_shard + out_shard

# Step 6: full_tensor
result_full = result.full_tensor()

# Reference
ref = t + t * 0.1
diff = (result_full - ref).abs().max().item()

if rank == 0:
    print(f'diff: {diff}', flush=True)
    print('PASS' if diff == 0.0 else f'FAIL (diff={diff})', flush=True)

dist.barrier()
