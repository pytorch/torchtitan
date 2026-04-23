"""Test Replicate→Shard(1) with 1 token (odd split)."""
import torch, torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)
mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('tp',))

# 1 token, same on both ranks
t = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device='cuda', dtype=torch.bfloat16)

# Wrap as Replicate
dt_rep = DTensor.from_local(t, mesh, (Replicate(),), run_check=False)
print(f"rank={rank}, Replicate shape={dt_rep.shape}, local={dt_rep._local_tensor.shape}", flush=True)

# Redistribute to Shard(1)
dt_shard = dt_rep.redistribute(placements=(Shard(1),))
print(f"rank={rank}, Shard(1) shape={dt_shard.shape}, local={dt_shard._local_tensor.shape}, local_data={dt_shard._local_tensor.tolist()}", flush=True)

# Now simulate residual: x_shard + dt_shard
# x was already Shard(1) with rank 0 having the 1 token
if rank == 0:
    x_local = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]], device='cuda', dtype=torch.bfloat16)
else:
    x_local = torch.empty(1, 0, 4, device='cuda', dtype=torch.bfloat16)

x_shard = DTensor.from_local(x_local, mesh, (Shard(1),))
print(f"rank={rank}, x_shard local={x_shard._local_tensor.shape}", flush=True)

result = x_shard + dt_shard
print(f"rank={rank}, result local={result._local_tensor.shape}, data={result._local_tensor.tolist()}", flush=True)

# full_tensor
result_full = result.full_tensor()
print(f"rank={rank}, full={result_full.shape}, data={result_full.tolist()}", flush=True)

dist.barrier()
