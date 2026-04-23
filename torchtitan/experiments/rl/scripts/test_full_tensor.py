"""Test full_tensor() with and without SequenceParallel norm."""
import torch, torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import RowwiseParallel, SequenceParallel, parallelize_module

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)
mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('tp',))

D = 16

# Test 1: bare Shard(1) full_tensor
print(f"\n--- Test 1: bare Shard(1) from RowwiseParallel ---", flush=True)
linear = torch.nn.Linear(D, D, bias=False, device='cuda')
parallelize_module(linear, mesh, RowwiseParallel(output_layouts=Shard(1), use_local_output=False))

x = DTensor.from_local(torch.randn(1, 1, D, device='cuda'), mesh, (Replicate(),), run_check=False)
with torch.no_grad():
    y = linear(x)

print(f"rank={rank}: y local={y._local_tensor.shape}", flush=True)
try:
    ft = y.full_tensor()
    print(f"rank={rank}: full_tensor={ft.shape}", flush=True)
except Exception as e:
    print(f"rank={rank}: full_tensor ERROR: {e}", flush=True)

dist.barrier()

# Test 2: Shard(1) → SequenceParallel norm → full_tensor
print(f"\n--- Test 2: Shard(1) → SequenceParallel norm ---", flush=True)
norm = torch.nn.RMSNorm(D, device='cuda')
parallelize_module(norm, mesh, SequenceParallel(use_local_output=False))

with torch.no_grad():
    y_normed = norm(y)

print(f"rank={rank}: y_normed local={y_normed._local_tensor.shape} placement={y_normed.placements}", flush=True)
try:
    ft2 = y_normed.full_tensor()
    print(f"rank={rank}: full_tensor after norm={ft2.shape}", flush=True)
except Exception as e:
    print(f"rank={rank}: full_tensor after norm ERROR: {e}", flush=True)

dist.barrier()
