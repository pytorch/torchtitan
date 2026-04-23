"""Test: Shard(1) from RowwiseParallel + Replicate from_local with 1 token."""
import torch, torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)
mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('tp',))

D = 8

# Create a RowwiseParallel linear to produce Shard(1) output
linear = torch.nn.Linear(D, D, bias=False, device='cuda')
parallelize_module(linear, mesh, RowwiseParallel(output_layouts=Shard(1), use_local_output=False))

# 1 token input
torch.manual_seed(42)
x_rep = DTensor.from_local(
    torch.randn(1, 1, D, device='cuda'), mesh, (Replicate(),), run_check=False
)

with torch.no_grad():
    x_shard = linear(x_rep)  # Shard(1) from RowwiseParallel

print(
    f"rank={rank}: x_shard local={x_shard._local_tensor.shape}, "
    f"global={x_shard.shape}, placements={x_shard.placements}",
    flush=True,
)

# Create Replicate moe_out (same as what MoE returns)
moe_data = torch.randn(1, 1, D, device='cuda')
moe_rep = DTensor.from_local(moe_data.clone(), mesh, (Replicate(),), run_check=False)

# The critical addition
result = x_shard + moe_rep

print(
    f"rank={rank}: result local={result._local_tensor.shape}, "
    f"global={result.shape}, placements={result.placements}",
    flush=True,
)

# Check: result should equal x_shard_local + moe_data (the replicated part)
ref = x_shard._local_tensor + moe_data
diff = (result._local_tensor - ref).abs().max().item()
print(f"rank={rank}: local diff={diff:.6f} {'OK' if diff < 1e-5 else 'BAD'}", flush=True)

dist.barrier()
