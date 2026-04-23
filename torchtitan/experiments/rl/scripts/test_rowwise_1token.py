"""Test RowwiseParallel with 1 token — what local shapes are produced?"""
import torch, torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)
mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('tp',))

for num_tokens in [1, 2, 3, 4, 5]:
    linear = torch.nn.Linear(8, 16, bias=False, device='cuda')
    parallelize_module(linear, mesh, RowwiseParallel(
        output_layouts=Shard(1), use_local_output=False
    ))

    x = DTensor.from_local(
        torch.randn(1, num_tokens, 8, device='cuda'),
        mesh, (Replicate(),), run_check=False,
    )

    with torch.no_grad():
        y = linear(x)

    print(
        f"rank={rank} tokens={num_tokens}: "
        f"y_local={y._local_tensor.shape} y_global={y.shape} "
        f"y_placement={y.placements}",
        flush=True,
    )
    dist.barrier()

dist.barrier()
