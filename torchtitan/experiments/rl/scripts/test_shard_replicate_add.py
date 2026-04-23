"""Test Replicate→Shard(1) redistribute for various token counts."""
import torch, torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)
mesh = init_device_mesh('cuda', (2,), mesh_dim_names=('tp',))

D = 8

for num_tokens in [1, 2, 3, 4, 5, 7, 8]:
    torch.manual_seed(42 + num_tokens)
    t = torch.randn(1, num_tokens, D, device='cuda')

    # Replicate → Shard(1)
    dt = DTensor.from_local(t.clone(), mesh, (Replicate(),), run_check=False)
    dt_shard = dt.redistribute(placements=(Shard(1),))

    local = dt_shard._local_tensor
    # Expected: tensor_split divides into ceil(N/2) and floor(N/2)
    expected_splits = torch.tensor_split(t, 2, dim=1)
    expected = expected_splits[rank].contiguous()

    if local.numel() > 0 and expected.numel() > 0:
        diff = (local - expected).abs().max().item()
    elif local.numel() == 0 and expected.numel() == 0:
        diff = 0.0
    else:
        diff = -1.0

    print(
        f"rank={rank} tokens={num_tokens}: "
        f"local={local.shape} expected={expected.shape} "
        f"diff={diff:.6f} {'OK' if 0 <= diff < 1e-5 else 'BAD'}",
        flush=True,
    )

    dist.barrier()

dist.barrier()
