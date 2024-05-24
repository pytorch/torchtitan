import torch
import torch.nn as nn

from torch.distributed._tensor import (
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)

from torchtitan.models.norms import create_norm, fused_rms_norm_fn

def get_device_type():
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )


world_size = 4
device_type = get_device_type()
device = torch.device(device_type)
mesh = init_device_mesh(device_type, (4,))
x = torch.randn(4, 4, 4, device=device)  # Shard(1)
w = torch.randn(4, device=device, requires_grad=True)  # Replicate

dx = distribute_tensor(x, mesh, [Shard(1)])
dw = distribute_tensor(w, mesh, [Replicate()])

# fused rmsnorm
out = fused_rms_norm_fn(dx, dw)
grad_out = torch.ones_like(out)
out.backward(grad_out)
print(grad_out)
