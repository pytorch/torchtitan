import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (2, 2))
ss = _StridedShard(0, split_factor=2)
source = distribute_tensor(torch.arange(8, device="cuda"), mesh, (ss, Shard(0)))
# result = source.redistribute(mesh, (ss, Replicate()))
result = source.redistribute(mesh, (ss, Partial("sum")))
dist.destroy_process_group()
