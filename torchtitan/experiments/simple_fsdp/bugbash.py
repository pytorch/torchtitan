import torch
import torch.nn as nn
import time
import os
import torch.distributed as dist
from simple_fsdp import data_parallel


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.matmul(x, self.net.weight.T) + self.net.bias + torch.matmul(x, self.net.weight.T)

input_dim, output_dim = 128, 10


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)

    dp_mesh = dist.device_mesh.init_device_mesh(
        device.type,
        mesh_shape=(2,),
        mesh_dim_names=("dp",),
    )

    module = SimpleMLP(input_dim, output_dim).to(device)
    module = data_parallel(module, dp_mesh, mode="fully_shard")
    module = torch.compile(module, mode="reduce-overhead")

    x = torch.randn(32, input_dim, device=device)
    output = module(x)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()

# USE_EXPANDABLE_SEGMENTS=False NGPU=8 MODULE=compiler_toolkit.llama3 CONFIG=compiler_toolkit_llama3_8b ./run_train.sh --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=4 --compile.passes autobucketing_reordering --debug.seed=0 --debug.deterministic
