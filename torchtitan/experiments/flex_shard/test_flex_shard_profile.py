#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run flex_shard per-layer wrapping test with profiler traces."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.profiler as torch_profiler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    lift_params_to_global_spmd_mesh,
    per_param_placements,
)

TRACE_DIR = "/data/users/weif/code-review/torchtitan/profiler_traces"


class MultiLayerMLP(nn.Module):
    def __init__(self, dim=4096, hidden_dim=8192, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(dim, 2048)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x)


def main():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=20))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))

    # Build model with per-layer wrapping
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = MultiLayerMLP().to(device)
    lift_params_to_global_spmd_mesh(model, mesh)

    # Wrap each layer first, then root
    for layer in model.layers:
        flex_shard(
            layer,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["0.*"]), BucketSpec(["2.*"])],
        )
    flex_shard(
        model,
        mesh,
        DataParallelMeshDims(shard="fsdp"),
        shard_placement_fn=per_param_placements,
        buckets=[BucketSpec(["embed.*"]), BucketSpec(["output.*"])],
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    os.makedirs(TRACE_DIR, exist_ok=True)

    prof = torch_profiler.profile(
        activities=[
            torch_profiler.ProfilerActivity.CPU,
            torch_profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch_profiler.schedule(
            wait=1,
            warmup=2,
            active=1,
            repeat=1,
            skip_first=1,
        ),
        on_trace_ready=torch_profiler.tensorboard_trace_handler(TRACE_DIR),
        record_shapes=True,
        with_stack=True,
    )

    prof.start()
    for iter_idx in range(6):
        torch.cuda.synchronize()
        dist.barrier()

        torch.manual_seed(42 + rank + iter_idx)
        inp = torch.randn(512, 4096, device=device)

        optimizer.zero_grad()
        loss = model(inp).sum()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"  iter {iter_idx}: loss={loss.item():.6f}")

        torch.cuda.synchronize()
        dist.barrier()
        prof.step()

    prof.stop()

    if rank == 0:
        print("PASSED: Per-layer wrapping with profiler traces")
        print(f"Traces saved to: {TRACE_DIR}")

    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
