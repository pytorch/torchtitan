# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal FSDP2 + synthetic CP all-to-all example for PR #5645.

Run each case separately on two or more CUDA GPUs:
  torchrun --standalone --nproc-per-node=2 scripts/repro_fsdp_cp.py --profile
  torchrun --standalone --nproc-per-node=2 scripts/repro_fsdp_cp.py --profile --separate-group

A full-world CP mesh aliases WORLD. --separate-group gives FSDP a new group.
Native forward/backward prefetch remain enabled. No launch order is reversed.
This demonstrates group aliasing; a scheduling-dependent hang is not guaranteed.
"""

import argparse
import os
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard


class Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        output = torch.empty_like(value)
        dist.all_to_all_single(output, value.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx, grad):
        output = torch.empty_like(grad)
        dist.all_to_all_single(output, grad.contiguous(), group=ctx.group)
        return output, None


class Block(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.linear = nn.Linear(512, 512, bias=False)
        self.group = group

    def forward(self, value):
        return Exchange.apply(self.linear(value).relu(), self.group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separate-group", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world >= 2
    mesh = init_device_mesh("cuda", (world,))
    cp_group = mesh.get_group()
    assert cp_group is dist.group.WORLD
    if args.separate_group:
        mesh = DeviceMesh.from_group(
            dist.new_group(timeout=timedelta(seconds=30)), "cuda"
        )
    print(
        f"rank={rank} shared={mesh.get_group() is cp_group} world={world} "
        f"torch={torch.__version__} nccl={torch.cuda.nccl.version()} "
        f"gpu={torch.cuda.get_device_name()} "
        f"implicit={os.getenv('NCCL_LAUNCH_ORDER_IMPLICIT', 'unset')}",
        flush=True,
    )
    torch.manual_seed(21)
    model = nn.Sequential(Block(cp_group), Block(cp_group)).cuda()
    for block in model:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    model[0].set_modules_to_forward_prefetch([model[1]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    torch.manual_seed(100 + rank)
    value = torch.randn(8 * world, 512, device="cuda")
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if args.profile
        else None
    )
    with profiler if profiler is not None else nullcontext():
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model(value).square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            for parameter in model.parameters():
                assert torch.isfinite(parameter.grad.to_local()).all()
            optimizer.step()
            torch.cuda.synchronize()
            print(f"rank={rank} step={step} loss={loss.item():.9g}", flush=True)
    if profiler is not None:
        group = "separate" if args.separate_group else "shared"
        profiler.export_chrome_trace(f"fsdp_cp_{group}_rank{rank}.json")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
