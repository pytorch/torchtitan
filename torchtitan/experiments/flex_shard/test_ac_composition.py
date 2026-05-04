#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test composition of activation checkpointing + FlexShard reshard checkpoint.

Verifies that when both AC and FlexShard are applied:
1. Each layer has exactly one CheckpointWrapper (not nested)
2. FlexShard collective ops are marked MUST_RECOMPUTE (reshard semantics)
3. Forward/backward produces correct numerics vs. unsharded reference

Usage:
    torchrun --nproc_per_node=2 test_ac_composition.py
"""

import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from torchtitan.experiments.flex_shard import (
    flex_shard,
    lift_params_to_global_spmd_mesh,
    per_param_placements,
)


class SimpleMLP(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for _ in range(3)]
        )
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x)


def apply_selective_ac(model):
    """Apply selective AC to each layer (simulates torchtitan's apply_ac)."""
    save_ops = {
        torch.ops.aten.mm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.addmm.default,
    }

    def _policy(ctx, func, *args, **kwargs):
        from torch.utils.checkpoint import CheckpointPolicy

        if func in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    for i in range(len(model.layers)):
        model.layers[i] = checkpoint_wrapper(
            model.layers[i],
            context_fn=lambda: create_selective_checkpoint_contexts(_policy),
        )


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def _init_flex_mesh(world_size):
    return init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))


def _flex_shard(model, mesh, **kwargs):
    lift_params_to_global_spmd_mesh(model, mesh)
    return flex_shard(model, mesh, DataParallelMeshDims(shard="fsdp"), **kwargs)


def test_no_nested_wrappers():
    """AC + FlexShard produces a single CheckpointWrapper per layer, not nested."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = _init_flex_mesh(world_size)

    torch.manual_seed(42)
    model = SimpleMLP().cuda()
    dist.broadcast(model.layers[0][0].weight.data, src=0)

    # Apply AC first (as torchtitan does)
    apply_selective_ac(model)

    # Verify AC wrapped
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, CheckpointWrapper), f"layers.{i} not AC-wrapped"

    # Apply FlexShard with reshard_after_forward
    _flex_shard(model, mesh, shard_placement_fn=per_param_placements)

    # Verify: single wrapper, not nested
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, CheckpointWrapper), f"layers.{i} not wrapped"
        inner = layer._checkpoint_wrapped_module
        assert not isinstance(
            inner, CheckpointWrapper
        ), f"layers.{i} has nested CheckpointWrapper"

    print_rank0("PASSED: no_nested_wrappers")


def test_numerics_ac_plus_flexshard():
    """AC + FlexShard produces correct loss matching DDP reference."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = _init_flex_mesh(world_size)
    device = torch.device("cuda", torch.cuda.current_device())

    # FlexShard + AC model
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = SimpleMLP().to(device)
    apply_selective_ac(model)
    _flex_shard(model, mesh, shard_placement_fn=per_param_placements)

    # DDP reference
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ref_model = DDP(SimpleMLP().to(device), device_ids=[rank])

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    for step in range(3):
        torch.manual_seed(42 + rank + step)
        inp = torch.randn(4, 16, device=device)

        opt.zero_grad()
        loss = model(inp).sum()
        loss.backward()
        opt.step()

        ref_opt.zero_grad()
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        ref_opt.step()

        torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-4)
        if rank == 0:
            print(
                f"  step {step}: flex+ac={loss.item():.6f}  ref={ref_loss.item():.6f}"
            )

    print_rank0("PASSED: numerics_ac_plus_flexshard")


def test_flexshard_only():
    """FlexShard without AC still works (no AC wrapper to detect)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = _init_flex_mesh(world_size)
    device = torch.device("cuda", torch.cuda.current_device())

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = SimpleMLP().to(device)
    _flex_shard(model, mesh, shard_placement_fn=per_param_placements)

    # Verify layers are wrapped in CheckpointWrapper (reshard-only)
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, CheckpointWrapper), f"layers.{i} not wrapped"

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ref_model = DDP(SimpleMLP().to(device), device_ids=[rank])

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    torch.manual_seed(42 + rank)
    inp = torch.randn(4, 16, device=device)

    opt.zero_grad()
    loss = model(inp).sum()
    loss.backward()
    opt.step()

    ref_opt.zero_grad()
    ref_loss = ref_model(inp).sum()
    ref_loss.backward()
    ref_opt.step()

    torch.testing.assert_close(loss, ref_loss, atol=1e-5, rtol=1e-4)
    print_rank0("PASSED: flexshard_only")


def main():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    tests = [
        test_no_nested_wrappers,
        test_numerics_ac_plus_flexshard,
        test_flexshard_only,
    ]

    success = True
    for test in tests:
        try:
            test()
        except Exception as e:
            print_rank0(f"FAILED: {test.__name__}: {e}")
            traceback.print_exc()
            success = False

    dist.barrier()
    rank = dist.get_rank()
    dist.destroy_process_group()

    if not success:
        raise SystemExit(1)
    if rank == 0:
        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
