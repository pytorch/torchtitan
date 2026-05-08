#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Test script for flex_shard API.

Usage:
    torchrun --nproc_per_node=2 test_flex_shard.py
    torchrun --nproc_per_node=4 test_flex_shard.py

NOTE: Meta device init + numerical parity with DDP requires
ThreadBasedRNGTracker from https://github.com/pytorch/pytorch/pull/174446.
That codepath is disabled until the PR lands.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.nn.parallel import DistributedDataParallel as DDP

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flat_shard_placements,
    flex_shard,
    FlexShardModule,
    lift_params_to_global_spmd_mesh,
    per_param_placements,
)


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=20))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.destroy_process_group()


def print_rank0(msg):
    """Print only on rank 0."""
    if dist.get_rank() == 0:
        print(msg)


class MultiLayerMLP(nn.Module):
    """Multi-layer model for testing per-layer wrapping."""

    def __init__(self, dim: int = 16, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(dim, 8)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.output(x)


def _unwrap_checkpoint(module: nn.Module) -> nn.Module:
    if isinstance(module, CheckpointWrapper):
        return module._checkpoint_wrapped_module
    return module


def _init_flex_mesh(world_size):
    return init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))


def run_test_per_layer_wrapping(shard_placement_fn=None, name="per_param"):
    """Test per-layer wrapping with DDP parity over multiple training steps."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = _init_flex_mesh(world_size)

    print_rank0(f"\n{'=' * 70}")
    print_rank0(f"Testing {name} (world_size={world_size})")
    print_rank0(f"{'=' * 70}")

    # Create model
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = MultiLayerMLP().to(device)
    lift_params_to_global_spmd_mesh(model, mesh)

    # Create DDP reference
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ref_model = DDP(MultiLayerMLP().to(device), device_ids=[rank])

    # Per-layer wrapping: wrap each layer first, then root
    for layer in model.layers:
        flex_shard(
            layer,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=shard_placement_fn,
            buckets=[BucketSpec(["*"])],
        )
    flex_shard(
        model,
        mesh,
        DataParallelMeshDims(shard="fsdp"),
        shard_placement_fn=shard_placement_fn,
        buckets=[BucketSpec(["embed.*"]), BucketSpec(["output.*"])],
    )

    # Root-level reshard checkpointing wraps FlexShard-managed layers in
    # CheckpointWrapper; the FlexShardModule remains the wrapped inner module.
    for i, layer in enumerate(model.layers):
        inner_layer = _unwrap_checkpoint(layer)
        assert isinstance(inner_layer, FlexShardModule), f"layers.{i} not wrapped"
    assert isinstance(model, FlexShardModule), "root not wrapped"

    # Print storage info
    if rank == 0:
        root_ds = model.dstorage
        print(f"  Root DStorage: {len(root_ds.param_infos)} params")
        for fqn in root_ds.param_infos:
            print(f"    {fqn}")
        for i, layer in enumerate(model.layers):
            layer_ds = _unwrap_checkpoint(layer).dstorage
            print(f"  Layer {i} DStorage: {len(layer_ds.param_infos)} params")

    # Optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    # Multiple training steps
    for step in range(3):
        torch.manual_seed(42 + rank + step)
        inp = torch.randn(4, 16, device=device)

        optimizer.zero_grad()
        loss = model(inp).sum()
        loss.backward()
        optimizer.step()

        ref_optimizer.zero_grad()
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        ref_optimizer.step()

        torch.testing.assert_close(loss, ref_loss, atol=1e-6, rtol=1e-5)
        if rank == 0:
            print(
                f"  step {step}: flex_shard={loss.item():.6f}  ref={ref_loss.item():.6f}"
            )

    # Verify all params have gradients
    for fqn, param in model.named_parameters():
        if param.requires_grad and param.numel() > 0:
            assert param.grad is not None, f"{fqn} has no grad"

    print_rank0(f"PASSED: {name}")
    torch.cuda.synchronize()


def main():
    """Run tests."""
    rank, world_size = setup_distributed()
    print_rank0(f"Running tests with world_size={world_size}")

    success = True
    tests = [
        (per_param_placements, "per_param"),
        (flat_shard_placements, "flat_shard"),
    ]
    for placement_fn, name in tests:
        try:
            run_test_per_layer_wrapping(shard_placement_fn=placement_fn, name=name)
        except Exception as e:
            print_rank0(f"FAILED: {name}: {e}")
            import traceback

            if rank == 0:
                traceback.print_exc()
            success = False

    cleanup_distributed()
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
