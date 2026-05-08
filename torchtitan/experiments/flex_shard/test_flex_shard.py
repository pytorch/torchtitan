#!/usr/bin/env python3
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

import torch
import torch.distributed as dist
import torch.nn as nn
from datetime import timedelta
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.experiments.flex_shard import (
    flex_shard,
    FlexShardModule,
    is_flex_shard_param,
    Owned,
    RaggedShard,
    Shard,
)
from torch.nn.parallel import DistributedDataParallel as DDP


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


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, 8)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()


def run_test_placement(name, placement_fn=None):
    """Test flex_shard with a given placement, verifying DDP parity.

    Args:
        name: test label for logging
        placement_fn: optional shard_placement_fn; None means default Shard(0)
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    print_rank0(f"\n{'=' * 70}")
    print_rank0(f"Testing {name} (world_size={world_size})")
    print_rank0(f"{'=' * 70}")

    # Create model
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = SimpleMLP().to(device)

    # Create reference model (DDP) with same seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ref_model = SimpleMLP().to(device)
    ref_model = DDP(ref_model, device_ids=[rank])

    # Apply flex_shard
    if placement_fn is not None:
        flex_shard(model, mesh, shard_placement_fn=placement_fn)
    else:
        flex_shard(model, mesh)
    assert isinstance(model, FlexShardModule)
    storage = model.dstorage

    # Print placement config
    for fqn, info in storage.param_infos.items():
        assert is_flex_shard_param(
            dict(model.named_parameters())[fqn]
        ), f"{fqn} missing metadata"
        if rank == 0:
            print(
                f"  {fqn:<20} | {repr(info.placements[0]):<45} "
                f"| local_numel={info.local_numel}"
            )

    # Forward parity
    batch_size = 4
    torch.manual_seed(42 + rank)
    inp = torch.randn(batch_size, 16, device=device)

    with torch.no_grad():
        ref_loss = ref_model(inp).sum()
        fsdp_loss = model(inp).sum()
    torch.testing.assert_close(ref_loss, fsdp_loss)
    print_rank0(f"  Forward:  ref={ref_loss.item():.6f}  fsdp={fsdp_loss.item():.6f}")

    # Backward — verify all params get gradients
    torch.manual_seed(42 + rank)
    inp = torch.randn(batch_size, 16, device=device)

    ref_model(inp).sum().backward()
    model(inp).sum().backward()

    for fqn, param in model.named_parameters():
        if param.requires_grad and param.numel() > 0:
            assert param.grad is not None, f"{fqn} has no grad"
    print_rank0("  Backward: all params have gradients")

    print_rank0(f"PASSED: {name}")
    torch.cuda.synchronize()


def main():
    """Run tests."""
    rank, world_size = setup_distributed()
    print_rank0(f"Running tests with world_size={world_size}")

    equal_units = tuple(1 for _ in range(world_size))

    tests = [
        ("Shard(0)", None),
        (
            "RaggedShard (equal units)",
            lambda fqn, p: (
                RaggedShard(dims=(0,), local_units=equal_units)
                if p.dim() >= 2
                else Shard(0)
            ),
        ),
        (
            "Mixed Shard + Owned",
            lambda fqn, p: Owned(0) if "fc1" in fqn else Shard(0),
        ),
        (
            "Mixed Shard + RaggedShard + Owned",
            lambda fqn, p: (
                Owned(0)
                if "fc3" in fqn
                else (
                    RaggedShard(dims=(0,), local_units=equal_units)
                    if p.dim() >= 2
                    else Shard(0)
                )
            ),
        ),
    ]

    success = True
    for name, placement_fn in tests:
        try:
            run_test_placement(name, placement_fn)
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
