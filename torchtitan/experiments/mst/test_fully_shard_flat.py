#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Test script for fully_shard_flat API.

Usage:
    torchrun --nproc_per_node=2 test_fully_shard_flat.py
    torchrun --nproc_per_node=4 test_fully_shard_flat.py

NOTE: Meta device init + numerical parity with DDP requires
ThreadBasedRNGTracker from https://github.com/pytorch/pytorch/pull/174446.
That codepath is disabled until the PR lands.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from datetime import timedelta
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.experiments.mst import fully_shard_flat
from torch.distributed.tensor import DTensor, Shard
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


def run_test():
    """Test fully_shard_flat with eager device initialization."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    print_rank0(f"\n{'='*70}")
    print_rank0(f"Testing fully_shard_flat (world_size={world_size})")
    print_rank0(f"{'='*70}")

    # Create model on GPU directly (no meta init)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = SimpleMLP().to(device)

    # Create reference model (DDP) with same seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ref_model = SimpleMLP().to(device)
    ref_model = DDP(ref_model, device_ids=[rank])

    # Apply fully_shard_flat
    storage = fully_shard_flat(model, mesh)
    print_rank0(f"  fully_shard_flat: applied (total_bytes={storage.total_bytes})")

    # Verify params are DTensors
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor"
    print_rank0("  Params are DTensors")

    print_rank0(f"\n=== Shard Placement Configuration ===")
    for fqn, info in storage.param_infos.items():
        placement = info.placements[0]
        placement_str = repr(placement)
        if rank == 0:
            print(f"  {fqn:<20} | {placement_str:<12} | local_numel={info.local_numel}")

    # Verify forward pass produces identical loss
    print_rank0(f"\n=== Forward pass comparison ===")
    batch_size = 4

    torch.manual_seed(42 + rank)
    inp = torch.randn(batch_size, 16, device=device)

    with torch.no_grad():
        ref_loss = ref_model(inp).sum()
        fsdp_loss = model(inp).sum()

    torch.testing.assert_close(ref_loss, fsdp_loss)
    print_rank0(f"  Forward pass: ref_loss={ref_loss.item():.6f}, fsdp_loss={fsdp_loss.item():.6f}")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED: fully_shard_flat matches DDP reference")
    print_rank0(f"{'='*70}")

    torch.cuda.synchronize()
    return True


def main():
    """Run tests."""
    rank, world_size = setup_distributed()

    print_rank0(f"Running tests with world_size={world_size}")

    try:
        run_test()
        success = True
    except Exception as e:
        print_rank0(f"FAILED: {e}")
        import traceback
        if rank == 0:
            traceback.print_exc()
        success = False

    cleanup_distributed()
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
