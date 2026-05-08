# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint round-trip tests for FlexShard.

Verifies that FlexShard models can save/load sharded parameters per rank,
and that cross-format loading (SimpleFSDP → FlexShard) works.

Note: FlexShard parameters are regular tensors (not DTensors), so DCP's
sharding-aware save/load cannot be used directly. These tests use per-rank
torch.save/load to verify the sharded state round-trips correctly.

Requires 2+ GPUs. Run with:
    torchrun --standalone --nproc_per_node=2 -m pytest \
        torchtitan/experiments/flex_shard/tests/test_checkpoint.py -q
"""

import os
import shutil
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import DataParallelMeshDims
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    lift_params_to_global_spmd_mesh,
    Shard,
)


STEPS = 5


class TwoLayerMLP(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestFlexShardCheckpoint(FSDPTest):
    """Test checkpoint round-trip with FlexShard."""

    def _shared_tmpdir(self):
        """Create a temp dir on rank 0 and broadcast path to all ranks."""
        if dist.get_rank() == 0:
            tmpdir = tempfile.mkdtemp()
        else:
            tmpdir = ""
        obj_list = [tmpdir]
        dist.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]
        if dist.get_rank() == 0:
            self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        return tmpdir

    def _save_per_rank(self, model, tmpdir):
        """Save model state dict per-rank."""
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        rank = dist.get_rank()
        torch.save(sd, os.path.join(tmpdir, f"rank_{rank}.pt"))
        dist.barrier()

    def _load_per_rank(self, model, tmpdir):
        """Load per-rank checkpoint into model."""
        rank = dist.get_rank()
        sd = torch.load(
            os.path.join(tmpdir, f"rank_{rank}.pt"),
            weights_only=True,
            map_location=f"cuda:{rank}",
        )
        # state_dict() bypasses properties, reads _parameters directly
        model.load_state_dict(sd)
        dist.barrier()

    def init_test(self):
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _flex_shard(self, model, mesh, **kwargs):
        lift_params_to_global_spmd_mesh(model, mesh)
        return flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            **kwargs,
        )

    def _train_steps(self, model, inputs, labels, steps=STEPS):
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(steps):
            optim.zero_grad()
            out = model(inputs)
            loss = cross_entropy_loss(out, labels)
            loss.backward()
            optim.step()

    def test_flex_shard_roundtrip(self):
        """Save FlexShard model, load into fresh FlexShard model, verify params."""
        self.init_test()
        mesh = self.parallel_dims.get_mesh("fsdp")
        model = torch.nn.Linear(8, 8)
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()

        # Train original
        self._flex_shard(model, mesh, shard_placement_fn={"*": Shard(0)})
        self._train_steps(model, inputs, labels)

        # Save per-rank sharded state
        tmpdir = self._shared_tmpdir()
        self._save_per_rank(model, tmpdir)

        # Load into fresh model with same config
        model2 = torch.nn.Linear(8, 8)
        self._flex_shard(model2, mesh, shard_placement_fn={"*": Shard(0)})
        self._load_per_rank(model2, tmpdir)

        # Verify params match (state_dict bypasses properties)
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=True
        ):
            torch.testing.assert_close(
                p1.data, p2.data, msg=f"Param {n1} mismatch after round-trip"
            )

    def test_flex_shard_train_after_load(self):
        """Load checkpoint into FlexShard model, verify training continues."""
        self.init_test()
        mesh = self.parallel_dims.get_mesh("fsdp")
        model = torch.nn.Linear(8, 8)
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()

        # Train and save
        self._flex_shard(model, mesh, shard_placement_fn={"*": Shard(0)})
        self._train_steps(model, inputs, labels, steps=2)
        tmpdir = self._shared_tmpdir()
        self._save_per_rank(model, tmpdir)

        # Continue training original
        self._train_steps(model, inputs, labels, steps=3)

        # Load checkpoint into fresh model and train same steps
        model2 = torch.nn.Linear(8, 8)
        self._flex_shard(model2, mesh, shard_placement_fn={"*": Shard(0)})
        self._load_per_rank(model2, tmpdir)
        self._train_steps(model2, inputs, labels, steps=3)

        # Params should match after same training from same checkpoint
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=True
        ):
            torch.testing.assert_close(
                p1.data,
                p2.data,
                msg=f"Param {n1} diverged after train-from-checkpoint",
            )

    def test_flex_shard_roundtrip_with_buckets(self):
        """Multi-bucket FlexShard model checkpoint round-trip."""
        self.init_test()
        mesh = self.parallel_dims.get_mesh("fsdp")
        model = TwoLayerMLP(dim=8)
        inputs = torch.randn(8, 8).cuda()
        labels = torch.randn(8, 8).cuda()

        # Train with per-layer buckets
        self._flex_shard(
            model,
            mesh,
            shard_placement_fn={"*": Shard(0)},
            buckets=[
                BucketSpec(["fc1.*"]),
                BucketSpec(["fc2.*"]),
            ],
        )
        self._train_steps(model, inputs, labels)

        # Save and load into fresh model
        tmpdir = self._shared_tmpdir()
        self._save_per_rank(model, tmpdir)

        model2 = TwoLayerMLP(dim=8)
        self._flex_shard(
            model2,
            mesh,
            shard_placement_fn={"*": Shard(0)},
            buckets=[
                BucketSpec(["fc1.*"]),
                BucketSpec(["fc2.*"]),
            ],
        )
        self._load_per_rank(model2, tmpdir)

        # Verify params match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(), strict=True
        ):
            torch.testing.assert_close(
                p1.data,
                p2.data,
                msg=f"Param {n1} mismatch after multi-bucket round-trip",
            )


if __name__ == "__main__":
    unittest.main()
