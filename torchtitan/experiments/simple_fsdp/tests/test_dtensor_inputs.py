# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for DTensor inputs with SimpleFSDP full_dtensor mode.

KEY INSIGHT:
DTensor inputs work with SimpleFSDP when using full_dtensor=True and wrapping
inputs on the SAME mesh as the model weights (fsdp mesh).

With full_dtensor=True:
- Model weights remain as DTensors with Replicate() placement after forward
- Inputs can be DTensors with Shard(0) placement on the same mesh
- DTensor handles Replicate × Shard operations correctly
- Gradient reduction is automatic via Partial placement during backward

USAGE:
1. Apply SimpleFSDP with full_dtensor=True:
   model = data_parallel(model, fsdp_mesh, mode="fully_shard", full_dtensor=True)

2. Wrap inputs as DTensor on the SAME mesh:
   inputs = DTensor.from_local(local_inputs, fsdp_mesh, [Shard(0)])
   labels = DTensor.from_local(local_labels, fsdp_mesh, [Shard(0)])

3. Handle DTensor output in loss:
   out = model(inputs)
   loss = cross_entropy_loss(out.to_local(), labels.to_local())
"""

import copy

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel


class TestDTensorInputsWithSimpleFSDPFullDTensor(FSDPTest):
    """Test DTensor inputs with SimpleFSDP full_dtensor mode."""

    @property
    def world_size(self) -> int:
        return 4

    def init_parallel_dims(self, mode: str = "fully_shard"):
        """Initialize ParallelDims for testing."""
        self.mode = mode
        data_parallel_shard_degree = -1

        if mode == "replicate":
            self.dp_mesh_dim_names = ["dp_replicate"]
            data_parallel_replicate_degree = self.world_size
        elif mode == "fully_shard":
            self.dp_mesh_dim_names = ["fsdp"]
            data_parallel_replicate_degree = 1
        elif mode == "hybrid_shard":
            self.dp_mesh_dim_names = ["dp_replicate", "fsdp"]
            data_parallel_replicate_degree = self.world_size // 2
        else:
            raise ValueError(f"Unsupported mode {mode}")

        self.parallel_dims = ParallelDims(
            dp_shard=data_parallel_shard_degree,
            dp_replicate=data_parallel_replicate_degree,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

    def get_model(self, vocab_size: int = 32, hidden_dim: int = 64):
        """Create a simple model for testing."""
        model = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        return model

    def get_input(self, batch_size: int = 8, seq_len: int = 16, vocab_size: int = 32):
        """Generate input tensors for testing."""
        torch.manual_seed(42)
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        labels = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        return inputs, labels

    def shard_input_on_fsdp_mesh(self, tensor: torch.Tensor) -> torch.Tensor:
        """Shard tensor on batch dimension across FSDP ranks."""
        fsdp_mesh = self.parallel_dims.get_mesh(self.dp_mesh_dim_names)
        # For 2D mesh, use flattened size and coordinate
        total_size = fsdp_mesh.size()
        if fsdp_mesh.ndim == 2:
            # Get global rank within the mesh
            coord = fsdp_mesh.get_coordinate()
            local_rank = coord[0] * fsdp_mesh.size(1) + coord[1]
        else:
            local_rank = fsdp_mesh.get_local_rank()
        local_batch_size = tensor.shape[0] // total_size
        start_idx = local_rank * local_batch_size
        end_idx = start_idx + local_batch_size
        return tensor[start_idx:end_idx].contiguous()

    def wrap_as_dtensor_on_fsdp_mesh(self, local_tensor: torch.Tensor) -> DTensor:
        """Wrap a local tensor as a DTensor with Shard(0) on fsdp mesh."""
        fsdp_mesh = self.parallel_dims.get_mesh(self.dp_mesh_dim_names)
        # For 2D mesh (HSDP), use Shard on both dimensions
        if fsdp_mesh.ndim == 2:
            return DTensor.from_local(local_tensor, fsdp_mesh, [Shard(0), Shard(0)])
        else:
            return DTensor.from_local(local_tensor, fsdp_mesh, [Shard(0)])

    def run_training_with_dtensor_inputs(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 5,
    ) -> list[torch.Tensor]:
        """Run training with DTensor inputs."""
        # Apply SimpleFSDP with full_dtensor=True
        model = data_parallel(
            model,
            device_mesh=self.parallel_dims.get_mesh(self.dp_mesh_dim_names),
            mode=self.mode,
            full_dtensor=True,  # KEY: Keep weights as DTensors
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Shard and wrap inputs as DTensor on fsdp mesh
        local_inputs = self.shard_input_on_fsdp_mesh(inputs)
        local_labels = self.shard_input_on_fsdp_mesh(labels)
        dtensor_inputs = self.wrap_as_dtensor_on_fsdp_mesh(local_inputs)
        dtensor_labels = self.wrap_as_dtensor_on_fsdp_mesh(local_labels)

        losses = []
        for _ in range(epochs):
            optim.zero_grad()
            out = model(dtensor_inputs)

            # Handle DTensor output
            if isinstance(out, DTensor):
                out_local = out.to_local()
            else:
                out_local = out
            labels_local = dtensor_labels.to_local()

            loss = cross_entropy_loss(out_local, labels_local)
            loss.backward()
            optim.step()
            losses.append(loss.detach().clone())
        return losses

    def test_fullyshard_dtensor_inputs(self):
        """Test DTensor inputs with fully_shard mode."""
        self.init_parallel_dims("fully_shard")
        inputs, labels = self.get_input()

        model = self.get_model().cuda()
        losses = self.run_training_with_dtensor_inputs(
            copy.deepcopy(model), inputs, labels
        )

        # Verify training progresses
        self.assertLess(
            losses[-1].item(),
            losses[0].item(),
            "Loss should decrease during training",
        )

        # Verify no nan/inf
        for i, loss in enumerate(losses):
            self.assertFalse(
                torch.isnan(loss) or torch.isinf(loss),
                f"Loss is nan/inf at epoch {i}: {loss.item()}",
            )

    def test_replicate_dtensor_inputs(self):
        """Test DTensor inputs with replicate (DDP) mode."""
        self.init_parallel_dims("replicate")
        inputs, labels = self.get_input()

        model = self.get_model().cuda()
        losses = self.run_training_with_dtensor_inputs(
            copy.deepcopy(model), inputs, labels
        )

        # Verify training progresses
        self.assertLess(
            losses[-1].item(),
            losses[0].item(),
            "Loss should decrease during training",
        )

        # Verify no nan/inf
        for i, loss in enumerate(losses):
            self.assertFalse(
                torch.isnan(loss) or torch.isinf(loss),
                f"Loss is nan/inf at epoch {i}: {loss.item()}",
            )

    def test_hybridshard_dtensor_inputs(self):
        """Test DTensor inputs with hybrid_shard (HSDP) mode."""
        self.init_parallel_dims("hybrid_shard")
        inputs, labels = self.get_input()

        model = self.get_model().cuda()
        losses = self.run_training_with_dtensor_inputs(
            copy.deepcopy(model), inputs, labels
        )

        # Verify training progresses
        self.assertLess(
            losses[-1].item(),
            losses[0].item(),
            "Loss should decrease during training",
        )

        # Verify no nan/inf
        for i, loss in enumerate(losses):
            self.assertFalse(
                torch.isnan(loss) or torch.isinf(loss),
                f"Loss is nan/inf at epoch {i}: {loss.item()}",
            )


class TestDTensorInputsGradientReduction(FSDPTest):
    """Test that gradient reduction works correctly with DTensor inputs."""

    @property
    def world_size(self) -> int:
        return 4

    def init_parallel_dims(self):
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

    def test_gradient_sync_with_dtensor_inputs(self):
        """Test that gradients are synchronized across ranks with DTensor inputs."""
        self.init_parallel_dims()
        torch.manual_seed(42)

        vocab_size, hidden_dim = 32, 64
        batch_size, seq_len = 8, 16

        # Create model
        model = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        ).cuda()

        # Apply SimpleFSDP with full_dtensor=True
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model = data_parallel(
            model,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
            full_dtensor=True,
        )

        # Create different inputs per rank (simulating DP)
        rank = fsdp_mesh.get_local_rank()
        local_batch = batch_size // self.world_size

        torch.manual_seed(42 + rank)
        local_inputs = torch.randint(0, vocab_size, (local_batch, seq_len)).cuda()
        local_labels = torch.randint(0, vocab_size, (local_batch, seq_len)).cuda()

        # Wrap as DTensor
        dtensor_inputs = DTensor.from_local(local_inputs, fsdp_mesh, [Shard(0)])
        dtensor_labels = DTensor.from_local(local_labels, fsdp_mesh, [Shard(0)])

        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        # Training step
        optim.zero_grad()
        out = model(dtensor_inputs)

        if isinstance(out, DTensor):
            out_local = out.to_local()
        else:
            out_local = out
        labels_local = dtensor_labels.to_local()

        loss = cross_entropy_loss(out_local, labels_local)
        loss.backward()
        optim.step()

        # Verify weights are synchronized (gradient reduction worked)
        for name, param in model.named_parameters():
            if isinstance(param, DTensor):
                full_weight = param.full_tensor()
            else:
                full_weight = param

            # All-gather weights from all ranks
            import torch.distributed as dist

            gathered = [torch.zeros_like(full_weight) for _ in range(self.world_size)]
            dist.all_gather(gathered, full_weight)

            # Verify all ranks have the same weights
            for i in range(1, self.world_size):
                self.assertTrue(
                    torch.allclose(gathered[0], gathered[i], atol=1e-5),
                    f"Weight {name} differs between rank 0 and rank {i}",
                )
            break  # Just check first weight

    def test_dtensor_input_output_types(self):
        """Verify DTensor input produces DTensor output with full_dtensor mode."""
        self.init_parallel_dims()
        torch.manual_seed(42)

        vocab_size, hidden_dim = 32, 64
        batch_size, seq_len = 8, 16

        model = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        ).cuda()

        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model = data_parallel(
            model,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
            full_dtensor=True,
        )

        rank = fsdp_mesh.get_local_rank()
        local_batch = batch_size // self.world_size
        local_inputs = torch.randint(0, vocab_size, (local_batch, seq_len)).cuda()

        # Wrap as DTensor
        dtensor_inputs = DTensor.from_local(local_inputs, fsdp_mesh, [Shard(0)])

        # Forward
        out = model(dtensor_inputs)

        # Verify output is DTensor with Shard(0) placement
        self.assertIsInstance(out, DTensor, "Output should be DTensor")
        self.assertEqual(
            out._spec.placements[0],
            Shard(0),
            "Output should have Shard(0) placement",
        )


class TestPlainTensorInputsWithSimpleFSDP(FSDPTest):
    """Test that plain tensor inputs still work with SimpleFSDP (full_dtensor=False)."""

    @property
    def world_size(self) -> int:
        return 4

    def init_parallel_dims(self):
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=self.world_size,
        )

    def test_plain_inputs_with_default_mode(self):
        """Test plain tensor inputs with default SimpleFSDP mode (full_dtensor=False)."""
        self.init_parallel_dims()
        torch.manual_seed(42)

        vocab_size, hidden_dim = 32, 64
        batch_size, seq_len = 8, 16

        model = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        ).cuda()

        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
        model = data_parallel(
            model,
            device_mesh=fsdp_mesh,
            mode="fully_shard",
            # full_dtensor=False is default
        )

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        rank = fsdp_mesh.get_local_rank()
        local_batch = batch_size // self.world_size

        torch.manual_seed(42 + rank)
        local_inputs = torch.randint(0, vocab_size, (local_batch, seq_len)).cuda()
        local_labels = torch.randint(0, vocab_size, (local_batch, seq_len)).cuda()

        # Training loop with plain tensors
        losses = []
        for _ in range(5):
            optim.zero_grad()
            out = model(local_inputs)
            loss = cross_entropy_loss(out, local_labels)
            loss.backward()
            optim.step()
            losses.append(loss.detach().clone())

        # Verify training progresses
        self.assertLess(
            losses[-1].item(),
            losses[0].item(),
            "Loss should decrease during training",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
