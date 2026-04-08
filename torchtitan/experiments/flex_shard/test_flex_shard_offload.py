#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard Phase 2d: CPU offloading.

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest test_flex_shard_offload.py -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 test_flex_shard_offload.py
"""

import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# OffloadPolicy tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestOffloadPolicy(unittest.TestCase):
    """Test OffloadPolicy dataclass."""

    def test_frozen_dataclass(self):
        """OffloadPolicy is immutable."""
        from torchtitan.experiments.flex_shard import OffloadPolicy

        policy = OffloadPolicy()
        with self.assertRaises(AttributeError):
            policy.pin_memory = False  # type: ignore[misc]

    def test_default_pin_memory(self):
        """Default policy pins memory."""
        from torchtitan.experiments.flex_shard import OffloadPolicy

        policy = OffloadPolicy()
        self.assertTrue(policy.pin_memory)

    def test_pin_memory_false(self):
        """Can disable pinning."""
        from torchtitan.experiments.flex_shard import OffloadPolicy

        policy = OffloadPolicy(pin_memory=False)
        self.assertFalse(policy.pin_memory)


# ---------------------------------------------------------------------------
# BucketSpec offload_policy wiring tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketSpecOffloadPolicy(unittest.TestCase):
    """Test offload_policy wiring from BucketSpec to parametrization."""

    def test_offload_policy_type(self):
        """BucketSpec.offload_policy accepts OffloadPolicy."""
        from torchtitan.experiments.flex_shard import BucketSpec, OffloadPolicy

        policy = OffloadPolicy()
        spec = BucketSpec(patterns=["*"], offload_policy=policy)
        self.assertIsNotNone(spec.offload_policy)
        self.assertTrue(spec.offload_policy.pin_memory)

    def test_offload_policy_default_none(self):
        """BucketSpec.offload_policy defaults to None."""
        from torchtitan.experiments.flex_shard import BucketSpec

        spec = BucketSpec(patterns=["*"])
        self.assertIsNone(spec.offload_policy)

    def test_parametrization_receives_compute_device(self):
        """Parametrization created with correct compute_device."""
        from torchtitan.experiments.flex_shard import ShardParametrization

        p = ShardParametrization(
            shard_dim=0,
            group_name="fake",
            world_size=2,
            compute_device=torch.device("cuda:0"),
        )
        self.assertEqual(p.compute_device, torch.device("cuda:0"))

    def test_parametrization_default_no_compute_device(self):
        """Parametrization without offload has None compute_device."""
        from torchtitan.experiments.flex_shard import ShardParametrization

        p = ShardParametrization(
            shard_dim=0,
            group_name="fake",
            world_size=2,
        )
        self.assertIsNone(p.compute_device)

    def test_guard_disabled_returns_raw_with_offload(self):
        """With guard disabled, parametrization returns input unchanged."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            ShardParametrization,
        )

        param = ShardParametrization(
            shard_dim=0,
            group_name="fake",
            world_size=2,
            compute_device=torch.device("cuda:0"),
        )
        shard = torch.randn(4, 8)  # CPU tensor
        with disable_active_parametrization():
            result = param(shard)
        # Should return raw shard, no H2D transfer
        self.assertIs(result, shard)
        self.assertEqual(result.device.type, "cpu")

    def test_offload_rejects_hooks_mode(self):
        """offload_policy + register_hooks=True raises ValueError."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            OffloadPolicy,
        )

        # Need a real mesh — but this test needs NCCL; skip if not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not torch.distributed.is_initialized():
            self.skipTest("Distributed not initialized")

        mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))
        model = nn.Linear(8, 4, bias=False, device="cuda")

        with self.assertRaises(ValueError, msg="register_hooks"):
            flex_shard(
                model,
                mesh,
                buckets=[BucketSpec(patterns=["*"], offload_policy=OffloadPolicy())],
                register_hooks=True,
            )


# ---------------------------------------------------------------------------
# H2D transfer tests (single-process, needs CUDA)
# ---------------------------------------------------------------------------


class TestH2DTransfer(unittest.TestCase):
    """Test H2D transfer in parametrization forward."""

    def test_shard_parametrization_no_transfer_when_no_compute_device(self):
        """No .to() when compute_device is None (non-offloaded case)."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            ShardParametrization,
        )

        p = ShardParametrization(
            shard_dim=0,
            group_name="fake",
            world_size=2,
            compute_device=None,
        )
        shard = torch.randn(4, 8)  # CPU tensor
        with disable_active_parametrization():
            result = p(shard)
        # No compute_device => no transfer, returns raw
        self.assertIs(result, shard)

    def test_flat_shard_parametrization_no_transfer_when_no_compute_device(self):
        """No .to() when compute_device is None (non-offloaded case)."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            FlatShardParametrization,
        )

        p = FlatShardParametrization(
            group_name="fake",
            world_size=2,
            original_shape=torch.Size([8, 8]),
            compute_device=None,
        )
        flat_shard = torch.randn(32)  # CPU tensor
        with disable_active_parametrization():
            result = p(flat_shard)
        self.assertIs(result, flat_shard)


# ---------------------------------------------------------------------------
# Distributed offloading tests (multi-process, needs NCCL)
# ---------------------------------------------------------------------------


class TestDistributedOffload(unittest.TestCase):
    """Multi-process correctness tests for CPU offloading.

    Run with: torchrun --nproc_per_node=2 test_flex_shard_offload.py
    """

    @classmethod
    def setUpClass(cls):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(cls.rank % torch.cuda.device_count())

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.cuda.synchronize()
            torch.distributed.destroy_process_group()

    def test_offloaded_param_on_cpu(self):
        """Offloaded params are stored on CPU."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            OffloadPolicy,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(patterns=["*"], offload_policy=OffloadPolicy())],
        )

        # Raw param (via state_dict / guard) should be on CPU
        sd = model.state_dict()
        self.assertEqual(sd["weight"].device.type, "cpu")

    def test_offloaded_forward_output_on_gpu(self):
        """Forward output is on GPU despite CPU param storage."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            OffloadPolicy,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(patterns=["*"], offload_policy=OffloadPolicy())],
        )

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        output = model(x)
        self.assertEqual(output.device.type, "cuda")

    def test_offloaded_gradient_on_cpu(self):
        """After backward, param.grad is on CPU."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            OffloadPolicy,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(patterns=["*"], offload_policy=OffloadPolicy())],
        )

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        output = model(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(model.weight.grad)
        self.assertEqual(model.weight.grad.device.type, "cpu")

    def test_offload_with_mixed_precision(self):
        """Offloading + mp_policy compose correctly."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            MixedPrecisionPolicy,
            OffloadPolicy,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(
                    patterns=["*"],
                    mp_policy=mp,
                    offload_policy=OffloadPolicy(),
                )
            ],
        )

        # Param access triggers H2D + all-gather + mp cast
        result = model.weight
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.device.type, "cuda")

        # state_dict returns CPU fp32
        sd = model.state_dict()
        self.assertEqual(sd["weight"].dtype, torch.float32)
        self.assertEqual(sd["weight"].device.type, "cpu")

    def test_per_bucket_offload(self):
        """Only offloaded buckets are on CPU; others stay on GPU."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            OffloadPolicy,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Sequential(
            nn.Linear(8, 8, bias=False, device="cuda"),
            nn.Linear(8, 4, bias=False, device="cuda"),
        )
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(patterns=["0.*"], offload_policy=OffloadPolicy()),
                ["1.*"],  # Not offloaded
            ],
        )

        sd = model.state_dict()
        # Layer 0 offloaded to CPU
        self.assertEqual(sd["0.weight"].device.type, "cpu")
        # Layer 1 stays on GPU
        self.assertEqual(sd["1.weight"].device.type, "cuda")

    def test_offloaded_numerics_match_gpu(self):
        """Offloaded forward/backward matches non-offloaded numerics."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            OffloadPolicy,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        # Create two identical models
        torch.manual_seed(42)
        model_gpu = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model_gpu.weight.data, src=0)

        torch.manual_seed(42)
        model_cpu = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model_cpu.weight.data, src=0)

        # Shard both
        flex_shard(model_gpu, mesh, buckets=[["*"]])
        flex_shard(
            model_cpu,
            mesh,
            buckets=[BucketSpec(patterns=["*"], offload_policy=OffloadPolicy())],
        )

        # Same input
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)

        # Forward
        out_gpu = model_gpu(x)
        out_cpu = model_cpu(x)
        torch.testing.assert_close(out_gpu, out_cpu)

        # Backward
        out_gpu.sum().backward()
        out_cpu.sum().backward()
        # Compare grads (gpu grad is on cuda, cpu model grad is on cpu)
        torch.testing.assert_close(
            model_gpu.weight.grad,
            model_cpu.weight.grad.to("cuda"),
        )


# ---------------------------------------------------------------------------
# Main: support both pytest and torchrun
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
