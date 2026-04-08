#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard parametrization (Phase 2a).

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest test_flex_shard_parametrization.py -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 test_flex_shard_parametrization.py
"""

import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Guard behavior tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestActiveParametrizationGuard(unittest.TestCase):
    """Test _active_parametrization guard and disable_active_parametrization."""

    def test_guard_disabled_returns_raw_shard(self):
        """With guard disabled, ShardParametrization returns input unchanged."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            ShardParametrization,
        )

        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        shard = torch.randn(4, 8)
        with disable_active_parametrization():
            result = param(shard)
        self.assertIs(result, shard)

    def test_guard_disabled_returns_raw_flat_shard(self):
        """With guard disabled, FlatShardParametrization returns input unchanged."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            FlatShardParametrization,
        )

        param = FlatShardParametrization(
            group_name="fake",
            world_size=2,
            original_shape=torch.Size([4, 8]),
        )
        flat_shard = torch.randn(16)
        with disable_active_parametrization():
            result = param(flat_shard)
        self.assertIs(result, flat_shard)

    def test_guard_restores_after_context(self):
        """Guard restores to True after context manager exits."""
        import importlib

        fs = importlib.import_module("torchtitan.experiments.flex_shard.flex_shard")

        self.assertTrue(fs._active_parametrization)
        with fs.disable_active_parametrization():
            self.assertFalse(fs._active_parametrization)
        self.assertTrue(fs._active_parametrization)

    def test_guard_restores_on_exception(self):
        """Guard restores to True even if exception is raised."""
        import importlib

        fs = importlib.import_module("torchtitan.experiments.flex_shard.flex_shard")

        try:
            with fs.disable_active_parametrization():
                raise RuntimeError("test")
        except RuntimeError:
            pass
        self.assertTrue(fs._active_parametrization)


# ---------------------------------------------------------------------------
# Property registration tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestRegisterParametrization(unittest.TestCase):
    """Test _register_parametrization creates correct property getters."""

    def test_property_created_on_module(self):
        """Property getter is created on the module's dynamic subclass."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4, bias=False)
        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param})

        # The module's class should be a dynamic subclass
        self.assertIn("FlexShard", type(module).__name__)
        # Property should exist on the class
        self.assertIsInstance(type(module).__dict__["weight"], property)

    def test_state_dict_bypasses_property(self):
        """state_dict reads _parameters directly, not through property."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4, bias=False)
        original_shape = module.weight.shape

        # Register parametrization with guard disabled so no NCCL needed
        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param})

        # state_dict should return the raw parameter (bypasses property)
        sd = module.state_dict()
        self.assertEqual(sd["weight"].shape, original_shape)

    def test_multiple_params_on_same_module(self):
        """Multiple parameters can be parametrized on the same module."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4)  # has weight and bias
        param_w = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        param_b = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param_w, "bias": param_b})

        self.assertIsInstance(type(module).__dict__["weight"], property)
        self.assertIsInstance(type(module).__dict__["bias"], property)


# ---------------------------------------------------------------------------
# Distributed correctness tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedParametrization(unittest.TestCase):
    """Multi-process correctness tests for parametrized FlexShard.

    Run with: torchrun --nproc_per_node=2 test_flex_shard_parametrization.py
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

    def test_param_access_triggers_allgather(self):
        """Accessing module.weight returns the full (unsharded) tensor."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        # Broadcast weights so all ranks start with the same full tensor
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()

        flex_shard(model, mesh, register_hooks=False)

        # Accessing model.weight should trigger all-gather via property
        result = model.weight
        torch.testing.assert_close(result, full_ref)

    def test_state_dict_returns_sharded(self):
        """state_dict() returns sharded params, not unsharded."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        flex_shard(model, mesh, register_hooks=False)

        sd = model.state_dict()
        # state_dict bypasses property, returns local shard
        expected_rows = 4 // self.world_size
        self.assertEqual(sd["weight"].shape, (expected_rows, 8))

    def test_disable_guard_returns_sharded(self):
        """With guard disabled, param access returns raw sharded tensor."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            flex_shard,
        )

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        flex_shard(model, mesh, register_hooks=False)

        with disable_active_parametrization():
            result = model.weight
        expected_rows = 4 // self.world_size
        self.assertEqual(result.shape, (expected_rows, 8))

    def test_forward_produces_correct_output(self):
        """Forward pass through parametrized model produces correct results."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        ref_weight = model.weight.data.clone()

        # Reference output
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = x @ ref_weight.t()

        flex_shard(model, mesh, register_hooks=False)
        output = model(x)

        torch.testing.assert_close(output, ref_output)


if __name__ == "__main__":
    unittest.main()
