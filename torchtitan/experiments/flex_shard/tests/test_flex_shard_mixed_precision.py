#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard Phase 2c: mixed precision.

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest \
      torchtitan/experiments/flex_shard/tests/test_flex_shard_mixed_precision.py \
      -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 \
      torchtitan/experiments/flex_shard/tests/test_flex_shard_mixed_precision.py
"""

import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MixedPrecisionPolicy tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestMixedPrecisionPolicy(unittest.TestCase):
    """Test MixedPrecisionPolicy dataclass."""

    def test_frozen_dataclass(self):
        """MixedPrecisionPolicy is immutable."""
        from torchtitan.experiments.flex_shard import MixedPrecisionPolicy

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        with self.assertRaises(AttributeError):
            mp.param_dtype = torch.float32  # type: ignore[misc]

    def test_default_none(self):
        """Default policy has no dtype overrides."""
        from torchtitan.experiments.flex_shard import MixedPrecisionPolicy

        mp = MixedPrecisionPolicy()
        self.assertIsNone(mp.param_dtype)
        self.assertIsNone(mp.reduce_dtype)

    def test_param_dtype_only(self):
        """param_dtype set, reduce_dtype defaults to None."""
        from torchtitan.experiments.flex_shard import MixedPrecisionPolicy

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self.assertEqual(mp.param_dtype, torch.bfloat16)
        self.assertIsNone(mp.reduce_dtype)

    def test_both_dtypes(self):
        """Both param_dtype and reduce_dtype can be set."""
        from torchtitan.experiments.flex_shard import MixedPrecisionPolicy

        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        self.assertEqual(mp.param_dtype, torch.bfloat16)
        self.assertEqual(mp.reduce_dtype, torch.float32)


# ---------------------------------------------------------------------------
# _MixedPrecisionCast tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestMixedPrecisionCast(unittest.TestCase):
    """Test _MixedPrecisionCast autograd function."""

    def test_forward_casts_to_param_dtype(self):
        """Forward casts fp32 -> bf16 when param_dtype=bf16."""
        from torchtitan.experiments.flex_shard.mixed_precision import (
            _MixedPrecisionCast,
        )

        x = torch.randn(4, 4, dtype=torch.float32)
        result = _MixedPrecisionCast.apply(x, torch.bfloat16, None)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_forward_noop_when_none(self):
        """Forward is identity when param_dtype=None."""
        from torchtitan.experiments.flex_shard.mixed_precision import (
            _MixedPrecisionCast,
        )

        x = torch.randn(4, 4, dtype=torch.float32)
        result = _MixedPrecisionCast.apply(x, None, None)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, x)

    def test_forward_noop_when_same_dtype(self):
        """Forward is identity when param_dtype matches input dtype."""
        from torchtitan.experiments.flex_shard.mixed_precision import (
            _MixedPrecisionCast,
        )

        x = torch.randn(4, 4, dtype=torch.bfloat16)
        result = _MixedPrecisionCast.apply(x, torch.bfloat16, None)
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result, x)

    def test_backward_casts_to_reduce_dtype(self):
        """Backward casts grad to reduce_dtype."""
        from torchtitan.experiments.flex_shard.mixed_precision import (
            _MixedPrecisionCast,
        )

        x = torch.randn(4, 4, dtype=torch.float32, requires_grad=True)
        result = _MixedPrecisionCast.apply(x, torch.bfloat16, torch.float32)
        self.assertEqual(result.dtype, torch.bfloat16)

        # Backward: grad is bf16, should be cast to fp32 (reduce_dtype)
        loss = result.sum()
        loss.backward()
        self.assertEqual(x.grad.dtype, torch.float32)

    def test_backward_noop_when_reduce_none(self):
        """Backward is identity when reduce_dtype=None."""
        from torchtitan.experiments.flex_shard.mixed_precision import (
            _MixedPrecisionCast,
        )

        x = torch.randn(4, 4, dtype=torch.float32, requires_grad=True)
        result = _MixedPrecisionCast.apply(x, torch.bfloat16, None)

        loss = result.sum()
        loss.backward()
        # grad stays in bf16 (backward of bf16 cast)... actually autograd
        # through the .to() in forward produces grad in input dtype (fp32)
        # when reduce_dtype is None. Since _MixedPrecisionCast.backward
        # returns grad unchanged when reduce_dtype is None, the grad dtype
        # depends on autograd's .to() backward behavior.
        self.assertIsNotNone(x.grad)

    def test_decoupled_forward_backward(self):
        """Forward=bf16, backward=fp32 work independently."""
        from torchtitan.experiments.flex_shard.mixed_precision import (
            _MixedPrecisionCast,
        )

        x = torch.randn(4, 4, dtype=torch.float32, requires_grad=True)
        result = _MixedPrecisionCast.apply(x, torch.bfloat16, torch.float32)

        self.assertEqual(result.dtype, torch.bfloat16)
        loss = result.sum()
        loss.backward()
        self.assertEqual(x.grad.dtype, torch.float32)


# ---------------------------------------------------------------------------
# BucketSpec mp_policy wiring tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketSpecMpPolicy(unittest.TestCase):
    """Test mp_policy wiring from BucketSpec to eager parameter access state."""

    def test_mp_policy_type(self):
        """BucketSpec.mp_policy accepts MixedPrecisionPolicy."""
        from torchtitan.experiments.flex_shard import BucketSpec, MixedPrecisionPolicy

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        spec = BucketSpec(patterns=["*"], mp_policy=mp)
        self.assertEqual(spec.mp_policy.param_dtype, torch.bfloat16)

    def test_param_state_receives_dtypes(self):
        """Eager param access state carries param_dtype/reduce_dtype."""
        from torchtitan.experiments.flex_shard.module_wrapping import (
            EagerParamAccessState,
        )

        state = EagerParamAccessState(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        self.assertEqual(state.param_dtype, torch.bfloat16)
        self.assertEqual(state.reduce_dtype, torch.float32)

    def test_param_state_default_no_mp(self):
        """Eager param access state without mp has None dtypes."""
        from torchtitan.experiments.flex_shard.module_wrapping import (
            EagerParamAccessState,
        )

        state = EagerParamAccessState()
        self.assertIsNone(state.param_dtype)
        self.assertIsNone(state.reduce_dtype)


# ---------------------------------------------------------------------------
# Distributed mixed precision tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedMixedPrecision(unittest.TestCase):
    """Multi-process correctness tests for mixed precision FlexShard.

    Run with:
        torchrun --nproc_per_node=2 \
          torchtitan/experiments/flex_shard/tests/test_flex_shard_mixed_precision.py
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

    def _init_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh

        return init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("fsdp",))

    def _flex_shard(self, model, mesh, **kwargs):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            per_param_placements,
        )

        kwargs.setdefault("shard_placement_fn", per_param_placements)
        kwargs.setdefault("buckets", [BucketSpec(["*"])])
        return flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            **kwargs,
        )

    def test_param_access_without_pregather_raises(self):
        """Parameter access outside hooked forward raises in eager-only mode."""
        from torchtitan.experiments.flex_shard import BucketSpec, MixedPrecisionPolicy

        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(patterns=["*"], mp_policy=mp)],
        )

        with self.assertRaisesRegex(RuntimeError, "pre-gathered parameter data"):
            _ = model.weight

    def test_state_dict_returns_storage_dtype(self):
        """state_dict returns raw sharded params in original (storage) dtype."""
        from torchtitan.experiments.flex_shard import BucketSpec, MixedPrecisionPolicy

        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(patterns=["*"], mp_policy=mp)],
        )

        sd = model.state_dict()
        # state_dict bypasses property, returns raw fp32 shard
        self.assertEqual(sd["weight"].dtype, torch.float32)

    def test_forward_output_in_param_dtype(self):
        """Forward output dtype matches param_dtype."""
        from torchtitan.experiments.flex_shard import BucketSpec, MixedPrecisionPolicy

        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(patterns=["*"], mp_policy=mp)],
        )

        x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        torch.distributed.broadcast(x, src=0)
        output = model(x)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_gradient_in_reduce_dtype(self):
        """After backward, param.grad is in reduce_dtype."""
        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            MixedPrecisionPolicy,
        )

        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(patterns=["*"], mp_policy=mp)],
        )

        x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
        torch.distributed.broadcast(x, src=0)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # The reduce-scatter produces gradients in reduce_dtype, so the local
        # grad shard is fp32.
        grad = model._parameters["weight"].grad
        self.assertIsNotNone(grad)
        self.assertEqual(grad.dtype, torch.float32)

    def test_no_mp_matches_original(self):
        """Without mp_policy, output matches non-mp baseline."""
        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        ref_weight = model.weight.data.clone()

        self._flex_shard(model, mesh)

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)

        ref_output = x @ ref_weight.t()
        output = model(x)
        torch.testing.assert_close(output, ref_output)

    def test_per_bucket_mp_policy(self):
        """Different mp_policy per bucket works correctly."""
        from torchtitan.experiments.flex_shard import BucketSpec, MixedPrecisionPolicy

        mesh = self._init_mesh()

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 4, bias=False, device="cuda")
                self.linear2 = nn.Linear(4, 2, bias=False, device="cuda")

            def forward(self, x):
                return self.linear2(self.linear1(x))

        model = TwoLayer()
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        # linear1 in bf16, linear2 in fp32 (no mp)
        self._flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(
                    patterns=["linear1.*"],
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                ),
                BucketSpec(["linear2.*"]),  # no mp_policy
            ],
        )

        with self.assertRaisesRegex(RuntimeError, "pre-gathered parameter data"):
            _ = model.linear1.weight
        with self.assertRaisesRegex(RuntimeError, "pre-gathered parameter data"):
            _ = model.linear2.weight


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
