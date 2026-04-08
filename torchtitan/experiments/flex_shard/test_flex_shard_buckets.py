#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard Phase 2b: BucketSpec and bucket validation.

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest test_flex_shard_buckets.py -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 test_flex_shard_buckets.py
"""

import unittest

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# fnmatch placement dict tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestFnmatchPlacementFn(unittest.TestCase):
    """Test _resolve_placement_fn with dict[str, Placement] input."""

    def test_dict_resolves_first_match(self):
        """First matching pattern wins."""
        from torchtitan.experiments.flex_shard import Shard
        from torchtitan.experiments.flex_shard.flex_shard import _resolve_placement_fn

        fn = _resolve_placement_fn({"attn.*": Shard(1), "*": Shard(0)})
        module = nn.Linear(8, 4, bias=False)
        named_params = [("attn.weight", module.weight)]

        # Mock mesh — we only need mesh for the function signature
        from unittest.mock import MagicMock

        mesh = MagicMock()
        result = fn(named_params, mesh)
        self.assertEqual(result["attn.weight"], (Shard(1),))

    def test_dict_catchall_pattern(self):
        """'*' catches unmatched params."""
        from torchtitan.experiments.flex_shard import Shard
        from torchtitan.experiments.flex_shard.flex_shard import _resolve_placement_fn

        fn = _resolve_placement_fn({"attn.*": Shard(1), "*": Shard(0)})
        module = nn.Linear(8, 4, bias=False)
        named_params = [("ffn.weight", module.weight)]

        from unittest.mock import MagicMock

        mesh = MagicMock()
        result = fn(named_params, mesh)
        self.assertEqual(result["ffn.weight"], (Shard(0),))

    def test_dict_no_match_raises(self):
        """Unmatched param raises ValueError."""
        from torchtitan.experiments.flex_shard import Shard
        from torchtitan.experiments.flex_shard.flex_shard import _resolve_placement_fn

        fn = _resolve_placement_fn({"attn.*": Shard(1)})
        module = nn.Linear(8, 4, bias=False)
        named_params = [("ffn.weight", module.weight)]

        from unittest.mock import MagicMock

        mesh = MagicMock()
        with self.assertRaises(ValueError, msg="does not match any"):
            fn(named_params, mesh)

    def test_none_returns_per_param(self):
        """None input returns per_param_placements."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _resolve_placement_fn,
            per_param_placements,
        )

        result = _resolve_placement_fn(None)
        self.assertIs(result, per_param_placements)

    def test_callable_passthrough(self):
        """Callable input is returned directly."""
        from torchtitan.experiments.flex_shard.flex_shard import _resolve_placement_fn

        def my_fn(named_params, mesh):
            return {}

        result = _resolve_placement_fn(my_fn)
        self.assertIs(result, my_fn)


# ---------------------------------------------------------------------------
# Bucket assignment tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketAssignment(unittest.TestCase):
    """Test _assign_params_to_buckets."""

    def test_assigns_params_to_correct_bucket(self):
        """Params match the right bucket by fnmatch."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "attn.bias", "ffn.weight", "ffn.bias"]
        buckets = [["attn.*"], ["ffn.*"]]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight", "attn.bias"])
        self.assertEqual(result[1], ["ffn.weight", "ffn.bias"])

    def test_rejects_orphan_params(self):
        """Params matching zero buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "norm.weight"]
        buckets = [["attn.*"]]
        with self.assertRaises(ValueError, msg="not covered by any bucket"):
            _assign_params_to_buckets(fqns, buckets)

    def test_rejects_overlapping_params(self):
        """Params matching multiple buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight"]
        buckets = [["attn.*"], ["*"]]
        with self.assertRaises(ValueError, msg="matched multiple buckets"):
            _assign_params_to_buckets(fqns, buckets)

    def test_bucket_spec_patterns(self):
        """BucketSpec.patterns are used for matching."""
        from torchtitan.experiments.flex_shard import BucketSpec
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "ffn.weight"]
        buckets = [BucketSpec(patterns=["attn.*"]), BucketSpec(patterns=["ffn.*"])]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight"])
        self.assertEqual(result[1], ["ffn.weight"])

    def test_wildcard_bucket(self):
        """Single ['*'] bucket catches all params."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["a.weight", "b.bias", "c.weight"]
        buckets = [["*"]]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], fqns)

    def test_multi_pattern_bucket(self):
        """Bucket with multiple patterns matches any of them."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "ffn.weight", "norm.weight"]
        buckets = [["attn.*", "ffn.*"], ["norm.*"]]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight", "ffn.weight"])
        self.assertEqual(result[1], ["norm.weight"])


# ---------------------------------------------------------------------------
# Placement consistency tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketPlacementValidation(unittest.TestCase):
    """Test _validate_bucket_placements."""

    def test_rejects_mixed_placement_types(self):
        """Shard + FlatShard in one bucket raises ValueError."""
        from torchtitan.experiments.flex_shard import FlatShard, Shard
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_bucket_placements,
        )

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (FlatShard(),),
        }
        buckets = [["*"]]
        with self.assertRaises(ValueError, msg="mixed placement types"):
            _validate_bucket_placements(assignments, placements, buckets)

    def test_rejects_mixed_shard_dims(self):
        """Shard(0) + Shard(1) in one bucket raises ValueError."""
        from torchtitan.experiments.flex_shard import Shard
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_bucket_placements,
        )

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(1),),
        }
        buckets = [["*"]]
        with self.assertRaises(ValueError, msg="mixed shard dimensions"):
            _validate_bucket_placements(assignments, placements, buckets)

    def test_accepts_same_placement(self):
        """Shard(0) + Shard(0) in one bucket passes."""
        from torchtitan.experiments.flex_shard import Shard
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_bucket_placements,
        )

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(0),),
        }
        buckets = [["*"]]
        # Should not raise
        _validate_bucket_placements(assignments, placements, buckets)

    def test_accepts_separate_buckets_different_placements(self):
        """Different placement types in separate buckets is fine."""
        from torchtitan.experiments.flex_shard import FlatShard, Shard
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_bucket_placements,
        )

        assignments = [["a.weight"], ["b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (FlatShard(),),
        }
        buckets = [["a.*"], ["b.*"]]
        # Should not raise
        _validate_bucket_placements(assignments, placements, buckets)

    def test_empty_bucket_skipped(self):
        """Empty bucket assignments are silently skipped."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_bucket_placements,
        )

        assignments = [[], ["a.weight"]]
        placements = {"a.weight": (None,)}
        buckets = [["x.*"], ["a.*"]]
        # Should not raise (first bucket is empty)
        _validate_bucket_placements(assignments, placements, buckets)


# ---------------------------------------------------------------------------
# Auto-bucket tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestAutoBuckets(unittest.TestCase):
    """Test auto_buckets helper."""

    def test_generates_per_child_buckets(self):
        """One bucket per direct child module."""
        from torchtitan.experiments.flex_shard import auto_buckets

        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        result = auto_buckets(model)
        self.assertEqual(result, [["0.*"], ["1.*"]])

    def test_named_children_buckets(self):
        """Named children produce named patterns."""
        from torchtitan.experiments.flex_shard import auto_buckets

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(8, 4)
                self.ffn = nn.Linear(4, 2)

            def forward(self, x):
                return self.ffn(self.attn(x))

        result = auto_buckets(Model())
        self.assertEqual(result, [["attn.*"], ["ffn.*"]])

    def test_flat_module_returns_catchall(self):
        """Module with no children returns [['*']]."""
        from torchtitan.experiments.flex_shard import auto_buckets

        module = nn.Linear(8, 4)
        result = auto_buckets(module)
        self.assertEqual(result, [["*"]])


# ---------------------------------------------------------------------------
# BucketSpec tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketSpec(unittest.TestCase):
    """Test BucketSpec dataclass."""

    def test_basic_creation(self):
        """BucketSpec stores patterns."""
        from torchtitan.experiments.flex_shard import BucketSpec

        spec = BucketSpec(patterns=["attn.*", "ffn.*"])
        self.assertEqual(spec.patterns, ["attn.*", "ffn.*"])
        self.assertIsNone(spec.mp_policy)
        self.assertIsNone(spec.offload_policy)

    def test_with_policies(self):
        """BucketSpec stores policy placeholders."""
        from torchtitan.experiments.flex_shard import BucketSpec

        spec = BucketSpec(
            patterns=["*"],
            mp_policy="placeholder",
            offload_policy="placeholder",
        )
        self.assertEqual(spec.mp_policy, "placeholder")
        self.assertEqual(spec.offload_policy, "placeholder")


# ---------------------------------------------------------------------------
# Distributed per-bucket DStorage tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedBuckets(unittest.TestCase):
    """Multi-process correctness tests for bucketed FlexShard.

    Run with: torchrun --nproc_per_node=2 test_flex_shard_buckets.py
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

    def test_multi_bucket_forward_correct(self):
        """Model with explicit buckets produces correct forward output."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 4, bias=False, device="cuda")
                self.linear2 = nn.Linear(4, 2, bias=False, device="cuda")

            def forward(self, x):
                return self.linear2(self.linear1(x))

        model = TwoLayer()
        # Broadcast weights so all ranks start identically
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        # Reference output before sharding
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = model(x).clone()

        flex_shard(
            model,
            mesh,
            buckets=[["linear1.*"], ["linear2.*"]],
            register_hooks=False,
        )

        output = model(x)
        torch.testing.assert_close(output, ref_output)

    def test_multi_bucket_state_dict_sharded(self):
        """state_dict returns sharded params across all buckets."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        model = nn.Sequential(
            nn.Linear(8, 4, bias=False, device="cuda"),
            nn.Linear(4, 2, bias=False, device="cuda"),
        )
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        flex_shard(
            model,
            mesh,
            buckets=[["0.*"], ["1.*"]],
            register_hooks=False,
        )

        sd = model.state_dict()
        # Both layers should be sharded along dim 0
        self.assertEqual(sd["0.weight"].shape, (4 // self.world_size, 8))
        self.assertEqual(sd["1.weight"].shape, (2 // self.world_size, 4))

    def test_single_bucket_default_matches_no_buckets(self):
        """buckets=None produces same result as explicit [['*']]."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

        # Model A: no buckets
        model_a = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model_a.weight.data, src=0)
        ref_weight = model_a.weight.data.clone()
        flex_shard(model_a, mesh, register_hooks=False)

        # Model B: explicit [["*"]]
        model_b = nn.Linear(8, 4, bias=False, device="cuda")
        model_b.weight.data.copy_(ref_weight)
        flex_shard(model_b, mesh, buckets=[["*"]], register_hooks=False)

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)

        out_a = model_a(x)
        out_b = model_b(x)
        torch.testing.assert_close(out_a, out_b)

    def test_multi_bucket_dstorages_count(self):
        """Module has one DStorage per non-empty bucket."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import flex_shard

        mesh = init_device_mesh("cuda", (self.world_size,))

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

        flex_shard(
            model,
            mesh,
            buckets=[["linear1.*"], ["linear2.*"]],
            register_hooks=False,
        )

        self.assertEqual(len(model.dstorages), 2)
        self.assertIs(model.dstorage, model.dstorages[0])


if __name__ == "__main__":
    unittest.main()
