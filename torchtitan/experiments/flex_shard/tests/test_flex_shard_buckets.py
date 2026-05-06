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
    python -m pytest \
      torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py \
      -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 \
      torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py
"""

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.flex_shard import BucketSpec


class _FakeMesh:
    """Minimal DeviceMesh stand-in for mesh-info unit tests."""

    def __init__(self, mesh_dim_names):
        self.mesh_dim_names = mesh_dim_names
        self.ndim = len(mesh_dim_names) if mesh_dim_names is not None else 1

    def __getitem__(self, names):
        if isinstance(names, tuple):
            return _FakeMesh(names)
        return _FakeMesh((names,))

    def _flatten(self, name):
        return _FakeMesh((name,))


# ---------------------------------------------------------------------------
# Mesh metadata tests (single-process, no PG required)
# ---------------------------------------------------------------------------


class TestFlexShardMeshInfo(unittest.TestCase):
    """Test FlexShard mesh metadata derivation."""

    def test_dp_mesh_dims_derives_shard_submesh(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.flex_shard import (
            _get_flex_shard_mesh_info,
        )

        mesh = _FakeMesh(("fsdp", "tp"))
        info = _get_flex_shard_mesh_info(mesh, DataParallelMeshDims(shard="fsdp"))

        self.assertEqual(info.dp_shard_mesh.mesh_dim_names, ("fsdp",))

    def test_dp_mesh_dims_flattens_multiple_shard_dims(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.flex_shard import (
            _get_flex_shard_mesh_info,
        )

        mesh = _FakeMesh(("dp0", "dp1", "tp"))
        info = _get_flex_shard_mesh_info(
            mesh, DataParallelMeshDims(shard=("dp0", "dp1"))
        )

        self.assertEqual(info.dp_shard_mesh.mesh_dim_names, ("dp0_dp1",))

    def test_dp_mesh_dims_requires_named_mesh(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.flex_shard import (
            _get_flex_shard_mesh_info,
        )

        with self.assertRaisesRegex(ValueError, "mesh_dim_names"):
            _get_flex_shard_mesh_info(
                _FakeMesh(None), DataParallelMeshDims(shard="fsdp")
            )

    def test_dp_mesh_dims_rejects_missing_dim(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.flex_shard import (
            _get_flex_shard_mesh_info,
        )

        with self.assertRaisesRegex(ValueError, "not found"):
            _get_flex_shard_mesh_info(
                _FakeMesh(("fsdp", "tp")),
                DataParallelMeshDims(shard="missing"),
            )

    def test_dp_mesh_dims_rejects_replicate_until_hsdp_supported(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.flex_shard import (
            _get_flex_shard_mesh_info,
        )

        with self.assertRaisesRegex(NotImplementedError, "replicate"):
            _get_flex_shard_mesh_info(
                _FakeMesh(("rep", "fsdp", "tp")),
                DataParallelMeshDims(shard="fsdp", replicate="rep"),
            )


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
        buckets = [BucketSpec(["attn.*"]), BucketSpec(["ffn.*"])]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight", "attn.bias"])
        self.assertEqual(result[1], ["ffn.weight", "ffn.bias"])

    def test_rejects_orphan_params(self):
        """Params matching zero buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "norm.weight"]
        buckets = [BucketSpec(["attn.*"])]
        with self.assertRaises(ValueError, msg="not covered by any bucket"):
            _assign_params_to_buckets(fqns, buckets)

    def test_rejects_overlapping_params(self):
        """Params matching multiple buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight"]
        buckets = [BucketSpec(["attn.*"]), BucketSpec(["*"])]
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
        buckets = [BucketSpec(["*"])]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], fqns)

    def test_multi_pattern_bucket(self):
        """Bucket with multiple patterns matches any of them."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "ffn.weight", "norm.weight"]
        buckets = [BucketSpec(["attn.*", "ffn.*"]), BucketSpec(["norm.*"])]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight", "ffn.weight"])
        self.assertEqual(result[1], ["norm.weight"])


# ---------------------------------------------------------------------------
# Placement consistency tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketPlacementValidation(unittest.TestCase):
    """Test _validate_bucket_placements."""

    @staticmethod
    def _named_params(
        dtypes: dict[str, torch.dtype] | None = None,
    ) -> list[tuple[str, nn.Parameter]]:
        dtypes = dtypes or {}
        return [
            (
                fqn,
                nn.Parameter(torch.empty(2, 2, dtype=dtypes.get(fqn, torch.float32))),
            )
            for fqn in ("a.weight", "b.weight")
        ]

    def test_rejects_non_shard0_placement(self):
        """FlatShard in a bucket raises ValueError."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import FlatShard, Shard

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (FlatShard(),),
        }
        buckets = [BucketSpec(["*"])]
        with self.assertRaisesRegex(ValueError, "only Shard\\(0\\)"):
            _validate_bucket_placements(
                assignments,
                placements,
                buckets,
                self._named_params(),
            )

    def test_rejects_nonzero_shard_dim(self):
        """Shard(1) in a bucket raises ValueError."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import Shard

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(1),),
        }
        buckets = [BucketSpec(["*"])]
        with self.assertRaisesRegex(ValueError, "only Shard\\(0\\)"):
            _validate_bucket_placements(
                assignments,
                placements,
                buckets,
                self._named_params(),
            )

    def test_accepts_same_placement(self):
        """Shard(0) + Shard(0) in one bucket passes."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import Shard

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(0),),
        }
        buckets = [BucketSpec(["*"])]
        # Should not raise
        _validate_bucket_placements(
            assignments,
            placements,
            buckets,
            self._named_params(),
        )

    def test_rejects_non_shard0_in_separate_bucket(self):
        """FlatShard is rejected even when isolated to its own bucket."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import FlatShard, Shard

        assignments = [["a.weight"], ["b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (FlatShard(),),
        }
        buckets = [BucketSpec(["a.*"]), BucketSpec(["b.*"])]
        with self.assertRaisesRegex(ValueError, "only Shard\\(0\\)"):
            _validate_bucket_placements(
                assignments,
                placements,
                buckets,
                self._named_params(),
            )

    def test_rejects_mixed_dtypes(self):
        """Parameters in one bucket must share the same storage dtype."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import Shard

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(0),),
        }
        buckets = [BucketSpec(["*"])]
        with self.assertRaisesRegex(ValueError, "mixed parameter dtypes"):
            _validate_bucket_placements(
                assignments,
                placements,
                buckets,
                self._named_params({"b.weight": torch.bfloat16}),
            )

    def test_empty_bucket_skipped(self):
        """Empty bucket assignments are silently skipped."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import Shard

        assignments = [[], ["a.weight"]]
        placements = {"a.weight": (Shard(0),)}
        buckets = [BucketSpec(["x.*"]), BucketSpec(["a.*"])]
        # Should not raise (first bucket is empty)
        _validate_bucket_placements(
            assignments,
            placements,
            buckets,
            self._named_params(),
        )


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
        self.assertEqual([bucket.patterns for bucket in result], [["0.*"], ["1.*"]])

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
        self.assertEqual(
            [bucket.patterns for bucket in result],
            [["attn.*"], ["ffn.*"]],
        )

    def test_flat_module_returns_catchall(self):
        """Module with no children returns [['*']]."""
        from torchtitan.experiments.flex_shard import auto_buckets

        module = nn.Linear(8, 4)
        result = auto_buckets(module)
        self.assertEqual([bucket.patterns for bucket in result], [["*"]])


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
        self.assertTrue(spec.reshard_after_forward)

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
        self.assertTrue(spec.reshard_after_forward)

    def test_reshard_after_forward_policy(self):
        """BucketSpec stores per-bucket reshard policy."""
        from torchtitan.experiments.flex_shard import BucketSpec

        spec = BucketSpec(patterns=["*"], reshard_after_forward=False)
        self.assertFalse(spec.reshard_after_forward)


# ---------------------------------------------------------------------------
# Distributed per-bucket DStorage tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedBuckets(unittest.TestCase):
    """Multi-process correctness tests for bucketed FlexShard.

    Run with:
        torchrun --nproc_per_node=2 \
          torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py
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

    def test_multi_bucket_forward_correct(self):
        """Model with explicit buckets produces correct forward output."""
        mesh = self._init_mesh()

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

        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(["linear1.*"]), BucketSpec(["linear2.*"])],
        )

        output = model(x)
        torch.testing.assert_close(output, ref_output)

    def test_multi_bucket_state_dict_sharded(self):
        """state_dict returns sharded params across all buckets."""
        mesh = self._init_mesh()

        model = nn.Sequential(
            nn.Linear(8, 4, bias=False, device="cuda"),
            nn.Linear(4, 2, bias=False, device="cuda"),
        )
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(["0.*"]), BucketSpec(["1.*"])],
        )

        sd = model.state_dict()
        # Both layers should be sharded along dim 0
        self.assertEqual(sd["0.weight"].shape, (4 // self.world_size, 8))
        self.assertEqual(sd["1.weight"].shape, (2 // self.world_size, 4))

    def test_single_bucket_explicit_matches_helper_default(self):
        """Explicit single bucket produces same result as the test helper default."""
        mesh = self._init_mesh()

        # Model A: no buckets
        model_a = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model_a.weight.data, src=0)
        ref_weight = model_a.weight.data.clone()
        self._flex_shard(model_a, mesh)

        # Model B: explicit single bucket
        model_b = nn.Linear(8, 4, bias=False, device="cuda")
        model_b.weight.data.copy_(ref_weight)
        self._flex_shard(model_b, mesh, buckets=[BucketSpec(["*"])])

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)

        out_a = model_a(x)
        out_b = model_b(x)
        torch.testing.assert_close(out_a, out_b)

    def test_multi_bucket_dstorages_count(self):
        """Module has one DStorage per non-empty bucket."""
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

        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(["linear1.*"]), BucketSpec(["linear2.*"])],
        )

        self.assertEqual(len(model.dstorages), 2)
        self.assertIs(model.dstorage, model.dstorages[0])

    def test_bucket_spec_reshard_policy_wired_to_dstorages(self):
        """Each bucket's reshard policy is stored on its DStorage."""
        from torchtitan.experiments.flex_shard import BucketSpec

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

        self._flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(["linear1.*"], reshard_after_forward=True),
                BucketSpec(["linear2.*"], reshard_after_forward=False),
            ],
        )

        self.assertTrue(model.dstorages[0]._reshard_after_forward)
        self.assertFalse(model.dstorages[1]._reshard_after_forward)


if __name__ == "__main__":
    unittest.main()
