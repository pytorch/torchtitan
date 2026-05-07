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

from torchtitan.experiments.flex_shard import BucketSpec, is_flex_shard_param
from torchtitan.experiments.flex_shard.bucket_storage import (
    _assign_params_to_buckets,
    _create_param_infos,
    _materialize_bucket_storages,
)
from torchtitan.experiments.flex_shard.placements import Shard
from torchtitan.experiments.flex_shard.tests.common import (
    make_transformer_model,
    transformer_bucket_specs,
    transformer_inputs,
)


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


class _FakeRankMesh:
    """DeviceMesh stand-in with rank/world-size methods for storage tests."""

    def __init__(self, rank: int = 0, world_size: int = 2) -> None:
        self.rank = rank
        self.world_size = world_size

    def get_local_rank(self) -> int:
        return self.rank

    def size(self) -> int:
        return self.world_size


# ---------------------------------------------------------------------------
# DP shard mesh tests (single-process, no PG required)
# ---------------------------------------------------------------------------


class TestDPShardMesh(unittest.TestCase):
    """Test FlexShard DP shard mesh derivation."""

    def test_dp_mesh_dims_derives_shard_submesh(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.utils import (
            _get_dp_shard_mesh,
        )

        mesh = _FakeMesh(("fsdp", "tp"))
        shard_mesh = _get_dp_shard_mesh(mesh, DataParallelMeshDims(shard="fsdp"))

        self.assertEqual(shard_mesh.mesh_dim_names, ("fsdp",))

    def test_dp_mesh_dims_flattens_multiple_shard_dims(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.utils import (
            _get_dp_shard_mesh,
        )

        mesh = _FakeMesh(("dp0", "dp1", "tp"))
        shard_mesh = _get_dp_shard_mesh(
            mesh,
            DataParallelMeshDims(shard=("dp0", "dp1")),
        )

        self.assertEqual(shard_mesh.mesh_dim_names, ("dp0_dp1",))

    def test_dp_mesh_dims_requires_named_mesh(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.utils import (
            _get_dp_shard_mesh,
        )

        with self.assertRaisesRegex(ValueError, "mesh_dim_names"):
            _get_dp_shard_mesh(_FakeMesh(None), DataParallelMeshDims(shard="fsdp"))

    def test_dp_mesh_dims_rejects_missing_dim(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.utils import (
            _get_dp_shard_mesh,
        )

        with self.assertRaisesRegex(ValueError, "not found"):
            _get_dp_shard_mesh(
                _FakeMesh(("fsdp", "tp")),
                DataParallelMeshDims(shard="missing"),
            )

    def test_dp_mesh_dims_rejects_replicate_until_hsdp_supported(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.utils import (
            _get_dp_shard_mesh,
        )

        with self.assertRaisesRegex(NotImplementedError, "replicate"):
            _get_dp_shard_mesh(
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
        from torchtitan.experiments.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "attn.bias", "ffn.weight", "ffn.bias"]
        buckets = [BucketSpec(["attn.*"]), BucketSpec(["ffn.*"])]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight", "attn.bias"])
        self.assertEqual(result[1], ["ffn.weight", "ffn.bias"])

    def test_rejects_orphan_params(self):
        """Params matching zero buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "norm.weight"]
        buckets = [BucketSpec(["attn.*"])]
        with self.assertRaises(ValueError, msg="not covered by any bucket"):
            _assign_params_to_buckets(fqns, buckets)

    def test_rejects_overlapping_params(self):
        """Params matching multiple buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight"]
        buckets = [BucketSpec(["attn.*"]), BucketSpec(["*"])]
        with self.assertRaises(ValueError, msg="matched multiple buckets"):
            _assign_params_to_buckets(fqns, buckets)

    def test_bucket_spec_patterns(self):
        """BucketSpec.patterns are used for matching."""
        from torchtitan.experiments.flex_shard import BucketSpec
        from torchtitan.experiments.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "ffn.weight"]
        buckets = [BucketSpec(patterns=["attn.*"]), BucketSpec(patterns=["ffn.*"])]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight"])
        self.assertEqual(result[1], ["ffn.weight"])

    def test_wildcard_bucket(self):
        """Single ['*'] bucket catches all params."""
        from torchtitan.experiments.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["a.weight", "b.bias", "c.weight"]
        buckets = [BucketSpec(["*"])]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], fqns)

    def test_multi_pattern_bucket(self):
        """Bucket with multiple patterns matches any of them."""
        from torchtitan.experiments.flex_shard.bucket_storage import (
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

    def test_rejects_missing_or_extra_placements(self):
        """Placement validation requires exact managed parameter coverage."""
        from torchtitan.experiments.flex_shard.utils import _validate_placements
        from torchtitan.experiments.flex_shard.placements import Shard

        named_params = self._named_params()
        with self.assertRaisesRegex(ValueError, "missing placements"):
            _validate_placements(
                {"a.weight": (Shard(0),)},
                named_params,
                _FakeMesh(("fsdp",)),
            )

        with self.assertRaisesRegex(ValueError, "unexpected placements"):
            _validate_placements(
                {
                    "a.weight": (Shard(0),),
                    "b.weight": (Shard(0),),
                    "extra.weight": (Shard(0),),
                },
                named_params,
                _FakeMesh(("fsdp",)),
            )

    def test_rejects_non_shard0_placement(self):
        """Unsupported placement in a bucket raises ValueError."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import Placement, Shard

        class UnsupportedPlacement(Placement):
            def __repr__(self):
                return "UnsupportedPlacement()"

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (UnsupportedPlacement(),),
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
        """Unsupported placement is rejected even when isolated to its own bucket."""
        from torchtitan.experiments.flex_shard.utils import (
            _validate_bucket_placements,
        )
        from torchtitan.experiments.flex_shard.placements import Placement, Shard

        class UnsupportedPlacement(Placement):
            def __repr__(self):
                return "UnsupportedPlacement()"

        assignments = [["a.weight"], ["b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (UnsupportedPlacement(),),
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
# Bucket storage layout tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketStorageLayout(unittest.TestCase):
    """Test ParamInfo and DStorage layout for bucket materialization."""

    def test_create_param_infos_uses_sequential_byte_offsets(self):
        mesh = _FakeRankMesh(rank=0, world_size=2)
        args, model = make_transformer_model()
        named_params = [
            ("tok_embeddings.weight", model.tok_embeddings.weight),
            ("pos_embeddings.weight", model.pos_embeddings.weight),
        ]
        placements = {fqn: (Shard(0),) for fqn, _ in named_params}

        infos, total_bytes = _create_param_infos(named_params, mesh, placements)

        tok_embeddings = infos["tok_embeddings.weight"]
        pos_embeddings = infos["pos_embeddings.weight"]
        self.assertEqual(
            tok_embeddings.local_shape,
            torch.Size([args.vocab_size // 2, args.dim]),
        )
        self.assertEqual(tok_embeddings.byte_offset, 0)
        self.assertEqual(
            pos_embeddings.local_shape,
            torch.Size([args.max_seq_len // 2, args.dim]),
        )
        self.assertEqual(
            pos_embeddings.byte_offset,
            (args.vocab_size // 2) * args.dim * torch.float32.itemsize,
        )
        self.assertEqual(
            total_bytes,
            ((args.vocab_size // 2) + (args.max_seq_len // 2))
            * args.dim
            * torch.float32.itemsize,
        )

    def test_materialized_params_are_views_into_bucket_storage(self):
        mesh = _FakeRankMesh(rank=1, world_size=2)
        args, model = make_transformer_model()
        named_params = list(model.named_parameters())
        placements = {fqn: (Shard(0),) for fqn, _ in named_params}
        buckets = transformer_bucket_specs(
            args.n_layers,
            reshard_after_forward=False,
        )
        assignments = _assign_params_to_buckets(
            [fqn for fqn, _ in named_params],
            buckets,
        )

        storages, fqn_to_bucket_spec = _materialize_bucket_storages(
            model,
            named_params,
            assignments,
            buckets,
            placements,
            mesh,
            torch.device("cpu"),
        )

        self.assertEqual(len(storages), len(buckets))
        self.assertIs(fqn_to_bucket_spec["tok_embeddings.weight"], buckets[0])
        self.assertIs(fqn_to_bucket_spec["output.weight"], buckets[-1])

        current_params = dict(model.named_parameters())
        for storage in storages:
            storage_ptr = storage.byte_storage.untyped_storage().data_ptr()
            for fqn, info in storage.param_infos.items():
                param = current_params[fqn]
                local_view = storage.get_local_view(fqn)

                self.assertEqual(param.shape, info.local_shape)
                self.assertEqual(
                    param.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                self.assertEqual(
                    local_view.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                torch.testing.assert_close(param, local_view)
                self.assertTrue(is_flex_shard_param(param))


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
        args, model = make_transformer_model(device="cuda")
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        x = transformer_inputs(args, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = model(x).clone()

        self._flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(
                args.n_layers,
                reshard_after_forward=False,
            ),
        )

        output = model(x)
        torch.testing.assert_close(output, ref_output)

    def test_multi_bucket_state_dict_sharded(self):
        """state_dict returns sharded params across all buckets."""
        mesh = self._init_mesh()
        args, model = make_transformer_model(device="cuda")
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(
                args.n_layers,
                reshard_after_forward=False,
            ),
        )

        sd = model.state_dict()
        self.assertEqual(
            sd["tok_embeddings.weight"].shape,
            (args.vocab_size // self.world_size, args.dim),
        )
        self.assertEqual(
            sd["output.weight"].shape,
            (args.vocab_size // self.world_size, args.dim),
        )

    def test_whole_transformer_bucket_requires_matching_hook_boundary(self):
        """A whole-Transformer RAF bucket has no matching recompute hook boundary."""
        mesh = self._init_mesh()

        args, model = make_transformer_model(device="cuda")
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
        self._flex_shard(model, mesh, buckets=[BucketSpec(["*"])])

        x = transformer_inputs(args, device="cuda")
        torch.distributed.broadcast(x, src=0)

        with self.assertRaisesRegex(RuntimeError, "pre-gathered parameter data"):
            model(x)

    def test_multi_bucket_dstorages_count(self):
        """Module has one DStorage per non-empty bucket."""
        mesh = self._init_mesh()
        args, model = make_transformer_model(device="cuda")
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        buckets = transformer_bucket_specs(args.n_layers, reshard_after_forward=False)
        self._flex_shard(
            model,
            mesh,
            buckets=buckets,
        )

        self.assertEqual(len(model.dstorages), len(buckets))
        self.assertIs(model.dstorage, model.dstorages[0])

    def test_bucket_spec_reshard_policy_wired_to_dstorages(self):
        """Each bucket's reshard policy is stored on its DStorage."""
        mesh = self._init_mesh()
        args, model = make_transformer_model(device="cuda")
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=0)

        buckets = transformer_bucket_specs(args.n_layers, reshard_after_forward=False)
        buckets[0] = BucketSpec(["tok_embeddings.*"], reshard_after_forward=True)
        self._flex_shard(
            model,
            mesh,
            buckets=buckets,
        )

        self.assertTrue(model.dstorages[0]._reshard_after_forward)
        for storage in model.dstorages[1:]:
            self.assertFalse(storage._reshard_after_forward)


if __name__ == "__main__":
    unittest.main()
