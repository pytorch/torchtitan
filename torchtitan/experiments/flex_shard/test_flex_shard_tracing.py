#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard FX-tracing support (Phase 1).

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest test_flex_shard_tracing.py -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 test_flex_shard_tracing.py
"""

import unittest

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode


# ---------------------------------------------------------------------------
# Step 1: FakeTensorMode blocking gate
# ---------------------------------------------------------------------------


class TestFakeTensorModeByteBuffer(unittest.TestCase):
    """Validate that byte-buffer view patterns work under FakeTensorMode.

    FlexShard stores all sharded parameters in a single uint8 buffer and
    creates typed views via .view(dtype).view(shape).  If FakeTensorMode
    cannot handle these operations, the parametrization approach is blocked
    and a typed-per-bucket fallback is needed (design doc Gap #5).
    """

    def test_view_uint8_to_bf16(self):
        """torch.empty(N, dtype=uint8).view(torch.bfloat16) adjusts shape."""
        with FakeTensorMode():
            buf = torch.empty(64, dtype=torch.uint8)
            typed = buf.view(torch.bfloat16)
            # bf16 is 2 bytes, so 64 bytes -> 32 elements
            self.assertEqual(typed.shape, (32,))
            self.assertEqual(typed.dtype, torch.bfloat16)

    def test_byte_offset_slice_and_view(self):
        """Byte offset slicing + view(dtype) + view(shape) works."""
        with FakeTensorMode():
            buf = torch.empty(256, dtype=torch.uint8)
            # Simulate a parameter at byte offset 64, shape (4, 8), dtype bf16
            num_elements = 4 * 8  # 32
            num_bytes = num_elements * 2  # 64 bytes for bf16
            local_view = buf[64 : 64 + num_bytes]
            typed_view = local_view.view(torch.bfloat16).view(4, 8)
            self.assertEqual(typed_view.shape, (4, 8))
            self.assertEqual(typed_view.dtype, torch.bfloat16)

    def test_mixed_dtype_regions(self):
        """Mixed-dtype regions (bf16 + fp32) in one uint8 buffer."""
        with FakeTensorMode():
            # bf16 param: shape (8, 4) = 32 elements * 2 bytes = 64 bytes
            # fp32 param: shape (4, 4) = 16 elements * 4 bytes = 64 bytes
            buf = torch.empty(128, dtype=torch.uint8)

            # First region: bf16
            region1 = buf[0:64].view(torch.bfloat16).view(8, 4)
            self.assertEqual(region1.shape, (8, 4))
            self.assertEqual(region1.dtype, torch.bfloat16)

            # Second region: fp32
            region2 = buf[64:128].view(torch.float32).view(4, 4)
            self.assertEqual(region2.shape, (4, 4))
            self.assertEqual(region2.dtype, torch.float32)


# ---------------------------------------------------------------------------
# Step 6: FX tracing tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestParametrizationTracing(unittest.TestCase):
    """Validate that parametrization classes trace correctly under FakeTensorMode."""

    def test_shard_dim0_traces(self):
        """ShardParametrization(dim=0) traces with make_fx under FakeTensorMode."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.flex_shard import ShardParametrization

        with FakeTensorMode():
            param = ShardParametrization(
                shard_dim=0, group_name="fake_pg", world_size=2
            )
            # Local shard: shape (4, 8) -> full shape (8, 8) after all-gather
            local_shard = torch.randn(4, 8)

            gm = make_fx(param, tracing_mode="fake")(local_shard)

        # Check graph contains expected ops (names include overload suffix
        # like ".default", so use substring matching)
        op_names = [
            n.target.__name__ if callable(n.target) else str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        ]
        op_names_str = " ".join(op_names)
        self.assertIn("all_gather_into_tensor", op_names_str)
        self.assertIn("wait_tensor", op_names_str)

    def test_shard_dim1_traces(self):
        """ShardParametrization(dim=1) traces with chunk+cat in graph."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.flex_shard import ShardParametrization

        with FakeTensorMode():
            param = ShardParametrization(
                shard_dim=1, group_name="fake_pg", world_size=2
            )
            # Local shard: shape (8, 4) -> full shape (8, 8) after all-gather
            local_shard = torch.randn(8, 4)

            gm = make_fx(param, tracing_mode="fake")(local_shard)

        op_names = [
            n.target.__name__ if callable(n.target) else str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        ]
        op_names_str = " ".join(op_names)
        self.assertIn("all_gather_into_tensor", op_names_str)
        self.assertIn("wait_tensor", op_names_str)
        # dim != 0 requires chunk/split + cat
        has_chunk_or_split = "chunk" in op_names_str or "split" in op_names_str
        self.assertTrue(
            has_chunk_or_split, f"Expected chunk/split in graph, got: {op_names}"
        )
        self.assertIn("cat", op_names_str)

    def test_flat_shard_traces(self):
        """FlatShardParametrization traces with all_gather + view."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.flex_shard import FlatShardParametrization

        with FakeTensorMode():
            param = FlatShardParametrization(
                group_name="fake_pg",
                world_size=2,
                original_shape=torch.Size([4, 8]),
            )
            # Flat local shard: 16 elements (32 total / 2 ranks)
            flat_shard = torch.randn(16)

            gm = make_fx(param, tracing_mode="fake")(flat_shard)

        op_names = [
            n.target.__name__ if callable(n.target) else str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        ]
        op_names_str = " ".join(op_names)
        self.assertIn("all_gather_into_tensor", op_names_str)
        self.assertIn("wait_tensor", op_names_str)
        # Should have a view/reshape to restore original shape
        has_reshape = "view" in op_names_str or "reshape" in op_names_str
        self.assertTrue(has_reshape, f"Expected view/reshape in graph, got: {op_names}")

    def test_owned_parametrization_traces(self):
        """OwnedParametrization traces with broadcast+wait_tensor in graph."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.flex_shard import OwnedParametrization

        with FakeTensorMode():
            param = OwnedParametrization(
                owner_rank=0,
                group_name="fake_pg",
                world_size=2,
            )
            # Full param (all ranks store full copy in parametrization mode)
            full_param = torch.randn(8, 8)

            gm = make_fx(param, tracing_mode="fake")(full_param)

        op_names = [
            n.target.__name__ if callable(n.target) else str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        ]
        op_names_str = " ".join(op_names)
        self.assertIn("broadcast", op_names_str)
        self.assertIn("wait_tensor", op_names_str)

    def test_uneven_shard_dim0_traces(self):
        """ShardParametrization with uneven split traces with pad+narrow."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.flex_shard import ShardParametrization

        with FakeTensorMode():
            param = ShardParametrization(
                shard_dim=0,
                group_name="fake_pg",
                world_size=2,
                padded_shard_size=4,  # ceil(7/2) = 4
                global_dim_size=7,
            )
            # Local shard: last rank gets 3, padded to 4
            local_shard = torch.randn(3, 8)

            gm = make_fx(param, tracing_mode="fake")(local_shard)

        op_names = [
            n.target.__name__ if callable(n.target) else str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        ]
        op_names_str = " ".join(op_names)
        self.assertIn("all_gather_into_tensor", op_names_str)
        self.assertIn("wait_tensor", op_names_str)
        # Padding cat and final slice/narrow
        self.assertIn("cat", op_names_str)
        has_slice_or_narrow = "slice" in op_names_str or "narrow" in op_names_str
        self.assertTrue(has_slice_or_narrow)

    def test_ragged_shard_parametrization_traces(self):
        """RaggedShardParametrization traces with pad+gather+chunk+narrow+cat."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.flex_shard import RaggedShardParametrization

        with FakeTensorMode():
            param = RaggedShardParametrization(
                shard_dim=0,
                split_sizes=[3, 5],  # rank 0 gets 3, rank 1 gets 5
                group_name="fake_pg",
                world_size=2,
            )
            # Rank 0's local shard: shape (3, 8), max is 5 so will be padded
            local_shard = torch.randn(3, 8)

            gm = make_fx(param, tracing_mode="fake")(local_shard)

        op_names = [
            n.target.__name__ if callable(n.target) else str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        ]
        op_names_str = " ".join(op_names)
        self.assertIn("all_gather_into_tensor", op_names_str)
        self.assertIn("wait_tensor", op_names_str)
        # Padding cat, chunk/split for reassembly, slice/narrow for each chunk,
        # final cat.
        self.assertIn("cat", op_names_str)
        has_chunk_or_split = "chunk" in op_names_str or "split" in op_names_str
        self.assertTrue(has_chunk_or_split)
        has_slice_or_narrow = "slice" in op_names_str or "narrow" in op_names_str
        self.assertTrue(has_slice_or_narrow)


# ---------------------------------------------------------------------------
# Step 7: Validation tests
# ---------------------------------------------------------------------------


class TestInitValidation(unittest.TestCase):
    """Test _validate_placements_for_tracing rejects invalid configs."""

    def _make_mock_mesh(self, ndim: int = 1, size: int = 4):
        """Create a minimal mock mesh for validation testing."""
        from unittest.mock import MagicMock

        mesh = MagicMock()
        mesh.ndim = ndim
        mesh.size.return_value = size
        return mesh

    def _make_mock_mesh_info(self, mesh):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard.flex_shard import FlexShardMeshInfo

        return FlexShardMeshInfo(
            dp_shard_mesh=mesh,
            spmd_mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="fsdp"),
            dp_shard_dim_names=("fsdp",),
            dp_dim_indices=(0,),
        )

    def test_accepts_multidim_mesh(self):
        """Multi-dim mesh is accepted (Phase 5: multi-mesh composition)."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            Shard,
        )

        mesh = self._make_mock_mesh(ndim=2)
        param = nn.Parameter(torch.randn(8, 8))
        # Should not raise (multi-mesh supported since Phase 5)
        _validate_placements_for_tracing(
            param_placements={"weight": (Shard(0),)},
            named_params=[("weight", param)],
            mesh_info=self._make_mock_mesh_info(mesh),
            mesh=mesh,
        )

    def test_rejects_owned_invalid_rank(self):
        """Owned with owner_rank >= world_size is rejected."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            Owned,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(8, 8))
        with self.assertRaises(ValueError, msg="owner_rank"):
            _validate_placements_for_tracing(
                param_placements={"weight": (Owned(4),)},
                named_params=[("weight", param)],
                mesh_info=self._make_mock_mesh_info(mesh),
                mesh=mesh,
            )

    def test_accepts_owned(self):
        """Owned with valid owner_rank is accepted."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            Owned,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(8, 8))
        # Should not raise
        _validate_placements_for_tracing(
            param_placements={"weight": (Owned(0),)},
            named_params=[("weight", param)],
            mesh_info=self._make_mock_mesh_info(mesh),
            mesh=mesh,
        )

    def test_accepts_ragged_shard(self):
        """RaggedShard with correct local_units length is accepted (Phase 5)."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            RaggedShard,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(8, 8))
        # Should not raise (RaggedShard supported since Phase 5)
        _validate_placements_for_tracing(
            param_placements={"weight": (RaggedShard((0,), (1, 2, 1, 1)),)},
            named_params=[("weight", param)],
            mesh_info=self._make_mock_mesh_info(mesh),
            mesh=mesh,
        )

    def test_rejects_ragged_shard_bad_local_units(self):
        """RaggedShard with wrong local_units length is rejected."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            RaggedShard,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(8, 8))
        with self.assertRaises(ValueError, msg="local_units"):
            _validate_placements_for_tracing(
                param_placements={"weight": (RaggedShard((0,), (1, 2)),)},
                named_params=[("weight", param)],
                mesh_info=self._make_mock_mesh_info(mesh),
                mesh=mesh,
            )

    def test_accepts_uneven_shard(self):
        """Uneven Shard is accepted (Phase 5: pad-to-uniform)."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            Shard,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(7, 8))  # dim 0 = 7, not divisible by 4
        # Should not raise (uneven supported since Phase 5)
        _validate_placements_for_tracing(
            param_placements={"weight": (Shard(0),)},
            named_params=[("weight", param)],
            mesh_info=self._make_mock_mesh_info(mesh),
            mesh=mesh,
        )

    def test_accepts_uneven_flat_shard(self):
        """Uneven FlatShard is accepted (Phase 5: pad-to-uniform)."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            FlatShard,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        # numel=49, not divisible by 4
        param = nn.Parameter(torch.randn(7, 7))
        # Should not raise (uneven supported since Phase 5)
        _validate_placements_for_tracing(
            param_placements={"weight": (FlatShard(),)},
            named_params=[("weight", param)],
            mesh_info=self._make_mock_mesh_info(mesh),
            mesh=mesh,
        )

    def test_rejects_shard_invalid_dim(self):
        """Shard with dim >= ndim is rejected."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            Shard,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(8, 8))  # 2D tensor
        with self.assertRaises(ValueError, msg="out of range"):
            _validate_placements_for_tracing(
                param_placements={"weight": (Shard(2),)},
                named_params=[("weight", param)],
                mesh_info=self._make_mock_mesh_info(mesh),
                mesh=mesh,
            )

    def test_accepts_valid_shard(self):
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            Shard,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(8, 8))  # dim 0 = 8, divisible by 4
        # Should not raise
        _validate_placements_for_tracing(
            param_placements={"weight": (Shard(0),)},
            named_params=[("weight", param)],
            mesh_info=self._make_mock_mesh_info(mesh),
            mesh=mesh,
        )

    def test_accepts_valid_flat_shard(self):
        from torchtitan.experiments.flex_shard.flex_shard import (
            _validate_placements_for_tracing,
            FlatShard,
        )

        mesh = self._make_mock_mesh(ndim=1, size=4)
        param = nn.Parameter(torch.randn(8, 8))  # numel=64, divisible by 4
        # Should not raise
        _validate_placements_for_tracing(
            param_placements={"weight": (FlatShard(),)},
            named_params=[("weight", param)],
            mesh_info=self._make_mock_mesh_info(mesh),
            mesh=mesh,
        )


# ---------------------------------------------------------------------------
# Distributed correctness tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedParametrization(unittest.TestCase):
    """Multi-process correctness tests for parametrization classes.

    Run with: torchrun --nproc_per_node=2 test_flex_shard_tracing.py
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

    def test_shard_dim0_correctness(self):
        """ShardParametrization(dim=0) produces correct full tensor."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import ShardParametrization

        mesh = init_device_mesh("cuda", (self.world_size,))
        group_name = mesh.get_group().group_name

        # Create reference full tensor on rank 0 and broadcast
        full_ref = torch.randn(8, 4, device="cuda")
        torch.distributed.broadcast(full_ref, src=0)

        # Each rank takes its shard
        chunk_size = full_ref.shape[0] // self.world_size
        local_shard = full_ref[
            self.rank * chunk_size : (self.rank + 1) * chunk_size
        ].contiguous()

        # Run parametrization
        param = ShardParametrization(
            shard_dim=0, group_name=group_name, world_size=self.world_size
        )
        result = param(local_shard)

        torch.testing.assert_close(result, full_ref)

    def test_shard_dim1_correctness(self):
        """ShardParametrization(dim=1) produces correct full tensor."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import ShardParametrization

        mesh = init_device_mesh("cuda", (self.world_size,))
        group_name = mesh.get_group().group_name

        full_ref = torch.randn(4, 8, device="cuda")
        torch.distributed.broadcast(full_ref, src=0)

        chunk_size = full_ref.shape[1] // self.world_size
        local_shard = full_ref[
            :, self.rank * chunk_size : (self.rank + 1) * chunk_size
        ].contiguous()

        param = ShardParametrization(
            shard_dim=1, group_name=group_name, world_size=self.world_size
        )
        result = param(local_shard)

        torch.testing.assert_close(result, full_ref)

    def test_flat_shard_correctness(self):
        """FlatShardParametrization produces correct full tensor."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan.experiments.flex_shard import FlatShardParametrization

        mesh = init_device_mesh("cuda", (self.world_size,))
        group_name = mesh.get_group().group_name

        original_shape = torch.Size([4, 8])
        full_ref = torch.randn(original_shape, device="cuda")
        torch.distributed.broadcast(full_ref, src=0)

        # Flatten and take shard
        flat_full = full_ref.flatten()
        shard_size = flat_full.shape[0] // self.world_size
        flat_shard = flat_full[
            self.rank * shard_size : (self.rank + 1) * shard_size
        ].contiguous()

        param = FlatShardParametrization(
            group_name=group_name,
            world_size=self.world_size,
            original_shape=original_shape,
        )
        result = param(flat_shard)

        torch.testing.assert_close(result, full_ref)


if __name__ == "__main__":
    unittest.main()
