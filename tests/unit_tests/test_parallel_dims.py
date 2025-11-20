# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed import ParallelDims


class TestParallelDimsValidation(unittest.TestCase):
    """Test ParallelDims validation logic without mesh building."""

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_basic_initialization(self):
        """Test basic initialization with valid parameters."""
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            etp=1,
            world_size=8,
        )
        self.assertEqual(parallel_dims.dp_replicate, 2)
        self.assertEqual(parallel_dims.dp_shard, 2)
        self.assertEqual(parallel_dims.cp, 1)
        self.assertEqual(parallel_dims.tp, 2)
        self.assertEqual(parallel_dims.pp, 1)
        self.assertEqual(parallel_dims.ep, 1)
        self.assertEqual(parallel_dims.etp, 1)
        self.assertEqual(parallel_dims.world_size, 8)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_auto_calculate_dp_shard(self):
        """Test automatic calculation of dp_shard when set to -1."""
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=-1,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            etp=1,
            world_size=8,
        )
        self.assertEqual(parallel_dims.dp_shard, 2)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_validation_invalid_world_size(self):
        """Test validation fails when parallelism degrees don't match world_size."""
        with self.assertRaises(AssertionError):
            ParallelDims(
                dp_replicate=2,
                dp_shard=2,
                cp=1,
                tp=2,
                pp=1,
                ep=1,
                etp=1,
                world_size=10,  # Invalid: 2*2*1*2*1 = 8, not 10
            )

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_validation_invalid_etp(self):
        """Test validation fails when etp is not equal to tp or 1."""
        with self.assertRaises(AssertionError):
            ParallelDims(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=4,
                pp=1,
                ep=2,
                etp=2,  # Invalid: etp must be tp or 1 when ep > 1
                world_size=8,
            )

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_validation_zero_parallelism(self):
        """Test validation fails when parallelism degree is 0."""
        with self.assertRaises(AssertionError):
            ParallelDims(
                dp_replicate=0,  # Invalid: must be >= 1
                dp_shard=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                etp=1,
                world_size=1,
            )

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_validation_invalid_dp_shard(self):
        """Test validation fails when dp_shard is invalid (not -1 and not >=1)."""
        with self.assertRaises(AssertionError):
            ParallelDims(
                dp_replicate=1,
                dp_shard=0,  # Invalid: must be -1 or >= 1
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                etp=1,
                world_size=1,
            )

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_enabled_properties(self):
        """Test all enabled properties."""
        # Test with DP enabled
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            etp=1,
            world_size=8,
        )
        self.assertTrue(parallel_dims.dp_enabled)
        self.assertTrue(parallel_dims.dp_replicate_enabled)
        self.assertTrue(parallel_dims.dp_shard_enabled)
        self.assertFalse(parallel_dims.cp_enabled)
        self.assertTrue(parallel_dims.tp_enabled)
        self.assertFalse(parallel_dims.pp_enabled)
        self.assertFalse(parallel_dims.ep_enabled)
        self.assertFalse(parallel_dims.etp_enabled)
        self.assertTrue(parallel_dims.fsdp_enabled)

        # Test with CP enabled
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=2,
        )
        self.assertFalse(parallel_dims.dp_enabled)
        self.assertTrue(parallel_dims.cp_enabled)
        self.assertTrue(parallel_dims.dp_cp_enabled)
        self.assertTrue(parallel_dims.fsdp_enabled)

        # Test with EP and ETP enabled (EP * ETP must not contribute to world_size)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            world_size=2,
        )
        self.assertTrue(parallel_dims.ep_enabled)
        self.assertFalse(parallel_dims.etp_enabled)

        # Test with PP enabled
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=2,
            ep=1,
            etp=1,
            world_size=2,
        )
        self.assertTrue(parallel_dims.pp_enabled)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_fsdp_gradient_divide_factor(self):
        """Test fsdp_gradient_divide_factor calculation."""
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=3,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=12,
        )
        # Should be dp_replicate * dp_shard * cp = 2 * 3 * 2 = 12
        self.assertEqual(parallel_dims.fsdp_gradient_divide_factor, 12)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_non_data_parallel_size(self):
        """Test non_data_parallel_size calculation."""
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=2,
            tp=3,
            pp=2,
            ep=1,
            etp=1,
            world_size=48,
        )
        # Should be cp * tp * pp = 2 * 3 * 2 = 12
        self.assertEqual(parallel_dims.non_data_parallel_size, 12)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_seq_len_divisor(self):
        """Test seq_len_divisor calculation."""
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=1,
            cp=2,
            tp=4,
            pp=1,
            ep=1,
            etp=1,
            world_size=16,
        )
        # Should be tp * (cp * 2) = 4 * 4 = 16
        self.assertEqual(parallel_dims.seq_len_divisor, 16)


class TestParallelDimsMeshOperations(unittest.TestCase):
    """Test ParallelDims mesh operations with single-rank distributed environment."""

    def setUp(self):
        """Initialize distributed environment for CPU testing."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12356",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        """Clean up distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_get_mesh_invalid_name(self):
        """Test getting mesh with invalid name raises error."""
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )
        parallel_dims.build_mesh()

        with self.assertRaises(ValueError) as context:
            parallel_dims.get_mesh("invalid_mesh")
        self.assertIn("Invalid mesh dim", str(context.exception))

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_get_mesh_lazy_initialization(self):
        """Test that get_optional_mesh triggers build_mesh if not built yet."""
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )
        # Don't call build_mesh explicitly
        self.assertEqual(len(parallel_dims._meshes), 0)

        # get_optional_mesh should trigger build_mesh
        result = parallel_dims.get_optional_mesh("tp")
        # Result is None because tp has size 1, but build_mesh should have been called
        self.assertGreater(len(parallel_dims._meshes), 0)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_single_rank_mesh_operations(self):
        """Comprehensive test for all single-rank mesh operations.

        This test verifies mesh building, mesh retrieval, mesh sizes, and property
        access when all parallelism dimensions are set to 1 (single rank).
        """
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )

        # Test mesh building
        world_mesh = parallel_dims.build_mesh()
        self.assertIsNotNone(world_mesh)
        self.assertEqual(world_mesh.size(), 1)

        # Verify all expected meshes are created
        self.assertIsNotNone(parallel_dims._meshes)
        self.assertIn("pp", parallel_dims._meshes)
        self.assertIn("batch", parallel_dims._meshes)
        self.assertIn("loss", parallel_dims._meshes)
        self.assertIn("dp_replicate", parallel_dims._meshes)
        self.assertIn("fsdp", parallel_dims._meshes)
        self.assertIn("cp", parallel_dims._meshes)
        self.assertIn("tp", parallel_dims._meshes)

        # Validate 1D mesh sizes - all should be 1 for single rank
        self.assertEqual(parallel_dims._meshes["dp_replicate"].size(), 1)
        self.assertEqual(parallel_dims._meshes["fsdp"].size(), 1)
        self.assertEqual(parallel_dims._meshes["tp"].size(), 1)
        self.assertEqual(parallel_dims._meshes["batch"].size(), 1)
        self.assertEqual(parallel_dims._meshes["loss"].size(), 1)
        self.assertEqual(parallel_dims._meshes["pp"].size(), 1)
        self.assertEqual(parallel_dims._meshes["cp"].size(), 1)
        self.assertEqual(parallel_dims._meshes["ep"].size(), 1)
        self.assertEqual(parallel_dims._meshes["etp"].size(), 1)
        self.assertEqual(parallel_dims._meshes["efsdp"].size(), 1)

        # Validate 2D mesh shapes
        dp_replicate_fsdp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "fsdp"]
        )
        self.assertIsNone(dp_replicate_fsdp_mesh)  # Both dimensions have size 1
        dp_replicate_efsdp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "efsdp"]
        )
        self.assertIsNone(dp_replicate_efsdp_mesh)  # Both dimensions have size 1
        ep_etp_mesh = parallel_dims.get_optional_mesh(["ep", "etp"])
        self.assertIsNone(ep_etp_mesh)  # Both dimensions have size 1

        # Test get_optional_mesh returns None when all dimensions have size 1
        self.assertIsNone(parallel_dims.get_optional_mesh("tp"))
        self.assertIsNone(parallel_dims.get_optional_mesh("dp_replicate"))
        self.assertIsNone(parallel_dims.get_optional_mesh("pp"))
        self.assertIsNone(parallel_dims.get_optional_mesh("cp"))
        self.assertIsNone(parallel_dims.get_optional_mesh("fsdp"))

        # Test get_optional_mesh with list input
        self.assertIsNone(parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"]))

        # Test get_all_one_dimensional_meshes returns empty when all dimensions have size 1
        one_d_meshes = parallel_dims.get_all_one_dimensional_meshes()
        self.assertEqual(len(one_d_meshes), 0)

        # Test world_mesh property
        world_mesh_property = parallel_dims.world_mesh
        self.assertIsNotNone(world_mesh_property)
        self.assertEqual(world_mesh_property.size(), 1)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_get_mesh_with_list_input(self):
        """Test get_optional_mesh accepts both string and list inputs."""
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )
        parallel_dims.build_mesh()

        # Should accept list input
        result = parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
        # Returns None because both dimensions have size 1
        self.assertIsNone(result)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_expert_parallelism_validation(self):
        """Test expert parallelism configurations."""
        # EP with ETP = 1 (valid) - world_size = dp_replicate * dp_shard * cp * tp * pp
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            world_size=2,  # 1 * 2 * 1 * 1 * 1 = 2
        )
        self.assertTrue(parallel_dims.ep_enabled)
        self.assertFalse(parallel_dims.etp_enabled)

        # Test with larger configuration
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=3,
            etp=1,
            world_size=4,  # 2 * 2 * 1 * 1 * 1 = 4
        )
        self.assertTrue(parallel_dims.ep_enabled)
        self.assertFalse(parallel_dims.etp_enabled)
        self.assertTrue(parallel_dims.dp_replicate_enabled)
        self.assertTrue(parallel_dims.dp_shard_enabled)


class TestParallelDimsWorld8MeshOperations(DTensorTestBase):
    """Test ParallelDims mesh operations with 8-rank distributed environment."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_world_size_8_mesh_operations(self):
        """Comprehensive test for world_size=8 mesh operations.

        This test validates mesh building, mesh retrieval, mesh sizes, and properties
        for a world_size=8 configuration with multiple parallelism dimensions enabled.
        Configuration: dp_replicate=2, dp_shard=2, cp=1, tp=2, pp=1 (2*2*1*2*1 = 8)
        """
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims = ParallelDims(
                dp_replicate=2,
                dp_shard=2,
                cp=1,
                tp=2,
                pp=1,
                ep=1,
                etp=1,
                world_size=8,
            )

            # Test mesh building
            world_mesh = parallel_dims.build_mesh()
            self.assertIsNotNone(world_mesh)
            self.assertEqual(world_mesh.size(), 8)

            # Verify all expected meshes are created
            self.assertIsNotNone(parallel_dims._meshes)
            self.assertIn("pp", parallel_dims._meshes)
            self.assertIn("batch", parallel_dims._meshes)
            self.assertIn("loss", parallel_dims._meshes)
            self.assertIn("dp_replicate", parallel_dims._meshes)
            self.assertIn("fsdp", parallel_dims._meshes)
            self.assertIn("cp", parallel_dims._meshes)
            self.assertIn("tp", parallel_dims._meshes)
            self.assertIn("ep", parallel_dims._meshes)
            self.assertIn("etp", parallel_dims._meshes)
            self.assertIn("efsdp", parallel_dims._meshes)

            # Validate 1D mesh sizes match parallelism configuration
            self.assertEqual(parallel_dims._meshes["pp"].size(), 1)
            self.assertEqual(
                parallel_dims._meshes["batch"].size(), 4
            )  # dp_replicate * dp_shard = 2 * 2
            self.assertEqual(
                parallel_dims._meshes["loss"].size(), 4
            )  # dp_replicate * dp_shard * cp = 2 * 2 * 1
            self.assertEqual(parallel_dims._meshes["dp_replicate"].size(), 2)
            self.assertEqual(
                parallel_dims._meshes["fsdp"].size(), 2
            )  # dp_shard * cp = 2 * 1
            self.assertEqual(parallel_dims._meshes["cp"].size(), 1)
            self.assertEqual(parallel_dims._meshes["tp"].size(), 2)
            self.assertEqual(parallel_dims._meshes["ep"].size(), 1)
            self.assertEqual(parallel_dims._meshes["etp"].size(), 1)
            self.assertEqual(
                parallel_dims._meshes["efsdp"].size(), 4
            )  # fsdp * tp / (etp * ep) = 2 * 2 / (1 * 1) = 4

            # Validate 2D mesh shapes
            dp_replicate_fsdp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
            self.assertIsNotNone(dp_replicate_fsdp_mesh)
            self.assertEqual(
                dp_replicate_fsdp_mesh.shape, (2, 2)
            )  # (dp_replicate, fsdp)
            # efsdp mesh only exists when ep > 1, so dp_replicate_efsdp should be None when ep=1
            dp_replicate_efsdp_mesh = parallel_dims.get_optional_mesh(
                ["dp_replicate", "efsdp"]
            )
            self.assertIsNone(dp_replicate_efsdp_mesh)  # efsdp disabled when ep=1
            ep_etp_mesh = parallel_dims.get_optional_mesh(["ep", "etp"])
            self.assertIsNone(ep_etp_mesh)  # Both dimensions have size 1

            # Test get_mesh returns valid meshes for enabled dimensions (size > 1)
            self.assertIsNotNone(parallel_dims.get_mesh("tp"))
            self.assertIsNotNone(parallel_dims.get_mesh("dp_replicate"))
            self.assertIsNotNone(parallel_dims.get_mesh("fsdp"))
            self.assertIsNotNone(parallel_dims.get_mesh("batch"))
            self.assertIsNotNone(parallel_dims.get_mesh("loss"))

            # Test get_optional_mesh returns None for disabled dimensions (size = 1)
            self.assertIsNone(parallel_dims.get_optional_mesh("pp"))
            self.assertIsNone(parallel_dims.get_optional_mesh("cp"))
            self.assertIsNone(parallel_dims.get_optional_mesh("ep"))

            # Test get_mesh with 2D mesh names
            self.assertIsNotNone(parallel_dims.get_mesh(["dp_replicate", "fsdp"]))
            hsdp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
            self.assertEqual(hsdp_mesh.shape, (2, 2))

            # Test get_all_one_dimensional_meshes returns only meshes with size > 1
            one_d_meshes = parallel_dims.get_all_one_dimensional_meshes()
            self.assertGreater(len(one_d_meshes), 0)
            # Should include: dp_replicate, fsdp, tp, batch, loss, efsdp (all with size > 1)
            self.assertIn("dp_replicate", one_d_meshes)
            self.assertIn("fsdp", one_d_meshes)
            self.assertIn("tp", one_d_meshes)
            self.assertIn("batch", one_d_meshes)
            self.assertIn("loss", one_d_meshes)
            self.assertIn("efsdp", one_d_meshes)
            # Should not include: pp, cp, ep, etp (all with size = 1)
            self.assertNotIn("pp", one_d_meshes)
            self.assertNotIn("cp", one_d_meshes)
            self.assertNotIn("ep", one_d_meshes)
            self.assertNotIn("etp", one_d_meshes)

            # Test that we can get 2D meshes via get_mesh() instead
            dp_replicate_fsdp = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
            self.assertIsNotNone(dp_replicate_fsdp)
            self.assertEqual(dp_replicate_fsdp.ndim, 2)

            # Test world_mesh property
            world_mesh_property = parallel_dims.world_mesh
            self.assertIsNotNone(world_mesh_property)
            self.assertEqual(world_mesh_property.size(), 8)

            # Validate enabled properties
            self.assertTrue(parallel_dims.dp_enabled)
            self.assertTrue(parallel_dims.dp_replicate_enabled)
            self.assertTrue(parallel_dims.dp_shard_enabled)
            self.assertTrue(parallel_dims.fsdp_enabled)
            self.assertTrue(parallel_dims.tp_enabled)
            self.assertFalse(parallel_dims.cp_enabled)
            self.assertFalse(parallel_dims.pp_enabled)
            self.assertFalse(parallel_dims.ep_enabled)

            # Validate calculated properties
            self.assertEqual(
                parallel_dims.fsdp_gradient_divide_factor, 4
            )  # dp_replicate * dp_shard * cp = 2 * 2 * 1
            self.assertEqual(
                parallel_dims.non_data_parallel_size, 2
            )  # cp * tp * pp = 1 * 2 * 1
            self.assertEqual(
                parallel_dims.seq_len_divisor, 4
            )  # tp * (cp * 2) = 2 * (1 * 2) = 2 * 2


if __name__ == "__main__":
    unittest.main()
