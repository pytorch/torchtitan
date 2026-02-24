# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from torchtitan.config import DebugConfig
from torchtitan.distributed.utils import set_determinism


class FakeParallelDims:
    """Fake ParallelDims for testing seed uniqueness.

    Args:
        mesh_dim_names: List of dimension names (e.g., ["dp", "pp", "tp"])
        mesh_sizes: List of sizes for each dimension (e.g., [4, 2, 8])
        rank_coords: Tuple of coordinates for this rank (e.g., (2, 1, 5))
    """

    def __init__(self, mesh_dim_names, mesh_sizes, rank_coords):
        self.mesh_dim_names = mesh_dim_names
        self.mesh_sizes = dict(zip(mesh_dim_names, mesh_sizes))
        self.rank_coords = dict(zip(mesh_dim_names, rank_coords))
        # Calculate world_size as product of all mesh sizes
        self.world_size = 1
        for size in mesh_sizes:
            self.world_size *= size

        # Add individual parallelism degree attributes to match real ParallelDims interface
        self.pp = self.mesh_sizes.get("pp", 1)
        self.tp = self.mesh_sizes.get("tp", 1)
        self.cp = self.mesh_sizes.get("cp", 1)
        self.dp_replicate = self.mesh_sizes.get("dp_replicate", 1)
        self.dp_shard = self.mesh_sizes.get("dp_shard", 1)
        self.ep = self.mesh_sizes.get("ep", 1)
        self.etp = self.mesh_sizes.get("etp", 1)

        # For backward compatibility with 'dp' dimension name
        if "dp" in self.mesh_sizes:
            self.dp_replicate = self.mesh_sizes["dp"]

        # Create a world_mesh mock
        self.world_mesh = MagicMock()
        self.world_mesh.device_type = "cpu"

    def get_mesh(self, key):
        """Return a submesh for the given dimension."""
        if isinstance(key, str):
            # Single dimension
            if key not in self.mesh_dim_names:
                return None
            submesh = MagicMock()
            submesh.get_local_rank.return_value = self.rank_coords[key]
            submesh.size.return_value = self.mesh_sizes[key]
            submesh.get_coordinate.return_value = self.rank_coords[key]
            submesh.device_type = "cpu"
            return submesh
        elif isinstance(key, list):
            # Multiple dimensions - check if all exist
            if not all(dim in self.mesh_dim_names for dim in key):
                return None
            submesh = MagicMock()
            # For multiple dimensions, get_coordinate should return None
            # since we're not testing this path
            submesh.get_coordinate.return_value = None
            submesh.device_type = "cpu"
            return submesh
        else:
            return None

    def get_optional_mesh(self, key):
        """Return a submesh for the given dimension, or None if not available.

        This is the same as get_mesh() for FakeParallelDims since get_mesh()
        already returns None for unavailable meshes.
        """
        return self.get_mesh(key)

    def get_all_meshes(self):
        """Return a dict of all meshes."""
        return {dim: self.get_mesh(dim) for dim in self.mesh_dim_names}

    def __getitem__(self, key):
        """Return a submesh for the given dimension(s) - for backward compatibility."""
        return self.get_mesh(key)

    def get_coordinate(self):
        """Return the coordinate tuple for this rank."""
        return tuple(self.rank_coords[dim] for dim in self.mesh_dim_names)


class TestSetDeterminismWithFakeMesh(unittest.TestCase):
    """Test set_determinism with fake mesh to verify seed uniqueness."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def tearDown(self):
        """Clean up after tests."""
        torch.use_deterministic_algorithms(False)
        if "PYTHONHASHSEED" in os.environ:
            del os.environ["PYTHONHASHSEED"]
        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]

    @patch("torch.distributed.distributed_c10d.get_world_size")
    @patch("torch.distributed.distributed_c10d.get_rank")
    def test_seed_uniqueness_2d_mesh(self, mock_get_rank, mock_get_world_size):
        """Test that different PP ranks get unique seeds, same DP ranks share seeds."""
        mock_get_world_size.return_value = 8  # 4 * 2

        mesh_dim_names = ["dp", "pp"]
        mesh_sizes = [4, 2]
        base_seed = 1000

        seeds_by_coord = {}

        # Test all possible rank coordinates
        for dp_rank in range(mesh_sizes[0]):
            for pp_rank in range(mesh_sizes[1]):
                mock_get_rank.return_value = dp_rank * mesh_sizes[1] + pp_rank

                # Create fake mesh for this rank
                rank_coords = (dp_rank, pp_rank)
                fake_mesh = FakeParallelDims(mesh_dim_names, mesh_sizes, rank_coords)

                # Call set_determinism with distinct seeds only on PP dimension
                debug_config = DebugConfig(seed=base_seed, deterministic=False)
                set_determinism(
                    parallel_dims=fake_mesh,
                    device=self.device,
                    debug_config=debug_config,
                    distinct_seed_mesh_dims=["pp"],
                )

                # Capture the seed that was set
                rng_state = torch.get_rng_state()
                actual_seed = rng_state[:8].view(torch.int64).item()

                # Store for verification
                coord_key = (dp_rank, pp_rank)
                seeds_by_coord[coord_key] = actual_seed

        # Verify that coordinates with same PP but different DP have same seed
        for pp_rank in range(mesh_sizes[1]):
            # All DP ranks should have same seed for this PP rank
            seeds_for_this_pp = [
                seeds_by_coord[(dp_rank, pp_rank)] for dp_rank in range(mesh_sizes[0])
            ]
            self.assertEqual(
                len(set(seeds_for_this_pp)),
                1,
                f"Different DP ranks at pp={pp_rank} should have same seed, "
                f"got {seeds_for_this_pp}",
            )

        # Verify that different PP ranks have different seeds
        unique_pp_seeds = set()
        for pp_rank in range(mesh_sizes[1]):
            seed = seeds_by_coord[(0, pp_rank)]  # Just check first DP rank
            self.assertNotIn(seed, unique_pp_seeds, f"Duplicate seed for pp={pp_rank}")
            unique_pp_seeds.add(seed)

        self.assertEqual(
            len(unique_pp_seeds),
            mesh_sizes[1],
            f"Expected {mesh_sizes[1]} unique seeds for PP dimension",
        )

    @patch("torch.distributed.distributed_c10d.get_world_size")
    @patch("torch.distributed.distributed_c10d.get_rank")
    def test_seed_uniqueness_3d_mesh(self, mock_get_rank, mock_get_world_size):
        """Test that different dp_shard and dp_replicate get unique seeds, TP shares seeds."""
        mesh_dim_names = ["dp_shard", "dp_replicate", "tp"]
        mesh_sizes = [3, 2, 4]
        mock_get_world_size.return_value = 3 * 2 * 4
        base_seed = 2000

        seeds_by_coord = {}

        # Test all possible rank coordinates
        for dp_shard_rank in range(mesh_sizes[0]):
            for dp_replicate_rank in range(mesh_sizes[1]):
                for tp_rank in range(mesh_sizes[2]):
                    global_rank = (
                        dp_shard_rank * (mesh_sizes[1] * mesh_sizes[2])
                        + dp_replicate_rank * mesh_sizes[2]
                        + tp_rank
                    )
                    mock_get_rank.return_value = global_rank

                    # Create fake mesh for this rank
                    rank_coords = (dp_shard_rank, dp_replicate_rank, tp_rank)
                    fake_mesh = FakeParallelDims(
                        mesh_dim_names, mesh_sizes, rank_coords
                    )

                    # Call set_determinism with distinct seeds on dp_shard and dp_replicate only
                    debug_config = DebugConfig(seed=base_seed, deterministic=False)
                    set_determinism(
                        parallel_dims=fake_mesh,
                        device=self.device,
                        debug_config=debug_config,
                        distinct_seed_mesh_dims=["dp_shard", "dp_replicate"],
                    )

                    # Capture the seed that was set
                    rng_state = torch.get_rng_state()
                    actual_seed = rng_state[:8].view(torch.int64).item()

                    # Store for verification
                    coord_key = (dp_shard_rank, dp_replicate_rank, tp_rank)
                    seeds_by_coord[coord_key] = actual_seed

        # Verify that coordinates with same (dp_shard, dp_replicate) but different TP have same seed
        for dp_shard_rank in range(mesh_sizes[0]):
            for dp_replicate_rank in range(mesh_sizes[1]):
                # All TP ranks should have same seed for this (dp_shard, dp_replicate)
                seeds_for_this_dp = [
                    seeds_by_coord[(dp_shard_rank, dp_replicate_rank, tp_rank)]
                    for tp_rank in range(mesh_sizes[2])
                ]
                self.assertEqual(
                    len(set(seeds_for_this_dp)),
                    1,
                    f"Different TP ranks at (dp_shard={dp_shard_rank}, dp_replicate={dp_replicate_rank}) "
                    f"should have same seed, got {seeds_for_this_dp}",
                )

        # Verify that different (dp_shard, dp_replicate) combinations have different seeds
        unique_dp_seeds = set()
        for dp_shard_rank in range(mesh_sizes[0]):
            for dp_replicate_rank in range(mesh_sizes[1]):
                seed = seeds_by_coord[
                    (dp_shard_rank, dp_replicate_rank, 0)
                ]  # Just check first TP rank
                self.assertNotIn(
                    seed,
                    unique_dp_seeds,
                    f"Duplicate seed for (dp_shard={dp_shard_rank}, dp_replicate={dp_replicate_rank})",
                )
                unique_dp_seeds.add(seed)

        self.assertEqual(
            len(unique_dp_seeds),
            mesh_sizes[0] * mesh_sizes[1],
            f"Expected {mesh_sizes[0] * mesh_sizes[1]} unique seeds for (dp_shard, dp_replicate) combinations",
        )

    @patch("torch.distributed.distributed_c10d.get_world_size")
    @patch("torch.distributed.distributed_c10d.get_rank")
    def test_set_determinism_single_gpu(self, mock_get_rank, mock_get_world_size):
        """Test set_determinism for single GPU (empty mesh)"""
        mock_get_world_size.return_value = 1
        mock_get_rank.return_value = 0

        base_seed = 42

        fake_mesh = MagicMock()
        fake_mesh.world_size = 1
        fake_mesh.world_mesh = MagicMock()
        fake_mesh.get_mesh.return_value = None
        fake_mesh.get_optional_mesh.return_value = None
        fake_mesh.get_all_meshes.return_value = {}

        debug_config = DebugConfig(seed=base_seed, deterministic=False)
        set_determinism(
            parallel_dims=fake_mesh,
            device=self.device,
            debug_config=debug_config,
            distinct_seed_mesh_dims=["pp"],
        )


if __name__ == "__main__":
    unittest.main()
