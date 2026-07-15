# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils._debug_mode import DebugMode
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.parallel_dims import (
    MeshAxisName,
    ParallelDims,
    SpmdLayout,
    unfold_dp_axes,
)
from torchtitan.distributed.spmd_types import (
    spmd_distribute_tensor,
    spmd_layout_to_dtensor_placements,
    spmd_validate_redistributions,
    set_current_spmd_mesh,
)
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.llama3 import model_registry
from torchtitan.protocols.sharding import PerAxisRedistribution, ShardingConfig


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
            world_size=8,
        )
        self.assertEqual(parallel_dims.dp_replicate, 2)
        self.assertEqual(parallel_dims.dp_shard, 2)
        self.assertEqual(parallel_dims.cp, 1)
        self.assertEqual(parallel_dims.tp, 2)
        self.assertEqual(parallel_dims.pp, 1)
        self.assertEqual(parallel_dims.ep, 1)
        self.assertEqual(parallel_dims.world_size, 8)

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_from_config(self):
        """Test constructing ParallelDims from a ParallelismConfig."""
        config = ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=-1,
            context_parallel_degree=1,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
        )
        parallel_dims = ParallelDims.from_config(config, world_size=8)
        self.assertEqual(parallel_dims.dp_replicate, 2)
        self.assertEqual(parallel_dims.dp_shard, 2)  # auto-calculated: 8 / (2*1*2*1)
        self.assertEqual(parallel_dims.cp, 1)
        self.assertEqual(parallel_dims.tp, 2)
        self.assertEqual(parallel_dims.pp, 1)
        self.assertEqual(parallel_dims.ep, 1)
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
                world_size=10,  # Invalid: 2*2*1*2*1 = 8, not 10
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
            world_size=8,
        )
        self.assertTrue(parallel_dims.dp_enabled)
        self.assertTrue(parallel_dims.dp_replicate_enabled)
        self.assertTrue(parallel_dims.dp_shard_enabled)
        self.assertFalse(parallel_dims.cp_enabled)
        self.assertTrue(parallel_dims.tp_enabled)
        self.assertFalse(parallel_dims.pp_enabled)
        self.assertFalse(parallel_dims.ep_enabled)
        self.assertTrue(parallel_dims.fsdp_enabled)

        # Test with CP enabled
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
        )
        self.assertFalse(parallel_dims.dp_enabled)
        self.assertTrue(parallel_dims.cp_enabled)
        self.assertTrue(parallel_dims.dp_cp_enabled)
        self.assertTrue(parallel_dims.fsdp_enabled)

        # Test with EP enabled (EP must not contribute to world_size)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=2,
        )
        self.assertTrue(parallel_dims.ep_enabled)

        # Test with PP enabled
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=2,
            ep=1,
            world_size=2,
        )
        self.assertTrue(parallel_dims.pp_enabled)

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
            world_size=16,
        )
        # Should be tp * (cp * 2) = 4 * 4 = 16
        self.assertEqual(parallel_dims.seq_len_divisor, 16)


class TestSpmdLayout(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def test_converts_partition_spec_to_dtensor_shard(self):
        """PartitionSpec refines V into concrete DTensor Shard placement."""
        layout = SpmdLayout(
            {MeshAxisName.TP: spmd.V},
            partition_spec=spmd.PartitionSpec(MeshAxisName.TP),
        )

        self.assertEqual(layout.per_axis_spmd_types(), {MeshAxisName.TP: spmd.S(0)})
        self.assertEqual(
            spmd_layout_to_dtensor_placements(layout),
            {MeshAxisName.TP: Shard(0)},
        )

    def test_seq_parallel_activation_per_axis_spmd_types(self):
        """PartitionSpec can map multiple mesh axes to one tensor dim."""
        layout = SpmdLayout(
            {
                MeshAxisName.DP: spmd.V,
                MeshAxisName.CP: spmd.V,
                MeshAxisName.TP: spmd.V,
            },
            partition_spec=(
                MeshAxisName.DP,
                (MeshAxisName.CP, MeshAxisName.TP),
                None,
            ),
        )

        self.assertEqual(
            layout.per_axis_spmd_types(),
            {
                MeshAxisName.DP: spmd.S(0),
                MeshAxisName.CP: spmd.S(1),
                MeshAxisName.TP: spmd.S(1),
            },
        )

    def test_unfold_dp_axes(self):
        """Logical DP expands only when resolving concrete mesh axes."""
        self.assertEqual(
            unfold_dp_axes([MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP]),
            ["dp_replicate", "dp_shard", "cp", "tp"],
        )

    def test_rejects_non_innermost_axis_redistribution(self):
        """For ``(DP, CP)``, CP can be unsharded directly but DP cannot."""
        src = SpmdLayout(
            {
                MeshAxisName.DP: spmd.V,
                MeshAxisName.CP: spmd.V,
            },
            partition_spec=spmd.PartitionSpec(
                (MeshAxisName.DP, MeshAxisName.CP), None
            ),
        )

        spmd_validate_redistributions(
            ShardingConfig(
                out_src_shardings=src,
                out_redist=PerAxisRedistribution.Config(
                    axis=MeshAxisName.CP,
                    src=spmd.S(0),
                    dst=spmd.R,
                ),
            )
        )

        with self.assertRaises(ValueError) as cm:
            spmd_validate_redistributions(
                ShardingConfig(
                    out_src_shardings=src,
                    out_redist=PerAxisRedistribution.Config(
                        axis=MeshAxisName.DP,
                        src=spmd.S(0),
                        dst=spmd.R,
                    ),
                )
            )
        self.assertExpectedInline(
            str(cm.exception),
            """output: unsharding axis 'dp' from tensor dim 0 is only valid for the innermost mesh axis; source shard order is ('dp', 'cp').""",
        )

    @with_comms
    def test_partition_spec_order_controls_state_shard(self):
        """Test spmd_distribute_tensor follows PartitionSpec order.

        ``(DP, CP)`` and ``(CP, DP)`` both shard dim 0 across the same 2x2
        mesh, but assign different global slices to ranks. This verifies local
        state sharding preserves that ordering instead of relying on unordered
        per-axis shard types.
        """
        mesh = init_device_mesh(self.device_type, (2, 2), mesh_dim_names=("dp", "cp"))
        global_weight = torch.arange(
            8, dtype=torch.float32, device=self.device_type
        ).reshape(8, 1)

        for axis_order in (
            (MeshAxisName.DP, MeshAxisName.CP),
            (MeshAxisName.CP, MeshAxisName.DP),
        ):
            with self.subTest(axis_order=axis_order):
                layout = SpmdLayout(
                    {
                        MeshAxisName.DP: spmd.V,
                        MeshAxisName.CP: spmd.V,
                    },
                    partition_spec=spmd.PartitionSpec(axis_order, None),
                )

                local_weight = spmd_distribute_tensor(
                    global_weight.clone(), mesh, layout
                )

                axis_ranks = {
                    MeshAxisName.DP: mesh.get_local_rank("dp"),
                    MeshAxisName.CP: mesh.get_local_rank("cp"),
                }
                axis_sizes = {
                    MeshAxisName.DP: mesh.size(0),
                    MeshAxisName.CP: mesh.size(1),
                }
                shard_idx = 0
                for axis_name in axis_order:
                    shard_idx = (
                        shard_idx * axis_sizes[axis_name] + axis_ranks[axis_name]
                    )
                local_rows = global_weight.shape[0] // self.world_size
                expected = global_weight.narrow(0, shard_idx * local_rows, local_rows)
                torch.testing.assert_close(local_weight, expected)

    def test_per_axis_redistribution_reduce_scatter_config(self):
        """PerAxisRedistribution forwards collective and dtype config."""
        if dist.is_initialized():
            self.skipTest("requires no pre-existing process group")

        dist.init_process_group("fake", rank=0, world_size=2)
        try:
            mesh = init_device_mesh(
                "cpu",
                (2,),
                mesh_dim_names=("tp",),
            )
            x = torch.ones(4, 4, dtype=torch.float16, requires_grad=True)
            redist = PerAxisRedistribution.Config(
                axis=MeshAxisName.TP,
                src=spmd.P,
                dst=spmd.S(1),
                fwd_op_dtype=torch.float32,
                fwd_out_dtype=torch.bfloat16,
                bwd_op_dtype=torch.float32,
                bwd_out_dtype=torch.float16,
            ).build()

            set_spmd_backend("spmd_types")
            try:
                with set_current_spmd_mesh(mesh):
                    spmd.assert_type(x, {MeshAxisName.TP: spmd.P})
                    with DebugMode(record_torchfunction=False) as debug_mode:
                        result = redist(x)
                        result.float().sum().backward()
            finally:
                set_spmd_backend("default")
        finally:
            dist.destroy_process_group()

        self.assertExpectedInline(
            debug_mode.debug_string(),
            """\
    aten::_to_copy(t: f16[4, 4], dtype=torch.float32)  ->  t: f32[4, 4]
    aten::permute(t: f32[4, 4], [1, 0])  ->  t: f32[4, 4]
    aten::clone(t: f32[4, 4], memory_format=torch.contiguous_format)  ->  t: f32[4, 4]
    aten::new_empty(t: f32[4, 4], [2, 4], pin_memory=False)  ->  t: f32[2, 4]
    c10d::_reduce_scatter_base_(t: f32[2, 4], t: f32[4, 4], ScriptObject <__torch__.torch.classes.c10d.ProcessGroup>, ScriptObject <__torch__.torch.classes.c10d.ReduceOp>, False)  ->  ('t: f32[2, 4]', 'ScriptObject <__torch__.torch.classes.c10d.Work>')
    aten::permute(t: f32[2, 4], [1, 0])  ->  t: f32[4, 2]
    aten::_to_copy(t: f32[4, 2], dtype=torch.bfloat16)  ->  t: bf16[4, 2]
    aten::_to_copy(t: bf16[4, 2], dtype=torch.float32)  ->  t: f32[4, 2]
    aten::sum(t: f32[4, 2])  ->  t: f32[]
    aten::ones_like(t: f32[], pin_memory=False, memory_format=torch.preserve_format)  ->  t: f32[]
    aten::expand(t: f32[], [4, 2])  ->  t: f32[4, 2]
    aten::_to_copy(t: f32[4, 2], dtype=torch.bfloat16, layout=torch.strided, device=cpu)  ->  t: bf16[4, 2]
    aten::_to_copy(t: bf16[4, 2], dtype=torch.float32)  ->  t: f32[4, 2]
    aten::empty.memory_format([8, 2], dtype=torch.float32, device=cpu, pin_memory=False)  ->  t: f32[8, 2]
    c10d::_allgather_base_(t: f32[8, 2], t: f32[4, 2], ScriptObject <__torch__.torch.classes.c10d.ProcessGroup>, False)  ->  ('t: f32[8, 2]', 'ScriptObject <__torch__.torch.classes.c10d.Work>')
    aten::split.Tensor(t: f32[8, 2], 4)  ->  ['t: f32[4, 2]', 't: f32[4, 2]']
    aten::cat(['t: f32[4, 2]', 't: f32[4, 2]'], 1)  ->  t: f32[4, 4]
    aten::_to_copy(t: f32[4, 4], dtype=torch.float16)  ->  t: f16[4, 4]
    aten::detach(t: f16[4, 4])  ->  t: f16[4, 4]""",
        )
        self.assertEqual(result.shape, torch.Size([4, 2]))
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.dtype, torch.float16)

    def test_per_axis_redistribution_passes_dtype_options(self):
        x = torch.ones(2, 2)
        mesh = unittest.mock.Mock()
        mesh.mesh_dim_names = ("tp",)
        mesh.size.return_value = 2
        mesh.get_group.return_value = "tp_group"
        redist = PerAxisRedistribution.Config(
            axis=MeshAxisName.TP,
            src=spmd.P,
            dst=spmd.S(1),
            fwd_op_dtype=torch.float32,
            fwd_out_dtype=torch.bfloat16,
            bwd_op_dtype=torch.float16,
            bwd_out_dtype=torch.float32,
        ).build()

        with (
            patch("torchtitan.protocols.sharding.current_spmd_mesh", return_value=mesh),
            patch("torchtitan.protocols.sharding.spmd_mesh_size", return_value=2),
            patch("torchtitan.protocols.sharding.spmd.redistribute") as mock,
        ):
            mock.return_value = x
            result = redist(x)

        self.assertIs(result, x)
        mock.assert_called_once_with(
            x,
            "tp_group",
            src=spmd.P,
            dst=spmd.S(1),
            op_dtype=torch.float32,
            out_dtype=torch.bfloat16,
            backward_options={
                "op_dtype": torch.float16,
                "out_dtype": torch.float32,
            },
        )


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
            world_size=1,
        )
        # Don't call build_mesh explicitly
        self.assertEqual(len(parallel_dims._single_axis_meshes), 0)

        # get_optional_mesh should trigger build_mesh
        # Result is None because tp has size 1, but build_mesh should have been called
        self.assertIsNone(parallel_dims.get_optional_mesh("tp"))
        self.assertGreater(len(parallel_dims._single_axis_meshes), 0)

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
            world_size=1,
        )

        # Test mesh building
        world_mesh = parallel_dims.build_mesh()
        self.assertIsNotNone(world_mesh)
        self.assertEqual(world_mesh.size(), 1)

        # Verify all expected meshes are created
        self.assertIsNotNone(parallel_dims._single_axis_meshes)
        self.assertIn("pp", parallel_dims._single_axis_meshes)
        self.assertIn("batch", parallel_dims._single_axis_meshes)
        self.assertIn("loss", parallel_dims._single_axis_meshes)
        self.assertIn("dp_replicate", parallel_dims._single_axis_meshes)
        self.assertIn("fsdp", parallel_dims._single_axis_meshes)
        self.assertIn("cp", parallel_dims._single_axis_meshes)
        self.assertIn("tp", parallel_dims._single_axis_meshes)

        # Validate 1D mesh sizes - all should be 1 for single rank
        self.assertEqual(parallel_dims._single_axis_meshes["dp_replicate"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["fsdp"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["tp"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["batch"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["loss"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["pp"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["cp"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["ep"].size(), 1)
        self.assertEqual(parallel_dims._single_axis_meshes["efsdp"].size(), 1)

        # Validate 2D mesh shapes
        dp_replicate_fsdp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "fsdp"]
        )
        self.assertIsNone(dp_replicate_fsdp_mesh)  # Both dimensions have size 1
        dp_replicate_efsdp_mesh = parallel_dims.get_optional_mesh(
            ["dp_replicate", "efsdp"]
        )
        self.assertIsNone(dp_replicate_efsdp_mesh)  # Both dimensions have size 1

        # Test get_optional_mesh returns None when all dimensions have size 1
        self.assertIsNone(parallel_dims.get_optional_mesh("tp"))
        self.assertIsNone(parallel_dims.get_optional_mesh("dp_replicate"))
        self.assertIsNone(parallel_dims.get_optional_mesh("pp"))
        self.assertIsNone(parallel_dims.get_optional_mesh("cp"))

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
        # EP enabled (valid) - world_size = dp_replicate * dp_shard * cp * tp * pp
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=2,  # 1 * 2 * 1 * 1 * 1 = 2
        )
        self.assertTrue(parallel_dims.ep_enabled)

        # Test with larger configuration
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=3,
            world_size=4,  # 2 * 2 * 1 * 1 * 1 = 4
        )
        self.assertTrue(parallel_dims.ep_enabled)
        self.assertTrue(parallel_dims.dp_replicate_enabled)
        self.assertTrue(parallel_dims.dp_shard_enabled)


class TestSpmdMeshesLegacy(DTensorTestBase):
    """spmd_meshes() under non-full_dtensor."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_legacy_spmd_meshes(self):
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            pd = ParallelDims(
                dp_replicate=2,
                dp_shard=2,
                cp=1,
                tp=2,
                pp=1,
                ep=1,
                world_size=8,
                spmd_backend="default",
            )
            pd.build_mesh()

            # Legacy mode pre-flattens dp_shard+cp into 'fsdp'; dp_shard
            # never appears as a single-axis mesh, so must not appear in any
            # SPMD mesh either.
            meshes = pd.spmd_meshes()
            flat = {axis for m in meshes for axis in (m.mesh_dim_names or ())}
            self.assertNotIn("dp_shard", flat)
            # Dense mesh names ``fsdp`` (the storage axis) instead of
            # ``dp_shard`` / ``cp`` under legacy.
            dense = next(m for m in meshes if "tp" in (m.mesh_dim_names or ()))
            self.assertEqual(set(dense.mesh_dim_names), {"dp_replicate", "fsdp", "tp"})


class TestSpmdMeshesFullDTensor(DTensorTestBase):
    """spmd_meshes() under full_dtensor."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_full_dtensor_spmd_meshes(self):
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            pd = ParallelDims(
                dp_replicate=2,
                dp_shard=2,
                cp=1,
                tp=2,
                pp=1,
                ep=1,
                world_size=8,
                spmd_backend="full_dtensor",
            )
            pd.build_mesh()

            # Dense mesh keeps dp_shard separate (no 'fsdp' flatten), in
            # canonical outer-to-inner order; cp filtered out (disabled).
            meshes = pd.spmd_meshes()
            dense = next(m for m in meshes if "tp" in (m.mesh_dim_names or ()))
            self.assertEqual(dense.mesh_dim_names, ("dp_replicate", "dp_shard", "tp"))


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
                world_size=8,
            )

            # Test mesh building
            world_mesh = parallel_dims.build_mesh()
            self.assertIsNotNone(world_mesh)
            self.assertEqual(world_mesh.size(), 8)

            # Verify all expected meshes are created
            self.assertIsNotNone(parallel_dims._single_axis_meshes)
            self.assertIn("pp", parallel_dims._single_axis_meshes)
            self.assertIn("batch", parallel_dims._single_axis_meshes)
            self.assertIn("loss", parallel_dims._single_axis_meshes)
            self.assertIn("dp_replicate", parallel_dims._single_axis_meshes)
            self.assertIn("fsdp", parallel_dims._single_axis_meshes)
            self.assertIn("cp", parallel_dims._single_axis_meshes)
            self.assertIn("tp", parallel_dims._single_axis_meshes)
            self.assertIn("ep", parallel_dims._single_axis_meshes)
            self.assertIn("efsdp", parallel_dims._single_axis_meshes)

            # Validate 1D mesh sizes match parallelism configuration
            self.assertEqual(parallel_dims._single_axis_meshes["pp"].size(), 1)
            self.assertEqual(
                parallel_dims._single_axis_meshes["batch"].size(), 4
            )  # dp_replicate * dp_shard = 2 * 2
            self.assertEqual(
                parallel_dims._single_axis_meshes["loss"].size(), 4
            )  # dp_replicate * dp_shard * cp = 2 * 2 * 1
            self.assertEqual(
                parallel_dims._single_axis_meshes["dp_replicate"].size(), 2
            )
            self.assertEqual(
                parallel_dims._single_axis_meshes["fsdp"].size(), 2
            )  # dp_shard * cp = 2 * 1
            self.assertEqual(parallel_dims._single_axis_meshes["cp"].size(), 1)
            self.assertEqual(parallel_dims._single_axis_meshes["tp"].size(), 2)
            self.assertEqual(parallel_dims._single_axis_meshes["ep"].size(), 1)
            self.assertEqual(
                parallel_dims._single_axis_meshes["efsdp"].size(), 4
            )  # fsdp * tp / ep = 2 * 2 / 1 = 4

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
            # Should not include: pp, cp, ep (all with size = 1)
            self.assertNotIn("pp", one_d_meshes)
            self.assertNotIn("cp", one_d_meshes)
            self.assertNotIn("ep", one_d_meshes)

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
                parallel_dims.non_data_parallel_size, 2
            )  # cp * tp * pp = 1 * 2 * 1
            self.assertEqual(
                parallel_dims.seq_len_divisor, 4
            )  # tp * (cp * 2) = 2 * (1 * 2) = 2 * 2


class TestSingleGPUMixedPrecisionFSDP(DTensorTestBase):
    """Verify apply_fsdp on Llama3 debugmodel matches single-device reference.

    Tests that torchtitan's apply_fsdp with MixedPrecisionPolicy at degree 1
    produces numerically identical results to a reference model with manually
    cast bf16 parameters, following the pattern in
    pytorch/test/distributed/_composable/fsdp/test_fully_shard_mixed_precision.py.

    See https://github.com/pytorch/torchtitan/issues/2886
    """

    @property
    def world_size(self):
        return 1

    @with_comms
    def test_apply_fsdp_mixed_precision_single_gpu(self):
        """apply_fsdp with bf16 on Llama3 debugmodel matches manual bf16 reference on a single GPU."""
        torch.manual_seed(42)

        model_spec = model_registry("debugmodel")
        model_config = model_spec.model

        # This test runs forward+backward on self.device_type (CPU in the
        # CPU CI job). The default FlexAttention backend has no CPU backward,
        # so use ScaledDotProductAttention, which runs on CPU without a mask.
        from torchtitan.models.common.attention import ScaledDotProductAttention

        for layer in model_config.layers:
            layer.attention.inner_attention = ScaledDotProductAttention.Config()

        with torch.device("meta"):
            model = model_config.build()
        model.to_empty(device=self.device_type)
        with torch.no_grad():
            model.init_states(buffer_device=None)

        ref_model = copy.deepcopy(model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-4)

        dp_mesh = init_device_mesh(self.device_type, (1,))
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Cast only parameters to bf16, matching MixedPrecisionPolicy behavior
        # (buffers like freqs_cis stay fp32)
        ref_model_bf16 = copy.deepcopy(ref_model)
        for p in ref_model_bf16.parameters():
            p.data = p.data.to(torch.bfloat16)

        tokens = torch.randint(
            0, model_config.vocab_size, (2, 32), device=self.device_type
        )
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            loss = model(tokens).sum()
            loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model_bf16(tokens).sum()
            ref_loss.backward()
            for p_fp32, p_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                p_fp32.grad = p_bf16.grad.to(p_fp32.dtype)
                p_bf16.grad = None
            ref_optim.step()
            for p_fp32, p_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                p_bf16.detach().copy_(p_fp32)

            # Validates that apply_fsdp with param_dtype=bf16 matches the manual
            # bf16 reference. Would fail if mp_policy used param_dtype=fp32 instead,
            # since the ref model runs forward/backward in bf16.
            self.assertEqual(loss, ref_loss)


if __name__ == "__main__":
    unittest.main()
