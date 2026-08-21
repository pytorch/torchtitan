# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_spmd_meshes, spmd_distribute_tensor
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.state_dict_adapter import PlainToDTensorStateDictAdapter


class DeepSeekV3StateDictAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls._owns_process_group = not dist.is_initialized()
        if cls._owns_process_group:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{cls._temporary_directory.name}/rendezvous",
                rank=0,
                world_size=1,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._owns_process_group:
            dist.destroy_process_group()
        cls._temporary_directory.cleanup()

    def test_to_hf_handles_replicated_grouped_experts(self) -> None:
        config = deepseekv3_configs["debugmodel"](
            attn_backend="flex",
            moe_comm_backend="standard",
        )
        adapter = DeepSeekV3StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh(
            "cpu",
            (1, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        local_weight = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
        grouped_expert_weight = DTensor.from_local(
            local_weight,
            mesh,
            (Replicate(), Shard(0)),
            run_check=False,
        )

        hf_state_dict = adapter.to_hf(
            {"layers.1.moe.routed_experts.inner_experts.w1_EFD": grouped_expert_weight}
        )

        expected_keys = {
            f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"
            for expert in range(8)
        }
        self.assertEqual(set(hf_state_dict), expected_keys)
        for expert in range(8):
            key = f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"
            self.assertIsInstance(hf_state_dict[key], DTensor)
            torch.testing.assert_close(
                hf_state_dict[key].to_local(),
                local_weight[expert],
            )


class PlainToDTensorStateDictAdapterTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_spmd_type_partition_order(self) -> None:
        global_weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "tp"))
        set_spmd_meshes(
            dense_mesh=mesh,
            dense_storage_mesh=mesh,
            sparse_mesh=None,
            sparse_storage_mesh=None,
        )
        dp_axis = spmd.MeshAxis.of(mesh.get_group("dp_shard"))
        tp_axis = spmd.MeshAxis.of(mesh.get_group("tp"))
        layout = spmd.SpmdType(
            {dp_axis: spmd.V, tp_axis: spmd.V},
            spmd.PartitionSpec((tp_axis, dp_axis), None),
        )
        local_weight = spmd_distribute_tensor(global_weight, mesh, layout)
        adapter = PlainToDTensorStateDictAdapter({"weight": layout})
        state_dict = adapter.convert_save_state_dict({"weight": local_weight})

        torch.testing.assert_close(state_dict["weight"].full_tensor(), global_weight)
        torch.testing.assert_close(
            adapter.convert_load_state_dict(state_dict)["weight"], local_weight
        )
