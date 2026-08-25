# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.checkpointer.state_dict_adapter import (
    PaddedDTensorStateDictAdapter,
)
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter


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


class PaddedDTensorStateDictAdapterTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_dcp_round_trip_across_tp_degrees(self) -> None:
        device_type = self.device_type

        class OptimizerState:
            def __init__(self, state_dict):
                self.state = state_dict

            def state_dict(self):
                return self.state

            def load_state_dict(self, state_dict):
                self.state = state_dict

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(6, device=device_type))
                self._spmd_logical_state_shapes = {"weight": (5,)}

        model = Model()
        checkpoint_id = f"{self.file_name}_checkpoint"

        save_mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("dp_shard", "tp"),
        )
        save_placements = (_StridedShard(0, sf=2), Shard(0))
        padded = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 1000.0],
            device=self.device_type,
        )
        local_shape, global_offset = compute_local_shape_and_global_offset(
            padded.shape,
            save_mesh,
            save_placements,
        )
        local = padded.narrow(0, global_offset[0], local_shape[0]).clone()
        save_weight = DTensor.from_local(
            local,
            save_mesh,
            save_placements,
            shape=padded.shape,
            stride=padded.stride(),
            run_check=False,
        )
        save_optimizer_value = DTensor.from_local(
            local + 10,
            save_mesh,
            save_placements,
            shape=padded.shape,
            stride=padded.stride(),
            run_check=False,
        )
        save_optimizer = OptimizerState(
            {
                "state.weight.exp_avg": save_optimizer_value,
                "state.weight.step": torch.tensor(1.0, device=self.device_type),
            }
        )
        save_adapter = PaddedDTensorStateDictAdapter([model], save_optimizer)
        dcp.save(
            save_adapter.to_dcp({"weight": save_weight, "optimizer": save_optimizer}),
            checkpoint_id=checkpoint_id,
        )

        load_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("tp",),
        )
        target_local = torch.full((2,), -1.0, device=self.device_type)
        load_weight = DTensor.from_local(
            target_local,
            load_mesh,
            (Shard(0),),
            shape=torch.Size((8,)),
            stride=(1,),
            run_check=False,
        )
        load_optimizer_value = DTensor.from_local(
            torch.full((2,), -1.0, device=self.device_type),
            load_mesh,
            (Shard(0),),
            shape=torch.Size((8,)),
            stride=(1,),
            run_check=False,
        )
        load_optimizer = OptimizerState(
            {
                "state.weight.exp_avg": load_optimizer_value,
                "state.weight.step": torch.tensor(0.0, device=self.device_type),
            }
        )
        load_adapter = PaddedDTensorStateDictAdapter([model], load_optimizer)
        load_state = load_adapter.to_dcp(
            {"weight": load_weight, "optimizer": load_optimizer}
        )
        dcp.load(load_state, checkpoint_id=checkpoint_id)
        restored = load_adapter.from_dcp(load_state)

        self.assertIs(restored["weight"], load_weight)
        expected = {
            0: torch.tensor([0.0, 1.0], device=self.device_type),
            1: torch.tensor([2.0, 3.0], device=self.device_type),
            2: torch.tensor([4.0, 0.0], device=self.device_type),
            3: torch.tensor([0.0, 0.0], device=self.device_type),
        }
        torch.testing.assert_close(load_weight.to_local(), expected[self.rank])
        expected_optimizer = {
            0: torch.tensor([10.0, 11.0], device=self.device_type),
            1: torch.tensor([12.0, 13.0], device=self.device_type),
            2: torch.tensor([14.0, 0.0], device=self.device_type),
            3: torch.tensor([0.0, 0.0], device=self.device_type),
        }
        torch.testing.assert_close(
            load_optimizer.state["state.weight.exp_avg"].to_local(),
            expected_optimizer[self.rank],
        )
        self.assertEqual(load_optimizer.state["state.weight.step"].item(), 1.0)
