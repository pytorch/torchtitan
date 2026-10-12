# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import ParallelismConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, resolve_fsdp_mesh
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.lora import _LoRAMixin, merge_lora_state_dict
from torchtitan.models.llama3 import build_model_config


class TestMergeLoRAStateDictFSDPTensorParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_merged_weights_match_the_gathered_merge(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=True,
        )
        parallelism_context.build_mesh()
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        config = LoRATransform(rank=8, alpha=16.0).transform(
            build_model_config(
                "debugmodel",
                parallelism=ParallelismConfig(
                    tensor_parallel_degree=2,
                    enable_sequence_parallel=True,
                ),
            )
        )
        with torch.device("meta"):
            model = config.build()
        model._parallelize(parallelism_context)
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            dp_mesh_dims=dp_mesh_dims,
        )
        model.to_empty(device=self.device_type)
        torch.manual_seed(0)
        model.init_states()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_b." in name:
                    param.normal_()

        state_dict = model.state_dict()
        merged = merge_lora_state_dict(model)

        checked = 0
        for name, module in model.named_modules():
            if not isinstance(module, _LoRAMixin):
                continue
            base = state_dict[f"{name}.weight"]
            lora_a = module.lora_a.weight.full_tensor().float()
            lora_b = module.lora_b.weight.full_tensor().float()
            expected = base.full_tensor().float() + module._lora_scaling * torch.matmul(
                lora_b, lora_a
            )
            actual = merged[f"{name}.weight"]
            self.assertIsInstance(actual, DTensor)
            self.assertEqual(actual.placements, base.placements)
            torch.testing.assert_close(actual.full_tensor(), expected.to(base.dtype))
            checked += 1
        self.assertGreater(checked, 0)
