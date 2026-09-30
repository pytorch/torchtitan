# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import Shard

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.linear import GroupedLinear, Linear


class TestApplySimpleFSDPSingleRank(unittest.TestCase):
    """Check single-rank parameter precision and shard placement policies."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12358",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_uses_dtensor_storage_and_local_compute(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        model = apply_simple_fsdp(
            nn.Linear(8, 8),
            parallelism_context=parallelism_context,
            training=training,
        )

        self.assertIsInstance(
            model._parameters["weight"], torch.distributed.tensor.DTensor
        )
        self.assertEqual(model._parameters["weight"].dtype, torch.float32)
        self.assertNotIsInstance(model.weight, torch.distributed.tensor.DTensor)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(
            model(torch.randn(2, 8, dtype=torch.bfloat16)).dtype, torch.bfloat16
        )

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_stacked_linear_and_routed_expert_shard_placements(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )

        class _StackedDecoder(Decoder):
            def __init__(self):
                nn.Module.__init__(self)
                self.layers = nn.ModuleDict({"0": nn.Module()})

        model = _StackedDecoder()
        block = model.layers["0"]
        block.stacked = Linear.Config(
            in_features=8, out_features=4, num_linears=2, bias=True
        ).build()
        block.ordinary = nn.Linear(8, 8)
        model.extra_stacked = Linear.Config(
            in_features=8, out_features=4, num_linears=2
        ).build()
        block.moe_enabled = True
        block.moe = nn.Module()
        block.moe.routed_experts = nn.Module()
        block.moe.routed_experts.w13 = GroupedLinear.Config(
            group_size=2, in_features=8, out_features=4, num_linears=2
        ).build()
        block.moe.routed_experts.w2 = GroupedLinear.Config(
            group_size=2, in_features=4, out_features=8
        ).build()

        apply_simple_fsdp(
            model, parallelism_context=parallelism_context, training=TrainingConfig()
        )

        assert block.stacked._parameters["weight"].placements == (Shard(1),)
        assert block.stacked._parameters["bias"].placements == (Shard(1),)
        assert block.ordinary._parameters["weight"].placements == (Shard(0),)
        assert model.extra_stacked._parameters["weight"].placements == (Shard(1),)
        assert block.moe.routed_experts.w13._parameters["weight"].placements == (
            Shard(0),
        )


if __name__ == "__main__":
    unittest.main()
