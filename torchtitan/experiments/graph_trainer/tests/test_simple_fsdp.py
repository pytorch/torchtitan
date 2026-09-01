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
from torch.distributed.tensor import DTensor

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp


class TestApplySimpleFSDPSingleRank(unittest.TestCase):
    """Verify GraphTrainer SimpleFSDP setup at NGPU=1."""

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

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_partial_dtensor_backend_is_rejected(self):
        set_spmd_backend("partial_dtensor")
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            spmd_backend="partial_dtensor",
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        with self.assertRaisesRegex(ValueError, "requires spmd_backend='spmd_types'"):
            apply_simple_fsdp(
                nn.Linear(8, 8),
                parallel_dims=parallel_dims,
                training=training,
            )

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_spmd_types_uses_dtensor_storage_and_local_compute(self):
        set_spmd_backend("spmd_types")
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            spmd_backend="spmd_types",
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        model = apply_simple_fsdp(
            nn.Linear(8, 8),
            parallel_dims=parallel_dims,
            training=training,
        )

        self.assertIsInstance(model._parameters["weight"], DTensor)
        self.assertNotIsInstance(model.weight, DTensor)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(
            model(torch.randn(2, 8, dtype=torch.bfloat16)).dtype, torch.bfloat16
        )


if __name__ == "__main__":
    unittest.main()
