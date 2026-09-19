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

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp


class TestGraphTrainerQwen3JitActivationCheckpoint(unittest.TestCase):
    def test_qwen3_jit_full_ac_wraps_layers(self):
        from torchtitan.config import ParallelismConfig, TrainingConfig
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )
        from torchtitan.experiments.graph_trainer.qwen3 import parallelize

        class FakeParallelDims:
            seq_len_divisor = 1
            cp_enabled = False
            tp_enabled = False
            ep_enabled = False

        class TinyQwen3LikeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2)])

        model = TinyQwen3LikeModel()
        original_layer = model.layers[0]
        with (
            patch.object(parallelize, "annotate_qwen3"),
            patch.object(
                parallelize, "apply_simple_fsdp", lambda module, **kwargs: module
            ),
            patch.object(parallelize, "apply_compile", lambda module, **kwargs: module),
            patch.object(parallelize, "maybe_apply_ep_overlap_eager_chunking"),
        ):
            result = parallelize.parallelize_qwen3(
                model,
                parallel_dims=FakeParallelDims(),
                training=TrainingConfig(seq_len=1),
                parallelism=ParallelismConfig(),
                compile_config=GraphTrainerCompileConfig(mode="jit"),
                ac_config=FullAC.Config(),
                dump_folder="dump",
            )

        self.assertIs(result, model)
        self.assertIsNot(model.layers[0], original_layer)
        self.assertTrue(hasattr(model.layers[0], "_checkpoint_wrapped_module"))
        self.assertIs(model.layers[0]._checkpoint_wrapped_module, original_layer)


class TestApplySimpleFSDPSingleRank(unittest.TestCase):
    """Verify simple_fsdp's MixedPrecisionPolicy actually casts params at NGPU=1."""

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
    def test_uses_dtensor_storage_and_local_compute(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
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

        self.assertIsInstance(
            model._parameters["weight"], torch.distributed.tensor.DTensor
        )
        self.assertEqual(model._parameters["weight"].dtype, torch.float32)
        self.assertNotIsInstance(model.weight, torch.distributed.tensor.DTensor)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(
            model(torch.randn(2, 8, dtype=torch.bfloat16)).dtype, torch.bfloat16
        )


if __name__ == "__main__":
    unittest.main()
