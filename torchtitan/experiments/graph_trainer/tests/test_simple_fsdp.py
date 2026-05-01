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
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp


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
    def test_param_cast_to_bf16_at_ngpu_1(self):
        """With ``mixed_precision_param=bfloat16``, the parametrized weight must
        yield bf16 — and a forward must run in bf16 — even when fsdp /
        dp_replicate / ep are all disabled. Without the unconditional
        simple_fsdp wrap, parameters silently stay in fp32 on a single GPU and
        any downstream bf16-only kernel (e.g. MXFP8) breaks.
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
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        model = nn.Linear(8, 8)
        self.assertEqual(model.weight.dtype, torch.float32)

        model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

        # Parametrization replaces ``weight`` access with a bf16 cast via
        # ``redistribute(forward_dtype=...)``.
        self.assertEqual(model.weight.dtype, torch.bfloat16)

        # Underlying storage stays in fp32 (the cast is applied per forward),
        # confirming this is true mixed precision rather than a one-shot
        # downcast that would lose master-weight precision.
        self.assertEqual(model._parameters["weight"].dtype, torch.float32)

        # End-to-end: forward against the parametrized weight produces bf16
        # activations.
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        y = model(x)
        self.assertEqual(y.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
