# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.mixtral import mixtral_configs, model_registry
from torchtitan.protocols.model_converter import ModelConvertersContainer


class TestMixtralModel(unittest.TestCase):
    """Test Mixtral model without distributed setup."""

    def test_model_registry(self):
        spec = model_registry("debugmodel")
        self.assertEqual(spec.name, "mixtral")
        self.assertEqual(spec.flavor, "debugmodel")
        self.assertIsNotNone(spec.parallelize_fn)
        self.assertIsNotNone(spec.build_loss_fn)

    def test_debugmodel_meta_device(self):
        with torch.device("meta"):
            model = mixtral_configs["debugmodel"].build()
        self.assertEqual(len(model.layers), 4)
        for layer in model.layers.values():
            self.assertTrue(layer.moe_enabled)
            self.assertIsInstance(layer.moe, nn.Module)
            self.assertFalse(hasattr(layer, "feed_forward"))

    def test_8x7b_meta_device(self):
        with torch.device("meta"):
            model = mixtral_configs["8x7b"].build()
        self.assertEqual(len(model.layers), 32)
        n_params = sum(p.numel() for p in model.parameters())
        # Mixtral-8x7B should be ~46.7B params
        self.assertGreater(n_params, 46e9)
        self.assertLess(n_params, 47e9)

    def test_forward_backward_cpu(self):
        model = mixtral_configs["debugmodel"].build()
        model.init_weights(buffer_device=torch.device("cpu"))

        tokens = torch.randint(0, 2048, (2, 32))
        logits = model(tokens)

        self.assertEqual(logits.shape, (2, 32, 2048))

        loss = torch.nn.functional.cross_entropy(logits.view(-1, 2048), tokens.view(-1))
        loss.backward()

        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total = sum(1 for _ in model.parameters())
        self.assertEqual(grad_count, total)

    def test_moe_routing(self):
        model = mixtral_configs["debugmodel"].build()
        model.init_weights(buffer_device=torch.device("cpu"))

        tokens = torch.randint(0, 2048, (2, 32))
        _ = model(tokens)

        # Check that tokens were distributed across experts
        layer0 = model.layers["0"]
        tokens_per_expert = layer0.moe.tokens_per_expert
        self.assertEqual(tokens_per_expert.shape[0], 4)  # 4 experts
        self.assertGreater(tokens_per_expert.sum().item(), 0)


@unittest.skipUnless(torch.cuda.is_available(), "FSDP2 requires NCCL (GPU)")
class TestMixtralParallelize(DTensorTestBase):
    """Test Mixtral parallelization with 2 ranks.

    Requires GPU: FSDP2 uses ReduceOp.PREMUL_SUM which gloo doesn't support.
    """

    @property
    def world_size(self):
        return 2

    @with_comms
    def test_fsdp(self):
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=2,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                etp=1,
                world_size=2,
            )
            parallel_dims.build_mesh()

            model = mixtral_configs["debugmodel"].build()
            model.init_weights(buffer_device=torch.device("cpu"))

            from torchtitan.models.mixtral.parallelize import parallelize_mixtral

            parallelize_mixtral(
                model,
                parallel_dims=parallel_dims,
                training=TrainingConfig(
                    local_batch_size=4,
                    seq_len=256,
                    steps=1,
                    mixed_precision_param="float32",
                    mixed_precision_reduce="float32",
                ),
                model_converters=ModelConvertersContainer.Config(),
                parallelism=ParallelismConfig(
                    data_parallel_shard_degree=2,
                ),
                compile_config=CompileConfig(),
                ac_config=ActivationCheckpointConfig(),
                dump_folder="/tmp/mixtral_test",
            )

            # Model should still be callable after FSDP wrapping
            device = f"cuda:{self.rank}"
            tokens = torch.randint(0, 2048, (2, 32), device=device)
            logits = model(tokens)
            self.assertEqual(logits.shape[-1], 2048)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 2048), tokens.view(-1)
            )
            loss.backward()

    @with_comms
    def test_ep(self):
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
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
            parallel_dims.build_mesh()

            model = mixtral_configs["debugmodel"].build()
            model.init_weights(buffer_device=torch.device("cpu"))

            from torchtitan.models.mixtral.parallelize import parallelize_mixtral

            parallelize_mixtral(
                model,
                parallel_dims=parallel_dims,
                training=TrainingConfig(
                    local_batch_size=4,
                    seq_len=256,
                    steps=1,
                    mixed_precision_param="float32",
                    mixed_precision_reduce="float32",
                ),
                model_converters=ModelConvertersContainer.Config(),
                parallelism=ParallelismConfig(
                    data_parallel_shard_degree=2,
                    expert_parallel_degree=2,
                ),
                compile_config=CompileConfig(),
                ac_config=ActivationCheckpointConfig(),
                dump_folder="/tmp/mixtral_test",
            )

            tokens = torch.randint(0, 2048, (2, 32))
            logits = model(tokens)
            self.assertEqual(logits.shape[-1], 2048)


if __name__ == "__main__":
    unittest.main()
