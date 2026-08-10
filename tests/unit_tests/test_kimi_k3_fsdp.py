# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from unittest.mock import patch

import torch
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims

# Skip instead of failing collection when FLA, a per-model dependency of
# kimi_k3, is not installed; see the matching guard in test_kimi_k3.py.
try:
    from torchtitan.models.kimi_k3 import parallelize_kimi_k3
    from torchtitan.models.kimi_k3.model import KimiK3Model
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        f"Kimi K3 optional dependency unavailable: {exc.name}"
    ) from exc

from tests.unit_tests.test_kimi_k3 import _small_model_config


class TestKimiK3FSDP(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @with_comms
    def test_single_rank_fsdp_matches_manual_bf16_reference(self):
        torch.manual_seed(3)
        # attn_res_block_size=2 makes the second layer pass the attention
        # residual through rather than extend it. That path routes a tensor
        # back out through the FSDP module boundary, and its gradient
        # accumulation order is what keeps FSDP bitwise equal to eager, so the
        # comparison below has to cover it.
        config = _small_model_config(attn_res_block_size=2)
        with torch.device("meta"):
            model = config.build()
        model.to_empty(device=self.device_type)
        model.init_states()
        with torch.no_grad():
            for transformer_block in model.layers.values():
                if transformer_block.moe is not None:
                    transformer_block.moe.router.gate.weight.zero_()

        reference = copy.deepcopy(model)
        for parameter in reference.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)

        parallelism = ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        )
        parallel_dims = ParallelDims.from_config(parallelism, world_size=1)
        with patch(
            "torchtitan.distributed.parallel_dims.device_type",
            self.device_type,
        ):
            parallel_dims.build_mesh()
        model = parallelize_kimi_k3(
            model,
            parallel_dims=parallel_dims,
            training=TrainingConfig(
                local_batch_size=1,
                seq_len=6,
                steps=1,
                dtype="bfloat16",
            ),
            parallelism=parallelism,
            compile_config=CompileConfig(),
            ac_config=None,
            dump_folder="",
        )

        assert isinstance(model, KimiK3Model)
        self.assertIsInstance(model, FSDPModule)
        self.assertIsInstance(model.vision_encoder, FSDPModule)

        inputs = {
            "tokens": torch.tensor(
                [[1, 7, 2, 3, 4, 5]],
                dtype=torch.long,
                device=self.device_type,
            ),
            "pixel_values": torch.randn(
                1,
                4,
                3 * 2 * 2,
                device=self.device_type,
            ),
            "grid_thw": torch.tensor(
                [[1, 2, 2]],
                dtype=torch.long,
                device=self.device_type,
            ),
            "special_tokens": {"image_id": 7},
        }

        actual_BLV = model(**inputs)  # pyrefly: ignore [not-callable]
        expected_BLV = reference(**inputs)
        torch.testing.assert_close(actual_BLV, expected_BLV, atol=0.0, rtol=0.0)

        actual_BLV.float().square().mean().backward()
        expected_BLV.float().square().mean().backward()

        reference_parameters = dict(reference.named_parameters())
        compared_gradients = 0
        for name, parameter in model.named_parameters():
            actual_grad = parameter.grad
            expected_grad = reference_parameters[name].grad
            self.assertEqual(actual_grad is None, expected_grad is None)
            if actual_grad is None:
                continue
            if isinstance(actual_grad, DTensor):
                actual_grad = actual_grad.to_local()
            assert expected_grad is not None
            torch.testing.assert_close(
                actual_grad.float(),
                expected_grad.float(),
                atol=0.0,
                rtol=0.0,
            )
            compared_gradients += 1
        self.assertGreater(compared_gradients, 0)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
