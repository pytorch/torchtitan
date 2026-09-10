# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optimizer import (
    OptimizersContainer,
    ParamGroupConfig,
    register_moe_quantile_balancing_hook,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.common import RouterGateLinear
from torchtitan.models.common.moe import MoE, QuantileBalancedTopKRouter


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestQuantileBalancingDistributed(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    @with_comms
    def test_optimizer_hook_reduces_stacked_histograms_over_loss_mesh(self) -> None:
        device = torch.device(self.device_type, self.rank)
        model = nn.Module()
        moe_layers = []
        for layer_idx in range(2):
            router = QuantileBalancedTopKRouter.Config(
                num_experts=4,
                top_k=1,
                gate=RouterGateLinear.Config(
                    in_features=4,
                    out_features=4,
                    bias=False,
                ),
                score_func="sigmoid",
                num_bins=10,
            ).build()
            moe = MoE.__new__(MoE)
            nn.Module.__init__(moe)
            moe.router = router
            moe.register_buffer("expert_bias_E", torch.zeros(4))
            model.add_module(f"moe_{layer_idx}", moe)
            moe_layers.append(moe)
        model.to(device)

        score_rows_LRE = torch.tensor(
            [
                [
                    [0.92, 0.68, 0.31, 0.07],
                    [0.07, 0.31, 0.92, 0.68],
                ],
                [
                    [0.68, 0.92, 0.07, 0.31],
                    [0.31, 0.07, 0.68, 0.92],
                ],
            ],
            device=device,
        )
        local_bias_LRE = torch.tensor(
            [
                [
                    [-0.5, -0.1, 0.1, 0.5],
                    [0.5, 0.1, -0.5, -0.1],
                ],
                [
                    [-0.1, -0.5, 0.5, 0.1],
                    [0.1, 0.5, -0.1, -0.5],
                ],
            ],
            device=device,
        )
        expected_global_bias_LE = torch.tensor(
            [
                [-0.2, 0.2, -0.2, 0.2],
                [0.2, -0.2, 0.2, -0.2],
            ],
            device=device,
        )
        histograms = []
        for layer_idx, moe in enumerate(moe_layers):
            router = moe.router
            local_scores_TE = score_rows_LRE[layer_idx, self.rank].expand(4, -1)
            with torch.no_grad():
                router.gate.weight.copy_(torch.eye(4, device=device))
            _, _, routing_map_TE = router(
                torch.logit(local_scores_TE), moe.expert_bias_E
            )

            histogram_EB = router.quantile_balancer.required_bias_histogram_EB
            histograms.append(histogram_EB)
            torch.testing.assert_close(
                routing_map_TE.sum(dim=-1),
                torch.ones(4, dtype=torch.int64, device=device),
            )
            torch.testing.assert_close(
                histogram_EB.sum(dim=-1),
                torch.full((4,), 4, dtype=torch.int64, device=device),
            )
            torch.testing.assert_close(
                router.quantile_balancer.estimate_expert_bias(
                    histogram_EB,
                    moe.expert_bias_E,
                ),
                local_bias_LRE[layer_idx, self.rank],
            )
            self.assertFalse(
                torch.allclose(
                    local_bias_LRE[layer_idx, self.rank],
                    expected_global_bias_LE[layer_idx],
                )
            )

        optimizers = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
                )
            ],
        ).build(model_parts=[model])
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        register_moe_quantile_balancing_hook(
            optimizers,
            [model],
            parallel_dims,
        )

        optimizers.step()

        for layer_idx, (moe, histogram_EB) in enumerate(
            zip(moe_layers, histograms, strict=True)
        ):
            torch.testing.assert_close(
                moe.expert_bias_E,
                expected_global_bias_LE[layer_idx],
            )
            self.assertEqual(histogram_EB.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
