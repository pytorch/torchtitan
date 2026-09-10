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

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.common import RouterGateLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.kimi_k3.quantile_balance import (
    QuantileBalancedTopKRouter,
    register_moe_quantile_balancing_hook,
)


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestKimiK3QuantileBalancingDistributed(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    @with_comms
    def test_optimizer_hook_reduces_histograms_over_loss_mesh(self) -> None:
        device = torch.device(self.device_type, self.rank)
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
        model = nn.Module()
        model.add_module("moe", moe)
        model.to(device)

        histogram_EB = router.quantile_balancer.required_bias_histogram_EB
        score_rows_RE = torch.tensor(
            [
                [0.92, 0.68, 0.31, 0.07],
                [0.07, 0.31, 0.92, 0.68],
            ],
            device=device,
        )
        local_scores_TE = score_rows_RE[self.rank].expand(4, -1)
        with torch.no_grad():
            router.gate.weight.copy_(torch.eye(4, device=device))
        router(torch.logit(local_scores_TE), moe.expert_bias_E)

        torch.testing.assert_close(
            histogram_EB.sum(dim=-1),
            torch.full((4,), 4, dtype=torch.int64, device=device),
        )
        local_bias_RE = torch.tensor(
            [
                [-0.5, -0.1, 0.1, 0.5],
                [0.5, 0.1, -0.5, -0.1],
            ],
            device=device,
        )
        torch.testing.assert_close(
            router.quantile_balancer.estimate_expert_bias(moe.expert_bias_E),
            local_bias_RE[self.rank],
        )
        expected_global_bias_E = torch.tensor(
            [-0.2, 0.2, -0.2, 0.2],
            device=device,
        )
        self.assertFalse(
            torch.allclose(local_bias_RE[self.rank], expected_global_bias_E)
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

        torch.testing.assert_close(
            moe.expert_bias_E,
            expected_global_bias_E,
        )
        self.assertEqual(histogram_EB.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
