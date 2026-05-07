# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.flex_shard import BucketSpec
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cpu,
    single_rank_cpu_mesh,
)


class TinyResidualMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(4, 8)
        self.hidden = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.in_proj(x))
        h = h + torch.relu(self.hidden(h))
        return self.out_proj(h)


def _flex_shard_training_model(model: nn.Module, mesh) -> nn.Module:
    return flex_shard_cpu(
        model,
        mesh,
        buckets=[
            BucketSpec(["in_proj.*"], reshard_after_forward=False),
            BucketSpec(["hidden.*"], reshard_after_forward=False),
            BucketSpec(["out_proj.*"], reshard_after_forward=False),
        ],
    )


class TestFlexShardTraining(unittest.TestCase):
    def test_two_microbatch_gradient_accumulation_matches_reference(self):
        with single_rank_cpu_mesh() as mesh:
            torch.manual_seed(0)
            model = TinyResidualMLP()
            reference = copy.deepcopy(model)
            _flex_shard_training_model(model, mesh)

            inputs = [torch.randn(3, 4), torch.randn(2, 4)]
            optim = torch.optim.SGD(model.parameters(), lr=0.1)
            ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)

            optim.zero_grad(set_to_none=True)
            ref_optim.zero_grad(set_to_none=True)
            for x in inputs:
                model(x).sum().backward()
                reference(x).sum().backward()

            for (name, param), (ref_name, ref_param) in zip(
                model.named_parameters(),
                reference.named_parameters(),
                strict=True,
            ):
                self.assertEqual(name, ref_name)
                self.assertIsNotNone(param.grad)
                self.assertIsNotNone(ref_param.grad)
                torch.testing.assert_close(param.grad, ref_param.grad)

            optim.step()
            ref_optim.step()

            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
