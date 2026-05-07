# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.flex_shard import BucketSpec
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cpu,
    single_rank_cpu_mesh,
)


def _make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def _flex_shard_model(model: nn.Module, mesh) -> nn.Module:
    return flex_shard_cpu(
        model,
        mesh,
        buckets=[
            BucketSpec(["0.*"], reshard_after_forward=False),
            BucketSpec(["2.*"], reshard_after_forward=False),
        ],
    )


class TestFlexShardStateDict(unittest.TestCase):
    def test_state_dict_load_round_trip_into_flex_sharded_model(self):
        with single_rank_cpu_mesh() as mesh:
            torch.manual_seed(0)
            source = _flex_shard_model(_make_model(), mesh)
            state_dict = {k: v.clone() for k, v in source.state_dict().items()}

            torch.manual_seed(1)
            target = _flex_shard_model(_make_model(), mesh)
            load_result = target.load_state_dict(state_dict)

            self.assertEqual(load_result.missing_keys, [])
            self.assertEqual(load_result.unexpected_keys, [])

            x = torch.randn(3, 4)
            torch.testing.assert_close(source(x), target(x))

            source_optim = torch.optim.SGD(source.parameters(), lr=0.05)
            target_optim = torch.optim.SGD(target.parameters(), lr=0.05)
            for model, optim in ((source, source_optim), (target, target_optim)):
                optim.zero_grad(set_to_none=True)
                model(x).sum().backward()
                optim.step()

            for key, value in source.state_dict().items():
                torch.testing.assert_close(value, target.state_dict()[key])

    def test_state_dict_is_stable_across_forward_and_backward(self):
        with single_rank_cpu_mesh() as mesh:
            torch.manual_seed(0)
            model = _flex_shard_model(_make_model(), mesh)
            before = {k: v.clone() for k, v in model.state_dict().items()}

            x = torch.randn(3, 4)
            _ = model(x)
            after_forward = {k: v.clone() for k, v in model.state_dict().items()}

            model(x).sum().backward()
            after_backward = model.state_dict()

            for key, value in before.items():
                torch.testing.assert_close(value, after_forward[key])
                torch.testing.assert_close(value, after_backward[key])

    def test_load_state_dict_rejects_incompatible_shapes(self):
        with single_rank_cpu_mesh() as mesh:
            source = _flex_shard_model(_make_model(), mesh)
            target = flex_shard_cpu(
                nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2)),
                mesh,
                buckets=[
                    BucketSpec(["0.*"], reshard_after_forward=False),
                    BucketSpec(["2.*"], reshard_after_forward=False),
                ],
            )

            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    unittest.main()
