# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.experiments.rl.actors.trainer import PolicyTrainer


class TestPolicyTrainer(unittest.TestCase):
    def test_reduce_forward_backward_metrics_aggregates_microbatches(self):
        loss_mesh = object()
        trainer = SimpleNamespace(
            parallel_dims=SimpleNamespace(
                get_optional_mesh=lambda axis: loss_mesh if axis == "loss" else None
            )
        )
        microbatch_metrics = [
            {"loss/mean": torch.tensor(1.0), "logprobs_diff/max": torch.tensor(4.0)},
            {"loss/mean": torch.tensor(2.0), "logprobs_diff/max": torch.tensor(3.0)},
        ]

        with (
            patch(
                "torchtitan.experiments.rl.actors.trainer.dist_utils.dist_sum",
                side_effect=lambda value, mesh: float(value.item()),
            ) as dist_sum,
            patch(
                "torchtitan.experiments.rl.actors.trainer.dist_utils.dist_max",
                side_effect=lambda value, mesh: float(value.item()),
            ) as dist_max,
        ):
            result = PolicyTrainer.reduce_forward_backward_metrics(
                trainer, microbatch_metrics
            )

        self.assertEqual(result, {"loss/mean": 3.0, "logprobs_diff/max": 4.0})
        self.assertIs(dist_sum.call_args.args[1], loss_mesh)
        self.assertIs(dist_max.call_args.args[1], loss_mesh)


if __name__ == "__main__":
    unittest.main()
