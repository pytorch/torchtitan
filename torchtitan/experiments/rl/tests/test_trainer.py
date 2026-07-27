# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from torchtitan.experiments.rl.actors.trainer import PolicyTrainer


class TestPolicyTrainer(unittest.TestCase):
    def test_pack_pipeline_targets_keeps_microbatch_inputs_aligned(self):
        labels = torch.arange(8).view(4, 2)
        generator_logprobs = torch.arange(8, dtype=torch.float32).view(4, 2)
        loss_mask = generator_logprobs > 2
        advantages = generator_logprobs.float() + 0.5

        packed_targets = PolicyTrainer._pack_pipeline_targets(
            labels,
            generator_logprobs,
            loss_mask,
            advantages,
        )
        chunks = torch.tensor_split(packed_targets, 2, dim=0)

        self.assertEqual(len(chunks), 2)
        torch.testing.assert_close(chunks[0][..., 0], labels[:2].double())
        torch.testing.assert_close(chunks[1][..., 1], generator_logprobs[2:].double())
        torch.testing.assert_close(chunks[0][..., 2].bool(), loss_mask[:2])
        torch.testing.assert_close(chunks[1][..., 3], advantages[2:].double())

        trainer = SimpleNamespace(
            loss_fn=Mock(
                return_value=(torch.tensor(1.0), {"loss/mean": torch.tensor(1.0)})
            ),
            _pipeline_loss_metrics=[],
        )
        PolicyTrainer._pipeline_loss_fn(
            trainer,
            torch.ones(2, 2, 4),
            chunks[1],
            10,
        )
        loss_args = trainer.loss_fn.call_args
        torch.testing.assert_close(loss_args.args[1], labels[2:])
        self.assertEqual(loss_args.kwargs["generator_logprobs"].dtype, torch.float32)
        self.assertEqual(loss_args.kwargs["loss_mask"].dtype, torch.bool)
        self.assertEqual(loss_args.kwargs["advantages"].dtype, torch.float32)

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
