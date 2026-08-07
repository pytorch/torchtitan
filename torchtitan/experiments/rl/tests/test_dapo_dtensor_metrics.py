# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed.metrics import collect_dtensor_metrics
from torchtitan.experiments.rl.losses.dapo import DAPOLoss


class TestDAPODTensorMetrics(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _inputs(self):
        logits = torch.zeros((1, 2, 4), device=self.device_type)
        labels = torch.zeros((1, 2), dtype=torch.int64, device=self.device_type)
        generator_logprobs = torch.full((1, 2), -math.log(4), device=self.device_type)
        advantages = torch.ones((1, 2), device=self.device_type)
        loss_mask = torch.ones((1, 2), dtype=torch.bool, device=self.device_type)
        return logits, labels, generator_logprobs, advantages, loss_mask

    def _assert_metrics(self, metrics: dict[str, DTensor]) -> None:
        for name, metric in metrics.items():
            expected_op = "max" if name == "bit_wise/logprob_diff/max" else "sum"
            self.assertEqual(metric.placements, (Partial(expected_op),))

        collected = collect_dtensor_metrics(metrics)
        self.assertEqual(collected["loss/mean"], -1.0)
        self.assertEqual(collected["loss/ratio_mean"], 1.0)
        self.assertEqual(collected["loss/ratio_clipped_frac"], 0.0)
        self.assertEqual(collected["loss/generator_logprob_nan_frac"], 0.0)
        self.assertEqual(collected["bit_wise/logprob_diff/mean"], 0.0)
        self.assertEqual(collected["bit_wise/ratio_tokens_different/mean"], 0.0)
        self.assertEqual(collected["bit_wise/logprob_diff/max"], 0.0)
        self.assertEqual(collected["trainer/entropy/mean"], math.log(4))

    @with_comms
    def test_direct_loss_produces_partial_metrics(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        logits, labels, generator_logprobs, advantages, loss_mask = self._inputs()
        loss_fn = DAPOLoss.Config().build()

        _, metrics = loss_fn(
            logits,
            labels,
            global_valid_tokens=4,
            generator_logprobs=generator_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            metric_mesh=mesh,
        )

        self._assert_metrics(metrics)

    @with_comms
    def test_chunked_loss_merges_metrics_by_placement(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("loss",)
        )
        logits, labels, generator_logprobs, advantages, loss_mask = self._inputs()
        loss_fn = ChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=DAPOLoss.Config(),
        ).build()
        assert isinstance(loss_fn, ChunkedLossWrapper)
        loss_fn.set_lm_head(nn.Identity())

        _, metrics = loss_fn(
            logits,
            labels,
            global_valid_tokens=4,
            generator_logprobs=generator_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            metric_mesh=mesh,
        )

        self._assert_metrics(metrics)


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
