# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from torchtitan.experiments.flex_shard import is_flex_shard_param
from torchtitan.experiments.flex_shard.placements import Shard
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_tiny_model,
    single_rank_cpu_mesh,
)


class TestFlexShardEagerOnly(unittest.TestCase):
    def test_eager_forward_backward_on_cpu_mesh(self):
        with single_rank_cpu_mesh() as mesh:
            model = flex_shard_tiny_model(mesh)

            loss = model(torch.randn(3, 4)).sum()
            loss.backward()

            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                self.assertIsNotNone(param.grad)

    def test_param_access_outside_forward_raises(self):
        with single_rank_cpu_mesh() as mesh:
            model = flex_shard_tiny_model(mesh)

            with self.assertRaisesRegex(RuntimeError, "pre-gathered parameter data"):
                _ = model[0].weight

    def test_graph_capture_raises(self):
        with single_rank_cpu_mesh() as mesh:
            model = flex_shard_tiny_model(mesh)

            with patch.object(torch.compiler, "is_compiling", return_value=True):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(torch.randn(3, 4))

    def test_graph_capture_error_does_not_poison_next_eager_forward(self):
        with single_rank_cpu_mesh() as mesh:
            model = flex_shard_tiny_model(mesh)
            inp = torch.randn(3, 4)

            with patch.object(torch.compiler, "is_compiling", return_value=True):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(inp)

            loss = model(inp).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)

    def test_graph_capture_raises_before_collectives(self):
        with single_rank_cpu_mesh() as mesh:
            model = flex_shard_tiny_model(mesh)

            with (
                patch.object(torch.compiler, "is_compiling", return_value=True),
                patch.object(
                    Shard,
                    "unshard",
                    side_effect=AssertionError("unshard should not run"),
                ) as unshard,
                patch.object(
                    Shard,
                    "begin_unshard",
                    side_effect=AssertionError("begin_unshard should not run"),
                ) as begin_unshard,
            ):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(torch.randn(3, 4))

            unshard.assert_not_called()
            begin_unshard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
