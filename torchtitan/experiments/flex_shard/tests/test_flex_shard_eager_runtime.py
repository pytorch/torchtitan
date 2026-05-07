# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch

from torchtitan.experiments.flex_shard import is_flex_shard_param
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_transformer_model,
    single_rank_cpu_mesh,
    transformer_inputs,
)


class TestFlexShardEagerRuntime(unittest.TestCase):
    def test_eager_forward_backward_on_cpu_mesh(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            loss = model(transformer_inputs(args)).sum()
            loss.backward()

            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                self.assertIsNotNone(param.grad)

    def test_param_access_outside_forward_raises(self):
        with single_rank_cpu_mesh() as mesh:
            _, model = flex_shard_transformer_model(mesh)

            with self.assertRaisesRegex(RuntimeError, "pre-gathered parameter data"):
                _ = model.output.weight

    def test_graph_capture_raises(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            with patch.object(torch.compiler, "is_compiling", return_value=True):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(transformer_inputs(args))

    def test_graph_capture_error_does_not_poison_next_eager_forward(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)
            inp = transformer_inputs(args)

            with patch.object(torch.compiler, "is_compiling", return_value=True):
                with self.assertRaisesRegex(ValueError, "eager execution only"):
                    model(inp)

            loss = model(inp).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
