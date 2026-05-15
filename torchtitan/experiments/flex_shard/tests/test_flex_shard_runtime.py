# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import is_flex_shard_param
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_transformer_model,
    single_rank_cuda_mesh,
    transformer_inputs,
)


class TestFlexShardEagerRuntime(TestCase):
    def test_eager_forward_backward_on_cuda_mesh(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            loss = model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                self.assertIsNotNone(param.grad)

    def test_param_access_outside_forward_raises(self):
        with single_rank_cuda_mesh() as mesh:
            _, model = flex_shard_transformer_model(mesh)

            with self.assertRaisesRegex(RuntimeError, "bucket unshard hook"):
                _ = model.output.weight

    def test_torch_compile_forward_backward_on_cuda_mesh(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            compiled_model = torch.compile(model, backend="eager")

            loss = compiled_model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    run_tests()
