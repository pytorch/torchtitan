# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch

from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cpu,
    make_transformer_model,
    single_rank_cpu_mesh,
    transformer_bucket_specs,
    transformer_inputs,
)


class TestFlexShardTraining(unittest.TestCase):
    def test_two_microbatch_gradient_accumulation_matches_reference(self):
        with single_rank_cpu_mesh() as mesh:
            torch.manual_seed(0)
            args, model = make_transformer_model()
            reference = copy.deepcopy(model)
            flex_shard_cpu(
                model,
                mesh,
                buckets=transformer_bucket_specs(
                    args.n_layers,
                    reshard_after_forward=False,
                ),
            )

            inputs = [
                transformer_inputs(args, batch_size=3),
                transformer_inputs(args, batch_size=2),
            ]
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
