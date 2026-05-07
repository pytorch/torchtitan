# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_fsdp import FSDPTestMultiThread

from torchtitan.experiments.flex_shard.tests.common import (
    expected_shard,
    flex_shard_cpu,
    make_transformer_model,
    transformer_bucket_specs,
)


def _init_params_deterministically(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(param.numel(), dtype=param.dtype).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


def _deterministic_inputs(args, batch_size: int) -> torch.Tensor:
    values = torch.arange(batch_size * args.max_seq_len)
    return values.view(batch_size, args.max_seq_len).remainder(args.vocab_size)


class TestFlexShardTraining(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    def test_two_microbatch_gradient_accumulation_matches_reference(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))

        args, model = make_transformer_model()
        _init_params_deterministically(model)
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
            _deterministic_inputs(args, batch_size=3),
            _deterministic_inputs(args, batch_size=2),
        ]
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        for x in inputs:
            loss = model(x).sum()
            ref_loss = reference(x).sum()
            self.assertEqual(loss, ref_loss)
            loss.backward()
            ref_loss.backward()

        for (name, param), (ref_name, ref_param) in zip(
            model.named_parameters(),
            reference.named_parameters(),
            strict=True,
        ):
            self.assertEqual(name, ref_name)
            self.assertIsNotNone(param.grad)
            self.assertIsNotNone(ref_param.grad)
            expected_grad = expected_shard(
                ref_param.grad,
                rank=self.rank,
                world_size=self.world_size,
            )
            self.assertEqual(param.grad, expected_grad)

        optim.step()
        ref_optim.step()

        ref_state_dict = reference.state_dict()
        for key, value in model.state_dict().items():
            expected = expected_shard(
                ref_state_dict[key],
                rank=self.rank,
                world_size=self.world_size,
            )
            self.assertEqual(value, expected)


if __name__ == "__main__":
    unittest.main()
