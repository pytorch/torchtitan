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


class BranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(4, 3)
        self.unused = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor):
        return {
            "used": self.used(x),
            "unused": self.unused(x),
            "metadata": "not-a-tensor",
        }


class OptionalBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active = nn.Linear(4, 3)
        self.inactive = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor, use_inactive: bool = False) -> torch.Tensor:
        out = self.active(x)
        if use_inactive:
            out = out + self.inactive(x)
        return out


class FrozenTailModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 2)
        self.frozen.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frozen(torch.relu(self.trainable(x)))


class SharedParamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4, bias=False)
        self.right = nn.Linear(4, 4, bias=False)
        self.right.weight = self.left.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) + self.right(x)


class TestFlexShardAutograd(unittest.TestCase):
    def test_unused_forward_output_in_shared_bucket_matches_reference(self):
        with single_rank_cpu_mesh() as mesh:
            torch.manual_seed(0)
            model = BranchModel()
            reference = copy.deepcopy(model)
            flex_shard_cpu(
                model,
                mesh,
                buckets=[BucketSpec(["*"], reshard_after_forward=False)],
            )

            x = torch.randn(5, 4)
            loss = model(x)["used"].sum()
            ref_loss = reference(x)["used"].sum()
            loss.backward()
            ref_loss.backward()

            for key, param in dict(model.named_parameters()).items():
                ref_param = dict(reference.named_parameters())[key]
                if key.startswith("used."):
                    self.assertIsNotNone(param.grad)
                    self.assertIsNotNone(ref_param.grad)
                    torch.testing.assert_close(param.grad, ref_param.grad)
                else:
                    self.assertIsNone(param.grad)
                    self.assertIsNone(ref_param.grad)

    def test_unused_module_in_forward_path_keeps_gradients_local_to_used_params(self):
        with single_rank_cpu_mesh() as mesh:
            torch.manual_seed(0)
            model = OptionalBranchModel()
            reference = copy.deepcopy(model)
            flex_shard_cpu(
                model,
                mesh,
                buckets=[BucketSpec(["*"], reshard_after_forward=False)],
            )

            x = torch.randn(5, 4)
            model(x, use_inactive=False).sum().backward()
            reference(x, use_inactive=False).sum().backward()

            for key, param in dict(model.named_parameters()).items():
                ref_param = dict(reference.named_parameters())[key]
                if key.startswith("active."):
                    self.assertIsNotNone(param.grad)
                    self.assertIsNotNone(ref_param.grad)
                    torch.testing.assert_close(param.grad, ref_param.grad)
                else:
                    self.assertIsNone(param.grad)
                    self.assertIsNone(ref_param.grad)

    def test_mixed_requires_grad_bucket_matches_reference(self):
        with single_rank_cpu_mesh() as mesh:
            torch.manual_seed(0)
            model = FrozenTailModel()
            reference = copy.deepcopy(model)
            flex_shard_cpu(
                model,
                mesh,
                buckets=[BucketSpec(["*"], reshard_after_forward=False)],
            )

            x = torch.randn(5, 4)
            model(x).sum().backward()
            reference(x).sum().backward()

            for key, param in dict(model.named_parameters()).items():
                ref_param = dict(reference.named_parameters())[key]
                if key.startswith("trainable."):
                    self.assertIsNotNone(param.grad)
                    self.assertIsNotNone(ref_param.grad)
                    torch.testing.assert_close(param.grad, ref_param.grad)
                else:
                    self.assertFalse(param.requires_grad)
                    self.assertFalse(ref_param.requires_grad)
                    self.assertIsNone(param.grad)
                    self.assertIsNone(ref_param.grad)

    def test_shared_parameters_are_rejected(self):
        with single_rank_cpu_mesh() as mesh:
            with self.assertRaisesRegex(ValueError, "shared parameters"):
                flex_shard_cpu(
                    SharedParamModel(),
                    mesh,
                    buckets=[BucketSpec(["*"], reshard_after_forward=False)],
                )


if __name__ == "__main__":
    unittest.main()
