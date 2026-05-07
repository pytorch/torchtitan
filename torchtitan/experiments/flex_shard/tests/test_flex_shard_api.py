# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
)
from torchtitan.experiments.flex_shard.module_wrapping import FlexShardModule
from torchtitan.experiments.flex_shard.bucket_storage import OffloadPolicy
from torchtitan.experiments.flex_shard.placements import Shard
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cpu,
    flex_shard_tiny_model,
    single_rank_cpu_mesh,
)


class TestFlexShardAPI(unittest.TestCase):
    def test_flex_shard_returns_same_module_with_public_storage_properties(self):
        with single_rank_cpu_mesh() as mesh:
            model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

            result = flex_shard_cpu(
                model,
                mesh,
                buckets=[
                    BucketSpec(["0.*"], reshard_after_forward=False),
                    BucketSpec(["2.*"], reshard_after_forward=False),
                ],
            )

            self.assertIs(result, model)
            self.assertIsInstance(model, FlexShardModule)
            self.assertEqual(len(model.dstorages), 2)
            self.assertIs(model.dstorage, model.dstorages[0])
            self.assertEqual(
                set(model.dstorage.param_infos),
                {"0.weight", "0.bias"},
            )

    def test_reapplying_flex_shard_to_same_module_raises(self):
        with single_rank_cpu_mesh() as mesh:
            model = flex_shard_tiny_model(mesh)

            with self.assertRaisesRegex(ValueError, "Cannot apply flex_shard twice"):
                flex_shard_cpu(
                    model,
                    mesh,
                    buckets=[BucketSpec(["*"], reshard_after_forward=False)],
                )

    def test_metadata_helpers_on_managed_and_unmanaged_tensors(self):
        with single_rank_cpu_mesh() as mesh:
            model = flex_shard_tiny_model(mesh)
            raw_param = model[0]._parameters["weight"]

            self.assertTrue(is_flex_shard_param(raw_param))
            self.assertEqual(get_placements(raw_param), (Shard(0),))
            self.assertEqual(get_global_shape(raw_param), torch.Size([4, 4]))

            unmanaged = torch.randn(2, 3)
            self.assertFalse(is_flex_shard_param(unmanaged))
            self.assertIsNone(get_placements(unmanaged))
            self.assertIsNone(get_global_shape(unmanaged))

    def test_root_wrap_excludes_already_flex_sharded_child(self):
        with single_rank_cpu_mesh() as mesh:
            model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))

            flex_shard_cpu(
                model[0],
                mesh,
                buckets=[BucketSpec(["*"], reshard_after_forward=False)],
            )
            child_storage = model[0].dstorage

            flex_shard_cpu(
                model,
                mesh,
                buckets=[BucketSpec(["1.*"], reshard_after_forward=False)],
            )

            self.assertIs(model[0].dstorage, child_storage)
            self.assertEqual(set(model.dstorage.param_infos), {"1.weight", "1.bias"})

    def test_root_wrap_raises_if_all_children_are_already_managed(self):
        with single_rank_cpu_mesh() as mesh:
            model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
            for child in model:
                flex_shard_cpu(
                    child,
                    mesh,
                    buckets=[BucketSpec(["*"], reshard_after_forward=False)],
                )

            with self.assertRaisesRegex(ValueError, "has no parameters to shard"):
                flex_shard_cpu(
                    model,
                    mesh,
                    buckets=[BucketSpec(["*"], reshard_after_forward=False)],
                )

    def test_offload_policy_is_rejected_until_supported(self):
        with single_rank_cpu_mesh() as mesh:
            with self.assertRaisesRegex(NotImplementedError, "offload_policy"):
                flex_shard_cpu(
                    nn.Linear(4, 4),
                    mesh,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            offload_policy=OffloadPolicy(pin_memory=False),
                            reshard_after_forward=False,
                        )
                    ],
                )


if __name__ == "__main__":
    unittest.main()
