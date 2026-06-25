# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    OffloadPolicy,
)
from torchtitan.experiments.flex_shard.example.shard import per_param_placements, Shard
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cpu_mesh,
    single_rank_cuda_mesh,
    transformer_bucket_specs,
)


class TestFlexShardAPI(TestCase):
    def test_reapplying_flex_shard_to_same_module_raises(self):
        with single_rank_cuda_mesh() as mesh:
            _, model = flex_shard_transformer_model(mesh)

            with self.assertRaisesRegex(ValueError, "Cannot apply flex_shard twice"):
                flex_shard_cuda(
                    model,
                    mesh,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_flex_shard_rejects_child_then_root_nested_wrapping(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

            flex_shard(
                model[0],
                buckets=[
                    BucketSpec(
                        ["*"],
                        placement_fn=per_param_placements,
                        mesh=mesh,
                        reshard_after_forward=False,
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "Nested flex_shard wrapping"):
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_flex_shard_rejects_root_then_child_nested_wrapping(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

            flex_shard(
                model,
                buckets=[
                    BucketSpec(
                        ["*"],
                        placement_fn=per_param_placements,
                        mesh=mesh,
                        reshard_after_forward=False,
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "Nested flex_shard wrapping"):
                flex_shard(
                    model[0],
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_metadata_helpers_on_managed_and_unmanaged_tensors(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)
            raw_param = model.tok_embeddings._parameters["weight"]

            self.assertTrue(is_flex_shard_param(raw_param))
            self.assertEqual(get_placements(raw_param), (Shard(0),))
            self.assertEqual(
                get_global_shape(raw_param),
                torch.Size([args.vocab_size, args.dim]),
            )

            unmanaged = torch.randn(2, 3)
            self.assertFalse(is_flex_shard_param(unmanaged))
            self.assertIsNone(get_placements(unmanaged))
            self.assertIsNone(get_global_shape(unmanaged))

    def test_offload_policy_is_rejected_until_supported(self):
        with single_rank_cuda_mesh() as mesh:
            _, model = make_transformer_model()
            with self.assertRaisesRegex(NotImplementedError, "offload_policy"):
                flex_shard_cuda(
                    model,
                    mesh,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            offload_policy=OffloadPolicy(pin_memory=False),
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_cpu_mesh_is_rejected(self):
        with single_rank_cpu_mesh() as mesh:
            _, model = make_transformer_model()

            with self.assertRaisesRegex(NotImplementedError, "CUDA DeviceMesh"):
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )


if __name__ == "__main__":
    run_tests()
