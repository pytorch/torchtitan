# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
)
from torchtitan.experiments.flex_shard.param_access import FlexShardModule
from torchtitan.experiments.flex_shard.bucket_storage import OffloadPolicy
from torchtitan.experiments.flex_shard.placements import Shard
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cpu,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cpu_mesh,
    transformer_bucket_specs,
)


class TestFlexShardAPI(TestCase):
    def test_flex_shard_returns_same_module_with_public_storage_properties(self):
        with single_rank_cpu_mesh() as mesh:
            args, model = make_transformer_model()

            result = flex_shard_cpu(
                model,
                mesh,
                buckets=transformer_bucket_specs(
                    args.n_layers,
                    reshard_after_forward=False,
                ),
            )

            self.assertIs(result, model)
            self.assertIsInstance(model, FlexShardModule)
            self.assertEqual(len(model.dstorages), 5)
            self.assertIs(model.dstorage, model.dstorages[0])
            self.assertEqual(
                set(model.dstorage.param_infos),
                {"tok_embeddings.weight"},
            )

    def test_reapplying_flex_shard_to_same_module_raises(self):
        with single_rank_cpu_mesh() as mesh:
            _, model = flex_shard_transformer_model(mesh)

            with self.assertRaisesRegex(ValueError, "Cannot apply flex_shard twice"):
                flex_shard_cpu(
                    model,
                    mesh,
                    buckets=[BucketSpec(["*"], reshard_after_forward=False)],
                )

    def test_metadata_helpers_on_managed_and_unmanaged_tensors(self):
        with single_rank_cpu_mesh() as mesh:
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
        with single_rank_cpu_mesh() as mesh:
            _, model = make_transformer_model()
            with self.assertRaisesRegex(NotImplementedError, "offload_policy"):
                flex_shard_cpu(
                    model,
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
    run_tests()
