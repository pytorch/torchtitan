# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.experiments.flex_shard import BucketSpec, flex_shard, OffloadPolicy
from torchtitan.experiments.flex_shard.example.shard import per_param_placements, Shard
from torchtitan.experiments.flex_shard.flex_shard.param_access import (
    FlexShardModule,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
)
from torchtitan.experiments.flex_shard.flex_shard.reshard_after_forward import (
    _compose_with_ac_policy,
    _ReshardAfterForwardRecomputeState,
)
from torchtitan.experiments.flex_shard.flex_shard.reshard_provenance import (
    _flex_shard_all_gather_region,
    _is_flex_shard_recompute_tensor,
    _mark_flex_shard_recompute_tensors,
)
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cpu_mesh,
    single_rank_cuda_mesh,
    transformer_bucket_specs,
)


class TestFlexShardAPI(TestCase):
    def test_flex_shard_returns_same_module_with_public_storage_properties(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = make_transformer_model()

            result = flex_shard_cuda(
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
            for param in model.parameters():
                self.assertEqual(param.device.type, "cuda")

    def test_reapplying_flex_shard_to_same_module_raises(self):
        with single_rank_cuda_mesh() as mesh:
            _, model = flex_shard_transformer_model(mesh)

            with self.assertRaisesRegex(ValueError, "Cannot apply flex_shard twice"):
                flex_shard_cuda(
                    model,
                    mesh,
                    buckets=[BucketSpec(["*"], reshard_after_forward=False)],
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
                    mesh,
                    shard_placement_fn=per_param_placements,
                    buckets=[BucketSpec(["*"], reshard_after_forward=False)],
                )

    def test_reshard_after_forward_requires_replayable_bucket_hook(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

            with self.assertRaisesRegex(RuntimeError, "recomputation-safe"):
                flex_shard(
                    model,
                    mesh,
                    shard_placement_fn=per_param_placements,
                    buckets=[BucketSpec(["*"])],
                )

    def test_reshard_policy_only_forces_flex_shard_derived_tensors(self):
        def must_save_context_fn():
            def must_save_policy(ctx, func, *args, **kwargs):
                return CheckpointPolicy.MUST_SAVE

            return create_selective_checkpoint_contexts(must_save_policy)

        context_fn = _compose_with_ac_policy(
            must_save_context_fn,
            _ReshardAfterForwardRecomputeState(),
            frozenset({0}),
        )
        forward_ctx, _ = context_fn()
        all_gather = torch.ops._c10d_functional.all_gather_into_tensor.default
        self.assertEqual(
            forward_ctx.policy_fn(None, all_gather),
            CheckpointPolicy.MUST_SAVE,
        )

        all_gather_output = torch.ones(2, 2)

        class AllGatherContext:
            op_output = all_gather_output

        with _flex_shard_all_gather_region():
            self.assertEqual(
                forward_ctx.policy_fn(AllGatherContext(), all_gather),
                CheckpointPolicy.MUST_RECOMPUTE,
            )
        self.assertTrue(_is_flex_shard_recompute_tensor(all_gather_output))

        full_param = torch.ones(2, 2)
        _mark_flex_shard_recompute_tensors(full_param)
        full_param_view = full_param.t()

        class ViewContext:
            op_output = full_param_view

        self.assertEqual(
            forward_ctx.policy_fn(
                ViewContext(),
                torch.ops.aten.t.default,
                full_param,
            ),
            CheckpointPolicy.MUST_RECOMPUTE,
        )
        self.assertTrue(_is_flex_shard_recompute_tensor(full_param_view))

        forward_ctx, _ = context_fn()
        chained_view_param = torch.ones(2, 2)
        _mark_flex_shard_recompute_tensors(chained_view_param)
        with forward_ctx:
            chained_view = chained_view_param.t().narrow(0, 0, 1)
        self.assertTrue(_is_flex_shard_recompute_tensor(chained_view))


if __name__ == "__main__":
    run_tests()
