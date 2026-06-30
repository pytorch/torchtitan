# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import is_flex_shard_param
from torchtitan.experiments.flex_shard.flex_shard.bucket_runtime import (
    _accumulate_sharded_grads,
    _BucketUnshard,
    _EAGER_COMM_CONTEXTS_ATTR,
    _get_raf_saved_full_params,
    _MAX_PENDING_REDUCE_GRADS_ATTR,
    BucketCommContext,
    ParamOwnerRef,
)
from torchtitan.experiments.flex_shard.flex_shard.flex_shard import FlexShardModule
from torchtitan.experiments.flex_shard.flex_shard.unsharded_param_getters import (
    _RafSavedTensorContext,
)
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cuda_mesh,
    transformer_inputs,
)


class TestRafSavedTensorContext(TestCase):
    def test_raf_saved_tensor_context_packs_view_of_registered_base(self):
        class _BaseHandle:
            def __init__(self, tensor: torch.Tensor) -> None:
                self.tensor = tensor

            def unpack_raf_saved_tensor(self) -> torch.Tensor:
                return self.tensor

        class _ParamHandle:
            def __init__(
                self,
                param: torch.Tensor,
                base: torch.Tensor,
            ) -> None:
                self.param = param
                self.base = base

            def unpack_raf_saved_tensor(self) -> torch.Tensor:
                return self.param

            def base_handle_for_raf_saved_tensor(
                self,
                tensor: torch.Tensor,
                base: torch.Tensor,
            ) -> _BaseHandle:
                self.registered_tensor = tensor
                self.registered_base = base
                return _BaseHandle(self.base)

        gathered = torch.arange(4 * 4 * 4, dtype=torch.float32)
        param = gathered.as_strided(
            (2, 4, 4),
            (16, 4, 1),
            storage_offset=0,
        )
        saved_transpose = param.transpose(-2, -1)
        self.assertIs(saved_transpose._base, gathered)
        self.assertIsNot(saved_transpose._base, param)

        recomputed_gathered = gathered + 1000
        recomputed_param = recomputed_gathered.as_strided(
            param.size(),
            param.stride(),
            storage_offset=param.storage_offset(),
        )
        handle = _ParamHandle(recomputed_param, recomputed_gathered)

        context = _RafSavedTensorContext()
        context.register(param, handle)

        packed = context.pack(saved_transpose)
        self.assertIsNot(packed, saved_transpose)
        unpacked = context.unpack(packed)
        expected = recomputed_param.transpose(-2, -1)
        self.assertEqual(unpacked, expected)
        self.assertEqual(unpacked.stride(), expected.stride())
        self.assertIs(handle.registered_tensor, param)
        self.assertIs(handle.registered_base, gathered)


class TestFlexShardEagerRuntime(TestCase):
    def test_set_max_pending_reduce_grads_updates_existing_contexts(self):
        class _Module(FlexShardModule, nn.Module):
            pass

        class _Context:
            def __init__(self, max_pending_reduce_grads: int) -> None:
                self.max_pending_reduce_grads = max_pending_reduce_grads

        root = _Module()
        child = _Module()
        root.child = child

        root_context = _Context(1)
        child_context = _Context(1)
        setattr(root, _EAGER_COMM_CONTEXTS_ATTR, {torch.device("cuda"): root_context})
        setattr(child, _EAGER_COMM_CONTEXTS_ATTR, {torch.device("cuda"): child_context})

        root.set_max_pending_reduce_grads(3)

        self.assertEqual(root_context.max_pending_reduce_grads, 3)
        self.assertEqual(child_context.max_pending_reduce_grads, 3)
        self.assertEqual(getattr(root, _MAX_PENDING_REDUCE_GRADS_ATTR), 3)
        self.assertEqual(getattr(child, _MAX_PENDING_REDUCE_GRADS_ATTR), 3)

        root.set_max_pending_reduce_grads(0, recurse=False)

        self.assertEqual(root_context.max_pending_reduce_grads, 0)
        self.assertEqual(child_context.max_pending_reduce_grads, 3)
        self.assertEqual(getattr(root, _MAX_PENDING_REDUCE_GRADS_ATTR), 0)
        self.assertEqual(getattr(child, _MAX_PENDING_REDUCE_GRADS_ATTR), 3)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            root.set_max_pending_reduce_grads(-1)

        with self.assertRaisesRegex(ValueError, "non-negative int"):
            root.set_max_pending_reduce_grads(True)

        future_root = _Module()
        future_root.set_max_pending_reduce_grads(5)

        class _DeviceHandle:
            def Stream(self, priority: int):
                return object()

        with patch(
            "torchtitan.experiments.flex_shard.flex_shard.bucket_runtime."
            "_get_device_handle",
            return_value=_DeviceHandle(),
        ):
            future_context = BucketCommContext.create(
                future_root,
                torch.device("cuda"),
            )
        self.assertEqual(future_context.max_pending_reduce_grads, 5)

    def test_bucket_unshard_backward_releases_raf_saved_unshard_cache(self):
        class _BucketStorage:
            _reshard_after_forward = True

        bucket_storage = _BucketStorage()
        bucket_id = id(bucket_storage)

        class _Context:
            release_raf_saved_unshard_cache = (
                BucketCommContext.release_raf_saved_unshard_cache
            )

            def __init__(self) -> None:
                self.raf_saved_unshard_cache = {bucket_id: [torch.ones(1)]}

        class _Bucket:
            pass

        bucket = _Bucket()
        bucket.context = _Context()
        bucket.bucket_storage = bucket_storage
        bucket.bucket_params = []

        ctx = type("_Ctx", (), {})()
        ctx.runtime = type("_Runtime", (), {"bucket": bucket})()
        ctx.num_inputs = 0
        ctx.local_shard_dtypes = ()

        self.assertIn(bucket_id, ctx.runtime.bucket.context.raf_saved_unshard_cache)
        result = _BucketUnshard.backward(ctx)
        self.assertEqual(result, (None,))
        self.assertNotIn(bucket_id, ctx.runtime.bucket.context.raf_saved_unshard_cache)

    def test_raf_saved_unshard_cache_keeps_delayed_multi_bucket_unpacks_until_release(
        self,
    ):
        class _BucketStorage:
            _reshard_after_forward = True
            _reshard_after_forward_recompute_state = None

        class _Context:
            set_raf_saved_unshard_cache = (
                BucketCommContext.set_raf_saved_unshard_cache
            )
            release_raf_saved_unshard_cache = (
                BucketCommContext.release_raf_saved_unshard_cache
            )

            def __init__(self) -> None:
                self.raf_saved_unshard_cache = {}
                self.peak_cache_entries = 0
                self.clear_requests = 0

            def queue_raf_saved_unshard_cache_clear(self) -> None:
                self.clear_requests += 1
                self.peak_cache_entries = max(
                    self.peak_cache_entries,
                    len(self.raf_saved_unshard_cache),
                )

        class _Bucket:
            def __init__(self, context: _Context, value: int) -> None:
                self.context = context
                self.bucket_storage = _BucketStorage()
                self.bucket_params = []
                self.full_params = [torch.tensor([value])]
                self.recompute_count = 0

            def recompute_unshard_for_saved_tensor(self) -> list[torch.Tensor]:
                self.recompute_count += 1
                return self.full_params

        def _make_backward_ctx(bucket: _Bucket):
            ctx = type("_Ctx", (), {})()
            ctx.runtime = type("_Runtime", (), {"bucket": bucket})()
            ctx.num_inputs = 0
            ctx.local_shard_dtypes = ()
            return ctx

        context = _Context()
        buckets = [_Bucket(context, idx) for idx in range(3)]
        bucket_ids = [id(bucket.bucket_storage) for bucket in buckets]

        full_params = [_get_raf_saved_full_params(bucket) for bucket in buckets]

        self.assertEqual(len(context.raf_saved_unshard_cache), 3)
        self.assertEqual(context.peak_cache_entries, 3)
        self.assertEqual(context.clear_requests, 3)
        self.assertEqual([bucket.recompute_count for bucket in buckets], [1, 1, 1])
        for bucket_id in bucket_ids:
            self.assertIn(bucket_id, context.raf_saved_unshard_cache)

        self.assertIs(_get_raf_saved_full_params(buckets[0]), full_params[0])
        self.assertEqual(buckets[0].recompute_count, 1)

        for bucket, bucket_id, expected_entries in zip(
            buckets,
            bucket_ids,
            [2, 1, 0],
            strict=True,
        ):
            result = _BucketUnshard.backward(_make_backward_ctx(bucket))
            self.assertEqual(result, (None,))
            self.assertNotIn(bucket_id, context.raf_saved_unshard_cache)
            self.assertEqual(len(context.raf_saved_unshard_cache), expected_entries)

    def test_accumulate_sharded_grads_matches_param_layout_for_fused_optimizer(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for fused AdamW.")
        module = nn.Module().cuda()
        module.weight = nn.Parameter(torch.randn(4, 3, 2, device="cuda"))
        base = torch.randn(4, 9, 2, device="cuda")
        strided_grad = base.as_strided((4, 3, 2), (18, 2, 1))

        stored_grads = _accumulate_sharded_grads(
            [ParamOwnerRef(module, "weight")],
            [strided_grad],
        )

        self.assertIs(module.weight.grad, stored_grads[0])
        self.assertEqual(module.weight.grad.dtype, module.weight.dtype)
        self.assertEqual(module.weight.grad.stride(), module.weight.stride())

        optim = torch.optim.AdamW(module.parameters(), lr=1e-3, fused=True)
        optim.step()

    def test_eager_forward_backward_on_cuda_mesh(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            loss = model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                self.assertIsNotNone(param.grad)

    def test_eager_forward_allows_repeated_param_reads(self):
        class DoubleReadWeight(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4, device="cuda"))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight + x @ self.weight

        with single_rank_cuda_mesh() as mesh:
            model = DoubleReadWeight()
            x = torch.randn(2, 4, device="cuda")
            with torch.no_grad():
                ref_out = model(x)

            flex_shard_cuda(model, mesh)

            out = model(x)
            self.assertEqual(out, ref_out)
            out.sum().backward()
            self.assertIsNotNone(next(model.parameters()).grad)

    def test_meta_to_empty_materializes_bucket_storage_and_runtime(self):
        with single_rank_cuda_mesh() as mesh:
            with torch.device("meta"):
                args, model = make_transformer_model()

            flex_shard_cuda(model, mesh)
            for storage in model.sharded_bucket_storages:
                self.assertEqual(storage.byte_storage.device.type, "meta")

            model.to_empty(device="cuda")
            for storage in model.sharded_bucket_storages:
                self.assertEqual(storage.byte_storage.device.type, "cuda")
            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                nn.init.uniform_(param, -0.1, 0.1)

            loss = model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
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
