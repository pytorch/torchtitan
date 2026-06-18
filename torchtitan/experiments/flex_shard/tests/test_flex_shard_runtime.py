# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import is_flex_shard_param
from torchtitan.experiments.flex_shard.flex_shard.bucket_runtime import (
    _accumulate_sharded_grads,
    ParamOwnerRef,
)
from torchtitan.experiments.flex_shard.flex_shard.unsharded_param_getters import (
    UnshardedParamSlot,
)
from torchtitan.experiments.flex_shard.tests.common import (
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cuda_mesh,
    transformer_inputs,
)


class TestUnshardedParamSlot(TestCase):
    def test_repeated_reads_return_same_tensor_until_pop(self):
        slot = UnshardedParamSlot(param_fqn="weight", bucket_fqn="bucket")
        param = torch.randn(2, 3, requires_grad=True)

        slot.push_unsharded_param(param)
        self.assertIs(slot.get_unsharded_param(), param)
        self.assertIs(slot.get_unsharded_param(), param)
        slot.pop_unsharded_param()

        with self.assertRaisesRegex(RuntimeError, "bucket unshard hook"):
            slot.get_unsharded_param()

    def test_mixed_precision_read_is_cached_per_frame(self):
        slot = UnshardedParamSlot(
            param_fqn="weight",
            bucket_fqn="bucket",
            param_dtype=torch.float16,
        )
        param = torch.randn(2, 3, requires_grad=True)

        slot.push_unsharded_param(param)
        first = slot.get_unsharded_param()
        second = slot.get_unsharded_param()
        self.assertIs(first, second)
        self.assertEqual(first.dtype, torch.float16)

    def test_nested_frames_restore_outer_param(self):
        slot = UnshardedParamSlot(param_fqn="weight", bucket_fqn="bucket")
        outer = torch.randn(2, 3)
        inner = torch.randn(2, 3)

        slot.push_unsharded_param(outer)
        self.assertIs(slot.get_unsharded_param(), outer)
        slot.push_unsharded_param(inner)
        self.assertIs(slot.get_unsharded_param(), inner)
        slot.pop_unsharded_param()
        self.assertIs(slot.get_unsharded_param(), outer)
        slot.pop_unsharded_param()


class TestFlexShardEagerRuntime(TestCase):
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
