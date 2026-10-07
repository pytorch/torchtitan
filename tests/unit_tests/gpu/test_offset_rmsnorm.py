# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.qwen3_5.model import OffsetRMSNorm
from torchtitan.models.qwen3_5.sharding import _qk_norm_sharding


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestOffsetRMSNormCompile(unittest.TestCase):
    def setUp(self):
        apply_local_compile(["offset_rmsnorm"])

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def test_forward_and_backward_emit_triton(self):
        from torch._inductor.utils import run_fw_bw_and_get_code

        module = (
            OffsetRMSNorm.Config(dim=128, eps=1e-6).build().cuda().to(torch.bfloat16)
        )
        with torch.no_grad():
            module.weight.normal_()
        x = torch.randn(
            2048,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        _, codes = run_fw_bw_and_get_code(lambda: module(x))

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)
        self.assertTrue(any("rsqrt" in code for code in codes))

    def test_forward_and_backward_match_eager(self):
        module = (
            OffsetRMSNorm.Config(dim=128, eps=1e-6).build().cuda().to(torch.bfloat16)
        )
        with torch.no_grad():
            module.weight.normal_()
        x = torch.randn(
            8,
            6,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        grad_output = torch.randn_like(x)

        output = module(x)
        grad_x, grad_weight = torch.autograd.grad(
            output,
            (x, module.weight),
            grad_output,
        )

        reference_x = x.detach().clone().requires_grad_()
        reference_weight = module.weight.detach().clone().requires_grad_()
        reference_x_fp32 = reference_x.float()
        variance = reference_x_fp32.square().mean(-1, keepdim=True)
        reference_output = (
            (1.0 + reference_weight.float())
            * reference_x_fp32
            * torch.rsqrt(variance + module.eps)
        ).to(reference_x.dtype)
        reference_grad_x, reference_grad_weight = torch.autograd.grad(
            reference_output,
            (reference_x, reference_weight),
            grad_output,
        )

        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(grad_x, reference_grad_x)
        torch.testing.assert_close(grad_weight, reference_grad_weight)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestOffsetRMSNormTensorParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_compiled_qk_norm(self):
        apply_local_compile(["offset_rmsnorm"])
        device = self.device_type
        config = OffsetRMSNorm.Config(
            dim=128,
            eps=1e-6,
            param_init={"weight": torch.nn.init.zeros_},
            sharding_config=_qk_norm_sharding(),
        )
        module = config.build().to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.zero_()

        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        with patch("torchtitan.distributed.parallelism_context.device_type", device):
            parallelism_context.build_mesh()
        module._parallelize(parallelism_context)

        x_full = torch.randn(16, 4, 128, device=device, dtype=torch.bfloat16)
        x_local = x_full.chunk(self.world_size, 1)[self.rank].contiguous()
        x_local.requires_grad_()
        mesh = parallelism_context.spmd_dense_mesh()
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=None, dense_sp_enabled=False)
        with set_current_spmd_mesh(mesh):
            output = module(x_local)
            output.sum().backward()

        self.assertEqual(output.shape, x_local.shape)
        apply_local_compile([])
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
