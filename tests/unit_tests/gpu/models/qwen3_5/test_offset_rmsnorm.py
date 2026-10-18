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
from torchtitan.models.common.decoder_sharding import head_norm_sharding_config
from torchtitan.models.qwen3_5.model import _residual_add_norm, OffsetRMSNorm


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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestResidualAddNormCompile(unittest.TestCase):
    def setUp(self):
        # A local generator leaves the global RNG stream unchanged for later tests.
        self.generator = torch.Generator(device="cuda").manual_seed(0)
        self.norm = OffsetRMSNorm.Config(dim=2048).build().cuda()
        with torch.no_grad():
            self.norm.weight.copy_(self.randn(2048))

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, generator=self.generator
        )

    def test_forward_matches_separate_ops(self):
        for num_tokens in (1, 34, 4096):
            x, h = self.randn(num_tokens, 2048), self.randn(num_tokens, 2048)
            with torch.no_grad():
                # Before: a separate add, then the compiled offset_rmsnorm region.
                apply_local_compile(["offset_rmsnorm"])
                expected_x = x + h
                expected_norm = self.norm(expected_x)
                apply_local_compile(["offset_rmsnorm", "residual_add_norm"])
                actual_x, actual_norm = _residual_add_norm(x, h, self.norm)
            torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
            torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)

    def test_grads_as_accurate_as_separate_ops(self):
        norm = self.norm
        x0, h0, grad_x, grad_norm = (self.randn(4096, 2048) for _ in range(4))

        def run(regions: list[str] | None) -> list[torch.Tensor]:
            norm.weight.grad = None
            if regions is None:  # FP64 reference
                weight = norm.weight.detach().double().requires_grad_()
                x, h = (t.double().requires_grad_() for t in (x0, h0))
                summed = x + h
                rms = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + norm.eps)
                normed = summed * rms * (1 + weight)
                (
                    summed * grad_x.double() + normed * grad_norm.double()
                ).sum().backward()
                return [summed, normed, x.grad, h.grad, weight.grad]
            apply_local_compile(regions)
            x, h = (t.clone().requires_grad_() for t in (x0, h0))
            summed, normed = _residual_add_norm(x, h, norm)
            (summed * grad_x + normed * grad_norm).float().sum().backward()
            return [summed, normed, x.grad, h.grad, norm.weight.grad]

        reference = run(None)
        separate = run(["offset_rmsnorm"])
        fused = run(["offset_rmsnorm", "residual_add_norm"])
        for name, ref, sep, fus in zip(
            ("sum", "norm", "grad_x", "grad_h", "grad_weight"),
            reference,
            separate,
            fused,
        ):
            sep_err = ((sep.double() - ref).norm() / ref.norm()).item()
            fus_err = ((fus.double() - ref).norm() / ref.norm()).item()
            # One fp32 sum instead of two bf16 roundings: never less accurate.
            self.assertLessEqual(fus_err, sep_err * 1.01, name)


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
            sharding_config=head_norm_sharding_config(),
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
