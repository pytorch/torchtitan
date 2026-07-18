# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    LocalTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_minimal_async_ep,
)
from torchtitan.overrides.fused_grouped_experts import (
    fused_grouped_experts,
    FusedGroupedExperts,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
    silu_and_mul_op,
)

_ACCELERATOR = torch.accelerator.current_accelerator()
DEVICE = _ACCELERATOR.type if _ACCELERATOR is not None else "cpu"


class TestFusedSwiGLUOverride(unittest.TestCase):
    def test_minimal_async_ep_config_imports_override(self):
        config = deepseek_v3_debugmodel_minimal_async_ep()

        self.assertIn(
            "torchtitan.overrides.fused_grouped_experts",
            config.override.imports,
        )

    def test_minimal_async_ep_grouped_experts_config_is_replaced(self):
        cfg = GroupedExperts.Config(
            dim=16,
            hidden_dim=32,
            num_experts=4,
            token_dispatcher=MinimalAsyncEPTokenDispatcher.Config(
                num_experts=4,
                top_k=1,
            ),
        )

        replacement = fused_grouped_experts(cfg)

        self.assertIsInstance(replacement, FusedGroupedExperts.Config)
        self.assertIs(replacement.token_dispatcher, cfg.token_dispatcher)

    def test_local_grouped_experts_config_is_replaced(self):
        cfg = GroupedExperts.Config(
            dim=16,
            hidden_dim=32,
            num_experts=4,
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=4,
                top_k=1,
            ),
        )

        replacement = fused_grouped_experts(cfg)

        self.assertIsInstance(replacement, FusedGroupedExperts.Config)
        self.assertIs(replacement.token_dispatcher, cfg.token_dispatcher)


@unittest.skipUnless(torch.accelerator.is_available(), "accelerator required")
class TestFusedSwiGLUOverrideKernels(unittest.TestCase):
    def test_silu_and_mul_custom_op_matches_reference_with_offsets(self):
        gate = torch.randn(3, 2, device=DEVICE, requires_grad=True)
        up = torch.randn(3, 2, device=DEVICE, requires_grad=True)
        offsets = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)

        out = silu_and_mul_op(gate, up, offsets)
        out[:2].sum().backward()

        ref_gate = gate.detach().clone().requires_grad_()
        ref_up = up.detach().clone().requires_grad_()
        expected = torch.nn.functional.silu(ref_gate) * ref_up
        expected[:2].sum().backward()

        assert gate.grad is not None
        assert up.grad is not None
        assert ref_gate.grad is not None
        assert ref_up.grad is not None
        torch.testing.assert_close(out[:2], expected[:2])
        torch.testing.assert_close(gate.grad[:2], ref_gate.grad[:2])
        torch.testing.assert_close(up.grad[:2], ref_up.grad[:2])

    def test_silu_and_mul_custom_op_matches_reference_without_offsets(self):
        gate = torch.randn(3, 2, device=DEVICE, requires_grad=True)
        up = torch.randn(3, 2, device=DEVICE, requires_grad=True)

        out = silu_and_mul_op(gate, up)
        out.sum().backward()

        ref_gate = gate.detach().clone().requires_grad_()
        ref_up = up.detach().clone().requires_grad_()
        expected = torch.nn.functional.silu(ref_gate) * ref_up
        expected.sum().backward()

        assert gate.grad is not None
        assert up.grad is not None
        assert ref_gate.grad is not None
        assert ref_up.grad is not None
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(gate.grad, ref_gate.grad)
        torch.testing.assert_close(up.grad, ref_up.grad)

    def test_silu_and_mul_kernels_match_reference_with_offsets(self):
        gate = torch.tensor(
            [
                [0.0, 1.0],
                [2.0, -3.0],
                [4.0, 5.0],
            ],
            device=DEVICE,
            requires_grad=True,
        )
        up = torch.tensor(
            [
                [2.0, 3.0],
                [5.0, 7.0],
                [11.0, 13.0],
            ],
            device=DEVICE,
            requires_grad=True,
        )
        offsets = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)

        out = silu_and_mul_forward_kernel(gate, up, offsets)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(out[:2], expected[:2])

        grad_out = torch.tensor(
            [
                [17.0, 19.0],
                [23.0, 29.0],
                [31.0, 37.0],
            ],
            device=DEVICE,
        )
        grad_gate, grad_up = silu_and_mul_backward_kernel(
            grad_out,
            gate,
            up,
            offsets,
        )
        expected[:2].backward(grad_out[:2])
        assert gate.grad is not None
        assert up.grad is not None
        torch.testing.assert_close(grad_gate[:2], gate.grad[:2])
        torch.testing.assert_close(grad_up[:2], up.grad[:2])

    def test_fused_experts_forward_matches_base_grouped_experts(self):
        # Guards against the override drifting out of sync with the base
        # GroupedExperts._experts_forward signature: MoE.forward passes
        # dispatch_metadata positionally, so a stale 2-arg override would raise
        # TypeError here. Also checks the fused activation matches the base
        # SiLU-and-mul grouped-GEMM path within bf16 tolerance.
        dim = 128
        hidden_dim = 256
        num_experts = 8

        def build(cls):
            return cls(
                cls.Config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    num_experts=num_experts,
                    token_dispatcher=LocalTokenDispatcher.Config(
                        num_experts=num_experts,
                        top_k=1,
                    ),
                )
            ).to(DEVICE)

        fused = build(FusedGroupedExperts)
        base = build(GroupedExperts)
        # Expert weights are allocated with torch.empty; initialize them and
        # share across both paths so the comparison is meaningful.
        with torch.no_grad():
            for p in fused.parameters():
                p.normal_(0.0, 0.02)
        base.load_state_dict(fused.state_dict())

        num_tokens_per_expert = torch.tensor(
            [10, 8, 12, 9, 11, 7, 10, 13], dtype=torch.int32, device=DEVICE
        )
        total_tokens = int(num_tokens_per_expert.sum().item())
        x = torch.randn(total_tokens, dim, device=DEVICE, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Call with the 3-arg signature MoE.forward uses (dispatch_metadata).
        out = fused._experts_forward(x, num_tokens_per_expert, None)
        out_ref = base._experts_forward(x_ref, num_tokens_per_expert, None)

        self.assertEqual(out.shape, (total_tokens, dim))
        torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)

        out.sum().backward()
        out_ref.sum().backward()
        assert x.grad is not None
        assert x_ref.grad is not None
        torch.testing.assert_close(x.grad, x_ref.grad, atol=5e-2, rtol=5e-2)
