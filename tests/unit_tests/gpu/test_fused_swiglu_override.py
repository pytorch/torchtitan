# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd
import torch

from torchtitan.models.common.config_utils import fused_grouped_experts_param_init
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.overrides.fused_swiglu import (
    fused_swiglu,
    FusedSwiGLU,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
    silu_and_mul_op,
)
from torchtitan.protocols.sharding import ShardingConfig

_DIM = 16
_HIDDEN = 32
_E = 4


def _build_fused_swiglu_grouped_experts() -> GroupedExperts:
    fused = GroupedExperts.Config(
        dim=_DIM,
        hidden_dim=_HIDDEN,
        num_experts=_E,
        activation_fn=FusedSwiGLU.Config(),
    ).build()
    with torch.no_grad():
        fused.w13_E2FD.copy_(torch.randn(_E, 2, _HIDDEN, _DIM))
        fused.w2_EDF.copy_(torch.randn(_E, _DIM, _HIDDEN))
    return fused


def _logical_w13(experts: GroupedExperts) -> torch.Tensor:
    return experts.w13_E2FD


class TestFusedSwiGLUOverride(unittest.TestCase):
    def test_grouped_experts_activation_is_replaced(self):
        cfg = GroupedExperts.Config(
            dim=16,
            hidden_dim=32,
            num_experts=4,
        )

        replacement = fused_swiglu(cfg.activation_fn)

        self.assertIsInstance(replacement, FusedSwiGLU.Config)


class TestFusedSwiGLUGroupedExperts(unittest.TestCase):
    """Checkpoint interop and configuration for the fused activation override."""

    def test_saves_native_layout_and_loads_legacy_layout(self):
        """Native state uses W13 while legacy W1/W3 checkpoints still load."""
        src = _build_fused_swiglu_grouped_experts()
        sd = src.state_dict()

        self.assertEqual(set(sd), {"w13_E2FD", "w2_EDF"})
        legacy_sd = {
            "w1_EFD": src.w13_E2FD[:, 0].contiguous(),
            "w2_EDF": src.w2_EDF,
            "w3_EFD": src.w13_E2FD[:, 1].contiguous(),
        }

        dst = _build_fused_swiglu_grouped_experts()
        dst.load_state_dict(legacy_sd)
        self.assertTrue(torch.equal(dst.w13_E2FD, src.w13_E2FD))
        self.assertTrue(torch.equal(dst.w2_EDF, src.w2_EDF))

    def test_built_module_has_only_fused_params(self):
        """The override keeps the default physical w13_E2FD parameter layout."""
        fused = _build_fused_swiglu_grouped_experts()
        names = {name for name, _ in fused.named_parameters(recurse=False)}
        self.assertEqual(names, {"w13_E2FD", "w2_EDF"})
        self.assertEqual(tuple(fused.w13_E2FD.shape), (_E, 2, _HIDDEN, _DIM))

    def test_param_init_and_native_sharding_use_w13_e2fd(self):
        """Building uses logical initialization and native w13_E2FD sharding."""
        colwise = dense_param_placement(tp=spmd.S(2))
        rowwise = dense_param_placement(tp=spmd.S(2))  # w2_EDF
        base_sharding = ShardingConfig(
            state_shardings={
                "w13_E2FD": colwise,
                "w2_EDF": rowwise,
            },
            in_src_shardings={"x_RD": colwise},
            local_spmd=True,
        )
        cfg = GroupedExperts.Config(
            dim=_DIM,
            hidden_dim=_HIDDEN,
            num_experts=_E,
            param_init=fused_grouped_experts_param_init(
                {
                    "w1_EFD": lambda t: torch.nn.init.constant_(t, 1.0),
                    "w2_EDF": lambda t: torch.nn.init.constant_(t, 0.0),
                    "w3_EFD": lambda t: torch.nn.init.constant_(t, 2.0),
                }
            ),
            sharding_config=base_sharding,
        )

        module = cfg.build()

        assert module._param_init is not None
        self.assertEqual(set(module._param_init), {"w13_E2FD", "w2_EDF"})
        module.init_states()
        logical_w13 = _logical_w13(module)
        self.assertTrue(torch.all(logical_w13[:, 0] == 1.0))
        self.assertTrue(torch.all(logical_w13[:, 1] == 2.0))
        self.assertTrue(torch.all(module.w2_EDF == 0.0))

        # The native w13_E2FD/w2 shardings and the rest of the config are preserved.
        sc = module._sharding_config
        assert sc is not None
        self.assertEqual(set(sc.state_shardings), {"w13_E2FD", "w2_EDF"})
        self.assertIs(sc.state_shardings["w13_E2FD"], colwise)
        self.assertIs(sc.state_shardings["w2_EDF"], rowwise)
        self.assertIs(sc.in_src_shardings, base_sharding.in_src_shardings)
        self.assertEqual(sc.local_spmd, base_sharding.local_spmd)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedSwiGLUGroupedExpertsNumerics(unittest.TestCase):
    """The Triton activation override must match the torch-native default."""

    def test_structured_w13_matches_interleaved_layout_bitwise(self):
        torch.manual_seed(0)
        logical_w13_E2FD = torch.randn(
            _E, 2, _HIDDEN, _DIM, device="cuda", dtype=torch.bfloat16
        )
        structured_w13_E2FD = logical_w13_E2FD.detach().clone().requires_grad_()
        interleaved_w13_E_2F_D = (
            logical_w13_E2FD.transpose(1, 2).contiguous().flatten(1, 2).requires_grad_()
        )
        num_tokens_per_expert_E = torch.tensor([3, 2, 1, 2], device="cuda")
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        x_RD = torch.randn(
            int(num_tokens_per_expert_E.sum()),
            _DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        structured_x_RD = x_RD.detach().clone().requires_grad_()
        interleaved_x_RD = x_RD.detach().clone().requires_grad_()

        structured_R2F = torch._grouped_mm(
            structured_x_RD,
            structured_w13_E2FD.flatten(1, 2).transpose(-2, -1),
            offs=offsets_E,
        ).unflatten(-1, (2, _HIDDEN))
        interleaved_RF2 = torch._grouped_mm(
            interleaved_x_RD,
            interleaved_w13_E_2F_D.transpose(-2, -1),
            offs=offsets_E,
        ).unflatten(-1, (_HIDDEN, 2))
        self.assertTrue(torch.equal(structured_R2F, interleaved_RF2.transpose(-2, -1)))

        grad_R2F = torch.randn_like(structured_R2F)
        structured_R2F.backward(grad_R2F)
        interleaved_RF2.backward(grad_R2F.transpose(-2, -1).contiguous())
        assert structured_x_RD.grad is not None
        assert interleaved_x_RD.grad is not None
        assert structured_w13_E2FD.grad is not None
        assert interleaved_w13_E_2F_D.grad is not None
        self.assertTrue(torch.equal(structured_x_RD.grad, interleaved_x_RD.grad))
        self.assertTrue(
            torch.equal(
                structured_w13_E2FD.grad,
                interleaved_w13_E_2F_D.grad.unflatten(1, (_HIDDEN, 2)).transpose(1, 2),
            )
        )

    def test_default_matches_unfused_reference(self):
        torch.manual_seed(0)
        experts = (
            GroupedExperts.Config(
                dim=_DIM,
                hidden_dim=_HIDDEN,
                num_experts=_E,
            )
            .build()
            .cuda()
        )

        w1_EFD = (0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")).requires_grad_()
        w2_EDF = (0.1 * torch.randn(_E, _DIM, _HIDDEN, device="cuda")).requires_grad_()
        w3_EFD = (0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")).requires_grad_()
        with torch.no_grad():
            logical_w13 = _logical_w13(experts)
            logical_w13[:, 0].copy_(w1_EFD)
            logical_w13[:, 1].copy_(w3_EFD)
            experts.w2_EDF.copy_(w2_EDF)

        num_tokens_per_expert_E = torch.tensor([3, 2, 1, 2], device="cuda")
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        x_RD = torch.randn(int(num_tokens_per_expert_E.sum()), _DIM, device="cuda")
        actual_input_RD = x_RD.detach().clone().requires_grad_()
        expected_input_RD = x_RD.detach().clone().requires_grad_()

        actual_RD = experts(actual_input_RD, num_tokens_per_expert_E)
        gate_RF = torch._grouped_mm(
            expected_input_RD.bfloat16(),
            w1_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        up_RF = torch._grouped_mm(
            expected_input_RD.bfloat16(),
            w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        expected_RD = torch._grouped_mm(
            torch.nn.functional.silu(gate_RF) * up_RF,
            w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(expected_input_RD)

        torch.testing.assert_close(actual_RD, expected_RD, atol=2e-2, rtol=2e-2)
        actual_RD.sum().backward()
        expected_RD.sum().backward()

        assert actual_input_RD.grad is not None
        assert expected_input_RD.grad is not None
        assert experts.w13_E2FD.grad is not None
        assert experts.w2_EDF.grad is not None
        assert w1_EFD.grad is not None
        assert w2_EDF.grad is not None
        assert w3_EFD.grad is not None
        torch.testing.assert_close(
            actual_input_RD.grad, expected_input_RD.grad, atol=2e-2, rtol=2e-2
        )
        logical_w13_grad = experts.w13_E2FD.grad
        torch.testing.assert_close(
            logical_w13_grad[:, 0], w1_EFD.grad, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            logical_w13_grad[:, 1], w3_EFD.grad, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            experts.w2_EDF.grad, w2_EDF.grad, atol=2e-2, rtol=2e-2
        )

    def test_fused_activation_matches_default(self):
        torch.manual_seed(0)
        stock = (
            GroupedExperts.Config(
                dim=_DIM,
                hidden_dim=_HIDDEN,
                num_experts=_E,
            )
            .build()
            .cuda()
        )
        fused = (
            GroupedExperts.Config(
                dim=_DIM,
                hidden_dim=_HIDDEN,
                num_experts=_E,
                activation_fn=FusedSwiGLU.Config(),
            )
            .build()
            .cuda()
        )

        with torch.no_grad():
            w1 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w3 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w2 = 0.1 * torch.randn(_E, _DIM, _HIDDEN, device="cuda")
            stock_logical_w13 = _logical_w13(stock)
            stock_logical_w13[:, 0].copy_(w1)
            stock_logical_w13[:, 1].copy_(w3)
            stock.w2_EDF.copy_(w2)
            fused_logical_w13 = _logical_w13(fused)
            fused_logical_w13[:, 0].copy_(w1)
            fused_logical_w13[:, 1].copy_(w3)
            fused.w2_EDF.copy_(w2)

        # Tokens grouped by expert (positional), summing to the row count.
        num_tokens = torch.tensor([3, 2, 1, 2], device="cuda")
        rows = int(num_tokens.sum())
        x = torch.randn(rows, _DIM, device="cuda")
        x_stock = x.detach().clone().requires_grad_()
        x_fused = x.detach().clone().requires_grad_()

        out_stock = stock(x_stock, num_tokens)
        out_fused = fused(x_fused, num_tokens)
        # bf16 grouped_mm + fp32 silu_and_mul kernel vs two GEMMs: close, not exact.
        torch.testing.assert_close(out_fused, out_stock, atol=2e-2, rtol=2e-2)

        out_stock.sum().backward()
        out_fused.sum().backward()
        assert x_stock.grad is not None and x_fused.grad is not None
        torch.testing.assert_close(x_fused.grad, x_stock.grad, atol=2e-2, rtol=2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedSwiGLUOverrideKernels(unittest.TestCase):
    def test_silu_and_mul_custom_op_matches_reference_with_offsets(self):
        gate = torch.randn(3, 2, device="cuda", requires_grad=True)
        up = torch.randn(3, 2, device="cuda", requires_grad=True)
        offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

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
        gate = torch.randn(3, 2, device="cuda", requires_grad=True)
        up = torch.randn(3, 2, device="cuda", requires_grad=True)

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
            device="cuda",
            requires_grad=True,
        )
        up = torch.tensor(
            [
                [2.0, 3.0],
                [5.0, 7.0],
                [11.0, 13.0],
            ],
            device="cuda",
            requires_grad=True,
        )
        offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

        out = silu_and_mul_forward_kernel(gate, up, offsets)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(out[:2], expected[:2])

        grad_out = torch.tensor(
            [
                [17.0, 19.0],
                [23.0, 29.0],
                [31.0, 37.0],
            ],
            device="cuda",
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

    def test_silu_and_mul_uses_int64_row_stride_arithmetic(self):
        row = 262_144
        row_stride = 8192
        storage_numel = row * row_stride + 2
        required_bytes = storage_numel * 2 + 512 * 1024**2
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < required_bytes:
            self.skipTest(
                f"need at least {required_bytes} free CUDA bytes, got {free_bytes}"
            )

        storage = torch.empty(storage_numel, device="cuda", dtype=torch.bfloat16)
        gate = torch.as_strided(storage, (row + 1, 1), (row_stride, 2))
        up = torch.as_strided(storage, (row + 1, 1), (row_stride, 2), 1)
        gate[row] = 1.0
        up[row] = 2.0
        offsets = torch.tensor([row + 1], device="cuda", dtype=torch.int32)

        out = silu_and_mul_forward_kernel(gate, up, offsets)
        grad_out = torch.zeros_like(out)
        grad_out[row] = 3.0
        grad_gate, grad_up = silu_and_mul_backward_kernel(
            grad_out,
            gate,
            up,
            offsets,
        )
        torch.cuda.synchronize()

        ref_gate = torch.tensor(
            [1.0], device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ref_up = torch.tensor(
            [2.0], device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        expected = torch.nn.functional.silu(ref_gate) * ref_up
        expected.backward(torch.tensor([3.0], device="cuda", dtype=torch.bfloat16))
        assert ref_gate.grad is not None
        assert ref_up.grad is not None
        torch.testing.assert_close(out[row], expected.detach())
        torch.testing.assert_close(grad_gate[row], ref_gate.grad)
        torch.testing.assert_close(grad_up[row], ref_up.grad)
