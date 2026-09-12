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
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_minimal_async_ep,
)
from torchtitan.overrides.fused_swiglu import (
    fused_grouped_experts,
    FusedSwiGLU,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
    silu_and_mul_op,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

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
        fused.w13.copy_(torch.randn(_E, 2 * _HIDDEN, _DIM))
        fused.w2_EDF.copy_(torch.randn(_E, _DIM, _HIDDEN))
    return fused


def _logical_w13(experts: GroupedExperts) -> torch.Tensor:
    return experts.w13.unflatten(1, (_HIDDEN, 2))


class TestFusedSwiGLUOverride(unittest.TestCase):
    def test_minimal_async_ep_config_imports_override(self):
        config = deepseek_v3_debugmodel_minimal_async_ep(seq_len=2048)

        self.assertIn(
            "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
            config.override.imports,
        )

    def test_grouped_experts_config_is_replaced(self):
        cfg = GroupedExperts.Config(
            dim=16,
            hidden_dim=32,
            num_experts=4,
        )

        replacement = fused_grouped_experts(cfg)

        self.assertIsInstance(replacement, GroupedExperts.Config)
        self.assertIsInstance(replacement.activation_fn, FusedSwiGLU.Config)


class TestFusedSwiGLUGroupedExperts(unittest.TestCase):
    """Checkpoint interop and configuration for the fused activation override."""

    def test_saves_and_loads_physical_layout(self):
        src = _build_fused_swiglu_grouped_experts()
        sd = src.state_dict()

        self.assertEqual(set(sd), {"w13", "w2_EDF"})

        dst = _build_fused_swiglu_grouped_experts()
        dst.load_state_dict(sd)
        self.assertTrue(torch.equal(dst.w13, src.w13))
        self.assertTrue(torch.equal(dst.w2_EDF, src.w2_EDF))

    def test_built_module_has_only_fused_params(self):
        """The override keeps the default physical w13 parameter layout."""
        fused = _build_fused_swiglu_grouped_experts()
        names = {name for name, _ in fused.named_parameters(recurse=False)}
        self.assertEqual(names, {"w13", "w2_EDF"})
        self.assertEqual(tuple(fused.w13.shape), (_E, 2 * _HIDDEN, _DIM))

    def test_param_init_and_sharding_remapped_to_w13(self):
        """Building remaps logical initialization and sharding onto w13."""
        colwise = dense_param_placement(tp=spmd.S(1))  # w1_EFD/w3_EFD: shard hidden
        rowwise = dense_param_placement(tp=spmd.S(2))  # w2_EDF
        base_sharding = ShardingConfig(
            state_shardings={
                "w1_EFD": colwise,
                "w2_EDF": rowwise,
                "w3_EFD": colwise,
            },
            in_src_shardings={"x_RD": colwise},
            local_map=LocalMapConfig(in_grad_placements=None),
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

        replacement = fused_grouped_experts(cfg)
        module = replacement.build()

        assert module._param_init is not None
        self.assertEqual(set(module._param_init), {"w13", "w2_EDF"})
        module.init_states()
        logical_w13 = _logical_w13(module)
        self.assertTrue(torch.all(logical_w13[:, :, 0, :] == 1.0))
        self.assertTrue(torch.all(logical_w13[:, :, 1, :] == 2.0))
        self.assertTrue(torch.all(module.w2_EDF == 0.0))

        # state_shardings: w13 inherits w1_EFD's placement; w2_EDF kept; the
        # rest of the sharding config is preserved (same objects via replace()).
        sc = module._sharding_config
        assert sc is not None
        self.assertEqual(set(sc.state_shardings), {"w13", "w2_EDF"})
        self.assertIs(sc.state_shardings["w13"], colwise)
        self.assertIs(sc.state_shardings["w2_EDF"], rowwise)
        self.assertIs(sc.in_src_shardings, base_sharding.in_src_shardings)
        self.assertIs(sc.local_map, base_sharding.local_map)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedSwiGLUGroupedExpertsNumerics(unittest.TestCase):
    """The Triton activation override must match the torch-native default."""

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
            logical_w13[:, :, 0, :].copy_(w1_EFD)
            logical_w13[:, :, 1, :].copy_(w3_EFD)
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
        assert experts.w13.grad is not None
        assert experts.w2_EDF.grad is not None
        assert w1_EFD.grad is not None
        assert w2_EDF.grad is not None
        assert w3_EFD.grad is not None
        torch.testing.assert_close(
            actual_input_RD.grad, expected_input_RD.grad, atol=2e-2, rtol=2e-2
        )
        logical_w13_grad = experts.w13.grad.unflatten(1, (_HIDDEN, 2))
        torch.testing.assert_close(
            logical_w13_grad[:, :, 0, :], w1_EFD.grad, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            logical_w13_grad[:, :, 1, :], w3_EFD.grad, atol=2e-2, rtol=2e-2
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
            stock_logical_w13[:, :, 0, :].copy_(w1)
            stock_logical_w13[:, :, 1, :].copy_(w3)
            stock.w2_EDF.copy_(w2)
            fused_logical_w13 = _logical_w13(fused)
            fused_logical_w13[:, :, 0, :].copy_(w1)
            fused_logical_w13[:, :, 1, :].copy_(w3)
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
