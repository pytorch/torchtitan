# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    LocalTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_minimal_async_ep,
)
from torchtitan.overrides.fused_swiglu import (
    fused_grouped_experts,
    FusedGroupedExperts,
    _fused_silu_and_mul,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
    silu_and_mul_op,
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

_DIM = 16
_HIDDEN = 32
_E = 4


def _build_fused_grouped_experts() -> FusedGroupedExperts:
    fused = FusedGroupedExperts.Config(
        dim=_DIM,
        hidden_dim=_HIDDEN,
        num_experts=_E,
        token_dispatcher=LocalTokenDispatcher.Config(num_experts=_E, top_k=1),
    ).build()
    with torch.no_grad():
        fused.w13.copy_(torch.randn(_E, _HIDDEN, 2, _DIM))
        fused.w2_EDF.copy_(torch.randn(_E, _DIM, _HIDDEN))
    return fused


class TestFusedSwiGLUOverride(unittest.TestCase):
    def test_minimal_async_ep_config_imports_override(self):
        config = deepseek_v3_debugmodel_minimal_async_ep()

        self.assertIn(
            "torchtitan.overrides.fused_swiglu",
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


class TestFusedGroupedExperts(unittest.TestCase):
    """Checkpoint interop (state_dict hooks) and override config remap for the
    w13-fused grouped experts"""

    def test_saves_and_loads_stock_layout(self):
        """tests save and load checkpoint"""
        src = _build_fused_grouped_experts()
        sd = src.state_dict()

        self.assertEqual(set(sd), {"w1_EFD", "w3_EFD", "w2_EDF"})
        self.assertTrue(torch.equal(sd["w1_EFD"], src.w13[:, :, 0, :]))
        self.assertTrue(torch.equal(sd["w3_EFD"], src.w13[:, :, 1, :]))

        dst = _build_fused_grouped_experts()
        dst.load_state_dict(sd)
        self.assertTrue(torch.equal(dst.w13, src.w13))
        self.assertTrue(torch.equal(dst.w2_EDF, src.w2_EDF))

    def test_built_module_has_only_fused_params(self):
        """__init__ replaces w1_EFD/w3_EFD with the fused w13 (E, hidden, 2, dim)."""
        fused = _build_fused_grouped_experts()
        names = {name for name, _ in fused.named_parameters(recurse=False)}
        self.assertEqual(names, {"w13", "w2_EDF"})
        self.assertEqual(tuple(fused.w13.shape), (_E, _HIDDEN, 2, _DIM))

    def test_param_init_and_sharding_remapped_to_w13(self):
        """The override remaps both per-param init and state shardings from the
        separate w1_EFD/w3_EFD onto w13, keeps w2_EDF, and preserves the rest of
        the sharding config (in/out shardings, local_map)."""
        colwise = dense_param_placement(tp=spmd.S(1))  # w1_EFD/w3_EFD: shard hidden
        rowwise = dense_param_placement(tp=spmd.S(2))  # w2_EDF
        base_sharding = ShardingConfig(
            state_shardings={
                "w1_EFD": colwise,
                "w2_EDF": rowwise,
                "w3_EFD": colwise,
            },
            in_src_shardings={"x_BLD": colwise},
            local_map=LocalMapConfig(in_grad_placements=None),
        )
        cfg = GroupedExperts.Config(
            dim=_DIM,
            hidden_dim=_HIDDEN,
            num_experts=_E,
            token_dispatcher=LocalTokenDispatcher.Config(num_experts=_E, top_k=1),
            param_init={
                "w1_EFD": lambda t: t.fill_(1.0),
                "w2_EDF": lambda t: t.fill_(0.0),
                "w3_EFD": lambda t: t.fill_(2.0),
            },
            sharding_config=base_sharding,
        )

        replacement = fused_grouped_experts(cfg)

        # param_init: w1_EFD/w3_EFD fused into w13, w2_EDF untouched.
        assert replacement.param_init is not None
        self.assertEqual(set(replacement.param_init), {"w13", "w2_EDF"})
        # The w13 initializer applies each half's initializer to the right slice.
        t = torch.zeros(_E, _HIDDEN, 2, _DIM)
        replacement.param_init["w13"](t)
        self.assertTrue(torch.all(t[:, :, 0, :] == 1.0))  # gate (w1_EFD)
        self.assertTrue(torch.all(t[:, :, 1, :] == 2.0))  # up (w3_EFD)

        # state_shardings: w13 inherits w1_EFD's placement; w2_EDF kept; the
        # rest of the sharding config is preserved (same objects via replace()).
        sc = replacement.sharding_config
        assert sc is not None
        self.assertEqual(set(sc.state_shardings), {"w13", "w2_EDF"})
        self.assertIs(sc.state_shardings["w13"], colwise)
        self.assertIs(sc.state_shardings["w2_EDF"], rowwise)
        self.assertIs(sc.in_src_shardings, base_sharding.in_src_shardings)
        self.assertIs(sc.local_map, base_sharding.local_map)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedGroupedExpertsNumerics(unittest.TestCase):
    """The w13-fused experts must match the stock (unfused) GroupedExperts given
    the same weights."""

    def test_matches_unfused_forward_and_backward(self):
        torch.manual_seed(0)
        dispatcher = LocalTokenDispatcher.Config(num_experts=_E, top_k=1)
        stock = (
            GroupedExperts.Config(
                dim=_DIM,
                hidden_dim=_HIDDEN,
                num_experts=_E,
                token_dispatcher=dispatcher,
            )
            .build()
            .cuda()
        )
        fused = (
            FusedGroupedExperts.Config(
                dim=_DIM,
                hidden_dim=_HIDDEN,
                num_experts=_E,
                token_dispatcher=dispatcher,
            )
            .build()
            .cuda()
        )

        with torch.no_grad():
            w1 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w3 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w2 = 0.1 * torch.randn(_E, _DIM, _HIDDEN, device="cuda")
            stock.w1_EFD.copy_(w1)
            stock.w3_EFD.copy_(w3)
            stock.w2_EDF.copy_(w2)
            fused.w13[:, :, 0, :].copy_(w1)
            fused.w13[:, :, 1, :].copy_(w3)
            fused.w2_EDF.copy_(w2)

        # Tokens grouped by expert (positional), summing to the row count.
        num_tokens = torch.tensor([3, 2, 1, 2], device="cuda")
        rows = int(num_tokens.sum())
        x = torch.randn(rows, _DIM, device="cuda")
        x_stock = x.detach().clone().requires_grad_()
        x_fused = x.detach().clone().requires_grad_()

        out_stock = stock._experts_forward(x_stock, num_tokens)
        out_fused = fused._experts_forward(x_fused, num_tokens)
        # bf16 grouped_mm + fp32 silu_and_mul kernel vs two GEMMs: close, not exact.
        torch.testing.assert_close(out_fused, out_stock, atol=2e-2, rtol=2e-2)

        out_stock.sum().backward()
        out_fused.sum().backward()
        assert x_stock.grad is not None and x_fused.grad is not None
        torch.testing.assert_close(x_fused.grad, x_stock.grad, atol=2e-2, rtol=2e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedSwiGLUSpmdTypes(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_fused_silu_and_mul_typechecked_bitwise(self):
        mesh = init_device_mesh(
            self.device_type,
            (1, 1, self.world_size),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        torch.manual_seed(1234)
        gate = torch.randn(
            2, 3, 16, device=self.device_type, dtype=torch.float32
        ).requires_grad_()
        up = torch.randn(
            2, 3, 16, device=self.device_type, dtype=torch.float32
        ).requires_grad_()
        grad = torch.randn_like(gate)

        ref_gate = gate.detach().clone().requires_grad_()
        ref_up = up.detach().clone().requires_grad_()
        ref_out = _fused_silu_and_mul(ref_gate, ref_up)
        ref_out.backward(grad)

        local_type = {
            "dp": spmd.S(0),
            "cp": spmd.S(1),
            "tp": spmd.S(2),
        }
        set_spmd_backend("spmd_types")
        try:
            with set_current_spmd_mesh(mesh):
                with typecheck(strict_mode="strict", local=True):
                    typed_gate = spmd.assert_type(gate, local_type)
                    typed_up = spmd.assert_type(up, local_type)
                    out = _fused_silu_and_mul(typed_gate, typed_up)
                    spmd.assert_type(out, local_type)
        finally:
            set_spmd_backend("default")

        out.backward(grad)
        self.assertTrue(torch.equal(out, ref_out))
        self.assertTrue(torch.equal(gate.grad, ref_gate.grad))
        self.assertTrue(torch.equal(up.grad, ref_up.grad))


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
