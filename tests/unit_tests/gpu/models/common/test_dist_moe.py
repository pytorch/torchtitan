# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity between stock and Dist-MoE routed experts."""

import unittest
from typing import Any, cast

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

dist_moe = pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE parity requires the optional dist_moe package",
)

from torchtitan.distributed.spmd_types import (  # noqa: E402
    set_current_spmd_mesh,
    set_spmd_meshes,
)
from torchtitan.models.base.module import Module  # noqa: E402
from torchtitan.models.common.dist_moe import (  # noqa: E402
    DistMoeRoutedExperts,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.dist_moe.mxfp8 import _prepare_mxfp8_weight  # noqa: E402
from torchtitan.models.common.linear import GroupedLinear  # noqa: E402
from torchtitan.models.common.moe import RoutedExperts  # noqa: E402
from torchtitan.models.common.moe.token_dispatcher import (  # noqa: E402
    AllToAllTokenDispatcher,
)
from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU  # noqa: E402


# Shape suffixes: T = local tokens, K = selected experts, E = local experts,
# F = expert intermediate dimension, and D = model dimension.
_T = 16
_D = 256
_F = 256
_K = 3
_E_LOCAL = 4
_TOL = 1e-4
# MXFP8 quantizes the input, W13, the SwiGLU output, and W2 with E4M3 values
# (3 mantissa bits, at most 2^-4 relative rounding error each) under shared
# power-of-two block scales. The output's normwise relative error against BF16
# measures about 0.07 at these shapes, while swapped gate/up rows measure about
# 0.9, so this bound separates quantization error from layout errors.
_MXFP8_REL_TOL = 0.12


def _stock_routed_experts(
    *,
    num_experts: int,
    w13_E2FD: torch.Tensor,
    w2_EDF: torch.Tensor,
) -> RoutedExperts:
    """Build the production stock expert path with rank-local weights.

    Production EP sharding reduces each grouped linear to local experts while
    the dispatcher retains the global expert count. This focused test builds
    that post-sharding module state directly.
    """
    experts = RoutedExperts.__new__(RoutedExperts)
    Module.__init__(experts)
    experts.w13 = (
        GroupedLinear.Config(
            group_size=_E_LOCAL,
            in_features=_D,
            out_features=_F,
            num_linears=2,
        )
        .build()
        .to(device=w13_E2FD.device, dtype=torch.bfloat16)
    )
    experts.w2 = (
        GroupedLinear.Config(
            group_size=_E_LOCAL,
            in_features=_F,
            out_features=_D,
        )
        .build()
        .to(device=w2_EDF.device, dtype=torch.bfloat16)
    )
    with torch.no_grad():
        experts.w13.weight.copy_(w13_E2FD)
        experts.w2.weight.copy_(w2_EDF)
    experts.activation_fn = FusedSwiGLU.Config().build()
    experts.output_postprocess = None
    experts.token_dispatcher = AllToAllTokenDispatcher.Config(
        num_experts=num_experts,
        top_k=_K,
    ).build()
    experts.token_dispatcher.init_buffer()
    return experts


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the normwise relative error of ``actual`` against ``expected``."""
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistMoeNumerics(DTensorTestBase):
    """Compare the complete stock and Annex BF16 expert modules."""

    @property
    def world_size(self) -> int:
        return 2

    def _set_up_ep_inputs(self, *, gate_std: float = 0.02) -> tuple[Any, ...]:
        """Build the EP2 mesh and one rank's tokens, expert weights, and routing."""
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Dist-MoE requires an SM100 or SM103 GPU")

        ep_size = self.world_size
        num_experts = _E_LOCAL * ep_size
        device = torch.device(self.device_type, self.rank)
        mesh = init_device_mesh(
            self.device_type,
            (ep_size,),
            mesh_dim_names=("ep",),
        )
        set_spmd_meshes(
            dense_mesh=mesh,
            sparse_mesh=mesh,
            dense_sp_enabled=False,
        )

        generator = torch.Generator(device=device).manual_seed(1234 + self.rank)
        x_TD = torch.randn(
            _T,
            _D,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        # Match the first DeepSeek MoE layer: W1 uses the base standard
        # deviation, while W3 and W2 use its depth-scaled value.
        w13_E2FD = torch.empty(
            _E_LOCAL,
            2,
            _F,
            _D,
            device=device,
            dtype=torch.bfloat16,
        )
        torch.nn.init.trunc_normal_(w13_E2FD[:, 0], std=gate_std, generator=generator)
        torch.nn.init.trunc_normal_(w13_E2FD[:, 1], std=0.01, generator=generator)
        w2_EDF = torch.empty(
            _E_LOCAL,
            _D,
            _F,
            device=device,
            dtype=torch.bfloat16,
        )
        torch.nn.init.trunc_normal_(w2_EDF, std=0.01, generator=generator)
        topk_scores_TK = torch.rand(
            _T,
            _K,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        topk_scores_TK /= topk_scores_TK.sum(dim=-1, keepdim=True)
        global_token_ids_T = self.rank * _T + torch.arange(_T, device=device)
        topk_expert_ids_TK = (
            global_token_ids_T[:, None] * _K + torch.arange(_K, device=device)[None, :]
        ) % num_experts
        routing_map_TE = torch.zeros(
            _T, num_experts, dtype=torch.bool, device=device
        ).scatter_(1, topk_expert_ids_TK, True)
        return (
            mesh,
            num_experts,
            device,
            generator,
            x_TD,
            w13_E2FD,
            w2_EDF,
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        )

    @with_comms
    def test_bf16_matches_stock_fused_swiglu(self) -> None:
        """Real EP2 forward and backward agree at BF16 accumulation tolerance."""
        (
            mesh,
            num_experts,
            device,
            generator,
            x_TD,
            w13_E2FD,
            w2_EDF,
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        ) = self._set_up_ep_inputs()
        grad_out_TD = 0.02 * torch.randn(
            _T,
            _D,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )

        stock = _stock_routed_experts(
            num_experts=num_experts,
            w13_E2FD=w13_E2FD,
            w2_EDF=w2_EDF,
        )
        stock_x_TD = x_TD.clone().requires_grad_()
        stock_scores_TK = topk_scores_TK.clone().requires_grad_()
        with set_current_spmd_mesh(mesh):
            stock_out_TD = stock(
                stock_x_TD,
                stock_scores_TK,
                topk_expert_ids_TK,
                routing_map_TE,
            )
        stock_out_TD.backward(grad_out_TD)

        config = dist_moe.Config(
            max_num_local_input_tokens=_T,
            hidden_dim=_D,
            intermediate_dim=_F,
            top_k=_K,
            num_experts=num_experts,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=1.0,
            activation_slot_capacity_factor=1.0,
            num_activation_slots=1,
        )
        context = dist_moe.create_context(
            group=torch.distributed.group.WORLD,
            config=config,
            device=device,
        )
        try:
            annex = (
                DistMoeRoutedExperts.Config(
                    w13=GroupedLinear.Config(
                        group_size=_E_LOCAL,
                        in_features=_D,
                        out_features=_F,
                        num_linears=2,
                    ),
                    w2=GroupedLinear.Config(
                        group_size=_E_LOCAL,
                        in_features=_F,
                        out_features=_D,
                    ),
                    top_k=_K,
                    inplace_wgrad_accum=False,
                )
                .build()
                .to(device=device, dtype=torch.bfloat16)
            )
            with torch.no_grad():
                annex.w13.weight.copy_(w13_E2FD)
                annex.w2.weight.copy_(w2_EDF)
            # Module execution consumes only the context; runtime construction
            # and teardown have separate focused coverage.
            runtime = cast(DistMoeRuntime, object.__new__(DistMoeRuntime))
            runtime.config = DistMoeRuntime.Config()
            runtime.context = context
            annex._runtime = runtime
            annex_x_TD = x_TD.clone().requires_grad_()
            annex_scores_TK = topk_scores_TK.clone().requires_grad_()
            annex_out_TD = annex(
                annex_x_TD,
                annex_scores_TK,
                topk_expert_ids_TK,
                routing_map_TE,
            )
            annex_out_TD.backward(grad_out_TD)
        finally:
            context.close()

        comparisons = (
            ("output", annex_out_TD, stock_out_TD),
            ("input gradient", annex_x_TD.grad, stock_x_TD.grad),
            ("routing-score gradient", annex_scores_TK.grad, stock_scores_TK.grad),
            ("W13 gradient", annex.w13.weight.grad, stock.w13.weight.grad),
            ("W2 gradient", annex.w2.weight.grad, stock.w2.weight.grad),
        )
        for name, actual, expected in comparisons:
            with self.subTest(name=name):
                assert actual is not None
                assert expected is not None
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=_TOL,
                    atol=_TOL,
                )

    @with_comms
    def test_mxfp8_mega_fused_inference_matches_stock(self) -> None:
        """MXFP8 inference weights match BF16 eagerly, replayed, and reloaded."""
        # A large gate std makes SwiGLU nonlinear, so swapped gate/up rows
        # change the output.
        (
            mesh,
            num_experts,
            device,
            _,
            x_TD,
            w13_E2FD,
            w2_EDF,
            *routing,
        ) = self._set_up_ep_inputs(gate_std=0.2)
        # Reloading the experts in another order changes each local expert's
        # weights.
        reload_w13_E2FD = w13_E2FD.roll(1, dims=0)
        reload_w2_EDF = w2_EDF.roll(1, dims=0)

        block_scaled_config = dist_moe.BlockScaledConfig(pipeline="mega")
        context = dist_moe.create_context(
            group=torch.distributed.group.WORLD,
            config=dist_moe.Config(
                max_num_local_input_tokens=_T,
                hidden_dim=_D,
                intermediate_dim=_F,
                top_k=_K,
                num_experts=num_experts,
                max_moe_layers_per_activation_slot=1,
                device_scratch_capacity_factor=1.0,
                activation_slot_capacity_factor=None,
                num_activation_slots=0,
                block_scaled=block_scaled_config,
                inference=True,
            ),
            device=device,
        )
        graph = torch.cuda.CUDAGraph()
        try:
            experts = (
                MXFP8DistMoeRoutedExperts.Config(
                    w13=GroupedLinear.Config(
                        group_size=_E_LOCAL,
                        in_features=_D,
                        out_features=_F,
                        num_linears=2,
                    ),
                    w2=GroupedLinear.Config(
                        group_size=_E_LOCAL,
                        in_features=_F,
                        out_features=_D,
                    ),
                    top_k=_K,
                    block_scaled_config=block_scaled_config,
                    mega_fused_inference=True,
                )
                .build()
                .to(device=device, dtype=torch.bfloat16)
            )
            experts.requires_grad_(False)
            with torch.no_grad():
                experts.w13.weight.copy_(w13_E2FD)
                experts.w2.weight.copy_(w2_EDF)
            # Module execution consumes only the context; runtime construction
            # and teardown have separate focused coverage.
            runtime = cast(DistMoeRuntime, object.__new__(DistMoeRuntime))
            runtime.config = DistMoeRuntime.Config(inference=True)
            runtime.context = context
            runtime.set_num_local_input_tokens_per_call(_T)
            experts._runtime = runtime

            # Dist-MoE inference contexts require grad mode to be disabled
            def forward() -> torch.Tensor:
                with torch.no_grad():
                    return experts(x_TD, *routing)

            # Without FSDP, each call prepares the weights from the parameters
            dynamic_TD = forward()

            # The generator keeps one FSDP unshard alive and refills it in place
            # on weight loads (see VLLMModelWrapper)
            fsdp_mesh = init_device_mesh(
                self.device_type,
                (self.world_size, 1),
                mesh_dim_names=("ep", "dp_shard"),
            )["dp_shard"]
            fully_shard(experts, mesh=fsdp_mesh, reshard_after_forward=False)
            experts.set_keep_unsharded_storage(True)
            eager_TD = forward()
            with torch.cuda.graph(graph):
                graph_TD = forward()
            graph.replay()
            replay_TD = graph_TD.clone()

            # Reload as a generator weight sync does: copy new weights into the
            # sharded parameters, then unshard into the retained operands
            experts.reshard()
            with torch.no_grad():
                experts.w13.weight.to_local().copy_(reload_w13_E2FD)
                experts.w2.weight.to_local().copy_(reload_w2_EDF)
            experts.unshard()
            graph.replay()
            reload_replay_TD = graph_TD.clone()
            reload_eager_TD = forward()
        finally:
            graph.reset()
            context.close()

        def stock_out(w13: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
            stock = _stock_routed_experts(
                num_experts=num_experts, w13_E2FD=w13, w2_EDF=w2
            )
            with torch.no_grad(), set_current_spmd_mesh(mesh):
                return stock(x_TD, *routing)

        error = _relative_error(eager_TD, stock_out(w13_E2FD, w2_EDF))
        assert error < _MXFP8_REL_TOL, error

        # Swapping the gate and up halves models a W13 layout error
        swapped_TD = stock_out(w13_E2FD.flip(1), w2_EDF)
        swapped_error = _relative_error(eager_TD, swapped_TD)
        assert swapped_error > 4 * _MXFP8_REL_TOL, swapped_error
        reload_TD = stock_out(reload_w13_E2FD, reload_w2_EDF)
        reload_error = _relative_error(reload_eager_TD, reload_TD)
        assert reload_error < _MXFP8_REL_TOL, reload_error

        # Every path prepares the same operands from the same weights
        torch.testing.assert_close(dynamic_TD, eager_TD, rtol=0, atol=0)
        torch.testing.assert_close(replay_TD, eager_TD, rtol=0, atol=0)
        torch.testing.assert_close(reload_replay_TD, reload_eager_TD, rtol=0, atol=0)


def _blocked_scales_to_rows(scale: torch.Tensor) -> torch.Tensor:
    """Read cuBLAS blocked scales as one row per weight row.

    Uses the layout's offset formula directly: 128x4 tiles are stored one
    after another in row-major tile order, and scale (r, c) of a tile sits at
    (r % 32) * 16 + (r // 32) * 4 + c within it.
    """
    num_rows, num_cols = scale.shape
    r = torch.arange(num_rows, device=scale.device)[:, None]
    c = torch.arange(num_cols, device=scale.device)[None, :]
    tile = (r // 128) * (num_cols // 4) + c // 4
    offset = tile * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4
    return scale.view(torch.uint8).flatten()[offset]


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in ((10, 0), (10, 3)),
    "Dist-MoE requires an SM100 or SM103 GPU",
)
class TestMXFP8InferenceWeights(unittest.TestCase):
    def test_32x32_inference_weights_match_training(self) -> None:
        """Inference W13/W2 operands equal the training FPROP operands.

        The qdata and scales must be bitwise identical to the training FPROP
        operands from ``prepare_block_scaled_weight``, with W13's gate/up rows
        reordered as gate 0-7, up 0-7, gate 8-15, ... and W2 unchanged. The
        expected row order and the scale layout are computed here independently
        of the annex.
        """
        num_experts, num_rows = 2, 2 * _F
        generator = torch.Generator(device="cuda").manual_seed(0)
        weight_EOI = torch.randn(
            num_experts, num_rows, _D, device="cuda", generator=generator
        )
        # Give each 32x32 block a distinct magnitude, so misplaced scales
        # cannot match by chance
        rows = torch.arange(num_rows, device="cuda")[:, None]
        cols = torch.arange(_D, device="cuda")[None, :]
        weight_EOI = (weight_EOI * 2.0 ** (rows // 32 % 8 + cols // 32 % 4)).bfloat16()

        operands = _prepare_mxfp8_weight(weight_EOI, gate_up=True, inference=True)
        qdata, scale = operands.qdata, operands.fprop_scale
        self.assertEqual(operands.dgrad_scale.numel(), 0)
        training = dist_moe.prepare_block_scaled_weight(
            weight_EOI, dist_moe.BlockScaledConfig()
        )

        # Gate rows are the first half of W13 and up rows the second
        gate_groups = torch.arange(num_rows // 2, device="cuda").view(-1, 8)
        interleaved_order = torch.stack(
            (gate_groups, gate_groups + num_rows // 2), dim=1
        ).flatten()
        self.assertEqual(qdata.dtype, training.fprop_data.dtype)
        self.assertEqual(scale.dtype, training.fprop_scale.dtype)
        self.assertEqual(scale.shape, training.fprop_scale.shape)
        torch.testing.assert_close(
            qdata.view(torch.uint8),
            training.fprop_data.view(torch.uint8)[:, interleaved_order],
            rtol=0,
            atol=0,
        )
        training_scale = _blocked_scales_to_rows(training.fprop_scale)
        torch.testing.assert_close(
            _blocked_scales_to_rows(scale).view(num_experts, num_rows, -1),
            training_scale.view(num_experts, num_rows, -1)[:, interleaved_order],
            rtol=0,
            atol=0,
        )

        # W2 is not interleaved, so it matches the training operands directly
        weight_EDF = weight_EOI.transpose(1, 2).contiguous()
        operands = _prepare_mxfp8_weight(weight_EDF, gate_up=False, inference=True)
        training = dist_moe.prepare_block_scaled_weight(
            weight_EDF, dist_moe.BlockScaledConfig()
        )
        self.assertEqual(operands.dgrad_scale.numel(), 0)
        for actual, expected in (
            (operands.qdata, training.fprop_data),
            (operands.fprop_scale, training.fprop_scale),
        ):
            torch.testing.assert_close(
                actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
