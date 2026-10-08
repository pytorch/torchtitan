# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity between stock and Dist-MoE routed experts."""

import unittest
from typing import cast

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
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
from torchtitan.models.common.dist_moe import (  # noqa: E402
    DistMoeRoutedExperts,
    DistMoeRuntime,
)
from torchtitan.models.common.linear import GroupedLinear  # noqa: E402
from torchtitan.models.common.moe import RoutedExperts  # noqa: E402
from torchtitan.models.common.token_dispatcher import (  # noqa: E402
    AllToAllTokenDispatcher,
)
from torchtitan.protocols.module import Module  # noqa: E402
from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU  # noqa: E402


# Shape suffixes: T = local tokens, K = selected experts, E = local experts,
# F = expert intermediate dimension, and D = model dimension.
_T = 16
_D = 256
_F = 256
_K = 3
_E_LOCAL = 4
_TOL = 1e-4


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


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistMoeNumerics(DTensorTestBase):
    """Compare the complete stock and Annex BF16 expert modules."""

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_bf16_matches_stock_fused_swiglu(self) -> None:
        """Real EP2 forward and backward agree at BF16 accumulation tolerance."""
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
        torch.nn.init.trunc_normal_(w13_E2FD[:, 0], std=0.02, generator=generator)
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
        num_local_tokens_per_expert_E = torch.bincount(
            topk_expert_ids_TK.flatten(),
            minlength=num_experts,
        )
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
                num_local_tokens_per_expert_E,
            )
        stock_out_TD.backward(grad_out_TD)

        config = dist_moe.Config(
            num_local_input_tokens=_T,
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
            runtime.context = context
            annex._runtime = runtime
            annex_x_TD = x_TD.clone().requires_grad_()
            annex_scores_TK = topk_scores_TK.clone().requires_grad_()
            annex_out_TD = annex(
                annex_x_TD,
                annex_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
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


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
