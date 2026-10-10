# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in DeepSeek V3 learned routing and auxiliary loss, with fused backward.

The gate stays native; the fusion consumes its FP32 logits. The specialization
and activation instructions are in ``docs/fused-dsv3-router.md``.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat
from torch.autograd.function import once_differentiable

from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_local_context, spmd_mesh_size
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

from torchtitan_recipes.overrides import _dsv3_router_kernels as kernels


@torch.library.custom_op(
    "torchtitan::dsv3_router_forward", mutates_args=(), device_types="cuda"
)
def router_forward_op(
    logits_TE: torch.Tensor, expert_bias_E: torch.Tensor | None
) -> list[torch.Tensor]:
    """Return routing outputs and the tensors needed by the fused backward."""
    return list(kernels.forward(logits_TE, expert_bias_E))


@router_forward_op.register_fake
def _router_forward_fake(logits_TE, expert_bias_E):
    tokens, experts = logits_TE.shape
    return [
        logits_TE.new_empty((tokens, 8)),
        logits_TE.new_empty((tokens, 8), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts), dtype=torch.bool),
        logits_TE.new_empty(()),
        logits_TE.new_empty((experts,), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts)),
        logits_TE.new_empty((tokens, 1)),
        logits_TE.new_empty((tokens, 8)),
        logits_TE.new_empty((tokens, 1)),
        logits_TE.new_empty((tokens, 1)),
        logits_TE.new_empty((experts,)),
    ]


@torch.library.custom_op(
    "torchtitan::dsv3_router_backward", mutates_args=(), device_types="cuda"
)
def router_backward_op(
    scores_TE: torch.Tensor,
    row_norm_T1: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    selected_scores_TK: torch.Tensor,
    route_denominator_T1: torch.Tensor,
    norm_denominator_T1: torch.Tensor,
    frequencies_E: torch.Tensor,
    grad_weights_TK: torch.Tensor | None,
    grad_raw_sum: torch.Tensor | None,
) -> torch.Tensor:
    """Differentiate the routing weights and raw loss with respect to logits."""
    return kernels.backward(
        scores_TE,
        row_norm_T1,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        norm_denominator_T1,
        frequencies_E,
        grad_weights_TK,
        grad_raw_sum,
    )


@router_backward_op.register_fake
def _router_backward_fake(
    scores_TE,
    row_norm_T1,
    expert_ids_TK,
    selected_scores_TK,
    route_denominator_T1,
    norm_denominator_T1,
    frequencies_E,
    grad_weights_TK,
    grad_raw_sum,
):
    return torch.empty_like(scores_TE, memory_format=torch.contiguous_format)


class FusedDSv3RouterFunction(torch.autograd.Function):
    """Learned top-k routing and raw auxiliary loss, with an explicit backward.

    Returns weights, expert IDs, routing map, raw token-sum loss, and counts.
    Only weights and raw loss are differentiable. Bias affects the discrete
    selection, so it receives no gradient, as in the native router.
    """

    @staticmethod
    def spmd_typecheck(outputs, *, logits_TE, expert_bias_E):
        weights_TK, ids_TK, routing_map_TE, raw_sum, counts_E = outputs
        operands = (logits_TE,) if expert_bias_E is None else (logits_TE, expert_bias_E)
        rows = "t_->t_" if expert_bias_E is None else "t_,_->t_"
        for output in (weights_TK, ids_TK, routing_map_TE):
            spmd.rules.einsum(rows, *operands, out=output)
        loss = "__->" if expert_bias_E is None else "__,_->"
        with spmd_local_context("dp"):
            spmd.rules.einsum(loss, *operands, out=raw_sum)
        spmd.rules.einsum("t_->_", routing_map_TE, out=counts_E)

    @staticmethod
    def forward(ctx, logits_TE, expert_bias_E):  # pyrefly: ignore[bad-override]
        outputs = router_forward_op(logits_TE, expert_bias_E)
        weights_TK, ids_TK, routing_map_TE, raw_sum, counts_E = outputs[:5]
        (
            scores_TE,
            row_norm_T1,
            selected_TK,
            route_denom_T1,
            norm_denom_T1,
            freq_E,
        ) = outputs[5:]
        ctx.save_for_backward(
            scores_TE,
            row_norm_T1,
            ids_TK,
            selected_TK,
            route_denom_T1,
            norm_denom_T1,
            freq_E,
        )
        ctx.mark_non_differentiable(ids_TK, routing_map_TE, counts_E)
        ctx.set_materialize_grads(False)
        return weights_TK, ids_TK, routing_map_TE, raw_sum, counts_E

    @staticmethod
    @once_differentiable
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_weights_TK, grad_ids, grad_map, grad_raw_sum, grad_counts
    ):
        if grad_weights_TK is None and grad_raw_sum is None:
            return None, None
        return (
            router_backward_op(*ctx.saved_tensors, grad_weights_TK, grad_raw_sum),
            None,
        )


class FusedDSv3Router(DeepSeekV3Router):
    """GB300 specialization of the 256-expert DSv3 training router."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Router.Config):
        pass

    def _supports_fusion(self, x_TD, expert_bias_E, padding_mask_T):
        return (
            self.training
            and x_TD.is_cuda
            and x_TD.ndim == 2
            and x_TD.shape[0] == 4096
            and type(self.gate) is HiMidLoLinear
            and not torch.is_autocast_enabled("cuda")
            and self.num_experts == 256
            and self.top_k == 8
            and self.num_expert_groups == 8
            and self.num_limited_groups == 4
            and type(self.score_func) is Sigmoid
            and self.route_norm
            and self.route_norm_epsilon == 1e-20
            and self.route_scale == 2.5
            and type(self.aux_loss) is MicrobatchWiseLoadBalanceLoss
            and padding_mask_T is None
            and spmd_mesh_size("cp") == 1
            and spmd_mesh_size("tp") == 1
            and torch.cuda.get_device_capability(x_TD.device) == (10, 3)
            and torch.cuda.get_device_properties(x_TD.device).multi_processor_count
            >= 128
            and (
                expert_bias_E is None
                or (
                    expert_bias_E.shape == (256,)
                    and expert_bias_E.dtype == torch.float32
                    and expert_bias_E.device == x_TD.device
                    and expert_bias_E.is_contiguous()
                )
            )
        )

    def forward(
        self,
        x_TD,
        expert_bias_E=None,
        *,
        padding_mask_T=None,
        aux_loss_denominator=None,
        **router_kwargs,
    ):
        if self._supports_fusion(x_TD, expert_bias_E, padding_mask_T):
            assert self.aux_loss is not None
            if aux_loss_denominator is None:
                raise ValueError("An auxiliary-loss denominator is required.")
            logits_TE = self.gate(x_TD)
            remat.recompute_needs_tensor(logits_TE)
            weights_TK, ids_TK, routing_map_TE, raw_sum, counts_E = remat.region(
                FusedDSv3RouterFunction.apply, "routing_decision", recompute=False
            )(logits_TE, expert_bias_E)
            remat.recompute_needs_tensor(ids_TK)
            if not remat.is_recomputing():
                with torch.no_grad():
                    self.tokens_per_expert_E.add_(counts_E)
            with spmd_local_context("dp"):
                weights_TK = self.aux_loss.inject(
                    raw_sum, carrier=weights_TK, denominator=aux_loss_denominator
                )
            return weights_TK, ids_TK, routing_map_TE
        return super().forward(
            x_TD,
            expert_bias_E,
            padding_mask_T=padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
            **router_kwargs,
        )


@override(
    target=DeepSeekV3Router.Config,
    exact=True,
    description="Fuse learned DeepSeek V3 routing, auxiliary loss, and their backward.",
)
def fused_dsv3_router(cfg: DeepSeekV3Router.Config) -> FusedDSv3Router.Config:
    return derive(cfg, FusedDSv3Router.Config)
