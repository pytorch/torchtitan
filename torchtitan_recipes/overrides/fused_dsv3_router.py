# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental DeepSeek V3 learned routing with fused forward/backward.

The gate stays native; the fusion consumes its FP32 [4096, 256] logits.
The specialized kernels require nvidia-cutlass-dsl >= 4.8.0.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat
from torch.autograd.function import once_differentiable

from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

if TYPE_CHECKING:
    from torchtitan_recipes.overrides import _dsv3_router_kernels as kernels

    _CUTEDSL_IMPORT_ERROR: ImportError | None = None
else:
    try:
        from torchtitan_recipes.overrides import _dsv3_router_kernels as kernels

        _CUTEDSL_IMPORT_ERROR = None
    except ImportError as e:
        _CUTEDSL_IMPORT_ERROR = e


@torch.library.custom_op(
    "torchtitan::dsv3_router_forward",
    mutates_args=("arrival_counter",),
    device_types="cuda",
)
def router_forward_op(
    logits_TE: torch.Tensor,
    expert_bias_E: torch.Tensor | None,
    arrival_counter: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if _CUTEDSL_IMPORT_ERROR is not None:
        raise ImportError(
            "DSv3 router fusion requires CuTeDSL; install nvidia-cutlass-dsl>=4.8.0."
        ) from _CUTEDSL_IMPORT_ERROR
    return kernels.forward(logits_TE, expert_bias_E, arrival_counter)


@router_forward_op.register_fake
def _router_forward_fake(logits_TE, expert_bias_E, arrival_counter):
    tokens, experts = logits_TE.shape
    return (
        logits_TE.new_empty((tokens, 8)),
        logits_TE.new_empty((tokens, 8), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts), dtype=torch.bool),
        logits_TE.new_empty((experts,), dtype=torch.int64),
        logits_TE.new_empty((tokens, experts)),
        logits_TE.new_empty((tokens, 8)),
        logits_TE.new_empty((tokens, 1)),
    )


@torch.library.custom_op(
    "torchtitan::dsv3_router_backward", mutates_args=(), device_types="cuda"
)
def router_backward_op(
    scores_TE: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    selected_scores_TK: torch.Tensor,
    route_denominator_T1: torch.Tensor,
    grad_weights_TK: torch.Tensor | None,
    grad_scores_TE: torch.Tensor | None,
) -> torch.Tensor:
    if _CUTEDSL_IMPORT_ERROR is not None:
        raise ImportError(
            "DSv3 router fusion requires CuTeDSL; install nvidia-cutlass-dsl>=4.8.0."
        ) from _CUTEDSL_IMPORT_ERROR
    return kernels.backward(
        scores_TE,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        grad_weights_TK,
        grad_scores_TE,
    )


@router_backward_op.register_fake
def _router_backward_fake(
    scores_TE,
    expert_ids_TK,
    selected_scores_TK,
    route_denominator_T1,
    grad_weights_TK,
    grad_scores_TE,
):
    return torch.empty_like(scores_TE, memory_format=torch.contiguous_format)


class FusedDSv3RouterFunction(torch.autograd.Function):
    """Learned top-k routing with differentiable scores for an external loss.

    Returns weights, expert IDs, routing map, counts, and sigmoid scores.
    Weights and scores are differentiable. Bias affects discrete selection,
    so it receives no gradient, as in the native router. Calls sharing the
    zero-initialized arrival counter must be stream-ordered.
    """

    @staticmethod
    def spmd_typecheck(outputs, *, logits_TE, expert_bias_E, arrival_counter):
        spmd.rules.ignore(arrival_counter)
        weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE = outputs
        operands = (logits_TE,) if expert_bias_E is None else (logits_TE, expert_bias_E)
        rows = "t_->t_" if expert_bias_E is None else "t_,_->t_"
        for output in (weights_TK, ids_TK, routing_map_TE, scores_TE):
            spmd.rules.einsum(rows, *operands, out=output)
        spmd.rules.einsum("t_->_", routing_map_TE, out=counts_E)

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, logits_TE, expert_bias_E, arrival_counter
    ):
        (
            weights_TK,
            ids_TK,
            routing_map_TE,
            counts_E,
            scores_TE,
            selected_TK,
            route_denominator_T1,
        ) = router_forward_op(logits_TE, expert_bias_E, arrival_counter)
        ctx.save_for_backward(scores_TE, ids_TK, selected_TK, route_denominator_T1)
        ctx.mark_non_differentiable(ids_TK, routing_map_TE, counts_E)
        ctx.set_materialize_grads(False)
        return weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE

    @staticmethod
    @once_differentiable
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_weights_TK, grad_ids, grad_map, grad_counts, grad_scores_TE
    ):
        if grad_weights_TK is None and grad_scores_TE is None:
            return None, None, None
        return (
            router_backward_op(*ctx.saved_tensors, grad_weights_TK, grad_scores_TE),
            None,
            None,
        )


class FusedDSv3Router(DeepSeekV3Router):
    """Router specialization with no knowledge of the auxiliary-loss formula."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Router.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.register_buffer(
            "arrival_counter", torch.zeros(1, dtype=torch.int32), persistent=False
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        super()._init_self_buffers(buffer_device=buffer_device)
        self.arrival_counter = torch.zeros(
            1, dtype=torch.int32, device=self.tokens_per_expert_E.device
        )

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
            logits_TE = self.gate(x_TD)
            remat.recompute_needs_tensor(logits_TE)
            weights_TK, ids_TK, routing_map_TE, counts_E, scores_TE = remat.region(
                FusedDSv3RouterFunction.apply, "routing_decision", recompute=False
            )(logits_TE, expert_bias_E, self.arrival_counter)
            remat.recompute_needs_tensor(ids_TK)
            if not remat.is_recomputing():
                with torch.no_grad():
                    self.tokens_per_expert_E.add_(counts_E)
            if self.aux_loss is not None:
                if aux_loss_denominator is None:
                    raise ValueError("An auxiliary-loss denominator is required.")
                remat.recompute_needs_tensor(scores_TE)
                remat.recompute_needs_tensor(routing_map_TE)
                weights_TK = self.aux_loss(
                    scores_TE,
                    routing_map_TE,
                    carrier=weights_TK,
                    padding_mask_T=padding_mask_T,
                    denominator=aux_loss_denominator,
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
    description="Fuse learned DeepSeek V3 routing and its backward.",
)
def fused_dsv3_router(cfg: DeepSeekV3Router.Config) -> FusedDSv3Router.Config:
    return derive(cfg, FusedDSv3Router.Config)
