# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in DeepSeek V3 sequence-wise loss with explicit fused forward/backward.

Current TorchTitan calls this objective ``MicrobatchWiseLoadBalanceLoss``.
The routing map, loss scaling, metric accumulation, and collectives keep their
native contracts. See ``docs/fused-dsv3-seqwise-loss.md`` for the specialization.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.autograd.function import once_differentiable

from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_local_context, spmd_mesh_size
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router

from torchtitan_recipes.overrides import _dsv3_seqwise_loss_kernels as kernels


@torch.library.custom_op(
    "torchtitan::dsv3_seqwise_loss_forward",
    mutates_args=("arrival_counter",),
    device_types="cuda",
)
def seqwise_loss_forward_op(
    scores_TE: torch.Tensor,
    routing_map_TE: torch.Tensor,
    arrival_counter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One CUDA launch; the caller owns the stream-ordered arrival counter."""
    return kernels.forward(scores_TE, routing_map_TE, arrival_counter)


@seqwise_loss_forward_op.register_fake
def _seqwise_loss_forward_fake(scores_TE, routing_map_TE, arrival_counter):
    return scores_TE.new_empty(()), scores_TE.new_empty((scores_TE.shape[1],))


@torch.library.custom_op(
    "torchtitan::dsv3_seqwise_loss_backward", mutates_args=(), device_types="cuda"
)
def seqwise_loss_backward_op(
    grad_raw_sum: torch.Tensor,
    scores_TE: torch.Tensor,
    frequencies_E: torch.Tensor,
) -> torch.Tensor:
    """One CUDA launch for the raw-loss derivative with respect to scores."""
    return kernels.backward(grad_raw_sum, scores_TE, frequencies_E)


@seqwise_loss_backward_op.register_fake
def _seqwise_loss_backward_fake(grad_raw_sum, scores_TE, frequencies_E):
    return torch.empty_like(scores_TE, memory_format=torch.contiguous_format)


class FusedDSv3SeqwiseLossFunction(torch.autograd.Function):
    """Raw loss and its first derivative; normalization/injection belongs to AuxLoss.

    ``arrival_counter`` is a zero-initialized, module-owned CUDA int32 scalar.
    Calls sharing a counter must be stream-ordered. Separate modules have
    separate counters, so independent layers can run on independent streams.
    """

    @staticmethod
    def spmd_typecheck(result, *, scores_TE, routing_map_TE, arrival_counter):
        spmd.rules.ignore(arrival_counter)
        # Both reductions require complete token/expert dimensions on the
        # non-local axes. The module makes DP local and gates CP/TP to size 1.
        spmd.rules.einsum("__,__->", scores_TE, routing_map_TE, out=result)

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, scores_TE, routing_map_TE, arrival_counter
    ):
        scores_TE = scores_TE.resolve_neg()
        raw_sum, frequencies_E = seqwise_loss_forward_op(
            scores_TE, routing_map_TE, arrival_counter
        )
        ctx.save_for_backward(scores_TE, frequencies_E)
        return raw_sum

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_raw_sum):  # pyrefly: ignore[bad-override]
        scores_TE, frequencies_E = ctx.saved_tensors
        return (
            seqwise_loss_backward_op(grad_raw_sum, scores_TE, frequencies_E),
            None,
            None,
        )


def _supported(scores_TE, routing_map_TE, padding_mask_T):
    return (
        scores_TE.is_cuda
        and scores_TE.dtype == torch.float32
        and scores_TE.shape == (4096, 256)
        and scores_TE.is_contiguous()
        and scores_TE.storage_offset() % 4 == 0
        and routing_map_TE.dtype == torch.bool
        and routing_map_TE.device == scores_TE.device
        and routing_map_TE.shape == scores_TE.shape
        and routing_map_TE.is_contiguous()
        and padding_mask_T is None
        and spmd_mesh_size("cp") == 1
        and spmd_mesh_size("tp") == 1
        and torch.cuda.get_device_capability(scores_TE.device) == (10, 3)
    )


class FusedDSv3SeqwiseLoss(MicrobatchWiseLoadBalanceLoss):
    """Specialize the loss while retaining native scaling and metric state."""

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchWiseLoadBalanceLoss.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.register_buffer(
            "arrival_counter", torch.zeros(1, dtype=torch.int32), persistent=False
        )

    @property
    def metric_name(self) -> str:
        return "microbatch_wise_load_balance_loss"

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        super()._init_self_buffers(buffer_device=buffer_device)
        self.arrival_counter = torch.zeros(
            1, dtype=torch.int32, device=self.instance_acc.device
        )

    def forward(
        self,
        scores_TE,
        routing_map_TE,
        *,
        carrier,
        padding_mask_T=None,
        denominator,
    ):
        if _supported(scores_TE, routing_map_TE, padding_mask_T):
            with spmd_local_context("dp"):
                raw_sum = FusedDSv3SeqwiseLossFunction.apply(
                    scores_TE, routing_map_TE, self.arrival_counter
                )
                return self.inject(raw_sum, carrier=carrier, denominator=denominator)
        return super().forward(
            scores_TE,
            routing_map_TE,
            carrier=carrier,
            padding_mask_T=padding_mask_T,
            denominator=denominator,
        )


@override(
    target=DeepSeekV3Router.Config,
    exact=True,
    description="Fuse the DeepSeek V3 sequence-wise auxiliary loss and its backward.",
)
def fused_dsv3_seqwise_loss(cfg: DeepSeekV3Router.Config) -> DeepSeekV3Router.Config:
    if type(cfg.aux_loss) is not MicrobatchWiseLoadBalanceLoss.Config:
        return cfg
    return derive(
        cfg,
        DeepSeekV3Router.Config,
        aux_loss=derive(cfg.aux_loss, FusedDSv3SeqwiseLoss.Config),
    )
