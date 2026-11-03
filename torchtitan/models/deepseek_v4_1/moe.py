# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Modality-specific, auxiliary-loss-free balancing with the shared MoE stack.

T: tokens, E: experts, K: selected experts, R: routed tokens, F: expert features,
L: MoE layers.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.activation import LimitedSwiGLU
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter


class Float32LimitedSwiGLU(LimitedSwiGLU):
    """Round once after the reference model's bounded FP32 activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(LimitedSwiGLU.Config):
        pass

    def _activation_fn(  # pyrefly: ignore [bad-param-name-override]
        self, gate_RF, up_RF
    ):
        return super()._activation_fn(gate_RF.float(), up_RF.float()).to(gate_RF.dtype)


class ModalityRouter(TokenChoiceTopKRouter):
    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate.float()
        self.register_buffer(
            "image_tokens_per_expert_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def _init_self_buffers(self, *, buffer_device=None):
        super()._init_self_buffers(buffer_device=buffer_device)
        self.image_tokens_per_expert_E = torch.zeros_like(self.tokens_per_expert_E)

    def _select_experts(
        self,
        scores_TE,
        expert_bias_E=None,
        *,
        image_mask_T=None,
        expert_bias_vl_E=None,
        **router_kwargs
    ):
        if image_mask_T is None:
            return super()._select_experts(scores_TE, expert_bias_E, **router_kwargs)
        if expert_bias_E is None or expert_bias_vl_E is None:
            raise ValueError(
                "Image routing requires both text and image expert biases."
            )
        bias_TE = torch.where(image_mask_T[:, None], expert_bias_vl_E, expert_bias_E)
        return (scores_TE + bias_TE).topk(self.top_k, -1, sorted=False).indices

    def forward(
        self,
        x_TD,
        expert_bias_E=None,
        *,
        padding_mask_T=None,
        image_mask_T=None,
        expert_bias_vl_E=None,
        aux_loss_denominator=None,
        **router_kwargs
    ):
        scores_TK, ids_TK, routing_TE = super().forward(
            x_TD,
            expert_bias_E,
            padding_mask_T=padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
            image_mask_T=image_mask_T,
            expert_bias_vl_E=expert_bias_vl_E,
            **router_kwargs,
        )
        if self.training and not remat.is_recomputing() and image_mask_T is not None:
            with spmd.no_typecheck(), torch.no_grad():
                valid_T = (
                    image_mask_T
                    if padding_mask_T is None
                    else image_mask_T & ~padding_mask_T
                )
                image_counts_E = (routing_TE & valid_T[:, None]).sum(0)
                self.image_tokens_per_expert_E.add_(image_counts_E)
                self.tokens_per_expert_E.sub_(image_counts_E)
        return scores_TK, ids_TK, routing_TE


class ModalityMoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        router: ModalityRouter.Config  # pyrefly: ignore [bad-override]

        def __post_init__(self):
            super(ModalityMoE.Config, self).__post_init__()
            if self.load_balance_coeff is None:
                raise ValueError(
                    "V4.1 modality routing requires expert biases; use freeze_expert_bias to freeze updates."
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.router: ModalityRouter
        self.register_buffer(
            "expert_bias_vl_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )

    def _init_self_buffers(self, *, buffer_device=None):
        super()._init_self_buffers(buffer_device=buffer_device)
        assert self.expert_bias_E is not None
        self.expert_bias_vl_E = torch.zeros_like(self.expert_bias_E)

    def forward(
        self,
        x_TD,
        *,
        padding_mask_T=None,
        aux_loss_denominator=None,
        image_mask_T=None,
        **router_kwargs
    ):
        tp_group = spmd_mesh_group("tp")
        if image_mask_T is not None and tp_group is not None:
            # Follow the token shard used by the common MoE's routed branch.
            image_mask_T = spmd.redistribute(
                image_mask_T, tp_group, src=spmd.R, dst=spmd.S(0)
            )
        return super().forward(
            x_TD,
            padding_mask_T=padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
            image_mask_T=image_mask_T,
            expert_bias_vl_E=self.expert_bias_vl_E,
            **router_kwargs,
        )


def update_image_expert_bias(model_parts, parallelism_context):
    """Reduce visual counts on the same axes as the shared text balancing hook."""
    moes = [
        moe
        for part in model_parts
        for moe in part.modules()
        if isinstance(moe, ModalityMoE) and not moe.freeze_expert_bias
    ]
    if not moes:
        return
    counts_LE = torch.vstack([moe.router.image_tokens_per_expert_E for moe in moes])
    if parallelism_context.ep_enabled and parallelism_context.tp > 1:
        torch.distributed.all_reduce(
            counts_LE, group=parallelism_context.get_dense_tp_mesh().get_group()
        )
    loss_mesh = parallelism_context.get_optional_mesh("loss")
    if loss_mesh is not None:
        torch.distributed.all_reduce(counts_LE, group=loss_mesh.get_group())
    with torch.no_grad():
        for moe, counts_E in zip(moes, counts_LE, strict=True):
            assert moe.load_balance_coeff is not None
            delta_E = moe.load_balance_coeff * torch.sign(counts_E.mean() - counts_E)
            moe.expert_bias_vl_E.add_(delta_E - delta_E.mean())
            moe.router.image_tokens_per_expert_E.zero_()
