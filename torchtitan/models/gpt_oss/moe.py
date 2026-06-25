# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.protocols.module import Module


class ScaleBiasForward(torch.autograd.Function):
    """
    Custom autograd function that scales bias in forward pass but not in backward.

    For tensor parallel MoE, we need to scale the bias by 1/tp_degree in forward
    to cancel the extra reduction effect, but keep the gradient unchanged in backward.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, bias, tp_degree):
        ctx.tp_degree = tp_degree
        if tp_degree > 1:
            return bias / tp_degree
        return bias

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # Don't scale the gradient - pass it through as-is
        return grad_output, None


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return torch.addcmul(out_glu, out_glu, x_linear)


class GptOssGroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        super().__init__()
        dim = config.dim
        hidden_dim = config.hidden_dim
        num_experts = config.num_experts
        self.num_experts = num_experts
        self.swiglu_limit = config.swiglu_limit

        self.mlp1_weight_EGD = nn.Parameter(
            torch.empty((num_experts, hidden_dim * 2, dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp1_bias_EG = nn.Parameter(torch.empty((num_experts, hidden_dim * 2)))
        self.mlp2_weight_EDF = nn.Parameter(
            torch.empty((num_experts, dim, hidden_dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp2_bias_ED = nn.Parameter(torch.empty((num_experts, dim)))

        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine.

        Shape suffixes here describe logical grouped-mm inputs, not physical
        sharding. Under EP, E may be a local shard of experts; under TP,
        expert weights shard hidden dimensions instead; under SP, R may be a
        local token shard. Keep logical capital suffixes here to avoid encoding
        a specific parallel layout in these local tensor names.
        """
        if isinstance(self.mlp1_weight_EGD, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            mlp1_weight_EGD = self.mlp1_weight_EGD.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp1_bias_EG = self.mlp1_bias_EG.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_weight_EDF = self.mlp2_weight_EDF.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_bias_ED = self.mlp2_bias_ED.to_local()
        else:
            mlp1_weight_EGD = self.mlp1_weight_EGD
            mlp1_bias_EG = self.mlp1_bias_EG
            mlp2_weight_EDF = self.mlp2_weight_EDF
            mlp2_bias_ED = self.mlp2_bias_ED

        # Determine tp_degree from device mesh if available
        tp_degree = 1
        if isinstance(self.mlp1_weight_EGD, DTensor):
            mesh_dim_names = self.mlp1_weight_EGD.device_mesh.mesh_dim_names
            # pyrefly: ignore[not-iterable]
            if "tp" in mesh_dim_names:
                # pyrefly: ignore [missing-attribute]
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight_EGD.device_mesh.size(tp_dim_idx)

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        # Pad num_tokens_per_expert_E with tail slack so that repeat_interleave
        # with output_size=x_RD.shape[0] directly produces a static-shaped output,
        # avoiding the D2H sync that repeat_interleave incurs without output_size.
        tail_slack = (
            (x_RD.shape[0] - offsets_E[-1])
            .unsqueeze(0)
            .to(num_tokens_per_expert_E.dtype)
        )
        # shape (E+1,): E expert counts + 1 tail slack for padding
        num_tokens_per_expert_long = torch.cat(
            [num_tokens_per_expert_E, tail_slack]
        ).long()

        # G = gate+up dimension (2*F)
        h_RG = torch._grouped_mm(
            x_RD.bfloat16(),
            mlp1_weight_EGD.transpose(-2, -1).bfloat16(),
            offs=offsets_E,
        )

        b1 = torch.cat(
            [mlp1_bias_EG, mlp1_bias_EG.new_zeros(1, mlp1_bias_EG.shape[-1])]
        )
        b1_RG = b1.repeat_interleave(
            num_tokens_per_expert_long, dim=0, output_size=x_RD.shape[0]
        )
        h_RG = h_RG + b1_RG.to(h_RG.dtype)

        h_RF = swiglu(h_RG, limit=self.swiglu_limit)
        h_RD = torch._grouped_mm(
            h_RF, mlp2_weight_EDF.transpose(-2, -1).bfloat16(), offs=offsets_E
        )

        # Apply custom autograd function to scale bias in forward but not in backward
        b2 = torch.cat(
            [mlp2_bias_ED, mlp2_bias_ED.new_zeros(1, mlp2_bias_ED.shape[-1])]
        )
        b2_RD = b2.repeat_interleave(
            num_tokens_per_expert_long, dim=0, output_size=x_RD.shape[0]
        )
        b2_RD = ScaleBiasForward.apply(b2_RD, tp_degree)
        return h_RD + b2_RD.to(h_RD.dtype)

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        num_local_tokens_after_seq_dim_padding: int,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add."""
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.size(-1)
        T = B * L
        local_seq_len_after_padding = num_local_tokens_after_seq_dim_padding // B
        x_TD = x_BLD.view(T, D)
        topk_scores_TK = topk_scores_BLK.view(T, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self._experts_forward(
                routed_input_RD, num_global_tokens_per_local_expert_e
            )

        out_TD = self.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
            num_local_tokens_after_padding=num_local_tokens_after_seq_dim_padding,
            local_seq_len_after_padding=local_seq_len_after_padding,
        )
        # Un-flatten back to 3-D (B, *, D) so the local_map output sharding
        # won't cause _StridedShard in the downstream view (e.g., CP is used).
        return out_TD.view(B, -1, D)

    def parallelize(self, parallel_dims) -> None:
        """Parallelize experts and wire dispatcher meshes.

        Mirrors ``GroupedExperts.parallelize``: after the base
        ``Module.parallelize`` distributes the expert weight params, install
        the EP / TP meshes on the non-Module ``token_dispatcher`` child via
        ``wire_meshes``. ``GptOssGroupedExperts`` inherits ``Module``
        directly (not ``GroupedExperts``) so it needs its own override.
        """
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
        )


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        # Initialize the base MoE class
        super().__init__(config)

        # Override the base GroupedExperts with GptOssGroupedExperts. Forward
        # every Module.Config slot from ``config.experts`` so the rebuilt
        # config carries ``sharding_config`` (set by
        # ``set_moe_sharding_config``) into the new instance.
        gptoss_experts_config = GptOssGroupedExperts.Config(
            dim=config.experts.dim,
            hidden_dim=config.experts.hidden_dim,
            num_experts=config.experts.num_experts,
            swiglu_limit=config.swiglu_limit,
            param_init=config.experts.param_init,
            sharding_config=config.experts.sharding_config,
            token_dispatcher=config.experts.token_dispatcher,
        )
        self.experts = gptoss_experts_config.build()
