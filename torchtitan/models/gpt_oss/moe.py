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
    return out_glu * (x_linear + 1)


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

        self.mlp1_weight = nn.Parameter(
            torch.empty((num_experts, hidden_dim * 2, dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp1_bias = nn.Parameter(torch.empty((num_experts, hidden_dim * 2)))
        self.mlp2_weight = nn.Parameter(
            torch.empty((num_experts, dim, hidden_dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp2_bias = nn.Parameter(torch.empty((num_experts, dim)))

        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self,
        x_ND: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.mlp1_weight, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            mlp1_weight = self.mlp1_weight.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp1_bias = self.mlp1_bias.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_weight = self.mlp2_weight.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_bias = self.mlp2_bias.to_local()
        else:
            mlp1_weight = self.mlp1_weight
            mlp1_bias = self.mlp1_bias
            mlp2_weight = self.mlp2_weight
            mlp2_bias = self.mlp2_bias

        tp_degree = 1
        if isinstance(self.mlp1_weight, DTensor):
            mesh_dim_names = self.mlp1_weight.device_mesh.mesh_dim_names
            # pyrefly: ignore[not-iterable]
            if "tp" in mesh_dim_names:
                # pyrefly: ignore [missing-attribute]
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        # Pad with tail slack so repeat_interleave with output_size
        # produces a static-shaped output without D2H sync.
        tail_slack = (
            (x_ND.shape[0] - offsets_E[-1])
            .unsqueeze(0)
            .to(num_tokens_per_expert_E.dtype)
        )
        num_tokens_per_expert_long = torch.cat(
            [num_tokens_per_expert_E, tail_slack]
        ).long()

        h_NF = torch._grouped_mm(
            x_ND.bfloat16(),
            mlp1_weight.transpose(-2, -1).bfloat16(),
            offs=offsets_E,
        )

        b1 = torch.cat([mlp1_bias, mlp1_bias.new_zeros(1, mlp1_bias.shape[-1])])
        b1_NF = b1.repeat_interleave(
            num_tokens_per_expert_long, dim=0, output_size=x_ND.shape[0]
        )
        h_NF = h_NF + b1_NF.to(h_NF.dtype)

        h_NF = swiglu(h_NF, limit=self.swiglu_limit)
        h_ND = torch._grouped_mm(
            h_NF, mlp2_weight.transpose(-2, -1).bfloat16(), offs=offsets_E
        )

        b2 = torch.cat([mlp2_bias, mlp2_bias.new_zeros(1, mlp2_bias.shape[-1])])
        b2_ND = b2.repeat_interleave(
            num_tokens_per_expert_long, dim=0, output_size=x_ND.shape[0]
        )
        b2_ND = ScaleBiasForward.apply(b2_ND, tp_degree)
        return h_ND + b2_ND.to(h_ND.dtype)

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_ids_BLK: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add."""
        x_TD = x_BLD.reshape(-1, x_BLD.size(-1))
        topk_scores_TK = topk_scores_BLK.reshape(-1, topk_scores_BLK.size(-1))
        topk_ids_TK = topk_ids_BLK.reshape(-1, topk_ids_BLK.size(-1))
        routed_input_ND, num_tokens_local_E, metadata = self.token_dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_ids_TK
        )
        routed_output_ND = self._experts_forward(routed_input_ND, num_tokens_local_E)
        return self.token_dispatcher.combine(routed_output_ND, metadata, x_TD)

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
