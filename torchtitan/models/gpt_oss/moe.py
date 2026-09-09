# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
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
    def forward(ctx, bias, tp_degree, dtype):
        ctx.tp_degree = tp_degree
        if tp_degree > 1:
            bias = bias / tp_degree
        return bias.to(dtype)

    @staticmethod
    def spmd_typecheck(out, *, bias):
        """
        Typecheck for bias scaling, already interleaved to num tokens shape.
        If EP enabled, V on all axes. If disabled, TP axis: R->V.
        Technically R->P, but easier to mix in local region as V.
        TODO(pianpwk): .to() dtype casts in LocalTokenDispatcher don't propagate Partial;
        we would like a spmd_types API where callers are conscious of numerics loss.
        """
        enable_ep = spmd_mesh_size("ep") > 1
        if enable_ep:
            in_type = out_type = spmd.V
        else:
            in_type = {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R}
            out_type = {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V}
        spmd.assert_type(bias, in_type)
        spmd.assert_type(out, out_type)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # Don't scale the gradient - pass it through as-is
        return grad_output, None, None


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return torch.addcmul(out_glu, out_glu, x_linear)


class GptOssGroupedExperts(GroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        Module.__init__(self)
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

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine.

        Shape suffixes here describe logical grouped-mm inputs, not physical
        sharding. Under EP, E may be a local shard of experts; under TP,
        expert weights shard hidden dimensions instead; under SP, R may be a
        local token shard. P is the E expert entries plus one tail-slack entry.
        Keep logical capital suffixes here to avoid encoding a specific
        parallel layout in these local tensor names.
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

        # Determine tp_degree from the active backend's device mesh.
        tp_degree = 1
        if get_spmd_backend() == "spmd_types":
            tp_degree = spmd_mesh_size("tp")
        elif isinstance(self.mlp1_weight_EGD, DTensor):
            mesh_dim_names = self.mlp1_weight_EGD.device_mesh.mesh_dim_names
            # pyrefly: ignore[not-iterable]
            if "tp" in mesh_dim_names:
                # pyrefly: ignore [missing-attribute]
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight_EGD.device_mesh.size(tp_dim_idx)

        if (
            get_spmd_backend() == "spmd_types"
            and spmd.is_type_checking()
            and spmd_mesh_size("ep") == 1
        ):
            spmd.mutate_type(
                num_tokens_per_expert_E,
                src=spmd.P,
                dst={"dp": spmd.V, "cp": spmd.V},
            )

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
        num_tokens_per_expert_P = torch.cat(
            [num_tokens_per_expert_E, tail_slack]
        ).long()

        h_RG = remat.region(
            self._mlp1_projection,
            self.remat_region_name("w13"),
            recompute=self.remat_should_recompute("w13"),
        )(
            x_RD,
            mlp1_weight_EGD,
            mlp1_bias_EG,
            offsets_E,
            num_tokens_per_expert_P,
        )
        remat.recompute_needs_tensor(h_RG)
        h_RF = swiglu(h_RG, limit=self.swiglu_limit)
        out_RD = remat.region(
            self._mlp2_projection,
            self.remat_region_name("w2"),
            recompute=self.remat_should_recompute("w2"),
        )(
            h_RF,
            mlp2_weight_EDF,
            mlp2_bias_ED,
            offsets_E,
            num_tokens_per_expert_P,
            tp_degree,
        )
        remat.recompute_needs_tensor(out_RD)
        return out_RD

    def _mlp1_projection(
        self,
        x_RD: torch.Tensor,
        weight_EGD: torch.Tensor,
        bias_EG: torch.Tensor,
        offsets_E: torch.Tensor,
        num_tokens_per_expert_P: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GPT-OSS's fused gate-up projection, including bias."""
        out_RG = self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=weight_EGD, offs=offsets_E
        )
        bias_PG = torch.cat([bias_EG, bias_EG.new_zeros(1, bias_EG.shape[-1])])
        bias_RG = bias_PG.repeat_interleave(
            num_tokens_per_expert_P, dim=0, output_size=x_RD.shape[0]
        )
        return out_RG + bias_RG.to(out_RG.dtype)

    def _mlp2_projection(
        self,
        h_RF: torch.Tensor,
        weight_EDF: torch.Tensor,
        bias_ED: torch.Tensor,
        offsets_E: torch.Tensor,
        num_tokens_per_expert_P: torch.Tensor,
        tp_degree: int,
    ) -> torch.Tensor:
        """Compute GPT-OSS's down projection, including its scaled bias."""
        out_RD = self._grouped_mm(A=h_RF, weight_EOI=weight_EDF, offs=offsets_E)
        bias_PD = torch.cat([bias_ED, bias_ED.new_zeros(1, bias_ED.shape[-1])])
        bias_RD = bias_PD.repeat_interleave(
            num_tokens_per_expert_P, dim=0, output_size=h_RF.shape[0]
        )
        bias_RD = ScaleBiasForward.apply(bias_RD, tp_degree, out_RD.dtype)
        return out_RD + bias_RD


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass
