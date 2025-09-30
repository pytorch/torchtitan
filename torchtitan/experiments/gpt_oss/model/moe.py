# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch import nn
from torchtitan.experiments.gpt_oss.infra.expert_parallel import expert_parallel
from torchtitan.protocols import model

from .args import GptOssModelArgs
from torchtitan.models.moe import MoE, MoEArgs, GroupedExperts

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)

class GptOssGroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        swiglu_limit: float,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_grouped_mm = use_grouped_mm
        self.swiglu_limit = swiglu_limit

        self.mlp1_weight = nn.Parameter(torch.empty((num_experts, dim, hidden_dim * 2))) # w1 and w3
        self.mlp1_bias = nn.Parameter(torch.empty((num_experts, hidden_dim * 2)))
        self.mlp2_weight = nn.Parameter(torch.empty((num_experts, hidden_dim, dim)))
        self.mlp2_bias = nn.Parameter(torch.empty((num_experts, dim)))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_grouped_mm:
            return GptOssGroupedExperts._run_experts_grouped_mm(
                self.mlp1_weight, self.mlp1_bias, self.mlp2_weight, self.mlp2_bias, self.swiglu_limit, x, num_tokens_per_expert
            )
        else:
            return GptOssGroupedExperts._run_experts_for_loop(
                self.mlp1_weight, self.mlp1_bias, self.mlp2_weight, self.mlp2_bias, self.swiglu_limit, x, num_tokens_per_expert
            )

    # TODO: keeping this for-loop implementation for comparison
    #       and readability, may remove later
    @expert_parallel
    @staticmethod
    def _run_experts_for_loop(
        mlp1_weight: torch.Tensor,
        mlp1_bias: torch.Tensor,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor,
        swiglu_limit: float,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            # NOTE: this would incur a synchronization between device and host
            num_tokens_per_expert = num_tokens_per_expert.tolist()

            # side-effect code due to the usage of generate_permute_indices
            num_padding = x.shape[0] - sum(num_tokens_per_expert)

            # a tuple of tensors indexed by experts
            # each with shape (tokens_per_expert(varying), dim)
            x = torch.split(
                x[: sum(num_tokens_per_expert)],
                split_size_or_sections=num_tokens_per_expert,
                dim=0,
            )
            out_experts_splits = []
            for expert_idx, x_expert in enumerate(x):
                h = torch.matmul(x_expert, mlp1_weight[expert_idx]) + mlp1_bias[expert_idx]
                h = swiglu(h, limit=swiglu_limit)
                h = torch.matmul(h, mlp2_weight[expert_idx]) + mlp2_bias[expert_idx]
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)

            # side-effect code due to the usage of generate_permute_indices
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        else:
            # x shape (num_experts, tokens_per_expert, dim)
            h = torch.bmm(x, mlp1_weight) + mlp1_bias.unsqueeze(1)
            h = swiglu(h, limit=swiglu_limit)
            out = torch.bmm(h, mlp2_weight) + mlp2_bias.unsqueeze(1)

        return out

    @expert_parallel  # NOTE: EP currently reduces 20B MFU from 17.8% to 16.5%!
    @staticmethod
    def _run_experts_grouped_mm(
        mlp1_weight: torch.Tensor,
        mlp1_bias: torch.Tensor,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor,
        swiglu_limit: float,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
            num_tokens_per_expert_long = num_tokens_per_expert.to(torch.long)
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        if isinstance(mlp1_weight, DTensor):
            mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = mlp1_weight.to_local(), mlp1_bias.to_local(), mlp2_weight.to_local(), mlp2_bias.to_local()

        h = torch._grouped_mm(x.bfloat16(), mlp1_weight.bfloat16(), offs=offsets)
        if offsets is not None:
            # TODO(jianiw): check what is this doing
            b1 = mlp1_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
            tail_slack = x.shape[0] - int(offsets[-1])
            if tail_slack:
                b1 = torch.cat([b1, b1.new_zeros((tail_slack, b1.shape[-1]))], dim=0)
            h = h + b1.to(h.dtype)

        h = swiglu(h, limit=swiglu_limit)
        # print(f"{h.shape} {mlp2_weight.shape}") # [rank0]:torch.Size([77507, 1440]) torch.Size([2, 2880, 128])
        h = torch._grouped_mm(h, mlp2_weight.bfloat16(), offs=offsets)
        if offsets is not None:
            b2 = mlp2_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
            tail_slack = x.shape[0] - int(offsets[-1])
            if tail_slack:
                b2 = torch.cat([b2, b2.new_zeros((tail_slack, b2.shape[-1]))], dim=0)
            h = h + b2.to(h.dtype)

        return h

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.mlp1_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp1_bias, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_bias, mean=0.0, std=init_std)

    def extra_repr(self):
        return (f"num_experts={self.num_experts}, "
                f"use_grouped_mm={self.use_grouped_mm}, "
                f"mlp1_weight={tuple(self.mlp1_weight.shape)}, "
                f"mlp1_bias={tuple(self.mlp1_bias.shape)}, "
                f"mlp2_weight={tuple(self.mlp2_weight.shape)}, "
                f"mlp2_bias={tuple(self.mlp2_bias.shape)}")


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""
    
    def __init__(self, model_args: GptOssModelArgs, dim: int, hidden_dim: int):
        # Convert GptOssModelArgs to MoEArgs for base class compatibility
        moe_args = model_args.moe_args
        
        # Initialize the base MoE class
        super().__init__(moe_args, dim, hidden_dim)
        
        # Override the base GroupedExperts with GptOssGroupedExperts
        self.experts = GptOssGroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=moe_args.num_experts,
            swiglu_limit=model_args.swiglu_limit,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
