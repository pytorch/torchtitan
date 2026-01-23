# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.models.moe.moe import MoE, TokenChoiceTopKRouter
from torchtitan.models.moe.utils import _permute, _unpermute

from .args import GptOssModelArgs


class GptOssTokenChoiceTopKRouter(TokenChoiceTopKRouter):
    """GPT-OSS router that supports learned router gate bias.

    GPT-OSS models use a router with a learned bias term. This class extends
    the base TokenChoiceTopKRouter to support bias in the gate linear layer.

    Note: Since softmax is a monotonically increasing function, the topk selection
    is the same whether applied before or after softmax. Therefore, we use the
    standard Score-then-TopK approach with route_norm=True to normalize over
    the selected experts (equivalent to softmax over just top-k).
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_expert_groups: int | None,
        num_limited_groups: int | None,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
        use_router_bias: bool = True,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__(
            dim=dim,
            num_experts=num_experts,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
            top_k=top_k,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=route_scale,
            _debug_force_load_balance=_debug_force_load_balance,
        )
        # Override the gate with bias support for GPT-OSS.
        # Note: The parent class creates gate with bias=False, and we recreate it here with bias.
        # This introduces an implicit ordering dependency - if the parent class changes to do
        # something with self.gate after creation, this override could cause subtle issues.
        self.gate = nn.Linear(dim, num_experts, bias=use_router_bias)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)


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


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.
    """

    def wrapper(
        mlp1_weight: torch.Tensor,
        mlp1_bias: torch.Tensor,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor,
        swiglu_limit: float,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        tp_degree: int = 1,
    ) -> torch.Tensor:
        num_local_experts = mlp1_weight.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert = _permute(
            x, num_tokens_per_expert, ep_degree, num_local_experts
        )

        out = func(
            mlp1_weight,
            mlp1_bias,
            mlp2_weight,
            mlp2_bias,
            swiglu_limit,
            x,
            num_tokens_per_expert,
            tp_degree,
        )

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def _run_experts_for_loop(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    # pyrefly: ignore [bad-assignment]
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    # pyrefly: ignore [bad-assignment]
    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        # pyrefly: ignore [bad-argument-type]
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = (
            torch.matmul(x_expert, mlp1_weight[expert_idx].transpose(-2, -1))
            + mlp1_bias[expert_idx]
        )
        h = swiglu(h, limit=swiglu_limit)
        # Apply custom autograd function to scale bias in forward but not in backward
        b2 = ScaleBiasForward.apply(mlp2_bias[expert_idx], tp_degree)
        h = torch.matmul(h, mlp2_weight[expert_idx].transpose(-2, -1)) + b2
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    # pyrefly: ignore [no-matching-overload]
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


def _run_experts_grouped_mm(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    num_tokens_per_expert_long = num_tokens_per_expert.to(torch.long)

    h = torch._grouped_mm(
        x.bfloat16(), mlp1_weight.transpose(-2, -1).bfloat16(), offs=offsets
    )

    b1 = mlp1_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
    tail_slack = x.shape[0] - int(offsets[-1])
    if tail_slack:
        b1 = torch.cat([b1, b1.new_zeros((tail_slack, b1.shape[-1]))], dim=0)
    h = h + b1.to(h.dtype)

    h = swiglu(h, limit=swiglu_limit)
    h = torch._grouped_mm(h, mlp2_weight.transpose(-2, -1).bfloat16(), offs=offsets)

    # Apply custom autograd function to scale bias in forward but not in backward
    b2_base = mlp2_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
    b2 = ScaleBiasForward.apply(b2_base, tp_degree)
    tail_slack = x.shape[0] - int(offsets[-1])
    if tail_slack:  # padding
        b2 = torch.cat([b2, b2.new_zeros((tail_slack, b2.shape[-1]))], dim=0)
    h = h + b2.to(h.dtype)

    return h


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

        self.mlp1_weight = nn.Parameter(
            torch.empty((num_experts, hidden_dim * 2, dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp1_bias = nn.Parameter(torch.empty((num_experts, hidden_dim * 2)))
        self.mlp2_weight = nn.Parameter(
            torch.empty((num_experts, dim, hidden_dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp2_bias = nn.Parameter(torch.empty((num_experts, dim)))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
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

        # Determine tp_degree from device mesh if available
        tp_degree = 1
        if isinstance(self.mlp1_weight, DTensor):
            mesh_dim_names = self.mlp1_weight.device_mesh.mesh_dim_names
            # pyrefly: ignore [not-iterable]
            if "tp" in mesh_dim_names:
                # pyrefly: ignore [missing-attribute]
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        if self.use_grouped_mm:
            if (
                not isinstance(self.mlp1_weight, DTensor)
                # pyrefly: ignore [not-iterable]
                or "ep" not in self.mlp1_weight.device_mesh.mesh_dim_names
            ):
                run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
            else:
                run_experts_fn = _run_experts_grouped_mm
            return run_experts_fn(
                mlp1_weight,
                mlp1_bias,
                mlp2_weight,
                mlp2_bias,
                self.swiglu_limit,
                x,
                num_tokens_per_expert,
                tp_degree,
            )
        else:
            return _run_experts_for_loop(
                mlp1_weight,
                mlp1_bias,
                mlp2_weight,
                mlp2_bias,
                self.swiglu_limit,
                x,
                num_tokens_per_expert,
                tp_degree,
            )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.mlp1_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp1_bias, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_bias, mean=0.0, std=init_std)


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class.

    GPT-OSS MoE supports:
    - Custom SwiGLU activation with expert biases (mlp1_bias, mlp2_bias)
    - Optional learned router gate bias (use_router_bias)
    - route_norm=True for normalizing over selected experts (equivalent to softmax over top-k)
    """

    def __init__(self, model_args: GptOssModelArgs, dim: int, hidden_dim: int):
        # Convert GptOssModelArgs to MoEArgs for base class compatibility
        moe_args = model_args.moe_args

        # Initialize the base MoE class
        super().__init__(moe_args, dim, hidden_dim)

        # Override the base GroupedExperts with GptOssGroupedExperts
        # pyrefly: ignore [bad-assignment]
        self.experts = GptOssGroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=moe_args.num_experts,
            swiglu_limit=model_args.swiglu_limit,
            use_grouped_mm=moe_args.use_grouped_mm,
        )

        # Override router with GPT-OSS router that supports learned gate bias
        if model_args.use_router_bias:
            # pyrefly: ignore [bad-assignment]
            self.router = GptOssTokenChoiceTopKRouter(
                dim=dim,
                num_experts=moe_args.num_experts,
                num_expert_groups=moe_args.num_expert_groups,
                num_limited_groups=moe_args.num_limited_groups,
                top_k=moe_args.top_k,
                score_func=moe_args.score_func,
                route_norm=moe_args.route_norm,
                route_scale=moe_args.route_scale,
                use_router_bias=True,
                _debug_force_load_balance=moe_args._debug_force_load_balance,
            )
