# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from collections.abc import Callable

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe.moe import MoE
from torchtitan.models.common.moe.utils import _permute, _unpermute
from torchtitan.models.common.utils import trunc_normal_


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


@torch.library.custom_op("torchtitan::repeat_interleave_with_pad", mutates_args=())
def _repeat_interleave_with_pad(
    bias: torch.Tensor,
    repeats: torch.Tensor,
    padded_len: int,
) -> torch.Tensor:
    if repeats.shape[0] > bias.shape[0]:
        bias = torch.cat([bias, bias.new_zeros(repeats.shape[0] - bias.shape[0], bias.shape[-1])])
    expanded = bias.repeat_interleave(repeats, dim=0)
    out = expanded.new_zeros((padded_len, expanded.shape[-1]))
    out[: expanded.shape[0]] = expanded
    return out


@_repeat_interleave_with_pad.register_fake
def _repeat_interleave_with_pad_fake(
    bias: torch.Tensor,
    repeats: torch.Tensor,
    padded_len: int,
) -> torch.Tensor:
    return bias.new_empty((padded_len, bias.shape[-1]))


@torch.library.custom_op("torchtitan::segment_sum", mutates_args=())
def _segment_sum(
    data: torch.Tensor,
    lengths: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
    n_valid = lengths.sum()
    data_trimmed = data[:n_valid]
    index = torch.arange(n_segments, device=data.device).repeat_interleave(lengths)
    return data.new_zeros((n_segments, data.shape[-1])).index_add(
        0, index, data_trimmed,
    )


@_segment_sum.register_fake
def _segment_sum_fake(
    data: torch.Tensor,
    lengths: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
    return data.new_empty((n_segments, data.shape[-1]))


def _repeat_interleave_with_pad_setup_context(ctx, inputs, output):
    bias, repeats, padded_len = inputs
    ctx.save_for_backward(repeats)
    ctx.bias_shape_0 = bias.shape[0]


def _repeat_interleave_with_pad_backward(ctx, grad_output):
    (repeats,) = ctx.saved_tensors
    n = ctx.bias_shape_0
    return _segment_sum(grad_output, repeats[:n], n), None, None


_repeat_interleave_with_pad.register_autograd(
    _repeat_interleave_with_pad_backward,
    setup_context=_repeat_interleave_with_pad_setup_context,
)


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
    # Pad num_tokens_per_expert with tail slack so that repeat_interleave
    # with output_size=x.shape[0] directly produces a static-shaped output,
    # avoiding the D2H sync that repeat_interleave incurs without output_size.
    tail_slack = (x.shape[0] - offsets[-1]).unsqueeze(0).to(num_tokens_per_expert.dtype)
    num_tokens_per_expert_long = torch.cat([num_tokens_per_expert, tail_slack]).long()

    h = torch._grouped_mm(
        x.bfloat16(), mlp1_weight.transpose(-2, -1).bfloat16(), offs=offsets
    )

    b1 = _repeat_interleave_with_pad(mlp1_bias, num_tokens_per_expert_long, x.shape[0])
    h = h + b1.to(h.dtype)

    h = swiglu(h, limit=swiglu_limit)
    h = torch._grouped_mm(h, mlp2_weight.transpose(-2, -1).bfloat16(), offs=offsets)

    b2 = _repeat_interleave_with_pad(mlp2_bias, num_tokens_per_expert_long, x.shape[0])
    b2 = ScaleBiasForward.apply(b2, tp_degree)
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
            # pyrefly: ignore[not-iterable]
            if "tp" in mesh_dim_names:
                # pyrefly: ignore [missing-attribute]
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        if self.use_grouped_mm:
            if (
                not isinstance(self.mlp1_weight, DTensor)
                # pyrefly: ignore[not-iterable]
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
        trunc_normal_(self.mlp1_weight, mean=0.0, std=init_std)
        trunc_normal_(self.mlp1_bias, mean=0.0, std=init_std)
        trunc_normal_(self.mlp2_weight, mean=0.0, std=init_std)
        trunc_normal_(self.mlp2_bias, mean=0.0, std=init_std)


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config, *, dim: int):
        # Initialize the base MoE class
        super().__init__(config, dim=dim)

        # Override the base GroupedExperts with GptOssGroupedExperts
        # pyrefly: ignore [bad-assignment]
        self.experts = GptOssGroupedExperts(
            dim=dim,
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            swiglu_limit=config.swiglu_limit,
            use_grouped_mm=config.use_grouped_mm,
        )
