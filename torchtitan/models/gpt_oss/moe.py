# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from collections.abc import Callable

from dataclasses import dataclass, field

import torch
import torch.library
import triton
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe.moe import MoE
from torchtitan.models.common.moe.utils import _permute, _unpermute
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


@torch.compile
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

    h = torch._grouped_mm(
        x.bfloat16(), mlp1_weight.transpose(-2, -1).bfloat16(), offs=offsets
    )
    h = expert_bias_add(h, mlp1_bias, offsets)

    h = swiglu(h, limit=swiglu_limit)
    h = torch._grouped_mm(h, mlp2_weight.transpose(-2, -1).bfloat16(), offs=offsets)

    # Apply custom autograd function to scale bias in forward but not in backward
    b2 = ScaleBiasForward.apply(mlp2_bias, tp_degree)
    h = expert_bias_add(h, b2, offsets)

    return h


class GptOssGroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = field(init=False)
        hidden_dim: int = field(init=False)
        num_experts: int = field(init=False)
        use_grouped_mm: bool = True
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        super().__init__()
        dim = config.dim
        hidden_dim = config.hidden_dim
        num_experts = config.num_experts
        self.num_experts = num_experts
        self.use_grouped_mm = config.use_grouped_mm
        self.swiglu_limit = config.swiglu_limit

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

    def init_weights(self, **kwargs) -> None:
        init_std = kwargs.get("init_std")
        assert init_std is not None
        nn.init.trunc_normal_(self.mlp1_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp1_bias, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_bias, mean=0.0, std=init_std)


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
        self.experts = GptOssGroupedExperts.Config(
            swiglu_limit=config.swiglu_limit,
            use_grouped_mm=config.experts.use_grouped_mm,
        ).build(
            dim=dim,
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
        )


@torch.library.custom_op("gpt_oss::expert_bias_add", mutates_args=())
def expert_bias_add(
    h: torch.Tensor,
    bias: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """Add per-expert bias to grouped-mm output without repeat_interleave.

    Replaces bias.repeat_interleave(...) + h + b with a fused Triton kernel
    that reads bias[e] once per expert and writes directly to the output.
    No intermediate (T, N) tensor is allocated.

    Implemented as a torch.library.custom_op rather than torch.autograd.Function
    so that torch.compile can inspect setup_context and know only `offs` is saved
    for backward. A torch.autograd.Function would force compile to conservatively
    save all inputs including the large (T, N) `h` tensor, negating the memory
    benefit. See register_autograd call below for details.
    """
    T, N = h.shape
    E = bias.shape[0]
    out = torch.empty_like(h)
    grid = lambda meta: (E, triton.cdiv(N, meta["BLOCK_D"]))  # noqa: E731
    _expert_bias_add_fwd_kernel[grid](h, bias.to(h.dtype), out, offs, N)
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 128, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_T": 64, "BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_T": 128, "BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_T": 64, "BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_T": 128, "BLOCK_D": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _expert_bias_add_fwd_kernel(
    h_ptr,  # (T, N) bfloat16
    bias_ptr,  # (E, N) bfloat16
    out_ptr,  # (T, N) bfloat16
    offs_ptr,  # (E,)   int32 — offsets[e] = exclusive end of expert e's tokens
    N,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    e = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Token range [t_start, t_end) for this expert
    prev_e = tl.maximum(e - 1, 0)
    t_start = tl.where(e == 0, 0, tl.load(offs_ptr + prev_e))
    t_end = tl.load(offs_ptr + e)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < N

    # Load bias[e] once for all tokens of this expert
    bias = tl.load(bias_ptr + e * N + d_offs, mask=d_mask, other=0.0)

    # Tile over tokens
    for t_base in range(t_start, t_end, BLOCK_T):
        t_offs = t_base + tl.arange(0, BLOCK_T)
        t_mask = t_offs < t_end

        ptrs = h_ptr + t_offs[:, None] * N + d_offs[None, :]
        h = tl.load(ptrs, mask=t_mask[:, None] & d_mask[None, :], other=0.0)
        tl.store(
            out_ptr + t_offs[:, None] * N + d_offs[None, :],
            h + bias[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
        )


# register_fake teaches torch.compile's fake-tensor / meta-device pass the output
# shape/dtype without running the real kernel. Required for fullgraph=True compilation.
@expert_bias_add.register_fake
def _(h, bias, offs):
    return torch.empty_like(h)


# The backward is also a custom_op rather than an inline Triton call so that
# torch.compile can trace through it without graph-breaking. A plain Triton call
# inside a torch.autograd.Function backward would be opaque to the compiler.
@torch.library.custom_op("gpt_oss::expert_bias_add_bwd", mutates_args=())
def _expert_bias_add_bwd_op(
    grad_out: torch.Tensor,
    offs: torch.Tensor,
    E: int,
    N: int,
) -> torch.Tensor:
    """Backward pass: grad_bias[e] = sum(grad_out[group_e], dim=0)."""
    grad_bias = torch.empty(E, N, dtype=torch.float32, device=grad_out.device)
    grid = lambda meta: (E, triton.cdiv(N, meta["BLOCK_D"]))  # noqa: E731
    _expert_bias_add_bwd_kernel[grid](grad_out.contiguous(), grad_bias, offs, N)
    return grad_bias


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 128, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 32, "BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_T": 64, "BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_T": 128, "BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_T": 64, "BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_T": 128, "BLOCK_D": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _expert_bias_add_bwd_kernel(
    grad_out_ptr,  # (T, N) bfloat16
    grad_bias_ptr,  # (E, N) float32 — output
    offs_ptr,  # (E,)   int32
    N,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    e = tl.program_id(0)
    pid_d = tl.program_id(1)

    prev_e = tl.maximum(e - 1, 0)
    t_start = tl.where(e == 0, 0, tl.load(offs_ptr + prev_e))
    t_end = tl.load(offs_ptr + e)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < N

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for t_base in range(t_start, t_end, BLOCK_T):
        t_offs = t_base + tl.arange(0, BLOCK_T)
        t_mask = t_offs < t_end
        g = tl.load(
            grad_out_ptr + t_offs[:, None] * N + d_offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(g, axis=0)

    tl.store(grad_bias_ptr + e * N + d_offs, acc, mask=d_mask)


@_expert_bias_add_bwd_op.register_fake
def _(grad_out, offs, E, N):
    return torch.empty(E, N, dtype=torch.float32, device=grad_out.device)


def _expert_bias_add_setup_context(ctx, inputs, output):
    _h, bias, offs = inputs
    ctx.save_for_backward(offs)
    ctx.E = bias.shape[0]
    ctx.N = bias.shape[1]


def _expert_bias_add_backward(ctx, grad_out):
    (offs,) = ctx.saved_tensors
    grad_bias = _expert_bias_add_bwd_op(grad_out, offs, ctx.E, ctx.N)
    # grad_out is passed through as grad_h (bias add is elementwise, grad is identity)
    return grad_out, grad_bias, None


# We use torch.library.register_autograd instead of torch.autograd.Function so
# that torch.compile can inspect setup_context and determine the minimal save set
# for backward. With torch.autograd.Function, compile must conservatively assume
# all inputs are saved, which would retain the large (T, N) `h` tensor in the
# activation cache — defeating the memory saving this kernel exists to provide.
# With register_autograd + setup_context, compile sees that only `offs` is saved,
# and `h` can be freed immediately after the forward kernel runs.
torch.library.register_autograd(
    "gpt_oss::expert_bias_add",
    _expert_bias_add_backward,
    setup_context=_expert_bias_add_setup_context,
)
