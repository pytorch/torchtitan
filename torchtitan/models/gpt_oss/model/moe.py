# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.models.moe.moe import MoE
from torchtitan.models.moe.utils import _permute, _unpermute

from .args import GptOssModelArgs

# Check for torch._grouped_mm availability (requires PyTorch 2.x+)
_GROUPED_MM_AVAILABLE = hasattr(torch, '_grouped_mm')


class ScaleBiasForward(torch.autograd.Function):
    """
    Custom autograd function that scales bias in forward pass but not in backward.

    For tensor parallel MoE, we need to scale the bias by 1/tp_degree in forward
    to cancel the extra reduction effect, but keep the gradient unchanged in backward.

    Note: tp_degree is expected to be a Python int (not a tensor). This is safe because
    the tensor parallel degree is fixed at model initialization time.
    """

    @staticmethod
    def forward(ctx, bias: torch.Tensor, tp_degree: int) -> torch.Tensor:
        # Store tp_degree for potential debugging; not used in backward since
        # we pass gradients through unchanged
        ctx.tp_degree = tp_degree
        if tp_degree > 1:
            return bias / tp_degree
        return bias

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
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
        mlp1_bias: torch.Tensor | None,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor | None,
        swiglu_limit: float,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        tp_degree: int = 1,
        compute_dtype: torch.dtype = torch.bfloat16,
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
            compute_dtype,
        )

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper


def swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def _run_experts_for_loop(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor | None,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor | None,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
    compute_dtype: torch.dtype = torch.bfloat16,  # unused in for-loop, kept for API consistency
) -> torch.Tensor:
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
        h = torch.matmul(x_expert, mlp1_weight[expert_idx].transpose(-2, -1))
        if mlp1_bias is not None:
            h = h + mlp1_bias[expert_idx]
        h = swiglu(h, limit=swiglu_limit)
        h = torch.matmul(h, mlp2_weight[expert_idx].transpose(-2, -1))
        if mlp2_bias is not None:
            # Apply custom autograd function to scale bias in forward but not in backward
            b2 = ScaleBiasForward.apply(mlp2_bias[expert_idx], tp_degree)
            h = h + b2
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


def _run_experts_grouped_mm(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor | None,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor | None,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if not _GROUPED_MM_AVAILABLE:
        raise RuntimeError(
            "torch._grouped_mm is not available. This requires PyTorch 2.x or later. "
            "Please upgrade PyTorch or set use_grouped_mm=False in MoEArgs."
        )

    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    num_tokens_per_expert_long = num_tokens_per_expert.to(torch.long)

    # Use configurable compute dtype instead of hardcoded bfloat16
    h = torch._grouped_mm(
        x.to(compute_dtype), mlp1_weight.transpose(-2, -1).to(compute_dtype), offs=offsets
    )

    if mlp1_bias is not None:
        b1 = mlp1_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
        tail_slack = x.shape[0] - int(offsets[-1])
        if tail_slack:
            b1 = torch.cat([b1, b1.new_zeros((tail_slack, b1.shape[-1]))], dim=0)
        h = h + b1.to(h.dtype)

    h = swiglu(h, limit=swiglu_limit)
    h = torch._grouped_mm(h, mlp2_weight.transpose(-2, -1).to(compute_dtype), offs=offsets)

    if mlp2_bias is not None:
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
        use_expert_bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_grouped_mm = use_grouped_mm
        self.swiglu_limit = swiglu_limit
        self.use_expert_bias = use_expert_bias
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.compute_dtype = compute_dtype

        # Cached values - computed once on first forward pass
        # This avoids repeated isinstance/device mesh lookups on every forward call
        self._cached_tp_degree: int | None = None
        self._is_dtensor: bool | None = None  # Whether weights are DTensors (set on first forward)

        self.mlp1_weight = nn.Parameter(
            torch.empty((num_experts, hidden_dim * 2, dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp2_weight = nn.Parameter(
            torch.empty((num_experts, dim, hidden_dim))
        )  # (num_experts, out_dim, in_dim)

        # Expert biases are part of the GPT-OSS architecture (gate_up_proj_bias, down_proj_bias)
        # They should be loaded from pretrained HF checkpoints when use_expert_bias=True
        if use_expert_bias:
            self.mlp1_bias = nn.Parameter(torch.empty((num_experts, hidden_dim * 2)))
            self.mlp2_bias = nn.Parameter(torch.empty((num_experts, dim)))
        else:
            self.mlp1_bias = None
            self.mlp2_bias = None

    def _get_tp_degree(self) -> int:
        """Get tensor parallel degree, caching the result for efficiency."""
        if self._cached_tp_degree is not None:
            return self._cached_tp_degree

        tp_degree = 1
        # Use cached _is_dtensor if available, otherwise check directly
        is_dtensor = self._is_dtensor if self._is_dtensor is not None else isinstance(self.mlp1_weight, DTensor)
        if is_dtensor:
            mesh_dim_names = self.mlp1_weight.device_mesh.mesh_dim_names
            if "tp" in mesh_dim_names:
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        self._cached_tp_degree = tp_degree
        return tp_degree

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        *args,  # Extra args for DeepEP hooks (selected_experts_indices, top_scores, num_experts)
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for grouped experts.

        Args:
            x: Input tensor of shape (num_tokens, dim)
            num_tokens_per_expert: Number of tokens assigned to each expert
            *args: Additional arguments passed by DeepEP hooks (ignored in base implementation)
            **kwargs: Additional keyword arguments (ignored in base implementation)

        Note: When using DeepEP (DeepEPExpertParallel), the parallelization hooks intercept
        this call and handle the extra routing arguments (selected_experts_indices, top_scores).
        In the base case without DeepEP, these extra args are unused.
        """
        # Cache DTensor check on first forward (avoids isinstance() on every forward)
        if self._is_dtensor is None:
            self._is_dtensor = isinstance(self.mlp1_weight, DTensor)

        if self._is_dtensor:
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            mlp1_weight = self.mlp1_weight.to_local()
            mlp2_weight = self.mlp2_weight.to_local()
            mlp1_bias = self.mlp1_bias.to_local() if self.mlp1_bias is not None else None
            mlp2_bias = self.mlp2_bias.to_local() if self.mlp2_bias is not None else None
        else:
            mlp1_weight = self.mlp1_weight
            mlp2_weight = self.mlp2_weight
            mlp1_bias = self.mlp1_bias
            mlp2_bias = self.mlp2_bias

        tp_degree = self._get_tp_degree()

        if self.use_grouped_mm:
            if (
                not self._is_dtensor
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
                self.compute_dtype,
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
                self.compute_dtype,
            )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.mlp1_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_weight, mean=0.0, std=init_std)
        if self.mlp1_bias is not None:
            nn.init.trunc_normal_(self.mlp1_bias, mean=0.0, std=init_std)
        if self.mlp2_bias is not None:
            nn.init.trunc_normal_(self.mlp2_bias, mean=0.0, std=init_std)


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    def __init__(self, model_args: GptOssModelArgs, dim: int, hidden_dim: int):
        # Convert GptOssModelArgs to MoEArgs for base class compatibility
        moe_args = model_args.moe_args

        # Initialize the base MoE class
        super().__init__(moe_args, dim, hidden_dim)

        # Expert biases are part of the GPT-OSS architecture and should be loaded
        # from pretrained models. Set use_expert_bias=True in MoEArgs for 20b/120b.
        # For backward compatibility, also enable biases when load_balance_coeff is set.
        use_expert_bias = moe_args.use_expert_bias or (moe_args.load_balance_coeff is not None)

        # Override the base GroupedExperts with GptOssGroupedExperts
        self.experts = GptOssGroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=moe_args.num_experts,
            swiglu_limit=model_args.swiglu_limit,
            use_grouped_mm=moe_args.use_grouped_mm,
            use_expert_bias=use_expert_bias,
        )


class GptOssDeepEPMoE(GptOssMoE):
    """
    GptOss MoE with DeepEP communication backend.

    Inherits from GptOssMoE but overrides forward() to pass routing info
    directly to experts, letting DeepEPExpertParallel hooks handle
    dispatch/combine instead of using the reorderer.

    IMPORTANT: This class requires DeepEPExpertParallel to be applied during
    model parallelization. The extra arguments passed to experts() are handled
    by the DeepEP hooks. If used without parallelization, the extra arguments
    are safely ignored by GptOssGroupedExperts.forward().
    """

    def __init__(self, model_args: GptOssModelArgs, dim: int, hidden_dim: int):
        super().__init__(model_args, dim, hidden_dim)
        # DeepEP doesn't use reorderer - routing handled by DeepEPExpertParallel
        self.reorderer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DeepEP communication.

        DeepEPExpertParallel hooks intercept experts() call and handle
        dispatch/combine via deepep functions. The extra routing arguments
        (selected_experts_indices, top_scores, num_experts) are passed to
        experts() and handled by the hooks.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, self.expert_bias
        )

        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Call experts with routing info - DeepEP hooks handle dispatch/combine
        # Extra args are passed through to hooks; ignored if hooks not applied
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        out = self.shared_experts(x) if self.shared_experts is not None else None

        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)
