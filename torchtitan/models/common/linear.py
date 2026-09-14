# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable linear modules.

``Linear`` uses diamond inheritance (``nn.Linear`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- Standard ``nn.Linear`` parameter and state-dict behavior is retained.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.
"""

import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.protocols.module import Module

# Shape suffix legend for the router gate:
#   T = num tokens, D = model dimension, E = num experts


class Linear(nn.Linear, Module):
    """Configurable linear with optional stacked output projections.

    With ``num_linears == 1``, the parameter and output use the standard
    ``[out_features, in_features]`` and ``[..., out_features]`` shapes. With
    ``num_linears > 1``, they use ``[num_linears, out_features, in_features]``
    and ``[..., num_linears, out_features]``. The stacked parameter is flattened
    without a copy for the GEMM, so each projection remains contiguous for
    blockwise weight quantization.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        num_linears: int = 1
        bias: bool = False

        def __post_init__(self) -> None:
            if self.in_features <= 0:
                raise ValueError(
                    f"in_features must be positive, got {self.in_features}"
                )
            if self.out_features <= 0:
                raise ValueError(
                    f"out_features must be positive, got {self.out_features}"
                )
            if self.num_linears <= 0:
                raise ValueError(
                    f"num_linears must be positive, got {self.num_linears}"
                )

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        if config.num_linears > 1:
            self.weight = nn.Parameter(
                self.weight.detach().unflatten(
                    0, (config.num_linears, config.out_features)
                ),
                requires_grad=self.weight.requires_grad,
            )
        if config.num_linears > 1 and self.bias is not None:
            self.bias = nn.Parameter(
                self.bias.detach().unflatten(
                    0, (config.num_linears, config.out_features)
                ),
                requires_grad=self.bias.requires_grad,
            )

    def reset_parameters(self) -> None:
        if self.weight.ndim == 2:
            nn.Linear.reset_parameters(self)
            return
        # init_states() calls this after meta materialization, when a stacked
        # weight is already 3D. Flatten first so fan-in remains in_features;
        # PyTorch's generic 3D fan calculation would incorrectly use
        # out_features * in_features.
        nn.init.kaiming_uniform_(self.weight.flatten(0, -2), a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.num_linears == 1:
            return F.linear(input, self.weight, self.bias)
        weight = self.weight.flatten(0, -2)
        bias = None if self.bias is None else self.bias.flatten()
        output = F.linear(input, weight, bias)
        return output.unflatten(-1, self.weight.shape[:-1])

    def extra_repr(self) -> str:
        result = nn.Linear.extra_repr(self)
        if self.num_linears > 1:
            result += f", num_linears={self.num_linears}"
        return result


@spmd.register_local_autograd_function
class _RouterGateLinearFunction(torch.autograd.Function):
    """Router projection with FP32 output and backward GEMMs."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, input_TD: torch.Tensor, weight_ED: torch.Tensor
    ) -> torch.Tensor:
        use_cuda_bf16_forward = (
            input_TD.device.type == "cuda"
            and input_TD.dtype is torch.bfloat16
            and weight_ED.dtype is torch.bfloat16
        )
        if use_cuda_bf16_forward:
            input_forward_TD = input_TD
            weight_forward_ED = weight_ED
            # CUDA supports BF16 matmul with FP32 accumulation and output via
            # out_dtype. The portable path below promotes the operands because
            # this mixed input/output dtype is not supported by all devices.
            output_TE = torch.mm(
                input_forward_TD, weight_forward_ED.T, out_dtype=torch.float32
            )
        else:
            input_forward_TD = input_TD.float()
            weight_forward_ED = weight_ED.float()
            output_TE = torch.mm(input_forward_TD, weight_forward_ED.T)

        ctx.save_for_backward(input_forward_TD, weight_forward_ED)
        ctx.input_dtype = input_TD.dtype
        ctx.weight_dtype = weight_ED.dtype
        return output_TE

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE: torch.Tensor):  # pyrefly: ignore[bad-override]
        input_forward_TD, weight_forward_ED = ctx.saved_tensors
        grad_output_fp32_TE = grad_output_TE.float()

        grad_input_TD = None
        if ctx.needs_input_grad[0]:
            grad_input_TD = torch.mm(grad_output_fp32_TE, weight_forward_ED.float()).to(
                ctx.input_dtype
            )

        grad_weight_ED = None
        if ctx.needs_input_grad[1]:
            grad_weight_ED = torch.mm(
                grad_output_fp32_TE.T, input_forward_TD.float()
            ).to(ctx.weight_dtype)

        return grad_input_TD, grad_weight_ED


class RouterGateLinear(Linear):
    """Router projection with FP32 output and backward compute.

    CUDA uses BF16 forward compute when both operands are BF16. All other
    forward paths use FP32 compute.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.num_linears == 1 else self.weight.flatten(0, -2)
        output_TE = _RouterGateLinearFunction.apply(input, weight)
        if self.bias is not None:
            bias = self.bias if self.num_linears == 1 else self.bias.flatten()
            output_TE = output_TE + bias.float()
        if self.num_linears == 1:
            return output_TE
        return output_TE.unflatten(-1, self.weight.shape[:-1])


class PartialBiasRowwiseLinear(Linear):
    """Rowwise linear whose invariant bias becomes TP-partial in forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def __init__(self, config: Config):
        if not config.bias:
            raise ValueError("PartialBiasRowwiseLinear requires bias=True")
        super().__init__(config)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        assert bias is not None
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
        if self.num_linears == 1:
            return F.linear(input, self.weight, bias)
        output = F.linear(input, self.weight.flatten(0, -2), bias.flatten())
        return output.unflatten(-1, self.weight.shape[:-1])


__all__ = [
    "Linear",
    "PartialBiasRowwiseLinear",
    "RouterGateLinear",
]
