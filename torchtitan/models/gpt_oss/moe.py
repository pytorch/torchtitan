# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch import nn

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.activation import ActivationFn
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import MoE


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
    return _swiglu_components(x_glu, x_linear, alpha=alpha, limit=limit)


def _swiglu_components(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Apply GPT-OSS SwiGLU to already separated gate and up projections."""
    # Clamp the input values
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    out_glu = gate * torch.sigmoid(alpha * gate)
    # Note we add an extra bias of 1 to the linear layer
    return torch.addcmul(out_glu, out_glu, up)


class GptOssSwiGLU(ActivationFn):
    """GPT-OSS clamped SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationFn.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        self.swiglu_limit = config.swiglu_limit

    def __call__(
        self,
        gate_RF: torch.Tensor,
        up_RF: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        return _swiglu_components(gate_RF, up_RF, limit=self.swiglu_limit)


class GptOssGroupedLinear(GroupedLinear):
    """Grouped linear with GPT-OSS per-expert bias."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.empty(config.group_size, *self.output_shape))

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        output_RO = super().forward(input_RI, offsets_E)
        bias_RO = self._expand_grouped_bias(
            self.bias.flatten(1), offsets_E, output_RO.shape[0]
        ).reshape_as(output_RO)
        return self._add_grouped_bias(output_RO, bias_RO)

    @staticmethod
    def _expand_grouped_bias(
        bias_EO: torch.Tensor,
        offsets_E: torch.Tensor,
        output_rows: int,
    ) -> torch.Tensor:
        """Expand expert bias across routed rows and zero-valued tail padding."""
        counts_E = torch.diff(torch.cat((offsets_E.new_zeros(1), offsets_E)))
        tail_count = (output_rows - offsets_E[-1]).unsqueeze(0).to(counts_E.dtype)
        padded_bias = torch.cat((bias_EO, bias_EO.new_zeros(1, bias_EO.shape[-1])))
        return padded_bias.repeat_interleave(
            torch.cat((counts_E, tail_count)).long(),
            dim=0,
            output_size=output_rows,
        )

    def _add_grouped_bias(
        self,
        output_RO: torch.Tensor,
        bias_RO: torch.Tensor,
    ) -> torch.Tensor:
        return output_RO + bias_RO.to(output_RO.dtype)


class GptOssDownGroupedLinear(GptOssGroupedLinear):
    """GPT-OSS grouped down projection with TP-aware forward bias."""

    @dataclass(kw_only=True, slots=True)
    class Config(GptOssGroupedLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_degree = 1

    def parallelize(self, parallel_dims) -> None:
        """Record the TP degree used to scale the local forward bias."""
        self.tp_degree = parallel_dims.tp
        super().parallelize(parallel_dims)

    def _add_grouped_bias(
        self,
        output_RO: torch.Tensor,
        bias_RO: torch.Tensor,
    ) -> torch.Tensor:
        bias_RO = ScaleBiasForward.apply(bias_RO, self.tp_degree, output_RO.dtype)
        return output_RO + bias_RO


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass
