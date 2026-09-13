# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import ExpertActivation, MoE


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


class GptOssExpertActivation(ExpertActivation):
    """GPT-OSS clamped SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(ExpertActivation.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(
        self,
        gate_RF: torch.Tensor,
        up_RF: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        del offsets_E
        return _swiglu_components(gate_RF, up_RF, limit=self.swiglu_limit)


class GptOssDownGroupedLinear(GroupedLinear):
    """Grouped down projection with GPT-OSS TP bias semantics."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
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
        output: torch.Tensor,
        bias: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        row_bias = self._expand_grouped_bias(bias, offsets, output.shape[0])
        row_bias = ScaleBiasForward.apply(row_bias, self.tp_degree, output.dtype)
        return output + row_bias


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass
