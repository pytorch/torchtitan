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

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.protocols.module import Module

# Shape suffix legend for the router gate:
#   T = num tokens, D = model dimension, E = num experts


class Linear(nn.Linear, Module):
    """Configurable linear that can store multiple stacked projections.

    A single projection keeps the standard ``[out_features, in_features]``
    parameter shape. Multiple projections use
    ``[num_linears, out_features, in_features]``, keeping each projection
    contiguous for blockwise weight quantization, and return
    ``[..., num_linears, out_features]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        num_linears: int = 1
        bias: bool = False

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
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().unflatten(
                        0, (config.num_linears, config.out_features)
                    ),
                    requires_grad=self.bias.requires_grad,
                )

    def reset_parameters(self) -> None:
        # Flattening handles both ordinary and stacked projections while
        # keeping fan-in equal to in_features.
        nn.init.kaiming_uniform_(self.weight.flatten(0, -2), a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _flatten_weight_and_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flatten stacked parameters for one linear operation."""
        weight = self.weight.flatten(0, -2)
        bias = None if self.bias is None else self.bias.flatten()
        return weight, bias

    def _unflatten_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore the logical stacked output dimensions after a linear operation."""
        if self.num_linears == 1:
            return output
        return output.unflatten(-1, self.weight.shape[:-1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self._flatten_weight_and_bias()
        output = self._linear(input, weight, bias)
        return self._unflatten_output(output)

    def extra_repr(self) -> str:
        result = nn.Linear.extra_repr(self)
        if self.num_linears > 1:
            result += f", num_linears={self.num_linears}"
        return result

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply local projection compute without outer communication.

        LoRA and quantized subclasses override this method so column- and
        row-parallel ``forward`` methods continue to own their collectives.
        Explicit operands let those boundaries adjust an operand's SPMD type
        before invoking the selected local compute implementation.
        """
        return F.linear(input, weight, bias)


class CastLinear(Linear):
    """``Linear`` whose forward matmul runs in ``compute_dtype``.

    Inputs, weight, and bias are cast to ``compute_dtype`` before
    ``F.linear`` and the output is returned in that dtype. The stored
    parameters retain their original dtype, including under weight tying.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        compute_dtype: str = "float32"
        """Dtype for the forward matmul (key into ``TORCH_DTYPE_MAP``)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_dtype = TORCH_DTYPE_MAP[config.compute_dtype]

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # The optimizer updates the weight each step, so training cannot cache
        # the upcast copy. Inference may be able to cache it between syncs.
        bias = None if bias is None else bias.to(self.compute_dtype)
        return F.linear(
            input.to(self.compute_dtype), weight.to(self.compute_dtype), bias
        )


class ColumnParallelLinear(Linear):
    """Prepare an input for a column-parallel Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. The same module handles both tensor-parallel modes.
    With sequence parallelism, ``Shard(0) -> Replicate`` is an input all-gather.
    Without sequence parallelism, ``Invariant -> Replicate`` is a forward no-op
    whose backward performs the required all-reduce.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            input = spmd.redistribute(
                input,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": input.dtype},
            )
        return super().forward(input)


class RowParallelLinear(Linear):
    """Reduce the partial output of an independently configured Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. ``Partial -> Shard(0)`` is a reduce-scatter, while
    ``Partial -> Invariant`` is an all-reduce without it. Dense SP state selects
    between the two. An invariant bias is converted to a partial contribution
    before local compute so the reduction adds it exactly once.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        weight, bias = self._flatten_weight_and_bias()
        if bias is not None and tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
            # The selected local compute may be native, LoRA, or quantized.
            # Its row-sharded operands and bias jointly produce a partial output.
            # TODO: Remove this suppression once spmd_types recognizes the
            # rowwise F.linear type combination [V, V, P] -> P.
            with spmd.no_typecheck():
                output = self._unflatten_output(self._linear(input, weight, bias))
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(self._linear(input, weight, bias))
        if tp_group is None:
            return output

        return spmd.redistribute(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
            backward_options={"op_dtype": output.dtype},
        )


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

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        output_TE = _RouterGateLinearFunction.apply(input, weight)
        if bias is not None:
            output_TE = output_TE + bias.float()
        return output_TE


__all__ = [
    "CastLinear",
    "ColumnParallelLinear",
    "Linear",
    "RowParallelLinear",
    "RouterGateLinear",
]
