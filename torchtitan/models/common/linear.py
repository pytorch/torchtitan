# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable linear modules.

``Linear`` uses diamond inheritance (``nn.Linear`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- All ``nn.Linear`` logic (forward, state_dict, etc.) is reused as-is.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.
"""

import functools
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types, spmd_mesh_group
from torchtitan.protocols.module import Module

# Shape suffix legend for the router gate:
#   T = num tokens, D = model dimension, E = num experts


class Linear(nn.Linear, Module):
    """Configurable nn.Linear."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.out_features,
            bias=config.bias,
        )

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        """Apply local projection compute without outer communication.

        LoRA and quantized subclasses override this method so column- and
        row-parallel ``forward`` methods continue to own their collectives.
        """
        return F.linear(input, self.weight, self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._linear(input)


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

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        # The optimizer updates the weight each step, so training cannot cache
        # the upcast copy. Inference may be able to cache it between syncs.
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        return F.linear(
            input.to(self.compute_dtype), self.weight.to(self.compute_dtype), bias
        )


def _tp_type(layout) -> spmd.PerMeshAxisSpmdType:
    """Return the TP-axis type from a boundary layout."""
    tp_type = _per_axis_types(layout).get(MeshAxisName.TP)
    assert tp_type is not None
    return tp_type


class ColumnParallelLinear(Linear):
    """Prepare an input for a column-parallel Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. The same module handles both tensor-parallel modes.
    With sequence
    parallelism, ``Shard(0) -> Replicate`` is an input all-gather. Without
    sequence parallelism, ``Invariant -> Replicate`` is a forward no-op whose
    backward performs the required all-reduce.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            sharding_config = self._sharding_config
            assert sharding_config is not None
            assert sharding_config.in_src_shardings is not None
            input_layout = sharding_config.in_src_shardings["input"]
            input = spmd.redistribute(
                input,
                tp_group,
                src=_tp_type(input_layout),
                dst=spmd.R,
                backward_options={"op_dtype": input.dtype},
            )
        return self._linear(input)


class RowParallelLinear(Linear):
    """Reduce the partial output of an independently configured Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. ``Partial -> Shard(0)`` is a reduce-scatter with
    ``Partial -> Invariant`` is an all-reduce without it. The output layout
    in this module's sharding config selects between the two.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self._linear(input)
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is None:
            return output

        sharding_config = self._sharding_config
        assert sharding_config is not None
        output_layout = sharding_config.out_src_shardings
        assert output_layout is not None and not isinstance(output_layout, tuple)
        return spmd.redistribute(
            output,
            tp_group,
            src=spmd.P,
            dst=_tp_type(output_layout),
            backward_options={"op_dtype": output.dtype},
        )


def get_parallel_linear_cls(
    config: Linear.Config,
) -> type[ColumnParallelLinear] | type[RowParallelLinear] | None:
    """Return the canonical column- or row-parallel class for a config."""
    owner = config._owner
    if owner is not None and issubclass(owner, ColumnParallelLinear):
        return ColumnParallelLinear
    if owner is not None and issubclass(owner, RowParallelLinear):
        return RowParallelLinear
    return None


@functools.cache
def compose_parallel_linear_cls(
    compute_cls: type[Module],
    parallel_cls: type[ColumnParallelLinear] | type[RowParallelLinear],
) -> type[Module]:
    """Combine one local Linear implementation with a column/row TP role.

    The parallel class owns ``forward`` and its collective. The compute class
    owns parameter construction and ``_linear``. This is used when LoRA or a
    quantization converter replaces the local compute without changing the
    projection's tensor-parallel role.
    """
    if compute_cls is Linear:
        return parallel_cls

    compute_config_cls = compute_cls.Config

    class SpecializedParallelLinear(parallel_cls, compute_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(compute_config_cls):  # type: ignore[misc]
            pass

        def __init__(self, config: Config):
            compute_cls.__init__(self, config)

        def _linear(self, input: torch.Tensor) -> torch.Tensor:
            return compute_cls._linear(self, input)  # type: ignore[attr-defined]

    compute_name = compute_cls.__name__.removesuffix("Linear")
    specialized_name = f"{compute_name}{parallel_cls.__name__}"
    SpecializedParallelLinear.__name__ = specialized_name
    SpecializedParallelLinear.__qualname__ = specialized_name
    SpecializedParallelLinear.__module__ = compute_cls.__module__
    SpecializedParallelLinear.Config.__qualname__ = f"{specialized_name}.Config"
    SpecializedParallelLinear.Config.__module__ = compute_cls.__module__
    return SpecializedParallelLinear


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

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        output_TE = _RouterGateLinearFunction.apply(input, self.weight)
        if self.bias is not None:
            output_TE = output_TE + self.bias.float()
        return output_TE


class PartialBiasRowwiseLinear(RowParallelLinear):
    """Row-parallel Linear whose invariant bias joins the partial output."""

    @dataclass(kw_only=True, slots=True)
    class Config(RowParallelLinear.Config):
        pass

    def __init__(self, config: Config):
        if not config.bias:
            raise ValueError("PartialBiasRowwiseLinear requires bias=True")
        super().__init__(config)

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
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
        return F.linear(input, self.weight, bias)


__all__ = [
    "CastLinear",
    "ColumnParallelLinear",
    "Linear",
    "RowParallelLinear",
    "PartialBiasRowwiseLinear",
    "RouterGateLinear",
    "get_parallel_linear_cls",
    "compose_parallel_linear_cls",
]
