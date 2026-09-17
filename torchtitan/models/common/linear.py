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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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


class _ParallelLinear(Module):
    """Communication boundary around an independently configurable Linear."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear: Linear.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.linear = config.linear.build()
        # Keep the wrapper transparent to checkpoint formats. Runtime module
        # composition uses ``linear.*`` while state dictionaries retain the
        # existing projection keys such as ``w13.weight``.
        self.register_state_dict_post_hook(self._flatten_linear_on_save)
        self.register_load_state_dict_pre_hook(self._nest_linear_on_load)

    @staticmethod
    def _flatten_linear_on_save(module, state_dict, prefix, local_metadata) -> None:
        nested_prefix = f"{prefix}linear."
        for key in tuple(state_dict):
            if key.startswith(nested_prefix):
                state_dict[f"{prefix}{key[len(nested_prefix) :]}"] = state_dict.pop(key)

    @staticmethod
    def _nest_linear_on_load(module, state_dict, prefix, *args) -> None:
        nested_prefix = f"{prefix}linear."
        for key in tuple(state_dict):
            if key.startswith(prefix) and not key.startswith(nested_prefix):
                state_dict[f"{nested_prefix}{key[len(prefix) :]}"] = state_dict.pop(key)


class ColumnParallelLinear(_ParallelLinear):
    """Prepare an input for a column-parallel Linear.

    The same module handles both tensor-parallel modes. With sequence
    parallelism, ``Shard(0) -> Replicate`` is an input all-gather. Without
    sequence parallelism, ``Invariant -> Replicate`` is a forward no-op whose
    backward performs the required all-reduce.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(_ParallelLinear.Config):
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
        return self.linear(input)


class RowParallelLinear(_ParallelLinear):
    """Reduce the partial output of an independently configured Linear.

    ``Partial -> Shard(0)`` is a reduce-scatter with sequence parallelism;
    ``Partial -> Invariant`` is an all-reduce without it. The output layout
    in this module's sharding config selects between the two.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(_ParallelLinear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.linear(input)
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


LinearConfig = Linear.Config | ColumnParallelLinear.Config | RowParallelLinear.Config


def canonical_linear_fqn(fqn: str, parent: object) -> str:
    """Hide a parallel wrapper's implementation child from a Linear FQN."""
    if isinstance(parent, _ParallelLinear.Config):
        wrapper_fqn, separator, attr = fqn.rpartition(".")
        assert attr == "linear"
        return wrapper_fqn if separator else ""
    return fqn


def is_column_parallel_linear_config(config: Module.Config) -> bool:
    """Return whether a config builds an explicit column-parallel boundary."""
    return isinstance(config, ColumnParallelLinear.Config)


def is_row_parallel_linear_config(config: Module.Config) -> bool:
    """Return whether a config builds an explicit row-parallel boundary."""
    return isinstance(config, RowParallelLinear.Config)


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
        output_TE = _RouterGateLinearFunction.apply(input, self.weight)
        if self.bias is not None:
            output_TE = output_TE + self.bias.float()
        return output_TE


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
        return F.linear(input, self.weight, bias)


__all__ = [
    "CastLinear",
    "canonical_linear_fqn",
    "ColumnParallelLinear",
    "Linear",
    "LinearConfig",
    "RowParallelLinear",
    "PartialBiasRowwiseLinear",
    "RouterGateLinear",
    "is_column_parallel_linear_config",
    "is_row_parallel_linear_config",
]
