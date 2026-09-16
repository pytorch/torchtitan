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
from functools import cache

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
        # nn.Linear.__init__ calls this override before self.num_linears is set,
        # so select the initialization path from the physical weight shape.
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

    def _flatten_weight_and_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flatten stacked parameters for one linear operation."""
        if self.num_linears == 1:
            return self.weight, self.bias
        bias = None if self.bias is None else self.bias.flatten()
        return self.weight.flatten(0, -2), bias

    def _unflatten_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore the logical stacked output dimensions after a linear operation."""
        if self.num_linears == 1:
            return output
        return output.unflatten(-1, self.weight.shape[:-1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self._flatten_weight_and_bias()
        output = F.linear(input, weight, bias)
        return self._unflatten_output(output)

    def extra_repr(self) -> str:
        result = nn.Linear.extra_repr(self)
        if self.num_linears > 1:
            result += f", num_linears={self.num_linears}"
        return result


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
        weight, bias = self._flatten_weight_and_bias()
        output = F.linear(
            input.to(self.compute_dtype),
            weight.to(self.compute_dtype),
            None if bias is None else bias.to(self.compute_dtype),
        )
        return self._unflatten_output(output)


def _tp_type(layout) -> spmd.PerMeshAxisSpmdType:
    """Return the TP-axis type from a boundary layout."""
    tp_type = _per_axis_types(layout).get(MeshAxisName.TP)
    assert tp_type is not None
    return tp_type


class _ColumnParallelLinearMixin(Module):
    """All-gather a TP-sharded input before the composed projection."""

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
        return super().forward(input)


class _RowParallelLinearMixin(Module):
    """Reduce a TP-partial output after the composed projection."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
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


class ColumnParallelLinear(_ColumnParallelLinearMixin, Linear):
    """Linear that explicitly all-gathers its TP-sharded input."""

    _underlying_linear_cls = Linear

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass


class RowParallelLinear(_RowParallelLinearMixin, Linear):
    """Linear that explicitly reduces its TP-partial output."""

    _underlying_linear_cls = Linear

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass


@cache
def specialize_column_parallel_linear(
    parent_cls: type[Module],
) -> type[Module]:
    """Add an outer input all-gather boundary to a Linear implementation.

    Quantization and LoRA converters use this after replacing the underlying
    Linear class. The mixin runs first in the MRO, all-gathers the input, and
    calls ``super().forward()`` to preserve the converted Linear computation.
    """
    if issubclass(parent_cls, _ColumnParallelLinearMixin):
        return parent_cls
    if parent_cls is Linear:
        return ColumnParallelLinear

    parent_config_cls = parent_cls.Config

    class SpecializedColumnParallelLinear(
        _ColumnParallelLinearMixin, parent_cls  # type: ignore[misc, valid-type]
    ):
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

    SpecializedColumnParallelLinear.__name__ = f"ColumnParallel{parent_cls.__name__}"
    SpecializedColumnParallelLinear.__qualname__ = (
        f"ColumnParallel{parent_cls.__qualname__}"
    )
    SpecializedColumnParallelLinear._underlying_linear_cls = parent_cls
    return SpecializedColumnParallelLinear


@cache
def specialize_row_parallel_linear(
    parent_cls: type[Module],
) -> type[Module]:
    """Add an outer output-reduction boundary to a Linear implementation.

    Quantization and LoRA converters use this after replacing the underlying
    Linear class. The mixin calls ``super().forward()`` for that computation,
    then reduce-scatters or all-reduces its partial output.
    """
    if issubclass(parent_cls, _RowParallelLinearMixin):
        return parent_cls
    if parent_cls is Linear:
        return RowParallelLinear

    parent_config_cls = parent_cls.Config

    class SpecializedRowParallelLinear(
        _RowParallelLinearMixin, parent_cls  # type: ignore[misc, valid-type]
    ):
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            pass

    SpecializedRowParallelLinear.__name__ = f"RowParallel{parent_cls.__name__}"
    SpecializedRowParallelLinear.__qualname__ = f"RowParallel{parent_cls.__qualname__}"
    SpecializedRowParallelLinear._underlying_linear_cls = parent_cls
    return SpecializedRowParallelLinear


def is_column_parallel_linear_config(config: Module.Config) -> bool:
    """Return whether a config builds an explicit column-parallel boundary."""
    return config._owner is not None and issubclass(
        config._owner, _ColumnParallelLinearMixin
    )


def is_row_parallel_linear_config(config: Module.Config) -> bool:
    """Return whether a config builds an explicit row-parallel boundary."""
    return config._owner is not None and issubclass(
        config._owner, _RowParallelLinearMixin
    )


def underlying_linear_cls(config: Module.Config) -> type[Module]:
    """Return the projection implementation inside a parallel boundary."""
    assert config._owner is not None
    return getattr(config._owner, "_underlying_linear_cls", config._owner)


def preserve_parallel_linear_role(
    replacement: type[Module], source_config: Module.Config
) -> type[Module]:
    """Apply ``source_config``'s column/row role to ``replacement``."""
    if is_column_parallel_linear_config(source_config):
        return specialize_column_parallel_linear(replacement)
    if is_row_parallel_linear_config(source_config):
        return specialize_row_parallel_linear(replacement)
    return replacement


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
        weight, bias = self._flatten_weight_and_bias()
        output_TE = _RouterGateLinearFunction.apply(input, weight)
        if bias is not None:
            output_TE = output_TE + bias.float()
        return self._unflatten_output(output_TE)


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
        weight, bias = self._flatten_weight_and_bias()
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
        output = F.linear(input, weight, bias)
        return self._unflatten_output(output)


__all__ = [
    "CastLinear",
    "ColumnParallelLinear",
    "Linear",
    "RowParallelLinear",
    "PartialBiasRowwiseLinear",
    "RouterGateLinear",
    "is_column_parallel_linear_config",
    "is_row_parallel_linear_config",
    "underlying_linear_cls",
    "preserve_parallel_linear_role",
    "specialize_column_parallel_linear",
    "specialize_row_parallel_linear",
]
