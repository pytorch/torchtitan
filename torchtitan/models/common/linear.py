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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor

from torchtitan.protocols.module import Module

# Shape suffix legend for the router gate:
#   T = num tokens, D = model dimension, E = num experts


class Linear(nn.Linear, Module):
    """Configurable nn.Linear."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # Converters such as LoRA use this hook to preserve independent state
        # for each logical projection while building one interleaved GEMM.
        _custom_interleaved_linear_builder: ClassVar[Callable[..., Any] | None] = None
        in_features: int
        out_features: int
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.out_features,
            bias=config.bias,
        )


def _make_interleaved_linear_config(
    logical_configs: tuple[tuple[str, Linear.Config], ...],
    param_init: dict[str, Callable] | None,
    *,
    config_type: type[Linear.Config] | None = None,
) -> Linear.Config:
    """Merge compatible logical output projections into one Linear config.

    Every option except output size and initialization must match so the result
    can be represented by one physical GEMM without changing either projection's
    requested behavior.
    """
    if not logical_configs:
        raise ValueError("At least one logical Linear config is required")

    first_name, first_config = logical_configs[0]
    merged_config_type = config_type or type(first_config)
    comparable_fields = {
        field.name
        for field in fields(merged_config_type)
        if field.init and field.name not in ("out_features", "param_init")
    }

    for logical_name, config in logical_configs:
        compatible_type = (
            isinstance(config, merged_config_type)
            if config_type is not None
            else type(config) is merged_config_type
        )
        if not compatible_type:
            raise ValueError(
                "Cannot fuse logical Linear projections with different "
                f"implementations: {first_name} uses {type(first_config).__name__}, "
                f"but {logical_name} uses {type(config).__name__}."
            )
        for field_name in comparable_fields:
            if getattr(config, field_name) != getattr(first_config, field_name):
                raise ValueError(
                    "Cannot fuse logical Linear projections with different "
                    f"{field_name}: {first_name} and {logical_name}."
                )

    config_kwargs = {
        field.name: getattr(first_config, field.name)
        for field in fields(merged_config_type)
        if field.init
    }
    config_kwargs["out_features"] = sum(
        config.out_features for _, config in logical_configs
    )
    config_kwargs["param_init"] = param_init
    return merged_config_type(**config_kwargs)


def _build_interleaved_linear(
    logical_configs: tuple[tuple[str, Linear.Config], ...],
    param_init: dict[str, Callable] | None,
) -> Any:
    """Build equal-sized logical projections as one interleaved Linear.

    ``_logical_output_slices`` lets checkpoint and serving integrations recover
    the logical projection names even though ``named_parameters()`` sees only the
    merged module.
    """
    if not logical_configs:
        raise ValueError("At least one logical Linear config is required")

    output_sizes = {config.out_features for _, config in logical_configs}
    if len(output_sizes) != 1:
        raise ValueError(
            "Interleaved logical Linear projections must have matching out_features"
        )

    custom_builders = {
        builder
        for _, config in logical_configs
        if (builder := type(config)._custom_interleaved_linear_builder) is not None
    }
    if len(custom_builders) > 1:
        raise ValueError(
            "Logical Linear projections specify different custom interleaved builders"
        )

    if custom_builders:
        custom_builder = custom_builders.pop()
        merged = custom_builder(logical_configs, param_init)
    else:
        merged = _make_interleaved_linear_config(logical_configs, param_init).build()

    merged._logical_output_slices = tuple(
        (name, config.out_features) for name, config in logical_configs
    )
    return merged


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


class ScaledBiasRowwiseLinear(Linear):
    """
    Rowwise linear whose local bias contribution is scaled by TP degree.
    TODO(pianpwk): this should work in decomposition in spmd_types, or as Partial
    init in DTensor. Today the local SPMD typecheck errors on the TP-axis
    input:V, weight:V, bias:P case; decomposing to input @ weight -> P, then P + P should pass.
    For DTensor, this errors because FSDP does not want to redistribute the incoming gradient
    from Replicate -> storage-time Partial.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_degree = 1

    def parallelize(self, parallel_dims) -> None:
        self.tp_degree = parallel_dims.tp
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
        bias = bias / self.tp_degree
        return F.linear(input, weight, bias)


__all__ = [
    "Linear",
    "RouterGateLinear",
    "ScaledBiasRowwiseLinear",
]
