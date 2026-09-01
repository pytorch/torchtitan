# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref
from dataclasses import dataclass, fields
from typing import cast, Literal, TypeVar

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


ConfigT = TypeVar("ConfigT", bound=Module.Config)


def _get_local_wgrad(grad: torch.Tensor) -> torch.Tensor:
    return grad.to_local() if isinstance(grad, DTensor) else grad


def _validate_wgrad_destination(
    grad: torch.Tensor,
    *,
    expected_shape: torch.Size,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> torch.Tensor:
    local_grad = _get_local_wgrad(grad)
    if local_grad.shape != expected_shape:
        raise ValueError(
            "existing weight gradient must be the full unsharded gradient; "
            f"expected shape {tuple(expected_shape)}, got {tuple(local_grad.shape)}"
        )
    if local_grad.dtype != expected_dtype:
        raise ValueError(
            "existing weight gradient must match the accumulation dtype; "
            f"expected {expected_dtype}, got {local_grad.dtype}"
        )
    if local_grad.device != expected_device:
        raise ValueError(
            "existing weight gradient must be on the weight device; "
            f"expected {expected_device}, got {local_grad.device}"
        )
    if not local_grad.is_contiguous():
        raise ValueError("existing weight gradient must be contiguous")
    return local_grad


class _FusedWGradAccumLinearFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        input: torch.Tensor,
        weight_NK: torch.Tensor,
        weight_ref: weakref.ReferenceType[torch.Tensor],
        wgrad_accum_dtype: torch.dtype,
    ) -> torch.Tensor:
        if input.dtype != torch.bfloat16 or weight_NK.dtype != torch.bfloat16:
            raise ValueError("fused WGRAD accumulation requires BF16 tensors")
        ctx.save_for_backward(input, weight_NK)
        ctx.weight_ref = weight_ref
        ctx.wgrad_accum_dtype = wgrad_accum_dtype
        return F.linear(input, weight_NK)

    @staticmethod
    @torch.autograd.function.once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        input, weight_NK = ctx.saved_tensors
        input_shape = input.shape
        grad_output_MN = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        input_MK = input.reshape(-1, input.shape[-1]).contiguous()

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output_MN.mm(weight_NK).reshape(input_shape)

        grad_weight_NK = None
        if ctx.needs_input_grad[1]:
            weight = ctx.weight_ref()
            if weight is None:
                raise RuntimeError("the Linear weight was released before backward")

            previous_grad = weight.grad
            if previous_grad is None:
                grad_weight_NK = torch.mm(
                    grad_output_MN.t(),
                    input_MK,
                    out_dtype=ctx.wgrad_accum_dtype,
                )
            else:
                local_grad_NK = _validate_wgrad_destination(
                    previous_grad,
                    expected_shape=weight_NK.shape,
                    expected_dtype=ctx.wgrad_accum_dtype,
                    expected_device=weight_NK.device,
                )
                torch.addmm(
                    local_grad_NK,
                    grad_output_MN.t(),
                    input_MK,
                    out=local_grad_NK,
                    out_dtype=ctx.wgrad_accum_dtype,
                )
                weight.grad = None
                grad_weight_NK = local_grad_NK

        return grad_input, grad_weight_NK, None, None


class FusedWGradAccumLinear(Linear):
    """BF16 Linear with in-place WGRAD accumulation across backward calls."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        wgrad_accum_dtype: Literal["bfloat16", "float32"] = "float32"

    def __init__(self, config: Config):
        super().__init__(config)
        self.wgrad_accum_dtype = TORCH_DTYPE_MAP[config.wgrad_accum_dtype]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "fused WGRAD accumulation does not support torch.compile because "
                "it mutates parameter.grad during backward"
            )
        weight_NK = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        weight_NK.grad_dtype = self.wgrad_accum_dtype
        output = _FusedWGradAccumLinearFunction.apply(
            input,
            weight_NK,
            weakref.ref(weight_NK),
            self.wgrad_accum_dtype,
        )
        if self.bias is not None:
            output = output + self.bias
        return output


def enable_fused_wgrad_accumulation(
    model_config: ConfigT,
    *,
    reduce_dtype: Literal["bfloat16", "float32"],
    fqns: list[str],
) -> ConfigT:
    """Replace selected standard Linear configs with fused WGRAD Linears."""
    for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
        if type(linear_config) is not Linear.Config:
            continue
        if fqns and not any(target_fqn in fqn for target_fqn in fqns):
            continue

        linear_kwargs = {
            config_field.name: getattr(linear_config, config_field.name)
            for config_field in fields(Linear.Config)
        }
        new_config = FusedWGradAccumLinear.Config(
            **linear_kwargs,
            wgrad_accum_dtype=reduce_dtype,
        )
        if parent is None:
            model_config = cast(ConfigT, new_config)
        elif isinstance(parent, list):
            assert isinstance(attr, int)
            parent[attr] = new_config
        else:
            assert isinstance(attr, str)
            setattr(parent, attr, new_config)

    logger.info("Enabled fused WGRAD accumulation on Linear layers")
    return model_config


__all__ = ["FusedWGradAccumLinear", "enable_fused_wgrad_accumulation"]
