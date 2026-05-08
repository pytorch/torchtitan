# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 quantization using TransformerEngine.

This module provides compile-compatible NVFP4 (E2M1 4-bit) training for Blackwell GPUs.
It uses custom ops wrapping TE's general_gemm to avoid graph breaks from te.autocast.

The approach follows this design:
1. Define custom ops for forward/backward GEMMs that create quantizers internally
2. Pass quantization settings as primitive types to avoid opaque type issues
3. Implement TENVFP4Linear module using these custom ops
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torchtitan.components.quantization import (
    QuantizationConverter,
    QuantizedLinearConfig,
)
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


# ---------------------------------------------------------------------------
# TransformerEngine imports
# ---------------------------------------------------------------------------
try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.module.linear import general_gemm

    _TE_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "TE NVFP4 requires transformer_engine library. "
        "Install from: https://github.com/NVIDIA/TransformerEngine"
    ) from e


# ---------------------------------------------------------------------------
# Custom ops for torch.compile compatibility
# ---------------------------------------------------------------------------
def _create_input_quantizer(with_rht: bool) -> te.NVFP4Quantizer:
    """Create quantizer for input activations."""
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=with_rht,
        with_post_rht_amax=True,
        stochastic_rounding=False,
    )


def _create_weight_quantizer(with_2d_quantization: bool) -> te.NVFP4Quantizer:
    """Create quantizer for weights."""
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_2d_quantization=with_2d_quantization,
        stochastic_rounding=False,
    )


def _create_grad_output_quantizer(with_rht: bool, stochastic_rounding: bool) -> te.NVFP4Quantizer:
    """Create quantizer for gradient outputs."""
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=with_rht,
        with_post_rht_amax=True,
        stochastic_rounding=stochastic_rounding,
    )


@torch.library.custom_op("te_nvfp4::linear_forward", mutates_args=())
def tenvfp4_linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    with_rht: bool,
    with_2d_quantization: bool,
) -> torch.Tensor:
    """Forward pass: y = x @ W.T with NVFP4 quantization."""
    input_quantizer = _create_input_quantizer(with_rht)
    weight_quantizer = _create_weight_quantizer(with_2d_quantization)

    q_input = input_quantizer.quantize(input)
    q_weight = weight_quantizer.quantize(weight)
    result = list(general_gemm(q_weight, q_input, out_dtype=input.dtype, layout="TN"))[0]
    return result


@tenvfp4_linear_forward.register_fake
def tenvfp4_linear_forward_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    with_rht: bool,
    with_2d_quantization: bool,
) -> torch.Tensor:
    # input: (*, in_features), weight: (out_features, in_features)
    # output: (*, out_features)
    out_shape = input.shape[:-1] + (weight.shape[0],)
    return input.new_empty(out_shape)


@torch.library.custom_op("te_nvfp4::linear_backward", mutates_args=())
def tenvfp4_linear_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    with_rht: bool,
    with_2d_quantization: bool,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass: compute grad_input and grad_weight with NVFP4 quantization."""
    input_quantizer = _create_input_quantizer(with_rht)
    weight_quantizer = _create_weight_quantizer(with_2d_quantization)
    grad_output_quantizer = _create_grad_output_quantizer(with_rht, stochastic_rounding)

    # grad_input = grad_output @ weight
    q_grad_output = grad_output_quantizer.quantize(grad_output)
    q_weight_t = weight_quantizer.quantize(weight.T.contiguous())
    grad_input = list(general_gemm(q_weight_t, q_grad_output, out_dtype=grad_output.dtype, layout="TN"))[0]

    # grad_weight = grad_output.T @ input
    q_input = input_quantizer.quantize(input)
    grad_weight = list(general_gemm(q_input, q_grad_output, out_dtype=grad_output.dtype, layout="NT"))[0]

    return grad_input, grad_weight


@tenvfp4_linear_backward.register_fake
def tenvfp4_linear_backward_fake(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    with_rht: bool,
    with_2d_quantization: bool,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_input = input.new_empty(input.shape)
    grad_weight = weight.new_empty(weight.shape)
    return grad_input, grad_weight


class TENVFP4LinearFunction(torch.autograd.Function):
    """Autograd function for TE NVFP4 linear layer."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        with_rht: bool,
        with_2d_quantization: bool,
        stochastic_rounding: bool,
    ) -> torch.Tensor:
        # Flatten input to 2D for GEMM
        input_shape = input.shape
        input_2d = input.view(-1, input.shape[-1])

        output = tenvfp4_linear_forward(input_2d, weight, with_rht, with_2d_quantization)

        # Reshape output back
        output = output.view(*input_shape[:-1], weight.shape[0])

        if bias is not None:
            output = output + bias

        # Save for backward
        ctx.save_for_backward(input_2d, weight, bias)
        ctx.input_shape = input_shape
        ctx.with_rht = with_rht
        ctx.with_2d_quantization = with_2d_quantization
        ctx.stochastic_rounding = stochastic_rounding

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_2d, weight, bias = ctx.saved_tensors
        input_shape = ctx.input_shape

        # Flatten grad_output to 2D
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        grad_input_2d, grad_weight = tenvfp4_linear_backward(
            grad_output_2d,
            input_2d,
            weight,
            ctx.with_rht,
            ctx.with_2d_quantization,
            ctx.stochastic_rounding,
        )

        # Reshape grad_input back
        grad_input = grad_input_2d.view(*input_shape)

        # Bias gradient
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_input, grad_weight, grad_bias, None, None, None


class TENVFP4Linear(Module):
    """NVFP4 quantized linear layer using TransformerEngine.

    This module is compatible with torch.compile(fullgraph=True) because it uses
    custom ops instead of te.autocast context managers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizedLinearConfig):
        """Drop-in replacement for Linear.Config that builds TENVFP4Linear."""

        disable_rht: bool = False
        """Disable random Hadamard transform for activations."""

        disable_stochastic_rounding: bool = False
        """Disable stochastic rounding in backward pass."""

        disable_2d_quantization: bool = False
        """Disable 2D quantization for weights (use 1D instead)."""

    def __init__(self, config: Config):
        super().__init__()
        self.in_features = config.in_features
        self.out_features = config.out_features

        self.weight = nn.Parameter(torch.empty(config.out_features, config.in_features))
        if config.bias:
            self.bias = nn.Parameter(torch.empty(config.out_features))
        else:
            self.register_parameter("bias", None)

        self._param_init = config.param_init

        # Store quantization settings as bools for the custom ops
        self.with_rht = not config.disable_rht
        self.with_2d_quantization = not config.disable_2d_quantization
        self.stochastic_rounding = not config.disable_stochastic_rounding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TENVFP4LinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.with_rht,
            self.with_2d_quantization,
            self.stochastic_rounding,
        )

    def _init_self_parameters(self) -> None:
        """Initialize parameters using param_init dict if available."""
        for name, param in self.named_parameters(recurse=False):
            if self._param_init and name in self._param_init:
                self._param_init[name](param)
            else:
                # Default initialization
                if name == "weight":
                    nn.init.kaiming_uniform_(param, a=5**0.5)
                elif name == "bias":
                    nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------
class TENVFP4LinearConverter(QuantizationConverter):
    """Apply NVFP4 quantization to Linear modules using TransformerEngine."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        disable_rht: bool = False
        """Disable random Hadamard transform for activations."""

        disable_stochastic_rounding: bool = False
        """Disable stochastic rounding in backward pass."""

        disable_2d_quantization: bool = False
        """Disable 2D quantization for weights."""

        filter_fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to skip NVFP4 quantization.
        Modules with dimensions not divisible by 16 are always skipped.
        """

    def __init__(self, config: Config):
        self.config = config

        # NVFP4 requires SM100+ (Blackwell) for native support
        if not has_cuda_capability(10, 0):
            raise ValueError(
                "TE NVFP4 training requires SM100 or later (Blackwell GPUs)."
            )

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile is recommended for NVFP4 training performance. "
                "Enable with --compile.enable"
            )

        logger.info(
            f"TE NVFP4 training active: "
            f"rht={'disabled' if config.disable_rht else 'enabled'}, "
            f"stochastic_rounding={'disabled' if config.disable_stochastic_rounding else 'enabled'}, "
            f"2d_quant={'disabled' if config.disable_2d_quantization else 'enabled'}"
        )

    def convert(self, model_config) -> None:
        filter_fqns = self.config.filter_fqns

        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            # Skip modules with dimensions not divisible by 16 (NVFP4 alignment requirement)
            if config.in_features % 16 != 0 or config.out_features % 16 != 0:
                logger.debug(f"Skipping {fqn}: dimensions not divisible by 16")
                continue

            # Skip filtered FQNs
            if any(filter_fqn in fqn for filter_fqn in filter_fqns):
                logger.debug(f"Skipping {fqn}: in filter_fqns")
                continue

            new_config = TENVFP4Linear.Config(
                in_features=config.in_features,
                out_features=config.out_features,
                bias=config.bias,
                param_init=config.param_init,
                disable_rht=self.config.disable_rht,
                disable_stochastic_rounding=self.config.disable_stochastic_rounding,
                disable_2d_quantization=self.config.disable_2d_quantization,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to TENVFP4Linear")
