# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pluggable MXFP8 weight quantization strategies.

Tensor shape suffixes:
    N: output features
    K: input features

The practical default uses square 32x32 scale tiles for weights and keeps
activation operands on the standard one-dimensional MXFP8 scale tiles. Square
weight tiles are orientation-symmetric: FPROP and DGRAD can share one qdata
allocation instead of selecting two independently quantized
representations.
Forward activation scales are computed within each token row, so scale
calculation does not mix information between causal positions.

Different weight numerics remain a module-level choice. A strategy produces
the four GEMM-ready weight operands and declares which tensors need independent
FSDP-managed storage. The only built-in strategy uses square weight tiles;
additional strategies can be registered by users.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import torch

from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_32x32_swizzle_dim0_and_dim1,
)


_MXFP8_WEIGHT_TILE_SIZE = 32


@dataclass(frozen=True)
class MXFP8WeightOperands:
    """The weight qdata and blocked scales consumed by FPROP and DGRAD."""

    q_weight_fprop_KN: torch.Tensor  # noqa: N815
    s_weight_fprop_blocked: torch.Tensor
    q_weight_dgrad_NK: torch.Tensor  # noqa: N815
    s_weight_dgrad_blocked: torch.Tensor


_ALL_MXFP8_WEIGHT_OPERAND_NAMES = tuple(
    field.name for field in fields(MXFP8WeightOperands)
)


class MXFP8WeightQuantizationStrategy(ABC):
    """Convert one BF16 logical weight into GEMM-ready MXFP8 operands.

    Every strategy must return FPROP and DGRAD qdata/scale pairs that can be
    passed directly as the B operand of ``torch.nn.functional.scaled_mm`` with
    ``ScalingType.BlockWise1x32`` and ``SwizzleType.SWIZZLE_32_4_4``. A strategy
    may use a different logical scale tile, such as 32x32, but it must expand
    and swizzle its scales into that hardware layout before returning them.
    Other scale recipes or layouts are incompatible with ``MXFP8Linear``.

    By default, FSDP manages all four operands as independent tensors. A
    strategy whose operands share storage must override
    ``fsdp_managed_tensor_names`` and
    ``reconstruct_operands_from_inner_tensors`` to describe that aliasing.
    """

    name: str
    fsdp_managed_tensor_names = _ALL_MXFP8_WEIGHT_OPERAND_NAMES

    @abstractmethod
    def quantize(self, weight_NK: torch.Tensor) -> MXFP8WeightOperands:
        raise NotImplementedError

    def reconstruct_operands_from_inner_tensors(
        self,
        inner_tensors: dict[str, torch.Tensor],
    ) -> MXFP8WeightOperands:
        """Reconstruct all operands from the independent FSDP inner tensors."""
        return MXFP8WeightOperands(
            q_weight_fprop_KN=inner_tensors["q_weight_fprop_KN"],
            s_weight_fprop_blocked=inner_tensors["s_weight_fprop_blocked"],
            q_weight_dgrad_NK=inner_tensors["q_weight_dgrad_NK"],
            s_weight_dgrad_blocked=inner_tensors["s_weight_dgrad_blocked"],
        )

    def fsdp_managed_tensors(
        self,
        operands: MXFP8WeightOperands,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(operands, name) for name in self.fsdp_managed_tensor_names)


def _validate_weight(weight_NK: torch.Tensor, strategy_name: str) -> None:
    if weight_NK.ndim != 2:
        raise ValueError(
            f"{strategy_name} MXFP8 weight quantization requires a 2D weight, "
            f"got {weight_NK.ndim} dimensions."
        )
    if weight_NK.dtype != torch.bfloat16:
        raise ValueError(
            f"{strategy_name} MXFP8 weight quantization requires BF16 weights, "
            f"got {weight_NK.dtype}."
        )
    if any(size % _MXFP8_WEIGHT_TILE_SIZE for size in weight_NK.shape):
        raise ValueError(
            f"{strategy_name} MXFP8 weight quantization requires both matrix "
            f"dimensions divisible by {_MXFP8_WEIGHT_TILE_SIZE}, got "
            f"{tuple(weight_NK.shape)}."
        )


class MXFP8WeightQuantization32x32(MXFP8WeightQuantizationStrategy):
    """Square weight scales with one shared qdata allocation."""

    name = "32x32"
    fsdp_managed_tensor_names = (
        "q_weight_fprop_KN",
        "s_weight_fprop_blocked",
        "s_weight_dgrad_blocked",
    )

    def quantize(self, weight_NK: torch.Tensor) -> MXFP8WeightOperands:
        _validate_weight(weight_NK, self.name)
        (
            q_weight_dgrad_NK,
            s_weight_fprop_blocked,
            _,
            s_weight_dgrad_blocked,
        ) = triton_to_mxfp8_32x32_swizzle_dim0_and_dim1(weight_NK.contiguous())
        # The fused kernel also materializes the transposed qdata for an
        # optimized DGRAD layout. Square quantization makes it an exact
        # transpose, so retain one allocation and use views in both GEMMs.
        q_weight_fprop_KN = q_weight_dgrad_NK.t()
        return MXFP8WeightOperands(
            q_weight_fprop_KN=q_weight_fprop_KN,
            s_weight_fprop_blocked=s_weight_fprop_blocked,
            q_weight_dgrad_NK=q_weight_dgrad_NK,
            s_weight_dgrad_blocked=s_weight_dgrad_blocked,
        )

    def reconstruct_operands_from_inner_tensors(
        self,
        inner_tensors: dict[str, torch.Tensor],
    ) -> MXFP8WeightOperands:
        q_weight_fprop_KN = inner_tensors["q_weight_fprop_KN"]
        return MXFP8WeightOperands(
            q_weight_fprop_KN=q_weight_fprop_KN,
            s_weight_fprop_blocked=inner_tensors["s_weight_fprop_blocked"],
            q_weight_dgrad_NK=q_weight_fprop_KN.t(),
            s_weight_dgrad_blocked=inner_tensors["s_weight_dgrad_blocked"],
        )


_WEIGHT_QUANTIZATION_STRATEGIES: dict[str, MXFP8WeightQuantizationStrategy] = {}


def register_mxfp8_weight_quantization_strategy(
    strategy: MXFP8WeightQuantizationStrategy,
    *,
    replace: bool = False,
) -> None:
    """Register a strategy name usable by ``MXFP8Linear.Config``.

    Custom strategies must be registered on every rank before model
    construction and before restoring a checkpoint that references the name.
    """
    if not strategy.name:
        raise ValueError("MXFP8 weight quantization strategy name cannot be empty")
    if strategy.name in _WEIGHT_QUANTIZATION_STRATEGIES and not replace:
        raise ValueError(
            f"MXFP8 weight quantization strategy {strategy.name!r} is already "
            "registered"
        )
    valid_names = {field.name for field in fields(MXFP8WeightOperands)}
    if not strategy.fsdp_managed_tensor_names or not set(
        strategy.fsdp_managed_tensor_names
    ).issubset(valid_names):
        raise ValueError(
            f"MXFP8 weight quantization strategy {strategy.name!r} has invalid "
            "FSDP-managed tensor names "
            f"{strategy.fsdp_managed_tensor_names}."
        )
    _WEIGHT_QUANTIZATION_STRATEGIES[strategy.name] = strategy


def get_mxfp8_weight_quantization_strategy(
    name: str,
) -> MXFP8WeightQuantizationStrategy:
    try:
        return _WEIGHT_QUANTIZATION_STRATEGIES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_WEIGHT_QUANTIZATION_STRATEGIES))
        raise ValueError(
            f"Unknown MXFP8 weight quantization strategy {name!r}; "
            f"available strategies: {choices}."
        ) from error


def quantize_mxfp8_weight(
    weight_NK: torch.Tensor,
    strategy_name: str,
) -> MXFP8WeightOperands:
    return get_mxfp8_weight_quantization_strategy(strategy_name).quantize(weight_NK)


register_mxfp8_weight_quantization_strategy(MXFP8WeightQuantization32x32())


__all__ = [
    "get_mxfp8_weight_quantization_strategy",
    "MXFP8WeightOperands",
    "MXFP8WeightQuantization32x32",
    "MXFP8WeightQuantizationStrategy",
    "quantize_mxfp8_weight",
    "register_mxfp8_weight_quantization_strategy",
]
