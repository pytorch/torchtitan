# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 weight representation for FlexShard all-gather output."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from pytorch.flex_shard.custom_placements.fp8_bucketed_block_shard import (
    _pad_2d_to_block_shape,
    _scale_shape,
    _validate_2d_non_empty,
    _validate_block_size,
)


def blockwise_dequant_weight(
    quant: torch.Tensor,
    recip_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize a square blockwise FP8 weight."""
    _validate_block_size(block_size)
    m, n = _validate_2d_non_empty(quant.shape, "blockwise_dequant_weight")
    expected_scale = _scale_shape(quant.shape, block_size)
    if tuple(recip_scale.shape) != expected_scale:
        raise ValueError(
            f"scale shape must be {expected_scale} for data {tuple(quant.shape)} "
            f"and block_size {block_size}, got {tuple(recip_scale.shape)}"
        )
    padded_quant = _pad_2d_to_block_shape(quant, block_size)
    padded_m, padded_n = padded_quant.shape
    scale_m, scale_n = expected_scale
    tiles = (
        padded_quant.reshape(scale_m, block_size, scale_n, block_size)
        .permute(0, 2, 1, 3)
        .reshape(-1, block_size * block_size)
        .to(torch.float32)
    )
    deq = tiles * recip_scale.reshape(-1, 1).to(torch.float32)
    deq = (
        deq.reshape(scale_m, scale_n, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(padded_m, padded_n)
    )
    return deq[:m, :n]


class BlockwiseFp8Weight(torch.Tensor):
    """Blockwise FP8 weight returned by the FlexShard all-gather."""

    __slots__ = ["_data", "_recip_scale", "_block_size", "_orig_dtype"]

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        orig_dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = False,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            device=data.device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        orig_dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = False,
    ) -> None:
        _ = requires_grad
        if data.ndim != 2:
            raise ValueError(f"BlockwiseFp8Weight expects 2D data, got {data.shape}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        expected_scale = _scale_shape(data.shape, block_size)
        if tuple(recip_scale.shape) != expected_scale:
            raise ValueError(
                f"scale shape must be {expected_scale} for data {tuple(data.shape)} "
                f"and block_size {block_size}, got {tuple(recip_scale.shape)}"
            )
        if data.device != recip_scale.device:
            raise ValueError(
                f"data and scale must be on the same device, got {data.device} and "
                f"{recip_scale.device}"
            )
        self._data = data
        self._recip_scale = recip_scale
        self._block_size = block_size
        self._orig_dtype = orig_dtype

    @property
    def fp8_data(self) -> torch.Tensor:
        return self._data

    @property
    def recip_scale(self) -> torch.Tensor:
        return self._recip_scale

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def orig_dtype(self) -> torch.dtype:
        return self._orig_dtype

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = blockwise_dequant_weight(
            self._data,
            self._recip_scale,
            self._block_size,
        )
        return result.to(dtype or self._orig_dtype)

    def __tensor_flatten__(self) -> tuple[list[str], dict[str, Any]]:
        return ["_data", "_recip_scale"], {
            "block_size": self._block_size,
            "orig_dtype": self._orig_dtype,
            "requires_grad": self.requires_grad,
        }

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "BlockwiseFp8Weight":
        _ = outer_size, outer_stride
        return BlockwiseFp8Weight(
            inner_tensors["_data"],
            inner_tensors["_recip_scale"],
            metadata["block_size"],
            metadata["orig_dtype"],
            requires_grad=metadata["requires_grad"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.alias.default:
            tensor = args[0]
            # Autograd attaches differentiable-view metadata after dispatch returns.
            result = BlockwiseFp8Weight(
                func(tensor._data),
                func(tensor._recip_scale),
                tensor._block_size,
                tensor._orig_dtype,
                requires_grad=False,
            )
            return return_and_correct_aliasing(func, args, kwargs or {}, result)

        if func == torch.ops.aten.detach.default:
            tensor = args[0]
            return BlockwiseFp8Weight(
                tensor._data.detach(),
                tensor._recip_scale.detach(),
                tensor._block_size,
                tensor._orig_dtype,
                requires_grad=False,
            )

        raise NotImplementedError(
            f"{cls.__name__} does not support {func}. "
            "Call dequantize() explicitly before applying this operation."
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


__all__ = [
    "BlockwiseFp8Weight",
    "blockwise_dequant_weight",
]
