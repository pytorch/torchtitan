# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent MXFP8 inference weights, shared with compute at FSDP degree 1."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

from .._fsdp_tensor import (
    _FSDPTensorBase,
    _unsharded_inner_tensors,
    _UnshardedFSDPTensor,
)
from .linear import MXFP8Linear
from .tensor import (
    _LinearShardedTensorWithMXFP8Compute,
    _MXFP8_BLOCK_SIZE,
    _MXFP8LinearOperands,
    _quantize_mxfp8_weight,
)


def _swizzle_scale(scale):
    """Arrange row-major E8M0 bytes in the scaled_mm 32_4_4 layout."""
    rows, cols = scale.shape
    padded = F.pad(scale.view(torch.uint8), (0, -cols % 4, 0, -rows % 128))
    row_blocks, col_blocks = padded.shape[0] // 128, padded.shape[1] // 4
    return (
        padded.view(row_blocks, 4, 32, col_blocks, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(row_blocks, col_blocks, 32, 16)
        .view(torch.float8_e8m0fnu)
    )


def _unswizzle_scale(scale, rows, cols):
    row_blocks, col_blocks = (rows + 127) // 128, (cols + 3) // 4
    return (
        scale.view(torch.uint8)
        .view(row_blocks, col_blocks, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(row_blocks * 128, col_blocks * 4)[:rows, :cols]
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )


class _MXFP8StorageTensor(_FSDPTensorBase):
    """Frozen FP8 data and GEMM-ready scales shared at FSDP degree 1.

    Logical BF16 metadata preserves the parameter and weight-loading interface.
    The persistent buffers are the existing MXFP8 compute operands; full-weight
    views share them, and BF16 copies quantize into them at stable addresses.
    """

    @staticmethod
    def __new__(cls, qdata, scale_fprop, scale_dgrad, *, shape=None, stride=None):
        with torch.inference_mode(qdata.is_inference()):
            return _FSDPTensorBase.__new__(
                cls,
                qdata,
                _logical_dtype=torch.bfloat16,
                _logical_requires_grad=False,
                _logical_size=qdata.shape if shape is None else shape,
                _logical_stride=qdata.stride() if stride is None else stride,
            )

    def __init__(self, qdata, scale_fprop, scale_dgrad, *, shape=None, stride=None):
        self._qdata = qdata
        self._scale_fprop = scale_fprop
        self._scale_dgrad = scale_dgrad
        self.operands = _MXFP8LinearOperands(qdata, scale_fprop, scale_dgrad)

    @classmethod
    def from_bf16(cls, weight):
        if weight.dtype != torch.bfloat16:
            raise ValueError("MXFP8 inference requires BF16 source weights")
        if weight.is_meta:
            matrix = weight.flatten(0, -2)
            rows, cols = matrix.shape
            operands = _MXFP8LinearOperands(
                torch.empty_like(matrix, dtype=torch.float8_e4m3fn),
                _swizzle_scale(
                    torch.empty(
                        rows, cols // 32, device="meta", dtype=torch.float8_e8m0fnu
                    )
                ),
                _swizzle_scale(
                    torch.empty(
                        cols, rows // 32, device="meta", dtype=torch.float8_e8m0fnu
                    )
                ),
            )
        else:
            operands = _quantize_mxfp8_weight(weight.flatten(0, -2).contiguous())
        return cls(
            *_unsharded_inner_tensors(operands),
            shape=weight.shape,
            stride=weight.stride(),
        )

    def dequantize(self):
        rows, cols = self._qdata.shape
        scale = _unswizzle_scale(self._scale_fprop, rows, cols // 32)
        return (
            (self._qdata.float().unflatten(-1, (-1, 32)) * scale.float().unsqueeze(-1))
            .flatten(-2)
            .view(self.shape)
            .bfloat16()
        )

    def __tensor_flatten__(self):
        return ["_qdata", "_scale_fprop", "_scale_dgrad"], (
            tuple(self.shape),
            self.stride(),
        )

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        shape, stride = metadata
        return cls(
            inner_tensors["_qdata"],
            inner_tensors["_scale_fprop"],
            inner_tensors["_scale_dgrad"],
            shape=shape if outer_size is None else outer_size,
            stride=stride if outer_stride is None else outer_stride,
        )

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = dict(kwargs or {})
        weight = args[0]
        assert isinstance(weight, cls)
        if func == torch.ops.aten.copy_.default:
            source = args[1]
            operands = (
                source.operands
                if isinstance(source, cls)
                else _quantize_mxfp8_weight(source.flatten(0, -2).contiguous())
            )
            for target, value in zip(
                _unsharded_inner_tensors(weight.operands),
                _unsharded_inner_tensors(operands),
                strict=True,
            ):
                target.copy_(value)
            return weight

        if func in {
            torch.ops.aten.detach.default,
            torch.ops.aten.alias.default,
            torch.ops.aten.view.default,
            torch.ops.aten.as_strided.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.split.Tensor,
        }:
            meta_args = pytree.tree_map_only(
                cls,
                lambda t: torch.empty_strided(
                    t.shape, t.stride(), dtype=t.dtype, device="meta"
                ),
                args,
            )

            def wrap(view):
                if (
                    view.numel() != weight.numel()
                    or view.storage_offset() != 0
                    or not view.is_contiguous()
                ):
                    raise ValueError(
                        "MXFP8 compute storage views must preserve the complete "
                        "TP-local weight at FSDP degree 1"
                    )
                return cls(
                    *_unsharded_inner_tensors(weight.operands),
                    shape=view.shape,
                    stride=view.stride(),
                )

            result = pytree.tree_map_only(
                torch.Tensor, wrap, func(*meta_args, **kwargs)
            )
            return return_and_correct_aliasing(func, args, kwargs, result)

        if func not in {
            torch.ops.aten.empty_like.default,
            torch.ops.aten.new_zeros.default,
            torch.ops.aten.clone.default,
            torch.ops.aten._to_copy.default,
            torch.ops.aten._pin_memory.default,
        }:
            raise NotImplementedError(f"MXFP8 compute storage does not support {func}")
        if "dtype" in kwargs and kwargs.pop("dtype") != torch.bfloat16:
            raise ValueError("MXFP8 inference weights require logical BF16 dtype")
        shape = weight.shape
        if func == torch.ops.aten.new_zeros.default:
            shape = args[1]
            if torch.Size(shape).numel() != weight.numel():
                raise ValueError("MXFP8 compute storage requires the full local weight")
        tensors = [
            func(
                tensor,
                *(
                    (tensor.shape,)
                    if func == torch.ops.aten.new_zeros.default
                    else args[1:]
                ),
                **kwargs,
            )
            for tensor in _unsharded_inner_tensors(weight.operands)
        ]
        return cls(
            *tensors,
            shape=shape,
            stride=torch.empty(shape, device="meta").stride(),
        )

    def fsdp_get_unsharded_view(self, mesh, module, mp_policy):
        if mesh.size() != 1:
            raise ValueError("MXFP8 compute storage requires FSDP degree 1")
        if mp_policy.param_dtype not in (None, torch.bfloat16):
            raise ValueError("MXFP8 inference requires BF16 compute metadata")
        return _UnshardedFSDPTensor(
            self._qdata,
            self.operands,
            _logical_size=self.shape,
            _logical_stride=self.stride(),
            _logical_dtype=torch.bfloat16,
        )


def _bf16_weight(weight, *, empty=False):
    """Materialize a BF16 tensor with the parameter's existing DTensor placement."""
    local = weight.to_local() if isinstance(weight, DTensor) else weight
    assert isinstance(local, _MXFP8StorageTensor)
    value = (
        torch.empty(local.shape, dtype=torch.bfloat16, device=local.device)
        if empty
        else local.dequantize()
    )
    if isinstance(weight, DTensor):
        return DTensor.from_local(
            value,
            weight.device_mesh,
            weight.placements,
            shape=weight.shape,
            stride=weight.stride(),
        )
    return value


class MXFP8InferenceLinear(MXFP8Linear):
    """Install frozen FP8 storage after TP sharding and before fully_shard.

    The generator uses FSDP degree 1. Persistent storage contains the existing
    MXFP8 compute operands, so BF16 loads update captured buffers in place.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8Linear.Config):
        pass

    def _parallelize(self, parallel_dims) -> None:
        shard_degree = parallel_dims.dp_shard * parallel_dims.cp
        if shard_degree != 1:
            raise ValueError(
                f"MXFP8 inference storage requires FSDP degree 1, got {shard_degree}"
            )
        super()._parallelize(parallel_dims)
        weight = self.weight
        assert isinstance(weight, _LinearShardedTensorWithMXFP8Compute)
        if weight.shape[-2] % _MXFP8_BLOCK_SIZE or weight.shape[-1] % _MXFP8_BLOCK_SIZE:
            raise ValueError(
                "MXFP8 inference requires each TP-local weight to contain complete "
                f"32x32 tiles; got TP-local shape {tuple(weight.shape)}."
            )
        quantized = nn.Parameter(
            _MXFP8StorageTensor.from_bf16(weight._tensor), requires_grad=False
        )
        spmd.assert_type_like(quantized, weight)
        self.weight = quantized

    def _init_self_parameters(self) -> None:
        weight = self.weight
        local = weight.to_local() if isinstance(weight, DTensor) else weight
        if not isinstance(local, _MXFP8StorageTensor):
            super()._init_self_parameters()
            return
        self.weight = nn.Parameter(
            _bf16_weight(weight, empty=True), requires_grad=False
        )
        try:
            super()._init_self_parameters()
            weight.copy_(self.weight)
        finally:
            self.weight = weight

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # A normal BF16 DTensor lets checkpoint readers and TorchStore discover
        # the generator's destination mesh/placements without handling subclasses.
        local = (
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        if isinstance(local, _MXFP8StorageTensor):
            destination[prefix + "weight"] = _bf16_weight(self.weight)
