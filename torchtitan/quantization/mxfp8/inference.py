# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent MXFP8 inference weights with byte-preserving FSDP all-gather."""

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils import _pytree as pytree

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


def _scale_size(size):
    if size[-1] % _MXFP8_BLOCK_SIZE:
        raise ValueError("MXFP8 storage views must preserve 32-element scale groups")
    return (*size[:-1], size[-1] // _MXFP8_BLOCK_SIZE)


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
    """Frozen E4M3 data and E8M0 scales, with logical BF16 parameter metadata.

    Scales are stored row-major, repeated over each tile's 32 rows. This costs
    1/32 byte per weight and lets FSDP slice both tensors on the same row dim,
    including the matrix-row dim of stacked projections. Quantization only
    happens when copying a BF16 update; all-gather only transports bytes.
    """

    @staticmethod
    def __new__(cls, qdata, scale):
        # Views of normal parameters stay normal even inside inference mode.
        with torch.inference_mode(qdata.is_inference()):
            return _FSDPTensorBase.__new__(
                cls, qdata, _logical_dtype=torch.bfloat16, _logical_requires_grad=False
            )

    def __init__(self, qdata, scale):
        self._qdata = qdata
        self._scale = scale

    @classmethod
    def from_bf16(cls, weight):
        if weight.dtype != torch.bfloat16:
            raise ValueError("MXFP8 inference storage requires BF16 source weights")
        qdata = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
        scale = torch.empty(
            _scale_size(weight.shape), device=weight.device, dtype=torch.float8_e8m0fnu
        )
        result = cls(qdata, scale)
        if not weight.is_meta:
            result.copy_(weight)
        return result

    def dequantize(self):
        return (
            (
                self._qdata.float().unflatten(-1, (-1, _MXFP8_BLOCK_SIZE))
                * self._scale.float().unsqueeze(-1)
            )
            .flatten(-2)
            .bfloat16()
        )

    def __tensor_flatten__(self):
        return ["_qdata", "_scale"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(inner_tensors["_qdata"], inner_tensors["_scale"])

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = dict(kwargs or {})
        weight = args[0]
        assert isinstance(weight, cls)
        if func == torch.ops.aten.copy_.default:
            source = args[1]
            if isinstance(source, cls):
                weight._qdata.copy_(source._qdata)
                weight._scale.copy_(source._scale)
            else:
                operands = _quantize_mxfp8_weight(source.flatten(0, -2).contiguous())
                weight._qdata.copy_(operands.weight_qdata_dgrad_NK.view(weight.shape))
                scale = _unswizzle_scale(
                    operands.weight_scale_fprop_swizzled,
                    source.numel() // source.shape[-1],
                    source.shape[-1] // _MXFP8_BLOCK_SIZE,
                )
                weight._scale.copy_(scale.view(weight._scale.shape))
            return weight

        qdata_args = pytree.tree_map_only(cls, lambda t: t._qdata, args)
        scale_args: list[Any] = list(
            pytree.tree_map_only(cls, lambda t: t._scale, args)
        )
        if func == torch.ops.aten.view.default:
            # Resolve -1 against the logical data shape before scaling it.
            shape = torch.empty(weight.shape, device="meta").view(args[1]).shape
            scale_args[1] = _scale_size(shape)
        elif func == torch.ops.aten.new_zeros.default:
            scale_args[1] = _scale_size(args[1])
        elif func == torch.ops.aten.slice.Tensor:
            dim = args[1] % weight.ndim
            if dim == weight.ndim - 1:
                start, end = args[2], min(args[3], weight.shape[dim])
                step = args[4] if len(args) > 4 else 1
                if start % 32 or end % 32 or step != 1:
                    raise ValueError(
                        "MXFP8 slices must preserve 32-element scale groups"
                    )
                scale_args[2:4] = [start // 32, end // 32]
        elif func == torch.ops.aten.split.Tensor:
            dim = args[2] if len(args) > 2 else 0
            if dim % weight.ndim == weight.ndim - 1:
                if args[1] % 32:
                    raise ValueError(
                        "MXFP8 splits must preserve 32-element scale groups"
                    )
                scale_args[1] //= 32
        elif func == torch.ops.aten.as_strided.default:
            size, stride = args[1:3]
            offset = args[3] if len(args) > 3 else weight.storage_offset()
            if stride[-1] != 1 or offset % 32 or any(s % 32 for s in stride[:-1]):
                raise ValueError("MXFP8 strided views must preserve scale groups")
            scale_args = [
                weight._scale,
                _scale_size(size),
                (*[s // 32 for s in stride[:-1]], 1),
                offset // 32,
            ]
        elif func not in {
            torch.ops.aten.detach.default,
            torch.ops.aten.alias.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.empty_like.default,
            torch.ops.aten._to_copy.default,
            torch.ops.aten._pin_memory.default,
        }:
            raise NotImplementedError(
                f"MXFP8 inference storage does not support {func}"
            )

        if "dtype" in kwargs and kwargs.pop("dtype") != torch.bfloat16:
            raise ValueError("MXFP8 inference weights require logical BF16 dtype")
        qdata = func(*qdata_args, **kwargs)
        scale = func(*scale_args, **kwargs)
        if isinstance(qdata, (list, tuple)):
            return type(qdata)(cls(q, s) for q, s in zip(qdata, scale, strict=True))
        return cls(qdata, scale)

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self):
        return True

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        sharded_dims = [
            dim
            for dim, (local, full) in enumerate(
                zip(self.shape, outer_size, strict=True)
            )
            if local != full
        ]
        if len(sharded_dims) > 1:
            raise ValueError(
                "MXFP8 FSDP storage supports sharding one tensor dimension"
            )
        shard_dim = sharded_dims[0] if sharded_dims else 0
        if shard_dim == len(outer_size) - 1 or outer_size[shard_dim] % mesh.size():
            raise ValueError(
                "MXFP8 FSDP requires even sharding on a non-contraction dimension"
            )
        if self.shape[-2] % 32 or self.shape[-1] % 32:
            raise ValueError(
                "MXFP8 FSDP shards must contain complete 32x32 weight tiles"
            )
        if mp_policy.param_dtype not in (None, torch.bfloat16):
            raise ValueError("MXFP8 inference requires BF16 compute metadata")
        # A single payload also works with FSDP's world-size-1 fast path.
        payload = torch.cat(
            [
                t.movedim(shard_dim, 0).contiguous().view(torch.uint8).flatten()
                for t in (self._qdata, self._scale)
            ]
        )
        return (payload,), (tuple(outer_size), shard_dim)

    def fsdp_post_all_gather(
        self, all_gather_outputs, metadata, param_dtype, *, out=None
    ):
        logical_size, shard_dim = metadata
        (payload,) = all_gather_outputs
        sizes = [self._qdata.numel(), self._scale.numel()]
        packed = payload.view(-1, sum(sizes))
        restored = []
        for tensor, size in zip(
            packed.split(sizes, dim=1),
            (logical_size, _scale_size(logical_size)),
            strict=True,
        ):
            comm_size = list(size)
            comm_size.insert(0, comm_size.pop(shard_dim))
            restored.append(
                tensor.reshape(comm_size).movedim(0, shard_dim).contiguous()
            )
        qdata, scale = restored
        qdata = qdata.view(torch.float8_e4m3fn)
        scale = scale.view(torch.float8_e8m0fnu).flatten(0, -2)
        scale_transposed = (
            scale.view(torch.uint8)[::32].t().repeat_interleave(32, dim=0)
        )
        operands = _MXFP8LinearOperands(
            weight_qdata_dgrad_NK=qdata.flatten(0, -2).clone(),
            weight_scale_fprop_swizzled=_swizzle_scale(scale),
            weight_scale_dgrad_swizzled=_swizzle_scale(scale_transposed),
        )
        if out is None:
            return (
                _UnshardedFSDPTensor(qdata, operands, _logical_dtype=torch.bfloat16),
                _unsharded_inner_tensors(operands),
            )
        for target, source in zip(
            _unsharded_inner_tensors(out.operands),
            _unsharded_inner_tensors(operands),
            strict=True,
        ):
            target.copy_(source)


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

    BF16 checkpoint and TorchStore updates are temporary. Copying them into
    the parameter quantizes each complete local tile before the next unshard.
    The ordinary MXFP8 forward consumes the existing FSDP operand container.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8Linear.Config):
        pass

    def _parallelize(self, parallel_dims) -> None:
        super()._parallelize(parallel_dims)
        weight = self.weight
        assert isinstance(weight, _LinearShardedTensorWithMXFP8Compute)
        shard_degree = parallel_dims.dp_shard * parallel_dims.cp
        if weight.shape[-2] % (32 * shard_degree) or weight.shape[-1] % 32:
            raise ValueError(
                "MXFP8 inference requires each TP/FSDP shard to contain complete "
                f"32x32 tiles; got TP-local shape {tuple(weight.shape)} and "
                f"FSDP degree {shard_degree}."
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
