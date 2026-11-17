# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized operands in the model state dict the trainer sends the generators.

With ``Generator.Config.receive_quantized_operands``, a generator receives each
parameter it holds quantized (e.g. MXFP8 Dist-MoE experts) as the operands its
forward reads, quantized by the trainer, instead of in ``precision.param_dtype``.
The generator reports each such parameter's format; the trainer quantizes its rows
of the parameter with it and sends the format's packed rows
(``_ShardedFSDPTensor._packed_operands``) as bytes, one key per operand tensor,
sharded on dim 0 like the parameter. E.g. MXFP8 W13 of 128 experts, trainer EP 4:

    trainer rank 1: w13.weight (128, 2, F, D), experts 32:64
        -> w13.weight.qdata        uint8 (128, 2FD),       rows 32:64
           w13.weight.fprop_scale  uint8 (128, 2FD / 32),  rows 32:64
    generator rank: copies its rows of each into its operands, in place

The generator gets the bytes its own unshard would have quantized, so the operands
are bitwise the same, and the transfer moves fp8 instead of bf16.
"""

import dataclasses
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

# TODO: _dtensor_spec and _StridedShard are private upstream APIs; DTensor has no
# public constructor that keeps a local tensor on another device than the mesh's.
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor

OperandFormat = tuple[type[_ShardedFSDPTensor], Any]
"""A quantized parameter's format: its class and its ``__tensor_flatten__`` context,
from which ``__tensor_unflatten__`` rebuilds it around another tensor."""


def operand_format(tensor: _ShardedFSDPTensor) -> OperandFormat:
    """The format of the quantized parameter ``tensor``."""
    return type(tensor), tensor.__tensor_flatten__()[1]


def packed_operand_tensors(
    key: str, rows: dict[str, torch.Tensor], *, like: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Name parameter ``key``'s packed operand ``rows`` and lay them out as bytes,
    sharded on dim 0 like this rank's part ``like`` of the parameter.

    With a DTensor ``like``, each tensor is a DTensor of global shape
    ``(like.shape[0], bytes per row)`` whose local tensor views the row bytes, on
    whatever device they are.

    Raises:
        ValueError: if ``like`` is sharded on a dim other than 0: a rank must hold
            whole rows, e.g. whole experts.
    """
    if isinstance(like, DTensor) and not all(
        isinstance(p, Replicate)
        or (isinstance(p, (Shard, _StridedShard)) and p.dim == 0)
        for p in like.placements
    ):
        raise ValueError(
            f"{key} is placed {like.placements}; sending its quantized operands "
            "needs every rank to hold whole rows of dim 0 (e.g. whole experts)"
        )
    local = like.to_local() if isinstance(like, DTensor) else like
    tensors = {}
    for name, row_tensor in rows.items():
        assert row_tensor.shape[0] == local.shape[0], (key, name, row_tensor.shape)
        row_bytes = row_tensor.view(torch.uint8)
        if isinstance(like, DTensor):
            shape = torch.Size((like.shape[0], row_bytes.shape[1]))
            spec = dataclasses.replace(
                like._spec,
                tensor_meta=TensorMeta(shape, (shape[1], 1), torch.uint8),
            )
            row_bytes = DTensor(row_bytes, spec, requires_grad=False)
        tensors[f"{key}.{name}"] = row_bytes
    return tensors


def pack_quantized_operands(
    model_state_dict: dict[str, torch.Tensor],
    formats: dict[str, OperandFormat],
    *,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Replace each parameter of ``formats`` in a trainer rank's ``model_state_dict``
    with the packed rows of its operands, quantized with its format from this rank's
    rows cast to ``dtype`` (None keeps theirs): the rows the generator would receive
    and quantize.

    Under PP a rank holds only its stages' parameters; it skips the others.
    """
    packed = dict(model_state_dict)
    with torch.no_grad():
        for key, (format_cls, metadata) in formats.items():
            tensor = packed.pop(key, None)
            if tensor is None:
                continue
            local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            if isinstance(local, _ShardedFSDPTensor):
                local = local._tensor
            logical = local.to(dtype or local.dtype).contiguous()
            quantizer = format_cls.__tensor_unflatten__(
                {"_tensor": logical}, metadata, logical.shape, logical.stride()
            )
            rows = quantizer._packed_operands(quantizer._build_operands(logical))
            packed.update(packed_operand_tensors(key, rows, like=tensor))
    return packed
