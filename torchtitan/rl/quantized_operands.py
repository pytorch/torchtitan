# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized parameters sent as the generators' operands
(``Generator.Config.receive_quantized_operands``): the trainer quantizes its rows with
the generators' format and sends ``_packed_operands`` as uint8 bytes, one key per
operand tensor, sharded on dim 0 like the parameter, e.g. ``w13.weight.qdata`` of shape
``(num_experts, bytes per expert)``. Generators copy their rows into their operands.
"""

import dataclasses
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

# TODO: TensorMeta and _StridedShard are private; DTensor has no public constructor
# that keeps a local tensor on another device than its mesh's.
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor

OperandFormat = tuple[type[_ShardedFSDPTensor], Any]
"""A quantized parameter's class and ``__tensor_flatten__`` context."""


def packed_operand_tensors(
    key: str, rows: dict[str, torch.Tensor], *, like: torch.Tensor
) -> dict[str, torch.Tensor]:
    """``rows`` of parameter ``key`` as bytes, sharded on dim 0 like this rank's part
    ``like`` of it: a DTensor ``like`` gives DTensors of global shape ``(like.shape[0],
    bytes per row)`` over ``rows``, on their device."""
    if isinstance(like, DTensor) and not all(
        isinstance(p, Replicate)
        or (isinstance(p, (Shard, _StridedShard)) and p.dim == 0)
        for p in like.placements
    ):
        raise ValueError(
            f"{key} is placed {like.placements}; sending its quantized operands "
            "needs every rank to hold whole rows of dim 0 (e.g. whole experts)"
        )
    tensors = {}
    for name, row_tensor in rows.items():
        row_bytes = row_tensor.view(torch.uint8)
        if isinstance(like, DTensor):
            shape = torch.Size((like.shape[0], row_bytes.shape[1]))
            meta = TensorMeta(shape, (shape[1], 1), torch.uint8)
            spec = dataclasses.replace(like._spec, tensor_meta=meta)
            row_bytes = DTensor(row_bytes, spec, requires_grad=False)
        tensors[f"{key}.{name}"] = row_bytes
    return tensors


@torch.no_grad()
def pack_quantized_operands(
    model_state_dict: dict[str, torch.Tensor],
    formats: dict[str, OperandFormat],
    *,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Replace each parameter of ``formats`` that this trainer rank holds with its
    packed operands, quantized from its rows cast to ``dtype`` (None: their own), as
    the generators would receive and quantize them."""
    packed = dict(model_state_dict)
    for key, (format_cls, metadata) in formats.items():
        if (tensor := packed.pop(key, None)) is None:
            continue  # held by another pipeline stage
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
