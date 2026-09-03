# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Hunks in this file are copied from upstream open PR 4322/4449/4450 (fegin's CP stack) to unblock running;
# pending rebase and reconcile.

from typing import Any, cast

import spmd_types as spmd
import torch
from spmd_types import SpmdType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import AttentionMasksType


def _cp_shard_dims(input_sharding: dict[str, SpmdType]) -> dict[str, int]:
    """Derive ``{name: seq_dim}`` for inputs whose CP mesh axis is a Shard.

    Inputs whose CP axis is Replicate/Partial (e.g. an image stream that is
    not sequence-sharded) are omitted and thus left untouched by CP.
    """
    dims: dict[str, int] = {}
    for name, layout in input_sharding.items():
        axis_type = _per_axis_types(layout).get(MeshAxisName.CP)
        if isinstance(axis_type, spmd.Shard):
            dims[name] = axis_type.dim
    return dims


def prepare_context_parallel_input(
    input_dict: dict[str, Any],
    input_shardings: dict[str, SpmdType] | None,
    cp_mesh: DeviceMesh,
    load_balancer_type: str | None = "headtail",
    ptrr_mask_key: str | None = None,
    *,
    shard_attention_mask: bool = True,
) -> dict[str, Any]:
    """Shard named tensors and attention masks for Context Parallel.

    Each tensor named in ``shard_dims`` (resolved against ``input_dict``) is
    sharded along its declared sequence dimension using a single shared load
    balancer. Attention masks (``BlockMask``) are sharded separately along their
    Q sequence dimension. Position resolution (per-document vs sequential) is
    handled upstream (the model's ``preprocess_inputs`` / the trainer).

    Args:
        input_dict: Model-forward inputs keyed by name, containing 'input',
            'labels', and any extra kwargs. Tensor entries named in
            ``shard_dims`` (e.g. 'input', 'labels', 'positions') are sharded and
            written back; 'attention_masks', if present, is sharded along its Q
            seq dim.
        input_shardings: Per-input SPMD layout; the CP sequence dim for each
            input is derived via ``_cp_shard_dims`` (inputs whose CP axis is
            Replicate/Partial are omitted and left untouched). When None,
            defaults to sharding ``{"input": 0, "labels": 0, "positions": 0}``
            (standard decoder inputs, for callers without a per-input layout).
        cp_mesh: Device mesh for the context parallel dimension.
        load_balancer_type: Type of load balancer to use for sharding.
            Options: "headtail", "ptrr", or None. Defaults to "headtail".
        ptrr_mask_key: When ``load_balancer_type`` is "ptrr" and the attention
            masks are a dict[str, BlockMask], selects which mask the
            PTRRLoadBalancer is built from. Ignored otherwise.
        shard_attention_mask: Whether to shard each mask's query dimension.

    Returns:
        The same ``input_dict`` object, mutated in place with its sharded tensor
        entries (e.g. 'input', 'labels', 'positions') and 'attention_masks'
        updated. When no named tensor is present to shard, it is returned
        unchanged.
    """
    if input_shardings is not None:
        shard_dims = _cp_shard_dims(input_shardings)
    else:
        shard_dims = {"input": 0, "labels": 0, "positions": 0}

    named: dict[str, torch.Tensor] = {
        k: v for k, v in input_dict.items() if isinstance(v, torch.Tensor)
    }

    shard_names = [n for n in shard_dims if n in named]
    if not shard_names:
        return input_dict
    buffers = tuple(named[n] for n in shard_names)
    seq_dims = tuple(shard_dims[n] for n in shard_names)

    attention_masks = input_dict.get("attention_masks", None)
    sharded_buffers, attention_masks = cp_shard(
        cp_mesh,
        buffers,
        attention_masks,
        load_balancer_type,
        input_seq_dims=seq_dims,
        ptrr_mask_key=ptrr_mask_key,
        shard_attention_mask=shard_attention_mask,
    )

    for n, buf in zip(shard_names, sharded_buffers):
        input_dict[n] = buf
    if attention_masks is not None:
        input_dict["attention_masks"] = attention_masks
    return input_dict


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dims: int | tuple[int, ...] = 0,
    ptrr_mask_key: str | None = None,
    *,
    shard_attention_mask: bool = True,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | None]:
    """
    Shard inputs and attention masks across the context parallel mesh.

    This function distributes input tensors across devices in the CP mesh
    along the sequence dimension, enabling efficient processing. It optionally
    uses a load balancer to handle uneven computation workload.

    Args:
        cp_mesh: Device mesh for context parallel dimension
        inputs: Tuple of input tensors to be sharded along the sequence
            dimension
        attention_masks: Attention masks to be sharded. Supports None,
            BlockMask, or dict[str, BlockMask]
        load_balancer_type: Type of load balancer to use. Options:
            - "headtail": Use HeadTailLoadBalancer (for SDPA)
            - "ptrr": Use PTRRLoadBalancer (for FlexAttention)
            - None: Disable load balancing
            Defaults to "headtail".
        input_seq_dims: Sequence dimension(s) for sharding. An int applies the
            same dim to every tensor in ``inputs``. Defaults to 0
            for folded text tensors with shape [num_tokens]. A tuple specifies a per-tensor
            sequence dim and must have the same length as ``inputs``.
        ptrr_mask_key: When ``load_balancer_type`` is "ptrr" and
            ``attention_masks`` is a dict[str, BlockMask], selects which mask in
            the dict the PTRRLoadBalancer is built from. The resulting balancer
            is used to shard every mask in the dict as well as the inputs.
            Required (must be a valid key) in that case; ignored otherwise.
        shard_attention_mask: Whether to shard each mask's query dimension.

    Returns:
        Tuple of (sharded_inputs, attention_masks) where:
            - sharded_inputs: Tuple of input tensors sharded along the
              sequence dimension
            - attention_masks: Sharded attention masks (BlockMask or
              dict[str, BlockMask]) or None

    Raises:
        ValueError: If load_balancer_type is "ptrr" and attention_masks
            is None, or is a dict and ``ptrr_mask_key`` is not a valid key
    """
    if isinstance(input_seq_dims, tuple):
        assert len(input_seq_dims) == len(inputs)
        seq_dims = input_seq_dims
    else:
        seq_dims = tuple(input_seq_dims for _ in inputs)
    seq_len = inputs[0].size(seq_dims[0])
    cp_world_size = cp_mesh.size(0)

    load_balancer = None
    if load_balancer_type:
        match load_balancer_type:
            case "headtail":
                # For SDPA, we use the _HeadTailLoadBalancer.
                load_balancer = _HeadTailLoadBalancer(
                    seq_len, cp_world_size, cp_mesh.device_type
                )
            case "ptrr":
                # For FlexAttention, we use _PTRRLoadBalancer.
                # _PTRRLoadBalancer is built from a single BlockMask. When the
                # attention masks are a dict[str, BlockMask], the caller must
                # specify which mask to build the balancer from via
                # ``ptrr_mask_key``; the resulting balancer is then used to
                # shard every mask in the dict as well as the inputs.
                if attention_masks is None:
                    raise ValueError(
                        "PTRRLoadBalancer requires attention_masks to be a "
                        "BlockMask or dict[str, BlockMask], but got None"
                    )
                if isinstance(attention_masks, dict):
                    if ptrr_mask_key is None:
                        raise ValueError(
                            "PTRRLoadBalancer received a dict[str, BlockMask] "
                            "but no mask key was specified. Set "
                            "--parallelism.context_parallel_ptrr_mask_key to "
                            f"one of: {sorted(attention_masks.keys())}"
                        )
                    if ptrr_mask_key not in attention_masks:
                        raise ValueError(
                            f"context_parallel_ptrr_mask_key '{ptrr_mask_key}' "
                            f"is not a key in attention_masks. Available keys: "
                            f"{sorted(attention_masks.keys())}"
                        )
                    ptrr_mask = attention_masks[ptrr_mask_key]
                else:
                    ptrr_mask = attention_masks
                if not isinstance(ptrr_mask, BlockMask):
                    raise ValueError(
                        f"PTRRLoadBalancer requires the mask to be a "
                        f"BlockMask, but got {type(ptrr_mask)}"
                    )
                load_balancer = _PTRRLoadBalancer(ptrr_mask, cp_world_size)
            case _:
                raise ValueError(
                    f"Invalid load_balancer_type '{load_balancer_type}'. "
                    f"Must be one of: 'headtail', 'ptrr', or None"
                )

    inputs = cast(
        tuple[torch.Tensor, ...],
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=inputs,
            seq_dims=seq_dims,
            load_balancer=load_balancer,
        ),
    )

    # BlockMask has shape [B, H, Q, KV]. Only Q can be sequence-sharded.
    MASK_Q_SEQ_DIM = 2
    if attention_masks is not None and shard_attention_mask:
        assert isinstance(attention_masks, (BlockMask, dict))
        masks: list[BlockMask] = []
        for mask in (
            [attention_masks]
            if isinstance(attention_masks, BlockMask)
            else attention_masks.values()
        ):
            if not isinstance(mask, BlockMask):
                raise ValueError(
                    "Context parallelism can only shard BlockMask attention "
                    f"masks, got {type(mask).__name__} in the mask dict."
                )
            masks.append(mask)
        sharded_masks = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=masks,
            seq_dims=(MASK_Q_SEQ_DIM,) * len(masks),
            load_balancer=load_balancer,
        )
        attention_masks = cast(
            (BlockMask | dict[str, BlockMask]),
            (
                sharded_masks[0]
                if isinstance(attention_masks, BlockMask)
                else {k: v for k, v in zip(attention_masks.keys(), sharded_masks)}
            ),
        )

    return inputs, attention_masks
