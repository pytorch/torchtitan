# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import spmd_types as spmd
import torch
from spmd_types import SpmdType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import AttentionMasksType

ContextParallelLoadBalancer = _LoadBalancer | None


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


def _create_context_parallel_load_balancer(
    cp_mesh: DeviceMesh,
    seq_len: int,
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    ptrr_mask_key: str | None = None,
) -> ContextParallelLoadBalancer:
    """Create the load balancer shared by input and metadata sharding."""
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
                # For FlexInnerAttention, we use _PTRRLoadBalancer.
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
    return load_balancer


def cp_shard_inputs(
    input_dict: dict[str, Any],
    input_shardings: dict[str, SpmdType] | None,
    cp_mesh: DeviceMesh,
    load_balancer_type: str | None = "headtail",
    ptrr_mask_key: str | None = None,
) -> tuple[dict[str, Any], ContextParallelLoadBalancer]:
    """Shard named tensors for Context Parallel.

    Each tensor named in ``shard_dims`` (resolved against ``input_dict``) is
    sharded along its declared sequence dimension using a single shared load
    balancer. Attention metadata is left unchanged for attention backends to
    shard. Position resolution (per-document vs sequential) is handled upstream
    (the model's ``preprocess_inputs`` / the trainer).

    Args:
        input_dict: Model-forward inputs keyed by name, containing 'input',
            'labels', and any extra kwargs. Tensor entries named in
            ``shard_dims`` (e.g. 'input', 'labels', 'positions') are sharded and
            written back; 'attention_masks', if present, is left unchanged.
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

    Returns:
        The same ``input_dict`` object with its named tensors sharded, together
        with the load balancer that attention backends must use for metadata.
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
        return input_dict, None

    buffers = tuple(named[n] for n in shard_names)
    seq_dims = tuple(shard_dims[n] for n in shard_names)
    load_balancer = _create_context_parallel_load_balancer(
        cp_mesh,
        buffers[0].size(seq_dims[0]),
        input_dict.get("attention_masks"),
        load_balancer_type,
        ptrr_mask_key,
    )
    sharded_buffers = cast(
        tuple[torch.Tensor, ...],
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=buffers,
            seq_dims=seq_dims,
            load_balancer=load_balancer,
        ),
    )
    for n, buf in zip(shard_names, sharded_buffers):
        input_dict[n] = buf
    return input_dict, load_balancer
