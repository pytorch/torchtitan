# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import AttentionMasksType


ContextParallelLoadBalancer = _LoadBalancer | None


def create_context_parallel_load_balancer(
    cp_mesh: DeviceMesh,
    seq_len: int,
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    ptrr_mask_key: str | None = None,
) -> ContextParallelLoadBalancer:
    """Create the load balancer shared by input and metadata sharding."""
    cp_world_size = cp_mesh.size(0)

    if load_balancer_type is None:
        return None
    if load_balancer_type == "headtail":
        return _HeadTailLoadBalancer(seq_len, cp_world_size, cp_mesh.device_type)
    if load_balancer_type != "ptrr":
        raise ValueError(
            f"Invalid load_balancer_type '{load_balancer_type}'. "
            "Must be one of: 'headtail', 'ptrr', or None"
        )

    if attention_masks is None:
        raise ValueError(
            "PTRRLoadBalancer requires attention_masks to be a "
            "BlockMask or dict[str, BlockMask], but got None"
        )
    if isinstance(attention_masks, Mapping):
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
            "PTRRLoadBalancer requires the mask to be a BlockMask, "
            f"but got {type(ptrr_mask).__name__}"
        )
    return _PTRRLoadBalancer(ptrr_mask, cp_world_size)


def shard_context_parallel_inputs(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    input_seq_dims: int | tuple[int, ...],
    load_balancer: ContextParallelLoadBalancer,
) -> tuple[torch.Tensor, ...]:
    """Shard input tensors along their sequence dimensions."""
    if isinstance(input_seq_dims, tuple):
        assert len(input_seq_dims) == len(inputs)
        seq_dims = input_seq_dims
    else:
        seq_dims = tuple(input_seq_dims for _ in inputs)
    return cast(
        tuple[torch.Tensor, ...],
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=inputs,
            seq_dims=seq_dims,
            load_balancer=load_balancer,
        ),
    )


def shard_context_parallel_attention_masks(
    cp_mesh: DeviceMesh,
    attention_masks: BlockMask | Mapping[str, BlockMask],
    load_balancer: ContextParallelLoadBalancer,
) -> BlockMask | dict[str, BlockMask]:
    """Shard BlockMask query dimensions using the input partition."""
    masks = (
        [attention_masks]
        if isinstance(attention_masks, BlockMask)
        else list(attention_masks.values())
    )
    mask_q_seq_dim = 2
    sharded_masks = cast(
        "tuple[BlockMask, ...]",
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=masks,
            seq_dims=(mask_q_seq_dim,) * len(masks),
            load_balancer=load_balancer,
        ),
    )
    if isinstance(attention_masks, BlockMask):
        return sharded_masks[0]
    return {key: mask for key, mask in zip(attention_masks, sharded_masks)}


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dims: int | tuple[int, ...] = 0,
    ptrr_mask_key: str | None = None,
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
            - "ptrr": Use PTRRLoadBalancer (for FlexInnerAttention)
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
    input_seq_dim = (
        input_seq_dims[0] if isinstance(input_seq_dims, tuple) else input_seq_dims
    )
    load_balancer = create_context_parallel_load_balancer(
        cp_mesh,
        inputs[0].size(input_seq_dim),
        attention_masks,
        load_balancer_type,
        ptrr_mask_key,
    )
    inputs = shard_context_parallel_inputs(
        cp_mesh, inputs, input_seq_dims, load_balancer
    )
    if attention_masks is not None:
        if isinstance(attention_masks, BlockMask):
            masks_to_shard: BlockMask | dict[str, BlockMask] = attention_masks
        elif isinstance(attention_masks, Mapping):
            mask_dict: dict[str, BlockMask] = {}
            for key, mask in attention_masks.items():
                if not isinstance(mask, BlockMask):
                    raise ValueError(
                        "Context parallelism can only shard BlockMask attention "
                        f"masks, got {type(mask).__name__} in the mask dict."
                    )
                mask_dict[key] = mask
            masks_to_shard = mask_dict
        else:
            raise ValueError(
                "Context parallelism can only shard BlockMask attention "
                f"masks, got {type(attention_masks).__name__}."
            )
        attention_masks = shard_context_parallel_attention_masks(
            cp_mesh, masks_to_shard, load_balancer
        )

    return inputs, attention_masks
