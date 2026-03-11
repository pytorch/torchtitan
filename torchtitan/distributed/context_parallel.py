# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _ContextParallel,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata
from torchtitan.tools.logging import logger


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
    attention_type: str,
) -> None:
    """
    Apply context parallelism to attention modules.

    CP splits the sequence dimension across devices to enable training with
    longer sequences. This function applies CP to the provided attention
    modules.

    Args:
        attention_modules: Sequence of attention modules to apply CP to
        cp_mesh: Device mesh for context parallel dimension
        attention_type: Type of attention mechanism. Must be one of:
            - "sdpa": scaled_dot_product_attention()
            - "flex": flex_attention()
            - "varlen": varlen_attn() with Magi Attention CP
    """
    # Apply context parallelism to every attention module
    # TODO: make seq_dim configurable once the implementation doesn't assume 2
    # internally.
    match attention_type:
        case "flex":
            cp_plan = _ContextParallel(
                seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX
            )
        case "sdpa":
            # Enable the DTensor dispatcher to route SDPA operations to the
            # Context Parallel implementation. This is required for CP to work
            # with SDPA (but not FlexAttention).
            # Note: Use _disable_context_parallel_dispatcher() if you need to
            # turn this off. In TorchTitan, we currently don't disable the CP
            # dispatcher.
            _enable_context_parallel_dispatcher()
            cp_plan = _ContextParallel(
                seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
            )
        case "varlen":
            # Varlen CP uses Magi Attention (arXiv:2505.13211) rather than
            # PyTorch's _ContextParallel (which doesn't support varlen).
            # Set cp_mesh on each VarlenAttentionWrapper so its forward
            # method can detect CP and use the Magi Attention path.
            for attention_module in attention_modules:
                attention_module._cp_mesh = cp_mesh
            logger.info("Applied Varlen Context Parallel (Magi Attention) to the model")
            return
        case _:
            raise ValueError(
                f"Invalid attention_type '{attention_type}'. "
                f"Must be one of: 'sdpa', 'flex', 'varlen'"
            )

    for attention_module in attention_modules:
        parallelize_module(
            module=attention_module,
            device_mesh=cp_mesh,
            parallelize_plan=cp_plan,
        )

    logger.info("Applied Context Parallel to the model")


def prepare_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
    load_balancer_type: str | None = "headtail",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Prepare inputs, labels, and attention masks for Context Parallel forward pass.

    This function prepares tensors for context parallel by:
    1. Creating position indices based on input sequence length
    2. Sharding inputs, labels, and positions across the CP mesh
    3. Sharding attention masks if present

    Args:
        inputs: Input tensor of shape [batch_size, seq_len]
        labels: Label tensor of shape [batch_size, seq_len]
        extra_kwargs: Dictionary that may contain 'attention_masks' to be sharded
        cp_mesh: Device mesh for context parallel dimension
        device: Device to create position tensor on
        load_balancer_type: Type of load balancer to use for sharding.
            Options: "headtail", "ptrr", or None. Defaults to "headtail".

    Returns:
        Tuple of (sharded_inputs, sharded_labels, updated_extra_kwargs) where:
            - sharded_inputs: Inputs sharded along sequence dimension
            - sharded_labels: Labels sharded along sequence dimension
            - updated_extra_kwargs: Dict with sharded 'positions' and optionally
              sharded 'attention_masks'
    """
    attention_masks = extra_kwargs.get("attention_masks", None)

    if isinstance(attention_masks, VarlenMetadata):
        # Varlen CP path: simple sequence sharding, keep global cu_seqlens.
        # The VarlenAttentionWrapper._forward_cp handles the ring attention.
        return _prepare_varlen_cp_input(
            inputs, labels, extra_kwargs, cp_mesh, device
        )

    # Standard CP path (SDPA/Flex)
    positions = torch.arange(
        0, inputs.shape[1], dtype=torch.int32, device=device
    ).expand(inputs.shape)
    (inputs, labels, positions), attention_masks = cp_shard(
        cp_mesh,
        (inputs, labels, positions),
        attention_masks,
        load_balancer_type,
    )
    extra_kwargs["positions"] = positions
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    return inputs, labels, extra_kwargs


def _prepare_varlen_cp_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Prepare inputs for varlen + context parallel.

    Shards inputs and labels along the sequence dimension. Keeps the
    global VarlenMetadata (cu_seqlens) intact — the VarlenAttentionWrapper
    with CP enabled will handle cross-rank attention via Magi Attention.
    """
    cp_rank = cp_mesh.get_local_rank()
    cp_world_size = cp_mesh.size(0)
    seq_len = inputs.shape[1]

    assert seq_len % cp_world_size == 0, (
        f"Sequence length {seq_len} must be divisible by CP world size "
        f"{cp_world_size} for varlen CP"
    )
    chunk_size = seq_len // cp_world_size

    chunk_start = cp_rank * chunk_size
    positions = torch.arange(
        chunk_start,
        chunk_start + chunk_size,
        dtype=torch.int32,
        device=device,
    ).expand(inputs.shape[0], -1)
    extra_kwargs["positions"] = positions

    inputs = inputs[:, chunk_start : chunk_start + chunk_size].contiguous()
    labels = labels[:, chunk_start : chunk_start + chunk_size].contiguous()

    return inputs, labels, extra_kwargs


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dim: int = 1,
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
        input_seq_dim: Sequence dimension index for sharding. Defaults to 1,
            which covers most use cases where tensors have shape
            [batch_size, seq_len]. Can be changed by passing a
            different value if your tensors use a different sequence
            dimension layout.

    Returns:
        Tuple of (sharded_inputs, attention_masks) where:
            - sharded_inputs: Tuple of input tensors sharded along the
              sequence dimension
            - attention_masks: Sharded attention masks (BlockMask or
              dict[str, BlockMask]) or None

    Raises:
        ValueError: If load_balancer_type is "ptrr" and attention_masks
            is None or a dict
    """
    seq_len = inputs[0].size(input_seq_dim)
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
                # _PTRRLoadBalancer requires attention_masks to be a BlockMask.
                # For dict[str, BlockMask], _PTRRLoadBalancer currently doesn't
                # support the case where there are multiple masks.
                if attention_masks is None or isinstance(attention_masks, dict):
                    raise ValueError(
                        "PTRRLoadBalancer requires attention_masks to be a "
                        "BlockMask, but got None or dict[str, BlockMask]"
                    )
                if not isinstance(attention_masks, BlockMask):
                    raise ValueError(
                        f"PTRRLoadBalancer requires attention_masks to be a "
                        f"BlockMask, but got {type(attention_masks)}"
                    )
                load_balancer = _PTRRLoadBalancer(attention_masks, cp_world_size)
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
            seq_dims=tuple(input_seq_dim for _ in inputs),
            load_balancer=load_balancer,
        ),
    )

    # BlockMask, has shape, [B, H, Q, KV], and we can only shard
    # on the Q seq dimension, not KV.
    MASK_Q_SEQ_DIM = 2
    if attention_masks is not None:
        assert isinstance(attention_masks, (BlockMask, dict[str, BlockMask]))
        masks = (
            [attention_masks]
            if isinstance(attention_masks, BlockMask)
            else list(attention_masks.values())
        )
        masks = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=masks,
            seq_dims=(MASK_Q_SEQ_DIM,) * len(masks),
            load_balancer=load_balancer,
        )
        attention_masks = cast(
            (BlockMask | dict[str, BlockMask]),
            masks[0]
            if isinstance(attention_masks, BlockMask)
            else {k: v for k, v in zip(attention_masks.keys(), masks)},
        )

    return inputs, attention_masks
