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

from torchtitan.protocols.model import AttentionMasksType
from torchtitan.tools.logging import logger


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
    use_flex_attn: bool,
) -> None:
    """
    Apply context parallelism to attention modules.

    Context Parallelism (CP) splits the sequence dimension across devices to enable
    training with longer sequences. This function applies CP to the provided attention
    modules.

    Args:
        attention_modules: Sequence of attention modules to apply CP to
        cp_mesh: Device mesh for context parallel dimension
        use_flex_attn: Whether to use FlexAttention (True) or SDPA (False)

    Note:
        - For FlexAttention: CP plan uses FLEX attention type
        - For SDPA: Enables CP dispatcher and uses SDPA attention type
    """
    # Apply context parallelism to every attention module
    # TODO: make seq_dim configurable once the implementation doesn't assume 2
    # internally.
    if use_flex_attn:
        cp_plan = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX
        )
    else:
        # Enable the DTensor dispatcher to route SDPA operations to the Context Parallel
        # implementation. This is required for CP to work with SDPA (but not FlexAttention).
        # Note: Use _disable_context_parallel_dispatcher() if you need to turn this off.
        #       In TorchTitan, we currently don't disable the CP dispatcher.
        _enable_context_parallel_dispatcher()
        cp_plan = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Prepare inputs, labels, and attention masks for Context Parallel forward pass.

    This function prepares tensors for Context Parallel by:
    1. Creating position indices based on input sequence length
    2. Sharding inputs, labels, and positions across the CP mesh
    3. Sharding attention masks if present

    Args:
        inputs: Input tensor of shape [batch_size, seq_len]
        labels: Label tensor of shape [batch_size, seq_len]
        extra_kwargs: Dictionary that may contain 'attention_masks' to be sharded
        cp_mesh: Device mesh for context parallel dimension
        device: Device to create position tensor on

    Returns:
        Tuple of (sharded_inputs, sharded_labels, updated_extra_kwargs) where:
            - sharded_inputs: Inputs sharded along sequence dimension
            - sharded_labels: Labels sharded along sequence dimension
            - updated_extra_kwargs: Dict with sharded 'positions' and optionally
              sharded 'attention_masks'
    """
    attention_masks = extra_kwargs.get("attention_masks", None)
    positions = torch.arange(
        0, inputs.shape[1], dtype=torch.int32, device=device
    ).expand(inputs.shape)
    (inputs, labels, positions), attention_masks = cp_shard(
        cp_mesh,
        (inputs, labels, positions),
        attention_masks,
    )
    extra_kwargs["positions"] = positions
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    return inputs, labels, extra_kwargs


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    disable_load_balancer: bool = False,
):

    INPUT_SEQ_DIM = 1
    seq_len = inputs[0].size(INPUT_SEQ_DIM)
    cp_world_size = cp_mesh.size(0)

    load_balancer = None
    if not disable_load_balancer:
        if isinstance(attention_masks, BlockMask):
            load_balancer = _PTRRLoadBalancer(attention_masks, cp_world_size)
        elif attention_masks is None or isinstance(attention_masks, dict):
            # For SDPA, we use the _HeadTailLoadBalancer.
            # TODO: For dict[str, BlockMask], _PTRRLoadBalancer currently doesn't
            # support the case where there are multiple masks. To address multiple
            # masks usage, _PTRRLoadBalancer also needs to take into account the
            # usage frequency of each mask. So we default to _HeadTailLoadBalancer
            # as we have not implemented this feature.
            load_balancer = _HeadTailLoadBalancer(
                seq_len, cp_world_size, cp_mesh.device_type
            )
        else:
            ValueError(
                "cp_shard only support attention_masks is "
                "None, BlockMask, dict[str, BlockMask]"
            )

    inputs = cast(
        tuple[torch.Tensor, ...],
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=inputs,
            seq_dims=tuple(1 for _ in inputs),
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
