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
)
from torch.distributed.tensor.parallel import parallelize_module

from torchtitan.protocols.model import AttentionMasksType
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
            - "varlen": varlen_attn() (not yet implemented)

    Raises:
        NotImplementedError: If attention_type is "varlen"
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
            raise NotImplementedError(
                "Variable-length attention CP is not yet supported"
            )
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
    input_seq_dim: int = 1,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | None]:
    """
    Shard inputs and attention masks across the context parallel mesh.

    This function distributes input tensors across devices in the CP mesh
    along the sequence dimension. It optionally uses a load balancer to
    handle uneven computation workload. Currently, HeadTailLoadBalancer is
    used for SDPA + CP, which is the only supported configuration.

    Args:
        cp_mesh: Device mesh for context parallel dimension
        inputs: Tuple of input tensors to be sharded along the sequence
            dimension
        attention_masks: Attention masks to be sharded (currently raises
            error as FlexAttention CP is not yet supported)
        disable_load_balancer: If True, disables load balancing. If False
            (default), uses HeadTailLoadBalancer for SDPA to handle uneven
            computation workload.
        input_seq_dim: Sequence dimension index for sharding. Defaults to 1,
            which covers most use cases where tensors have shape
            [batch_size, seq_len, ...]. Can be changed by passing a
            different value if your tensors use a different sequence
            dimension layout.

    Returns:
        Tuple of (sharded_inputs, attention_masks) where:
            - sharded_inputs: Tuple of input tensors sharded along the
              sequence dimension
            - attention_masks: Attention masks (currently unchanged/None)
    """
    seq_len = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size(0)
    if attention_masks is not None:
        raise ValueError(
            "FlexAttention CP is not supported yet. Will come in the next PR."
        )
    else:
        # For SDPA, we use the _HeadTailLoadBalancer.
        load_balancer = (
            None
            if disable_load_balancer
            else _HeadTailLoadBalancer(seq_len, cp_world_size, cp_mesh.device_type)
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

    return inputs, attention_masks
