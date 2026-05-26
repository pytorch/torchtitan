# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.distributed.tensor.experimental._context_parallel._attention import (
    flex_cp_allgather,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.varlen_cp import _VarlenPTRRLoadBalancer, CPVarlenMetadata
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.tools.logging import logger


def apply_cp_to_forward(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
) -> None:
    """Wrap inner attention ``forward`` with CP logic.

    Must be called **before** ``Module.parallelize()`` so the CP wrapper
    is captured inside parallelize's ``local_map`` wrapping.

    The attention type is inferred via isinstance on the first module.

    TODO: This is a temporary workaround that manually allgathers K/V
    (FlexAttention) or wraps inputs as CP-sharded DTensors (SDPA).
    Once all models adopt config-based sharding with full DTensor,
    CP redistribution should be expressed declaratively via
    ShardingConfig and this function should be removed.

    Args:
        attention_modules: Sequence of inner attention modules to apply CP to.
        cp_mesh: Device mesh for context parallel dimension.
    """
    first = attention_modules[0]
    if isinstance(first, FlexAttention):
        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                pg_name = dist._get_process_group_name(mesh.get_group())

                def cp_forward(q, k, v, **kwargs):
                    k = k.contiguous()
                    v = v.contiguous()
                    global_k, global_v = flex_cp_allgather(k, v, 1, pg_name)
                    return orig_fn(q, global_k, global_v, **kwargs)

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, ScaledDotProductAttention):
        _enable_context_parallel_dispatcher()

        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                placement = [Shard(2)]

                def cp_forward(q, k, v, **kwargs):
                    if not isinstance(q, DTensor):
                        q = DTensor.from_local(q, mesh, placement, run_check=False)
                    if not isinstance(k, DTensor):
                        k = DTensor.from_local(k, mesh, placement, run_check=False)
                    if not isinstance(v, DTensor):
                        v = DTensor.from_local(v, mesh, placement, run_check=False)
                    output = orig_fn(q, k, v, **kwargs)
                    return output.to_local() if isinstance(output, DTensor) else output

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, VarlenAttention):
        raise NotImplementedError(
            "VarlenAttention CP requires parallelism.full_dtensor=True; "
            "the legacy forward-wrapping path does not all-gather K/V."
        )
    else:
        raise NotImplementedError(
            f"Context Parallel forward wrapping is not supported for "
            f"{type(first).__name__}"
        )

    logger.info("Applied Context Parallel (forward wrapping) to the model")


def prepare_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
    load_balancer_type: str | None = "headtail",
    ptrr_block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Shard inputs, labels, positions, and attention masks for Context Parallel.

    The caller must provide ``extra_kwargs["positions"]`` before calling this
    function.  Position resolution (per-document vs sequential) is handled
    upstream in ``post_dataloading_process``.

    Args:
        inputs: Input tensor of shape [batch_size, seq_len]
        labels: Label tensor of shape [batch_size, seq_len]
        extra_kwargs: Dictionary containing 'positions' (required) and
            optionally 'attention_masks' to be sharded.
        cp_mesh: Device mesh for context parallel dimension
        device: Device for the tensors
        load_balancer_type: Type of load balancer to use for sharding.
            Options: "headtail", "ptrr", or None. Defaults to "headtail".
        ptrr_block_size: Block size used by the varlen PTRR load balancer
            (only consulted when ``load_balancer_type='ptrr'`` and
            ``attention_masks`` is a ``VarlenMetadata``).

    Returns:
        Tuple of (sharded_inputs, sharded_labels, updated_extra_kwargs) where:
            - sharded_inputs: Inputs sharded along sequence dimension
            - sharded_labels: Labels sharded along sequence dimension
            - updated_extra_kwargs: Dict with sharded 'positions' and optionally
              sharded 'attention_masks'
    """
    attention_masks = extra_kwargs.get("attention_masks", None)
    positions = extra_kwargs["positions"]
    (inputs, labels, positions), attention_masks = cp_shard(
        cp_mesh,
        (inputs, labels, positions),
        attention_masks,
        load_balancer_type,
        ptrr_block_size=ptrr_block_size,
    )
    extra_kwargs["positions"] = positions
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    return inputs, labels, extra_kwargs


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dim: int = 1,
    ptrr_block_size: int = 1024,
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
            BlockMask, dict[str, BlockMask], or VarlenMetadata.
        load_balancer_type: Type of load balancer to use. Options:
            - "headtail": Use HeadTailLoadBalancer (works for SDPA, varlen,
              and FlexAttention)
            - "ptrr": Use PTRRLoadBalancer (FlexAttention) or
              _VarlenPTRRLoadBalancer (varlen)
            - None: Disable load balancing
            Defaults to "headtail".
        input_seq_dim: Sequence dimension index for sharding. Defaults to 1,
            which covers most use cases where tensors have shape
            [batch_size, seq_len]. Can be changed by passing a
            different value if your tensors use a different sequence
            dimension layout.
        ptrr_block_size: Block size used by the varlen PTRR load balancer
            (only consulted when ``load_balancer_type='ptrr'`` and
            ``attention_masks`` is a ``VarlenMetadata``).

    Returns:
        Tuple of (sharded_inputs, attention_masks) where:
            - sharded_inputs: Tuple of input tensors sharded along the
              sequence dimension
            - attention_masks: Sharded attention masks (BlockMask,
              dict[str, BlockMask], or CPVarlenMetadata when the input
              was a VarlenMetadata) or None

    Raises:
        (Listed in the order they fire at runtime.)

        ValueError: If ``cp_mesh.ndim != 1``.
        ValueError: If attention_masks is a VarlenMetadata and
            input_seq_dim is not 1 (varlen sharding assumes the
            (B, S, ...) layout).
        ValueError: If load_balancer_type is "ptrr" and attention_masks
            is None or a dict (FlexAttention PTRR requires a BlockMask;
            VarlenMetadata is routed to _VarlenPTRRLoadBalancer).
        ValueError: When inputs are inconsistent for varlen sharding
            (cu_seq_q malformed, cross-attention, divisibility,
            double-shard). See CPVarlenMetadata.from_global for the
            full list.
    """
    if cp_mesh.ndim != 1:
        raise ValueError(f"cp_shard expects a 1-D CP mesh, got ndim={cp_mesh.ndim}.")
    if isinstance(attention_masks, VarlenMetadata) and input_seq_dim != 1:
        raise ValueError(
            "VarlenMetadata sharding assumes inputs are (B, S, ...) "
            f"with input_seq_dim=1; got input_seq_dim={input_seq_dim}."
        )
    seq_len = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size()

    # batch_size is only meaningful for VarlenMetadata branches (varlen
    # PTRR balancer construction and CPVarlenMetadata.from_global), which
    # both require the (B, S, ...) layout asserted above.
    batch_size = inputs[0].size(0)

    load_balancer = None
    if load_balancer_type:
        match load_balancer_type:
            case "headtail":
                load_balancer = _HeadTailLoadBalancer(
                    seq_len, cp_world_size, cp_mesh.device_type
                )
            case "ptrr":
                # FlexAttention takes a BlockMask; varlen takes the global cu_seq_q
                # via _VarlenPTRRLoadBalancer. The varlen balancer assumes
                # document-causal masking, which is the only varlen path currently in
                # the codebase (VarlenAttention.forward also rejects non-causal
                # window_size under CP).
                if isinstance(attention_masks, VarlenMetadata):
                    load_balancer = _VarlenPTRRLoadBalancer(
                        attention_masks.cu_seq_q,
                        batch_size=batch_size,
                        seq_length=seq_len,
                        world_size=cp_world_size,
                        block_size=ptrr_block_size,
                    )
                else:
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
            seq_dims=(input_seq_dim,) * len(inputs),
            load_balancer=load_balancer,
        ),
    )

    # BlockMask path uses the upstream dispatcher (shard Q seq dim only; KV stays
    # full). VarlenMetadata path uses the local CPVarlenMetadata builder since the
    # CP-aware varlen logic lives entirely in torchtitan.
    MASK_Q_SEQ_DIM = 2
    if attention_masks is not None:
        if isinstance(attention_masks, VarlenMetadata):
            attention_masks = CPVarlenMetadata.from_global(
                attention_masks,
                device_mesh=cp_mesh,
                batch_size=batch_size,
                seq_length=seq_len,
                load_balancer=load_balancer,
            )
        else:
            assert isinstance(attention_masks, (BlockMask, dict))
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
                (
                    masks[0]
                    if isinstance(attention_masks, BlockMask)
                    else {k: v for k, v in zip(attention_masks.keys(), masks)}
                ),
            )

    return inputs, attention_masks
