# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
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

from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.tools.logging import logger


@dataclasses.dataclass(kw_only=True, slots=True)
class CPVarlenMetadata(VarlenMetadata):
    """Pre-computed VarlenMetadata for the CP forward pass.

    Slice bounds are plain Python ints computed once per batch in
    ``_shard_varlen_metadata`` (outside the model loop) to avoid GPU→CPU
    synchronizations inside ``cp_forward``.
    """
    k_slice_start: int  # start index into allgathered global K
    k_slice_end: int  # end index into allgathered global K

def _shard_varlen_metadata(
    metadata: VarlenMetadata,
    cp_mesh: DeviceMesh,
    seq_len: int,
    cp_world_size: int,
) -> CPVarlenMetadata:
    """Shard VarlenMetadata for this rank's contiguous token range.

    All expensive Python-int extraction (.item() calls) happens here, once per
    batch, outside the compiled model loop.  The returned ``CPVarlenMetadata``
    carries plain Python ints so ``cp_forward`` can slice K/V without any
    GPU→CPU synchronization.

    NOTE: assumes contiguous token assignment (``load_balancer_type=None``).
    Interleaved sharding (headtail / ptrr) is not compatible with the
    ``cu_seq`` document-boundary structure.
    """
    cp_rank = dist.get_rank(group=cp_mesh.get_group())
    local_start = cp_rank * seq_len // cp_world_size
    local_end = (cp_rank + 1) * seq_len // cp_world_size

    local_cu_seq_q, count_cu_seq_q = metadata.cu_seq_q.clamp(local_start, local_end).unique_consecutive(return_counts=True)
    local_cu_seq_k = metadata.cu_seq_k[count_cu_seq_q[0]-1 : (-count_cu_seq_q[-1]+1) or None].clamp(max=local_end)
    local_max_q = int(local_cu_seq_q.diff().max().item())
    local_max_k = int(local_cu_seq_k.diff().max().item())

    local_cu_seq_k_start = int(local_cu_seq_k[0].item())
    local_cu_seq_k_end = int(local_cu_seq_k[-1].item())
    shifted_local_cu_seq_q = local_cu_seq_q - local_start
    shifted_local_cu_seq_k = local_cu_seq_k - local_cu_seq_k_start

    return CPVarlenMetadata(
        cu_seq_q=shifted_local_cu_seq_q,
        cu_seq_k=shifted_local_cu_seq_k,
        max_q=local_max_q,
        max_k=local_max_k,
        k_slice_start=local_cu_seq_k_start,
        k_slice_end=local_cu_seq_k_end,
    )


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
        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                pg_name = dist._get_process_group_name(mesh.get_group())

                def cp_forward(q, k, v, **kwargs):
                    attn_masks = kwargs.get("attention_masks")
                    assert isinstance(attn_masks, CPVarlenMetadata), "Expected CPVarlenMetadata in attention_masks"
                    k = k.contiguous()
                    v = v.contiguous()
                    global_k, global_v = flex_cp_allgather(k, v, 1, pg_name)
                    # k_slice_start / k_slice_end are plain Python ints
                    # pre-computed in _shard_varlen_metadata — no GPU→CPU sync.
                    global_k = global_k[:, attn_masks.k_slice_start:attn_masks.k_slice_end]
                    global_v = global_v[:, attn_masks.k_slice_start:attn_masks.k_slice_end]
                    return orig_fn(q, global_k, global_v, **kwargs)

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)
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

    batch_size = inputs[0].size(0)
    match attention_masks:
        case None | BlockMask() | dict():
            inputs = cast(
                tuple[torch.Tensor, ...],
                _context_parallel_shard(
                    mesh=cp_mesh,
                    buffers=inputs,
                    seq_dims=tuple(input_seq_dim for _ in inputs),
                    load_balancer=load_balancer,
                ),
            )
        case VarlenMetadata():
            # Instead of sharding directly in the sequence length dimension,
            # we first reshape to [1, batch_size * seq_len] and shard.
            # this let us treat the entire batch as a single sequence, which is necessary for
            # correct sharding with variable-length inputs.
            inputs = cast(
                tuple[torch.Tensor, ...],
                _context_parallel_shard(
                    mesh=cp_mesh,
                    buffers=tuple(buf.reshape(1, -1) for buf in inputs),
                    seq_dims=tuple(input_seq_dim for _ in inputs),
                    load_balancer=None,  # VarlenMetadata sharding is handled separately in _shard_varlen_metadata
                ),
            )
        case _:
            raise ValueError(
                f"Unsupported attention_masks type for CP sharding: "
                f"{type(attention_masks)}. Must be None, VarlenMetadata, "
                f"BlockMask, or dict[str, BlockMask]."
            )

    # BlockMask, has shape, [B, H, Q, KV], and we can only shard
    # on the Q seq dimension, not KV.
    MASK_Q_SEQ_DIM = 2

    match attention_masks:
        case VarlenMetadata():
            if load_balancer is not None:
                raise ValueError("Load balancer must be disabled for variable-length attention")
            attention_masks = _shard_varlen_metadata(
                attention_masks, cp_mesh, seq_len * batch_size, cp_world_size
            )
        case BlockMask() | dict():
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
        case None:
            pass
        case _:
            raise ValueError(
                f"Unsupported attention_masks type for CP sharding: "
                f"{type(attention_masks)}. Must be None, BlockMask, or dict[str, BlockMask]."
            )

    return inputs, attention_masks
