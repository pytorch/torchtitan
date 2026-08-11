# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, TYPE_CHECKING

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.models.common.attention import AttentionMasksType

if TYPE_CHECKING:
    from torchtitan.config import ParallelismConfig


def validate_cp_backend(parallelism: "ParallelismConfig") -> None:
    """Validate CP backend compatibility for ShardingConfig-based models."""
    if (
        parallelism.context_parallel_degree > 1
        and parallelism.spmd_backend != "spmd_types"
    ):
        raise ValueError(
            "Context Parallel requires parallelism.spmd_backend='spmd_types', "
            f"got '{parallelism.spmd_backend}'."
        )


def cp_shard_dims(input_sharding: dict[str, SpmdLayout]) -> dict[str, int]:
    """Derive ``{name: seq_dim}`` for inputs whose CP mesh axis is a Shard.

    Inputs whose CP axis is Replicate/Partial (e.g. an image stream that is
    not sequence-sharded) are omitted and thus left untouched by CP.
    """
    dims: dict[str, int] = {}
    for name, layout in input_sharding.items():
        cp_dim = layout.shard_dim(MeshAxisName.CP)
        if cp_dim is not None:
            dims[name] = cp_dim
    return dims


def prepare_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
    load_balancer_type: str | None = "headtail",
    ptrr_mask_key: str | None = None,
    input_sharding: dict[str, SpmdLayout] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Shard named tensors and attention masks for Context Parallel.

    Each tensor named in ``shard_dims`` (resolved against
    ``{"input": inputs, "labels": labels, **extra_kwargs}``) is sharded along
    its declared sequence dimension using a single shared load balancer.
    Attention masks (``BlockMask``) are sharded separately along their Q
    sequence dimension. Position resolution (per-document vs sequential) is
    handled upstream (the model's ``preprocess_inputs`` / the trainer).

    Args:
        inputs: Input tensor of shape [batch_size, seq_len].
        labels: Label tensor of shape [batch_size, seq_len].
        extra_kwargs: Additional model-forward kwargs. Tensor entries named in
            ``shard_dims`` (e.g. 'positions') are sharded and written back;
            'attention_masks', if present, is sharded along its Q seq dim.
        cp_mesh: Device mesh for the context parallel dimension.
        device: Device for the tensors.
        load_balancer_type: Type of load balancer to use for sharding.
            Options: "headtail", "ptrr", or None. Defaults to "headtail".
        ptrr_mask_key: When ``load_balancer_type`` is "ptrr" and the attention
            masks are a dict[str, BlockMask], selects which mask the
            PTRRLoadBalancer is built from. Ignored otherwise.
        input_sharding: Per-input SPMD layout; the CP sequence dim for each
            input is derived via ``cp_shard_dims`` (inputs whose CP axis is
            Replicate/Partial are omitted and left untouched). When None,
            defaults to sharding ``{"input": 1, "labels": 1, "positions": 1}``
            (standard decoder inputs, for callers without a per-input layout).

    Returns:
        Tuple of (sharded_inputs, sharded_labels, updated_extra_kwargs) where:
            - sharded_inputs: Inputs sharded along the sequence dimension.
            - sharded_labels: Labels sharded along the sequence dimension.
            - updated_extra_kwargs: ``extra_kwargs`` with its sharded tensor
              entries (e.g. 'positions') and 'attention_masks' updated.
    """
    if input_sharding is not None:
        shard_dims = cp_shard_dims(input_sharding)
    else:
        shard_dims = {"input": 1, "labels": 1, "positions": 1}

    named: dict[str, torch.Tensor] = {"input": inputs, "labels": labels}
    for k, v in extra_kwargs.items():
        if isinstance(v, torch.Tensor):
            named[k] = v

    shard_names = [n for n in shard_dims if n in named]
    if not shard_names:
        return inputs, labels, extra_kwargs
    buffers = tuple(named[n] for n in shard_names)
    seq_dims = tuple(shard_dims[n] for n in shard_names)

    attention_masks = extra_kwargs.get("attention_masks", None)
    sharded_buffers, attention_masks = cp_shard(
        cp_mesh,
        buffers,
        attention_masks,
        load_balancer_type,
        input_seq_dims=seq_dims,
        ptrr_mask_key=ptrr_mask_key,
    )

    result = dict(zip(shard_names, sharded_buffers))
    out_inputs = result.get("input", inputs)
    out_labels = result.get("labels", labels)
    for n in shard_names:
        if n not in ("input", "labels"):
            extra_kwargs[n] = result[n]
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks
    return out_inputs, out_labels, extra_kwargs


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dims: int | tuple[int, ...] = 1,
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
            - "ptrr": Use PTRRLoadBalancer (for FlexAttention)
            - None: Disable load balancing
            Defaults to "headtail".
        input_seq_dims: Sequence dimension(s) for sharding. An int applies the
            same dim to every tensor in ``inputs`` (default 1, covering the
            common [batch_size, seq_len] layout). A tuple specifies a per-tensor
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

    # BlockMask, has shape, [B, H, Q, KV], and we can only shard
    # on the Q seq dimension, not KV.
    MASK_Q_SEQ_DIM = 2
    if attention_masks is not None:
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
