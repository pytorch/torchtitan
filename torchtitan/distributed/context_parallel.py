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

from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.tools.logging import logger


class _AllToAllWithInverse(torch.autograd.Function):
    """All-to-all that scatters one tensor dimension and gathers another.

    Forward chunks ``input_tensor`` on ``scatter_dim``, exchanges chunks across
    ``group``, and concatenates the received chunks on ``gather_dim``. Backward
    performs the inverse exchange, so the primitive can be used inside
    sequence/head layout swaps for context-parallel modules.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        world_size: int,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        return _all_to_all_single_dim_swap(
            input_tensor,
            group,
            world_size,
            scatter_dim,
            gather_dim,
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _all_to_all_single_dim_swap(
            grad_output,
            ctx.group,
            ctx.world_size,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return grad_input, None, None, None, None


def _all_to_all_single_dim_swap(
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Scatter one dimension and concatenate peers along another dimension."""

    send = input_tensor.movedim(scatter_dim, 0).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    recv_chunks = recv.chunk(world_size, dim=0)
    return torch.cat(
        [chunk.movedim(0, scatter_dim) for chunk in recv_chunks],
        dim=gather_dim,
    )


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"Dimension out of range: got {dim} for tensor rank {ndim}.")
    return dim


def _cp_world_size(cp_mesh: DeviceMesh) -> int:
    if cp_mesh.ndim != 1:
        raise ValueError(
            "Context-parallel exchanges require a 1D DeviceMesh, but got "
            f"a {cp_mesh.ndim}D mesh."
        )
    return cp_mesh.size()


def _all_to_all_with_inverse(
    input_tensor: torch.Tensor,
    cp_mesh: DeviceMesh,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Exchange tensor chunks across a CP mesh with inverse autograd."""

    scatter_dim = _normalize_dim(scatter_dim, input_tensor.ndim)
    gather_dim = _normalize_dim(gather_dim, input_tensor.ndim)
    if scatter_dim == gather_dim:
        raise ValueError("scatter_dim and gather_dim must be different dimensions.")
    world_size = _cp_world_size(cp_mesh)
    if world_size == 1:
        return input_tensor
    if input_tensor.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"Context-parallel all-to-all requires dim {scatter_dim} size "
            f"{input_tensor.shape[scatter_dim]} to be divisible by CP size {world_size}."
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "Context-parallel all-to-all requires torch.distributed to be initialized."
        )
    return _AllToAllWithInverse.apply(
        input_tensor,
        cp_mesh.get_group(),
        world_size,
        scatter_dim,
        gather_dim,
    )


def sequence_to_head_all_to_all(
    tensor: torch.Tensor,
    cp_mesh: DeviceMesh,
    *,
    sequence_dim: int = 1,
    head_dim: int = 2,
) -> torch.Tensor:
    """Convert a sequence-sharded tensor into a head-sharded full-sequence tensor."""

    return _all_to_all_with_inverse(
        tensor,
        cp_mesh,
        scatter_dim=head_dim,
        gather_dim=sequence_dim,
    )


def head_to_sequence_all_to_all(
    tensor: torch.Tensor,
    cp_mesh: DeviceMesh,
    *,
    sequence_dim: int = 1,
    head_dim: int = 2,
) -> torch.Tensor:
    """Convert a head-sharded full-sequence tensor back into sequence shards."""

    return _all_to_all_with_inverse(
        tensor,
        cp_mesh,
        scatter_dim=sequence_dim,
        gather_dim=head_dim,
    )


class _PreviousSequenceShardTail(torch.autograd.Function):
    """Receive the previous CP rank's sequence tail and route gradients back."""

    @staticmethod
    def forward(
        ctx,
        local_tail: torch.Tensor,
        group: dist.ProcessGroup,
        local_rank: int,
        world_size: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.local_rank = local_rank
        ctx.world_size = world_size

        local_tail = local_tail.contiguous()
        recv_tail = (
            torch.empty_like(local_tail)
            if local_rank > 0
            else torch.zeros_like(local_tail)
        )

        recv_req = None
        send_req = None
        if local_rank > 0:
            recv_req = dist.irecv(recv_tail, group_src=local_rank - 1, group=group)
        if local_rank < world_size - 1:
            send_req = dist.isend(local_tail, group_dst=local_rank + 1, group=group)
        if recv_req is not None:
            recv_req.wait()
        if send_req is not None:
            send_req.wait()
        return recv_tail

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        grad_input = (
            torch.empty_like(grad_output)
            if ctx.local_rank < ctx.world_size - 1
            else torch.zeros_like(grad_output)
        )

        recv_req = None
        send_req = None
        if ctx.local_rank < ctx.world_size - 1:
            recv_req = dist.irecv(
                grad_input, group_src=ctx.local_rank + 1, group=ctx.group
            )
        if ctx.local_rank > 0:
            send_req = dist.isend(
                grad_output, group_dst=ctx.local_rank - 1, group=ctx.group
            )
        if recv_req is not None:
            recv_req.wait()
        if send_req is not None:
            send_req.wait()
        return grad_input.contiguous(), None, None, None


def previous_sequence_shard_tail(
    local_tail: torch.Tensor,
    cp_mesh: DeviceMesh,
) -> torch.Tensor:
    """Return the previous CP rank's local sequence tail with autograd support.

    Rank 0 receives zeros. In the single-rank case this returns a zero tensor
    connected to ``local_tail`` with zero gradient, which keeps small unit tests
    and non-CP code paths simple.
    """

    world_size = _cp_world_size(cp_mesh)
    if world_size == 1:
        return local_tail * 0
    if not dist.is_initialized():
        raise RuntimeError(
            "previous_sequence_shard_tail requires torch.distributed to be initialized."
        )
    return _PreviousSequenceShardTail.apply(
        local_tail,
        cp_mesh.get_group(),
        cp_mesh.get_local_rank(),
        world_size,
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
                    if kwargs.get("score_mod") is not None:
                        raise NotImplementedError(
                            "FlexAttention score_mod is not supported with "
                            "Context Parallel yet. It must be sharded before "
                            "use with CP."
                        )
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
                placement = [Shard(1)]

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
                def cp_forward(q, k, v, **kwargs):
                    q = sequence_to_head_all_to_all(q, mesh, sequence_dim=1, head_dim=2)
                    k = sequence_to_head_all_to_all(k, mesh, sequence_dim=1, head_dim=2)
                    v = sequence_to_head_all_to_all(v, mesh, sequence_dim=1, head_dim=2)
                    output = orig_fn(q, k, v, **kwargs)
                    return head_to_sequence_all_to_all(
                        output, mesh, sequence_dim=1, head_dim=2
                    )

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
            BlockMask, dict[str, BlockMask], or replicated VarlenMetadata.
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
        ValueError: If load_balancer_type is incompatible with attention_masks.
    """
    if isinstance(attention_masks, VarlenMetadata) and load_balancer_type is not None:
        raise ValueError(
            "VarlenMetadata requires contiguous sequence shards; set "
            "load_balancer_type=None."
        )

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
                    f"Invalid load_balancer_type {load_balancer_type!r}. "
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
        if isinstance(attention_masks, VarlenMetadata):
            return inputs, attention_masks
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
                else {k: v for k, v in zip(attention_masks.keys(), masks, strict=True)}
            ),
        )

    return inputs, attention_masks
