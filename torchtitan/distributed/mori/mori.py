# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MORI-EP primitives for MoE Expert Parallel on AMD GPUs.

MORI (https://github.com/ROCm/mori) provides GPU-initiated EP dispatch/combine
kernels for ROCm, analogous to DeepEP: ``EpDispatchCombineOp.dispatch`` sends
each token once per destination rank (not once per routed expert, as the
all-to-all does), and ``combine`` sums the per-rank partial outputs back on the
source rank. Within a node it uses the ``IntraNode`` kernel (XGMI peer access);
across nodes the ``InterNodeV1`` kernel (RDMA).

Layout. MORI's dispatch output is deduplicated: one row per received token,
with that token's global expert ids and routing scores. ``dispatch_tokens``
gathers it into expert-major order for the grouped GEMM (optionally padding
every expert's group to ``pad_multiple`` for quantized grouped GEMMs), and
``combine_tokens`` applies the routing scores and sums each received token's
local expert outputs before the combine op. Both are plain PyTorch, so the
custom ops below stay pure communication and autograd handles the score
gradient.

Autograd. Dispatch backward is a combine of the gradients; passing the received
score gradients as combine's weights also reduces the routing-score gradients
to their source tokens. Combine backward is a dispatch that replays the
forward's routing handle, so the gradients follow the forward's layout.

Activation checkpointing. MORI fills receive slots with atomics, so a replayed
dispatch can receive rows in another order than the forward's handle records,
and MORI has no deterministic mode. The ``ep_communication`` regions are
therefore always retained (``recompute=False``), under every policy including
FullAC: activation checkpointing replays the expert computation and the layout
around the communication, but never the communication itself.
"""

import functools
import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_remat as remat
from torch.distributed import ProcessGroup

try:
    import mori
except ImportError as e:
    raise ImportError(
        "MORI is required for MoriTokenDispatcher. "
        "Install from: https://github.com/ROCm/mori"
    ) from e


logger = logging.getLogger(__name__)

# One op per process, shared by every MoE layer: layers run one at a time and
# each dispatch is paired with its combine before the next layer's dispatch.
_op: "mori.ops.EpDispatchCombineOp | None" = None
_op_group: ProcessGroup | None = None
_op_key: tuple | None = None
_shmem_group: ProcessGroup | None = None

# Routing handles by id. Custom ops can only pass tensors across the op
# boundary, so the opaque MORI handle travels as a CPU int64 handle_id plus
# this cache (as in torchtitan.distributed.deepep.deepep). The dispatch
# backward, the handle's last user, removes it; a combine that no backward
# will follow (forward under no_grad) removes it instead.
_handle_cache: dict[int, "_Routing"] = {}
_handle_counter: int = 0


@dataclass
class _Routing:
    handle: object  # MORI routing handle returned by dispatch(return_routing=True)
    topk_idx: torch.Tensor  # int32 [num_tokens, top_k], needed by every replay


def _get_next_handle_id() -> torch.Tensor:
    """Generate a unique handle_id tensor on CPU to avoid a GPU-CPU sync."""
    global _handle_counter
    _handle_counter += 1
    return torch.tensor([_handle_counter], dtype=torch.int64, device="cpu")


def _get_op() -> "mori.ops.EpDispatchCombineOp":
    assert _op is not None, "MORI op must be initialized before dispatch"
    return _op


def _get_routing(handle_id: torch.Tensor, *, pop: bool) -> _Routing:
    key = int(handle_id.item())
    routing = _handle_cache.pop(key, None) if pop else _handle_cache.get(key)
    assert routing is not None, f"MORI routing handle {key} not found"
    return routing


# ============================================================================
# Custom ops: torch.compile, activation checkpointing and autograd
# ============================================================================

_lib = torch.library.Library("mori_ep", "DEF")

# dispatch returns (recv_x [R, D], recv_scores [R, K] fp32, recv_topk_idx [R, K]
# int32 global expert ids, handle_id [1] cpu). R, the received token count, is
# data-dependent.
_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)
# combine returns the [num_tokens, D] sum of the per-rank partial outputs.
# ``will_backward`` is the caller's outer grad state: inside a custom-op forward
# autograd disables grad, so the op cannot tell training from inference itself.
_lib.define(
    "combine(Tensor x, Tensor handle_id, SymInt num_tokens, bool will_backward) "
    "-> Tensor"
)
# Backward halves, as ops so a traced backward graph never calls into MORI from
# Python: dispatch backward is a combine (also reducing the score gradients),
# combine backward is a dispatch replaying the forward's routing handle.
_lib.define(
    "dispatch_backward(Tensor grad_recv_x, Tensor grad_recv_scores, "
    "Tensor handle_id, SymInt num_tokens) -> (Tensor, Tensor)"
)
_lib.define(
    "combine_backward(Tensor grad_out, Tensor handle_id, SymInt num_recv_tokens) "
    "-> Tensor"
)
# Ordered effects, so graph-level selective checkpointing saves the ops instead
# of replaying them in backward (see the module docstring).
for _name in ("dispatch", "combine", "dispatch_backward", "combine_backward"):
    _lib._register_effectful_op(f"mori_ep::{_name}", torch.library.EffectType.ORDERED)


@torch.library.impl(_lib, "dispatch", "CUDA")
def _dispatch_op_impl(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    op = _get_op()
    topk_idx = topk_idx.to(torch.int32).contiguous()
    recv_x, recv_scores, _scales, recv_topk_idx, num_recv, routing = op.dispatch(
        x.contiguous(),
        topk_weights.to(torch.float32).contiguous(),
        None,
        topk_idx,
        return_routing=True,
    )
    # The outputs are views into buffers the next MORI call reuses: copy the
    # received rows out. num_recv is a device scalar; this is the host sync.
    num_recv_tokens = int(num_recv.item())
    handle_id = _get_next_handle_id()
    _handle_cache[int(handle_id.item())] = _Routing(handle=routing, topk_idx=topk_idx)
    return (
        recv_x[:num_recv_tokens].clone(),
        recv_scores[:num_recv_tokens].clone(),
        recv_topk_idx[:num_recv_tokens].clone(),
        handle_id,
    )


@torch.library.register_fake("mori_ep::dispatch")
def _dispatch_op_fake(x, topk_idx, topk_weights):
    num_recv_tokens = torch.library.get_ctx().new_dynamic_size()
    top_k = topk_idx.shape[1]
    return (
        x.new_empty(num_recv_tokens, x.shape[1]),
        x.new_empty(num_recv_tokens, top_k, dtype=torch.float32),
        x.new_empty(num_recv_tokens, top_k, dtype=torch.int32),
        torch.empty(1, dtype=torch.int64, device="cpu"),
    )


@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(
    x: torch.Tensor, handle_id: torch.Tensor, num_tokens: int, will_backward: bool
) -> torch.Tensor:
    routing = _get_routing(handle_id, pop=not will_backward)
    combined, _ = _get_op().combine(
        x.contiguous(), None, routing.topk_idx, routing=routing.handle
    )
    return combined[:num_tokens].clone()


@torch.library.register_fake("mori_ep::combine")
def _combine_op_fake(x, handle_id, num_tokens, will_backward):
    return x.new_empty(num_tokens, x.shape[1])


@torch.library.impl(_lib, "dispatch_backward", "CUDA")
def _dispatch_backward_op_impl(
    grad_recv_x: torch.Tensor,
    grad_recv_scores: torch.Tensor,
    handle_id: torch.Tensor,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The last user of the handle.
    routing = _get_routing(handle_id, pop=True)
    grad_x, grad_scores = _get_op().combine(
        grad_recv_x.contiguous(),
        grad_recv_scores.contiguous(),
        routing.topk_idx,
        routing=routing.handle,
    )
    return grad_x[:num_tokens].clone(), grad_scores[:num_tokens].clone()


@torch.library.register_fake("mori_ep::dispatch_backward")
def _dispatch_backward_op_fake(grad_recv_x, grad_recv_scores, handle_id, num_tokens):
    return (
        grad_recv_x.new_empty(num_tokens, grad_recv_x.shape[1]),
        grad_recv_scores.new_empty(num_tokens, grad_recv_scores.shape[1]),
    )


@torch.library.impl(_lib, "combine_backward", "CUDA")
def _combine_backward_op_impl(
    grad_out: torch.Tensor, handle_id: torch.Tensor, num_recv_tokens: int
) -> torch.Tensor:
    routing = _get_routing(handle_id, pop=False)
    # Pass zero weights, not None: the InterNodeV1 dispatch kernel reads the
    # weights buffer without a null check.
    zero_scores = torch.zeros(
        routing.topk_idx.shape, dtype=torch.float32, device=grad_out.device
    )
    grad_recv_x, *_ = _get_op().dispatch(
        grad_out.contiguous(),
        zero_scores,
        None,
        routing.topk_idx,
        routing=routing.handle,
    )
    return grad_recv_x[:num_recv_tokens].clone()


@torch.library.register_fake("mori_ep::combine_backward")
def _combine_backward_op_fake(grad_out, handle_id, num_recv_tokens):
    return grad_out.new_empty(num_recv_tokens, grad_out.shape[1])


def _dispatch_setup_context(ctx, inputs, output):
    x, topk_idx, topk_weights = inputs
    *_, handle_id = output
    ctx.x_dtype = x.dtype
    ctx.scores_dtype = topk_weights.dtype
    ctx.num_tokens = x.shape[0]
    ctx.top_k = topk_idx.shape[1]
    ctx.save_for_backward(handle_id)


def _dispatch_backward(ctx, grad_recv_x, grad_recv_scores, _grad_idx, _grad_id):
    (handle_id,) = ctx.saved_tensors
    if grad_recv_scores is None:
        grad_recv_scores = grad_recv_x.new_zeros(
            grad_recv_x.shape[0], ctx.top_k, dtype=torch.float32
        )
    grad_x, grad_scores = torch.ops.mori_ep.dispatch_backward(
        grad_recv_x, grad_recv_scores.float(), handle_id, ctx.num_tokens
    )
    return grad_x.to(ctx.x_dtype), None, grad_scores.to(ctx.scores_dtype)


def _combine_setup_context(ctx, inputs, output):
    x, handle_id, _num_tokens, _will_backward = inputs
    ctx.num_recv_tokens = x.shape[0]
    ctx.save_for_backward(handle_id)


def _combine_backward(ctx, grad_out):
    (handle_id,) = ctx.saved_tensors
    grad_x = torch.ops.mori_ep.combine_backward(
        grad_out, handle_id, ctx.num_recv_tokens
    )
    return grad_x, None, None, None


torch.library.register_autograd(
    "mori_ep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "mori_ep::combine", _combine_backward, setup_context=_combine_setup_context
)


# ============================================================================
# Op construction
# ============================================================================


def _init_shmem(group: ProcessGroup) -> None:
    """Initialize MORI symmetric memory over ``group``.

    Equivalent to ``mori.shmem.shmem_torch_process_group_init``, but broadcasts
    the unique id with a group-local source rank. The upstream helper passes
    ``src=0``, a global rank, so it fails for an EP group that does not contain
    global rank 0 (e.g. EP within a node on every node but the first).
    """
    global _shmem_group
    if _shmem_group is not None:
        if _shmem_group is not group:
            raise ValueError("MORI supports a single EP process group per process.")
        return
    from mori.shmem import api as shmem_api

    shmem_api._ensure_shmem_module()
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    unique_id = [shmem_api.shmem_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(unique_id, group=group, group_src=0)
    shmem_api.shmem_init_attr(
        shmem_api.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world_size, unique_id[0]
    )
    _shmem_group = group


def get_op(
    group: ProcessGroup,
    *,
    hidden_dim: int,
    num_experts: int,
    num_topk: int,
    num_max_tokens_per_rank: int,
    dtype: torch.dtype,
    num_qp_per_pe: int,
    block_num: int | None,
    rdma_block_num: int | None,
    warp_num_per_block: int,
) -> "mori.ops.EpDispatchCombineOp":
    """Get or create the process-global MORI ``EpDispatchCombineOp``.

    The op preallocates its symmetric buffers for ``num_max_tokens_per_rank``
    input tokens of ``hidden_dim`` x ``dtype``; it is created once and shared by
    every MoE layer. The symmetric heap size comes from the
    ``MORI_SHMEM_HEAP_SIZE`` environment variable, read when the first op is
    created.

    The kernel is chosen from the EP group's extent: ``IntraNode`` when the
    group fits on one node (``torch.cuda.device_count()`` GPUs), else
    ``InterNodeV1``. This assumes consecutive EP ranks share a node, as
    torchtitan's EP mesh axis is innermost.
    """
    global _op, _op_group, _op_key
    config_key = (hidden_dim, num_experts, num_topk, num_max_tokens_per_rank, dtype)
    if _op is not None:
        if _op_group is not group or _op_key != config_key:
            raise ValueError(
                "MORI supports one EP group and one MoE shape per process; "
                f"have {_op_key}, requested {config_key}."
            )
        return _op

    _init_shmem(group)
    world_size = group.size()
    # MORI requires world_size to be a multiple of gpu_per_node; an EP group
    # smaller than a node spans one partial node.
    gpus_per_node = min(world_size, torch.cuda.device_count())
    internode = world_size > gpus_per_node
    kernel_type = mori.ops.EpDispatchCombineKernelType
    config = mori.ops.EpDispatchCombineConfig(
        data_type=dtype,
        rank=group.rank(),
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=4,
        max_num_inp_token_per_rank=num_max_tokens_per_rank,
        num_experts_per_rank=num_experts // world_size,
        num_experts_per_token=num_topk,
        block_num=block_num if block_num is not None else (96 if internode else 80),
        warp_num_per_block=warp_num_per_block,
        use_external_inp_buf=True,
        gpu_per_node=gpus_per_node,
        rdma_block_num=(
            (rdma_block_num if rdma_block_num is not None else 64) if internode else 0
        ),
        num_qp_per_pe=num_qp_per_pe,
        kernel_type=(kernel_type.InterNodeV1 if internode else kernel_type.IntraNode),
    )
    _op = mori.ops.EpDispatchCombineOp(config)
    _op_group = group
    _op_key = config_key
    return _op


@functools.cache
def warn_retained_communication() -> None:
    """Warn, once per process, that MORI retains communication a policy recomputes."""
    logger.warning(
        "The activation checkpointing policy recomputes ep_communication, but "
        "MoriTokenDispatcher retains its dispatch and combine communication: "
        "MORI has no deterministic mode, so a replay could reorder received tokens."
    )


# ============================================================================
# Dispatch / combine with the receiver-side expert-major layout
# ============================================================================


@dataclass
class DispatchState:
    """State from dispatch needed for combine."""

    handle_id: torch.Tensor  # CPU tensor naming the cached MORI routing handle
    num_tokens: int  # input tokens on this rank
    num_recv_tokens: int  # deduplicated tokens received by this rank
    # Received token of each expert-major row (padding rows excluded).
    token_indices: torch.Tensor
    # Expert-major row of each entry of token_indices.
    row_indices: torch.Tensor
    # Routing score of each entry of token_indices.
    scores: torch.Tensor
    num_rows: int  # expert-major rows, including padding


def _permute_tokens(
    recv_x_RD: torch.Tensor,
    recv_topk_idx_RK: torch.Tensor,
    recv_scores_RK: torch.Tensor,
    *,
    rank: int,
    num_local_experts: int,
    pad_multiple: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the deduplicated received tokens into expert-major order.

    A received token appears once per local expert it is routed to; picks of
    experts on other ranks are skipped. With ``pad_multiple``, each expert's
    group is padded with zero rows to a multiple of ``pad_multiple``.

    Returns:
        routed_input: ``[num_rows, D]`` expert-major tokens.
        num_tokens_per_expert: ``[num_local_experts]`` (padded) group sizes.
        token_indices, row_indices, scores: ``[num_valid]`` received token,
            expert-major row and routing score of each (token, local expert)
            pick, for ``_unpermute_tokens``.
    """
    local_expert_RK = recv_topk_idx_RK.to(torch.int64) - rank * num_local_experts
    mask_RK = (local_expert_RK >= 0) & (local_expert_RK < num_local_experts)
    valid_experts = local_expert_RK[mask_RK]
    sort_order = torch.argsort(valid_experts, stable=True)
    sorted_experts = valid_experts[sort_order]
    token_indices = (
        torch.arange(recv_x_RD.shape[0], device=recv_x_RD.device)
        .unsqueeze(1)
        .expand_as(mask_RK)[mask_RK][sort_order]
    )
    scores = recv_scores_RK[mask_RK][sort_order]
    counts = torch.bincount(sorted_experts, minlength=num_local_experts)

    if pad_multiple is None:
        row_indices = torch.arange(token_indices.shape[0], device=recv_x_RD.device)
        routed_input = recv_x_RD.index_select(0, token_indices)
        return routed_input, counts, token_indices, row_indices, scores

    padded_counts = (counts + pad_multiple - 1) // pad_multiple * pad_multiple
    starts = torch.cumsum(counts, 0) - counts
    padded_starts = torch.cumsum(padded_counts, 0) - padded_counts
    rank_in_expert = (
        torch.arange(sorted_experts.shape[0], device=recv_x_RD.device)
        - starts[sorted_experts]
    )
    row_indices = padded_starts[sorted_experts] + rank_in_expert
    # Host sync for the padded row count.
    num_rows = int(padded_counts.sum().item())
    routed_input = recv_x_RD.new_zeros(num_rows, recv_x_RD.shape[1])
    routed_input = routed_input.index_copy(
        0, row_indices, recv_x_RD.index_select(0, token_indices)
    )
    return routed_input, padded_counts, token_indices, row_indices, scores


def _unpermute_tokens(
    routed_output: torch.Tensor, state: DispatchState
) -> torch.Tensor:
    """Weight each expert output by its routing score and sum per received token."""
    rows = routed_output.index_select(0, state.row_indices)
    rows = (rows * state.scores.reshape(-1, 1).to(torch.float32)).to(rows.dtype)
    out = rows.new_zeros(state.num_recv_tokens, rows.shape[1])
    return out.index_add(0, state.token_indices, rows)


def dispatch_tokens(
    hidden_states: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    *,
    rank: int,
    num_local_experts: int,
    pad_multiple: int | None,
    remat_region_name: str,
) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
    """Dispatch tokens to experts via MORI.

    Args:
        hidden_states: Input tokens ``[num_tokens, hidden_dim]``.
        selected_experts_indices: Expert indices per token ``[num_tokens, top_k]``.
        top_scores: Routing scores per token ``[num_tokens, top_k]``.
        rank: This rank's index in the EP group.
        num_local_experts: Number of experts on this rank.
        pad_multiple: If set, pad each expert's token group to a multiple of
            this, for quantized grouped GEMMs.
        remat_region_name: Name for the dispatch communication region. The
            region is always saved (``recompute=False``); see the module
            docstring.

    Returns:
        (routed_tokens ``[num_rows, hidden_dim]``,
        tokens_per_expert ``[num_local_experts]``, state)
    """
    recv_x, recv_scores, recv_topk_idx, handle_id = remat.region(
        torch.ops.mori_ep.dispatch,
        remat_region_name,
        recompute=False,
    )(hidden_states, selected_experts_indices, top_scores)
    # The saved region is skipped during replay, while the layout below still
    # runs and therefore needs its original outputs.
    remat.recompute_needs_tensor(recv_x, recv_scores, recv_topk_idx, handle_id)

    (
        routed_input,
        num_tokens_per_expert,
        token_indices,
        row_indices,
        scores,
    ) = _permute_tokens(
        recv_x,
        recv_topk_idx,
        recv_scores,
        rank=rank,
        num_local_experts=num_local_experts,
        pad_multiple=pad_multiple,
    )
    state = DispatchState(
        handle_id=handle_id,
        num_tokens=hidden_states.shape[0],
        num_recv_tokens=recv_x.shape[0],
        token_indices=token_indices,
        row_indices=row_indices,
        scores=scores,
        num_rows=routed_input.shape[0],
    )
    return routed_input, num_tokens_per_expert, state


def combine_tokens(
    hidden_states: torch.Tensor,
    state: DispatchState,
    *,
    remat_region_name: str,
) -> torch.Tensor:
    """Combine expert outputs back to their source tokens via MORI.

    Routing scores are applied here, in plain PyTorch, while summing each
    received token's local expert outputs; MORI then sums the per-rank partial
    outputs on the source rank.

    Args:
        hidden_states: Expert outputs ``[num_rows, hidden_dim]`` in the
            expert-major order of ``dispatch_tokens``.
        state: Dispatch state from ``dispatch_tokens``.
        remat_region_name: Name for the combine communication region (always
            saved, like the dispatch region).

    Returns:
        Combined tokens ``[num_tokens, hidden_dim]``.
    """
    # Outer grad state decides whether the combine op frees the routing handle
    # (no backward) or leaves it for the dispatch backward.
    will_backward = torch.is_grad_enabled()
    partial_RD = _unpermute_tokens(hidden_states, state)
    combined = remat.region(
        torch.ops.mori_ep.combine,
        remat_region_name,
        recompute=False,
    )(partial_RD, state.handle_id, state.num_tokens, will_backward)
    # The caller consumes this output outside another remat region.
    remat.recompute_needs_tensor(combined)
    return combined
