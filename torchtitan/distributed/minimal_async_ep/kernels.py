# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

import torch
import triton
import triton.language as tl


_COUNT_COPY_BLOCK_SIZE = 1024
_METADATA_BLOCK_SIZE = 256
_MAX_BLOCK_N = 2048

# MinimalAsyncEP hidden buffers use TrainingConfig.mixed_precision_param,
# currently restricted to bfloat16 or float32.
_HIDDEN_ROW_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit(do_not_specialize=["rank"])
def _copy_full_counts_to_peer_ptrs_kernel(
    counts: tl.pointer_type(tl.int64),
    dst_ptrs: tl.pointer_type(tl.int64),
    rank: tl.int64,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    DST_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Copy this rank's global expert counts into every peer count buffer.

    Each peer exposes its symmetric counts buffer through ``dst_ptrs``. This
    kernel writes ``counts[:]`` into row ``rank`` of each peer's buffer.

    Example:
        With ``rank=1`` and ``counts=[2, 0, 1, 3]``, every peer buffer has
        row 1 updated to ``[2, 0, 1, 3]`` after the kernel finishes.
    """
    peer = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (peer < EP_SIZE) & (offsets < NUM_EXPERTS)
    base = tl.load(dst_ptrs + peer, mask=peer < EP_SIZE, other=0)
    dst = base.to(tl.pointer_type(tl.int64)) + rank * DST_ROW_STRIDE + offsets
    values = tl.load(counts + offsets, mask=mask, other=0)
    tl.store(dst, values, mask=mask)


@triton.jit
def _fill_dispatch_metadata_kernel(
    counts: tl.pointer_type(tl.int64),
    local_dest_offsets: tl.pointer_type(tl.int64),
    local_count_starts: tl.pointer_type(tl.int64),
    dst_ranks: tl.pointer_type(tl.int64),
    dst_rows: tl.pointer_type(tl.int64),
    NUM_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Build per-routed-row peer destinations for the dispatch copy.

    Rows are in this rank's expert-sorted routed-token order. ``dst_rows``
    indexes the destination peer's expert-major receive buffer.

    Example:
        With ``NUM_LOCAL_EXPERTS=2``, ``counts=[2, 0, 1, 1]``,
        ``local_count_starts=[0, 2, 2, 3]``, and
        ``local_dest_offsets=[0, 0, 5, 9]``, the outputs are
        ``dst_ranks=[0, 0, 1, 1]`` and ``dst_rows=[0, 1, 5, 9]``.
    """
    expert = tl.program_id(0)
    block = tl.program_id(1)
    offset = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    count = tl.load(counts + expert)
    mask = (expert < NUM_EXPERTS) & (offset < count)
    row = tl.load(local_count_starts + expert) + offset
    tl.store(dst_ranks + row, expert // NUM_LOCAL_EXPERTS, mask=mask)
    tl.store(dst_rows + row, tl.load(local_dest_offsets + expert) + offset, mask=mask)


@triton.jit(do_not_specialize=["ep_rank"])
def _fill_combine_metadata_kernel(
    segment_lens: tl.pointer_type(tl.int64),
    output_starts: tl.pointer_type(tl.int64),
    source_input_starts: tl.pointer_type(tl.int64),
    dst_ranks: tl.pointer_type(tl.int64),
    dst_rows: tl.pointer_type(tl.int64),
    ep_rank: tl.int64,
    EP_SIZE: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Build per-active-row peer destinations for the combine copy.

    Rows are local expert-major rows in ``x_recv``. ``dst_rows`` maps each
    active received row back to the source rank's expert-sorted routed rows.

    Example:
        With ``ep_rank=1``, ``EP_SIZE=2``, ``NUM_LOCAL_EXPERTS=2``,
        ``segment_lens=[2, 1, 0, 1]``, ``output_starts=[0, 2, 3, 3]``,
        and source starts for global experts 2 and 3 equal to
        ``[[4, 6], [8, 10]]``, the active prefix is
        ``dst_ranks=[0, 0, 1, 1]`` and ``dst_rows=[4, 5, 8, 10]``.
    """
    segment = tl.program_id(0)
    block = tl.program_id(1)
    offset = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    count = tl.load(segment_lens + segment)
    mask = offset < count

    output_start = tl.load(output_starts + segment)
    row = output_start + offset
    src_rank = segment % EP_SIZE
    local_expert = segment // EP_SIZE
    global_expert = ep_rank * NUM_LOCAL_EXPERTS + local_expert
    source_start = tl.load(source_input_starts + src_rank * NUM_EXPERTS + global_expert)

    tl.store(dst_ranks + row, src_rank, mask=mask)
    tl.store(dst_rows + row, source_start + offset, mask=mask)


@triton.jit
def _invert_flat_indices_kernel(
    flat_indices: tl.pointer_type(tl.int64),
    slot_to_row: tl.pointer_type(tl.int64),
    NUM_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Invert routed-row to top-k-slot indices into top-k-slot to routed-row.

    Example:
        ``flat_indices=[2, 0, 3, 1]`` means rows 0..3 belong to slots
        2, 0, 3, 1. The output is ``slot_to_row=[1, 3, 0, 2]``.
    """
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < NUM_ROWS
    slot = tl.load(flat_indices + row, mask=mask, other=0)
    tl.store(slot_to_row + slot, row, mask=mask)


@triton.jit
def _reduce_topk_slots_kernel(
    routed_output,
    slot_to_row: tl.pointer_type(tl.int64),
    scores,
    out,
    NUM_COLS: tl.constexpr,
    TOP_K: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    SCORES_ARE_SLOT_ORDERED: tl.constexpr,
    ROUTED_ROW_STRIDE: tl.constexpr,
    ROUTED_COL_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    OUT_COL_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Reduce expert-sorted routed rows back to origin token rows.

    This runs over local origin tokens and their ``TOP_K`` routed slots, not
    the worst-case receive-capacity buffer. Accumulation is done in fp32.

    Example:
        With ``TOP_K=2``, ``routed_output=[[10], [20], [30], [40]]``,
        ``slot_to_row=[1, 3, 0, 2]``, and slot-ordered
        ``scores=[0.1, 0.2, 0.3, 0.4]``, the output is
        ``[[10], [15]]``.
    """
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col < NUM_COLS
    acc = tl.zeros((BLOCK_N,), tl.float32)

    for slot in tl.static_range(0, TOP_K):
        row = tl.load(slot_to_row + token * TOP_K + slot)
        value = tl.load(
            routed_output + row * ROUTED_ROW_STRIDE + col * ROUTED_COL_STRIDE,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_SCORES:
            score_index = token * TOP_K + slot if SCORES_ARE_SLOT_ORDERED else row
            value *= tl.load(scores + score_index).to(tl.float32)
        acc += value

    tl.store(
        out + token * OUT_ROW_STRIDE + col * OUT_COL_STRIDE,
        acc,
        mask=col_mask,
    )


@triton.jit
def _expand_topk_grad_kernel(
    grad_out,
    flat_indices: tl.pointer_type(tl.int64),
    scores,
    grad_routed,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    TOP_K: tl.constexpr,
    HAS_SCORES: tl.constexpr,
    SCORES_ARE_SLOT_ORDERED: tl.constexpr,
    GRAD_OUT_ROW_STRIDE: tl.constexpr,
    GRAD_OUT_COL_STRIDE: tl.constexpr,
    GRAD_ROUTED_ROW_STRIDE: tl.constexpr,
    GRAD_ROUTED_COL_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Expand token gradients into expert-sorted routed-row gradients.

    This is the backward of ``_reduce_topk_slots_kernel`` with respect to the
    routed rows. It covers the local routed rows described by ``flat_indices``.

    Example:
        With ``TOP_K=2``, ``grad_out=[[100], [200]]``,
        ``flat_indices=[2, 0, 3, 1]``, and slot-ordered
        ``scores=[0.1, 0.2, 0.3, 0.4]``, the output is
        ``grad_routed=[[60], [10], [80], [20]]``.
    """
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col < NUM_COLS
    flat_index = tl.load(flat_indices + row)
    token = flat_index // TOP_K
    value = tl.load(
        grad_out + token * GRAD_OUT_ROW_STRIDE + col * GRAD_OUT_COL_STRIDE,
        mask=(row < NUM_ROWS) & col_mask,
        other=0.0,
    )
    if HAS_SCORES:
        score_index = flat_index if SCORES_ARE_SLOT_ORDERED else row
        value = value.to(tl.float32) * tl.load(scores + score_index).to(tl.float32)
    tl.store(
        grad_routed + row * GRAD_ROUTED_ROW_STRIDE + col * GRAD_ROUTED_COL_STRIDE,
        value,
        mask=(row < NUM_ROWS) & col_mask,
    )


@triton.jit
def _topk_scores_grad_kernel(
    routed_output,
    grad_out,
    flat_indices: tl.pointer_type(tl.int64),
    grad_scores,
    NUM_COLS: tl.constexpr,
    TOP_K: tl.constexpr,
    ROUTED_ROW_STRIDE: tl.constexpr,
    ROUTED_COL_STRIDE: tl.constexpr,
    GRAD_OUT_ROW_STRIDE: tl.constexpr,
    GRAD_OUT_COL_STRIDE: tl.constexpr,
    SCORES_ARE_SLOT_ORDERED: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute routing-score gradients for each local routed row.

    The score gradient is the dot product of that routed row's output and the
    gradient of its origin token output.

    Example:
        With ``TOP_K=2``, ``routed_output=[[10], [20], [30], [40]]``,
        ``grad_out=[[100], [200]]``, and
        ``flat_indices=[2, 0, 3, 1]``, the slot-ordered output is
        ``grad_scores=[2000, 4000, 2000, 6000]``.
    """
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    col_mask = col < NUM_COLS
    flat_index = tl.load(flat_indices + row)
    token = flat_index // TOP_K
    routed = tl.load(
        routed_output + row * ROUTED_ROW_STRIDE + col * ROUTED_COL_STRIDE,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    grad = tl.load(
        grad_out + token * GRAD_OUT_ROW_STRIDE + col * GRAD_OUT_COL_STRIDE,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    score_index = flat_index if SCORES_ARE_SLOT_ORDERED else row
    tl.store(grad_scores + score_index, tl.sum(routed * grad, axis=0))


@triton.jit
def _copy_rows_to_peer_ptrs_kernel(
    src,
    dst_ptrs: tl.pointer_type(tl.int64),
    dst_ranks: tl.pointer_type(tl.int64),
    dst_rows: tl.pointer_type(tl.int64),
    num_valid_rows: tl.pointer_type(tl.int64),
    src_rows: tl.pointer_type(tl.int64),
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    SRC_ROW_STRIDE: tl.constexpr,
    SRC_COL_STRIDE: tl.constexpr,
    DST_ROW_STRIDE: tl.constexpr,
    DST_DTYPE: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
    HAS_SRC_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Copy rows into peer symmetric hidden buffers through pointer tables.

    ``dst_ranks`` selects the peer buffer, ``dst_rows`` selects the row within
    that peer buffer, and ``src_rows`` optionally gathers rows from ``src``.
    ``num_valid_rows`` optionally limits the copy to the active prefix.

    Example:
        With ``src=[[10], [20], [30]]``, ``dst_ranks=[1, 0]``,
        ``dst_rows=[3, 4]``, and ``src_rows=[2, 0]``, peer 1 row 3 receives
        ``[30]`` and peer 0 row 4 receives ``[10]``.
    """
    row_start = tl.program_id(0) * BLOCK_M
    row_limit = NUM_ROWS
    if HAS_NUM_VALID_ROWS:
        row_limit = tl.load(num_valid_rows)
        if row_start >= row_limit:
            return
    row = row_start + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row < row_limit
    col_mask = col < NUM_COLS
    mask = row_mask[:, None] & col_mask[None, :]
    src_row = row
    if HAS_SRC_ROWS:
        src_row = tl.load(src_rows + row, mask=row_mask, other=0)
    dst_rank = tl.load(dst_ranks + row, mask=row_mask, other=-1)
    dst_row = tl.load(dst_rows + row, mask=row_mask, other=0)
    dst_rank_mask = row_mask & (dst_rank >= 0)
    values = tl.load(
        src + src_row[:, None] * SRC_ROW_STRIDE + col[None, :] * SRC_COL_STRIDE,
        mask=mask,
    )
    dst_base = tl.load(dst_ptrs + dst_rank, mask=dst_rank_mask, other=0)
    dst_ptr = (
        dst_base.to(tl.pointer_type(DST_DTYPE))[:, None]
        + dst_row[:, None] * DST_ROW_STRIDE
        + col[None, :]
    )
    tl.store(dst_ptr, values, mask=mask & dst_rank_mask[:, None])


def copy_full_counts_to_peers_kernel(
    counts: torch.Tensor,
    dsts: list[torch.Tensor],
    *,
    rank: int,
    ep_size: int,
    num_experts: int,
    dst_ptrs: torch.Tensor,
) -> None:
    if counts.dtype != torch.int64:
        raise ValueError(f"counts must be torch.int64, got {counts.dtype}.")
    if len(dsts) != ep_size:
        raise ValueError(f"expected {ep_size} count buffers, got {len(dsts)}.")
    if dsts[0].dtype != torch.int64:
        raise ValueError(
            "destination count buffers must be torch.int64, " f"got {dsts[0].dtype}."
        )

    block_size = _COUNT_COPY_BLOCK_SIZE
    grid = (ep_size, triton.cdiv(num_experts, block_size))
    _copy_full_counts_to_peer_ptrs_kernel[grid](
        counts,
        dst_ptrs,
        rank=rank,
        EP_SIZE=ep_size,
        NUM_EXPERTS=num_experts,
        DST_ROW_STRIDE=dsts[0].stride(0),
        BLOCK_SIZE=block_size,
    )


def copy_rows_to_peers_kernel(
    src: torch.Tensor,
    dsts: list[torch.Tensor],
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    *,
    ep_size: int,
    num_rows: int,
    num_cols: int,
    dst_ptrs: torch.Tensor,
    block_m: int = 1,
    num_warps: int = 4,
    src_rows: torch.Tensor | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    if len(dsts) != ep_size:
        raise ValueError(f"expected {ep_size} destination buffers, got {len(dsts)}.")

    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(num_cols))
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(num_cols, block_n))
    dst_dtype = _HIDDEN_ROW_DTYPES.get(src.dtype)
    if dst_dtype is None:
        raise ValueError(f"Unsupported MinimalAsyncEP row-copy dtype: {src.dtype}.")
    _copy_rows_to_peer_ptrs_kernel[grid](
        src,
        dst_ptrs,
        dst_ranks,
        dst_rows,
        num_valid_rows if num_valid_rows is not None else dst_rows[:1],
        src_rows if src_rows is not None else dst_rows,
        NUM_ROWS=num_rows,
        NUM_COLS=num_cols,
        SRC_ROW_STRIDE=src.stride(0),
        SRC_COL_STRIDE=src.stride(1),
        DST_ROW_STRIDE=dsts[0].stride(0),
        DST_DTYPE=dst_dtype,
        HAS_NUM_VALID_ROWS=num_valid_rows is not None,
        HAS_SRC_ROWS=src_rows is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


def fill_dispatch_metadata_kernel(
    counts: torch.Tensor,
    local_dest_offsets: torch.Tensor,
    local_count_starts: torch.Tensor,
    *,
    num_routed_tokens: int,
    num_local_experts: int,
    max_tokens_per_segment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = counts.contiguous()
    local_dest_offsets = local_dest_offsets.contiguous()
    local_count_starts = local_count_starts.contiguous()
    dst_ranks = torch.empty(
        num_routed_tokens,
        device=counts.device,
        dtype=torch.int64,
    )
    dst_rows = torch.empty_like(dst_ranks)
    block_size = _METADATA_BLOCK_SIZE
    grid = (
        counts.numel(),
        triton.cdiv(max_tokens_per_segment, block_size),
    )
    _fill_dispatch_metadata_kernel[grid](
        counts,
        local_dest_offsets,
        local_count_starts,
        dst_ranks,
        dst_rows,
        NUM_EXPERTS=counts.numel(),
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return dst_ranks, dst_rows


def fill_combine_metadata_kernel(
    segment_lens: torch.Tensor,
    output_starts: torch.Tensor,
    source_input_starts: torch.Tensor,
    *,
    ep_rank: int,
    ep_size: int,
    num_local_experts: int,
    receive_capacity: int,
    max_tokens_per_segment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_lens = segment_lens.contiguous()
    output_starts = output_starts.contiguous()
    source_input_starts = source_input_starts.contiguous()
    dst_ranks = torch.empty(
        receive_capacity,
        device=segment_lens.device,
        dtype=torch.int64,
    )
    dst_rows = torch.empty_like(dst_ranks)
    block_size = _METADATA_BLOCK_SIZE
    grid = (
        segment_lens.numel(),
        triton.cdiv(max_tokens_per_segment, block_size),
    )
    _fill_combine_metadata_kernel[grid](
        segment_lens,
        output_starts,
        source_input_starts,
        dst_ranks,
        dst_rows,
        ep_rank=ep_rank,
        EP_SIZE=ep_size,
        NUM_LOCAL_EXPERTS=num_local_experts,
        NUM_EXPERTS=ep_size * num_local_experts,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return (
        dst_ranks,
        dst_rows,
        (output_starts[-1:] + segment_lens[-1:]).to(torch.int64),
    )


def invert_flat_indices_kernel(
    flat_indices: torch.Tensor,
    *,
    num_rows: int,
) -> torch.Tensor:
    slot_to_row = flat_indices.new_empty(num_rows)

    block_size = _COUNT_COPY_BLOCK_SIZE
    _invert_flat_indices_kernel[(triton.cdiv(num_rows, block_size),)](
        flat_indices,
        slot_to_row,
        NUM_ROWS=num_rows,
        BLOCK_SIZE=block_size,
    )
    return slot_to_row


def reduce_topk_slots_kernel(
    routed_output: torch.Tensor,
    slot_to_row: torch.Tensor,
    scores: torch.Tensor | None,
    *,
    num_tokens: int,
    top_k: int,
    scores_are_slot_ordered: bool = False,
) -> torch.Tensor:
    num_cols = routed_output.shape[1]
    out = torch.empty(
        num_tokens,
        num_cols,
        device=routed_output.device,
        dtype=routed_output.dtype,
    )
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(num_cols))
    grid = (num_tokens, triton.cdiv(num_cols, block_n))
    _reduce_topk_slots_kernel[grid](
        routed_output,
        slot_to_row,
        scores if scores is not None else slot_to_row,
        out,
        NUM_COLS=num_cols,
        TOP_K=top_k,
        HAS_SCORES=scores is not None,
        SCORES_ARE_SLOT_ORDERED=scores_are_slot_ordered,
        ROUTED_ROW_STRIDE=routed_output.stride(0),
        ROUTED_COL_STRIDE=routed_output.stride(1),
        OUT_ROW_STRIDE=out.stride(0),
        OUT_COL_STRIDE=out.stride(1),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


def expand_topk_grad_kernel(
    grad_out: torch.Tensor,
    flat_indices: torch.Tensor,
    scores: torch.Tensor | None,
    *,
    top_k: int,
    dtype: torch.dtype,
    scores_are_slot_ordered: bool = False,
) -> torch.Tensor:
    num_rows = flat_indices.numel()
    num_cols = grad_out.shape[1]

    grad_routed = torch.empty(
        num_rows,
        num_cols,
        device=grad_out.device,
        dtype=dtype,
    )
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(num_cols))
    grid = (num_rows, triton.cdiv(num_cols, block_n))
    _expand_topk_grad_kernel[grid](
        grad_out,
        flat_indices,
        scores if scores is not None else flat_indices,
        grad_routed,
        NUM_ROWS=num_rows,
        NUM_COLS=num_cols,
        TOP_K=top_k,
        HAS_SCORES=scores is not None,
        SCORES_ARE_SLOT_ORDERED=scores_are_slot_ordered,
        GRAD_OUT_ROW_STRIDE=grad_out.stride(0),
        GRAD_OUT_COL_STRIDE=grad_out.stride(1),
        GRAD_ROUTED_ROW_STRIDE=grad_routed.stride(0),
        GRAD_ROUTED_COL_STRIDE=grad_routed.stride(1),
        BLOCK_N=block_n,
        num_warps=8,
    )
    return grad_routed


def topk_scores_grad_kernel(
    routed_output: torch.Tensor,
    grad_out: torch.Tensor,
    flat_indices: torch.Tensor,
    *,
    top_k: int,
    dtype: torch.dtype,
    scores_are_slot_ordered: bool = False,
) -> torch.Tensor:
    num_rows, num_cols = routed_output.shape
    grad_scores = torch.empty(
        num_rows,
        device=routed_output.device,
        dtype=dtype,
    )
    block_n = triton.next_power_of_2(num_cols)
    _topk_scores_grad_kernel[(num_rows,)](
        routed_output,
        grad_out,
        flat_indices,
        grad_scores,
        NUM_COLS=num_cols,
        TOP_K=top_k,
        ROUTED_ROW_STRIDE=routed_output.stride(0),
        ROUTED_COL_STRIDE=routed_output.stride(1),
        GRAD_OUT_ROW_STRIDE=grad_out.stride(0),
        GRAD_OUT_COL_STRIDE=grad_out.stride(1),
        SCORES_ARE_SLOT_ORDERED=scores_are_slot_ordered,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return grad_scores
