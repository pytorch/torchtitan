# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import linecache

import torch
import triton
import triton.language as tl


_copy_full_counts_to_peers_kernel_cache: dict[int, object] = {}
_copy_rows_to_peers_kernel_cache: dict[int, object] = {}
_copy_rows_to_peer_ptrs_kernel_cache: dict[torch.dtype, object] = {}

_POINTER_TABLE_EP_THRESHOLD = 8


_TRITON_DTYPE_NAMES = {
    torch.bool: "tl.int1",
    torch.uint8: "tl.uint8",
    torch.int8: "tl.int8",
    torch.int16: "tl.int16",
    torch.int32: "tl.int32",
    torch.int64: "tl.int64",
    torch.float16: "tl.float16",
    torch.bfloat16: "tl.bfloat16",
    torch.float32: "tl.float32",
    torch.float64: "tl.float64",
}


@triton.jit
def _count_expert_ids_kernel(
    expert_ids,
    counts,
    NUM_IDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < NUM_IDS
    expert = tl.load(expert_ids + offset, mask=mask, other=0)
    tl.atomic_add(counts + expert, 1, mask=mask, sem="relaxed")


@triton.jit
def _scatter_expert_indices_kernel(
    expert_ids,
    starts,
    write_offsets,
    token_indices,
    flat_indices,
    NUM_IDS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < NUM_IDS
    expert = tl.load(expert_ids + offset, mask=mask, other=0)
    local_offset = tl.atomic_add(write_offsets + expert, 1, mask=mask, sem="relaxed")
    dst = tl.load(starts + expert, mask=mask, other=0) + local_offset
    tl.store(flat_indices + dst, offset, mask=mask)
    tl.store(token_indices + dst, offset // TOP_K, mask=mask)


@triton.jit
def _copy_full_counts_to_peer_ptrs_kernel(
    counts,
    dst_ptrs,
    RANK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    DST_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    peer = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (peer < EP_SIZE) & (offsets < NUM_EXPERTS)
    base = tl.load(dst_ptrs + peer, mask=peer < EP_SIZE, other=0)
    dst = (base + (RANK * DST_ROW_STRIDE + offsets) * 8).to(
        tl.pointer_type(tl.int64)
    )
    values = tl.load(counts + offsets, mask=mask, other=0)
    tl.store(dst, values, mask=mask)


@triton.jit
def _fill_dispatch_metadata_kernel(
    counts,
    local_dest_offsets,
    local_count_starts,
    dst_ranks,
    dst_rows,
    NUM_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    MAX_TOKENS_PER_SEGMENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    expert = tl.program_id(0)
    block = tl.program_id(1)
    offset = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    count = tl.load(counts + expert)
    mask = (expert < NUM_EXPERTS) & (offset < count)
    row = tl.load(local_count_starts + expert) + offset
    tl.store(dst_ranks + row, expert // NUM_LOCAL_EXPERTS, mask=mask)
    tl.store(dst_rows + row, tl.load(local_dest_offsets + expert) + offset, mask=mask)


@triton.jit
def _fill_combine_metadata_kernel(
    segment_lens,
    output_starts,
    source_input_starts,
    dst_ranks,
    dst_rows,
    EP_RANK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_TOKENS_PER_SEGMENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    segment = tl.program_id(0)
    block = tl.program_id(1)
    offset = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    count = tl.load(segment_lens + segment)
    mask = offset < count

    output_start = tl.load(output_starts + segment)
    row = output_start + offset
    src_rank = segment % EP_SIZE
    local_expert = segment // EP_SIZE
    global_expert = EP_RANK * NUM_LOCAL_EXPERTS + local_expert
    source_start = tl.load(source_input_starts + src_rank * NUM_EXPERTS + global_expert)

    tl.store(dst_ranks + row, src_rank, mask=mask)
    tl.store(dst_rows + row, source_start + offset, mask=mask)


@triton.jit
def _invert_flat_indices_kernel(
    flat_indices,
    slot_to_row,
    NUM_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row < NUM_ROWS
    slot = tl.load(flat_indices + row, mask=mask, other=0)
    tl.store(slot_to_row + slot, row, mask=mask)


@triton.jit
def _reduce_topk_slots_kernel(
    routed_output,
    slot_to_row,
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
    flat_indices,
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
        value *= tl.load(scores + score_index)
    tl.store(
        grad_routed + row * GRAD_ROUTED_ROW_STRIDE + col * GRAD_ROUTED_COL_STRIDE,
        value,
        mask=(row < NUM_ROWS) & col_mask,
    )


@triton.jit
def _topk_scores_grad_kernel(
    routed_output,
    grad_out,
    flat_indices,
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
def _active_swiglu_forward_kernel(
    gate,
    up,
    out,
    active_rows_ptr,
    NUM_COLS: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    UP_ROW_STRIDE: tl.constexpr,
    UP_COL_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    OUT_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row_start = tl.program_id(0) * BLOCK_M
    active_rows = tl.load(active_rows_ptr)
    if row_start >= active_rows:
        return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < active_rows) & (cols[None, :] < NUM_COLS)

    gate_values = tl.load(
        gate + rows[:, None] * GATE_ROW_STRIDE + cols[None, :] * GATE_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * UP_ROW_STRIDE + cols[None, :] * UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu = gate_values * tl.sigmoid(gate_values)
    tl.store(
        out + rows[:, None] * OUT_ROW_STRIDE + cols[None, :] * OUT_COL_STRIDE,
        silu * up_values,
        mask=mask,
    )


@triton.jit
def _active_swiglu_backward_kernel(
    grad_out,
    gate,
    up,
    grad_gate,
    grad_up,
    active_rows_ptr,
    NUM_COLS: tl.constexpr,
    GRAD_OUT_ROW_STRIDE: tl.constexpr,
    GRAD_OUT_COL_STRIDE: tl.constexpr,
    GATE_ROW_STRIDE: tl.constexpr,
    GATE_COL_STRIDE: tl.constexpr,
    UP_ROW_STRIDE: tl.constexpr,
    UP_COL_STRIDE: tl.constexpr,
    GRAD_GATE_ROW_STRIDE: tl.constexpr,
    GRAD_GATE_COL_STRIDE: tl.constexpr,
    GRAD_UP_ROW_STRIDE: tl.constexpr,
    GRAD_UP_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row_start = tl.program_id(0) * BLOCK_M
    active_rows = tl.load(active_rows_ptr)
    if row_start >= active_rows:
        return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < active_rows) & (cols[None, :] < NUM_COLS)

    grad_values = tl.load(
        grad_out
        + rows[:, None] * GRAD_OUT_ROW_STRIDE
        + cols[None, :] * GRAD_OUT_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_values = tl.load(
        gate + rows[:, None] * GATE_ROW_STRIDE + cols[None, :] * GATE_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up + rows[:, None] * UP_ROW_STRIDE + cols[None, :] * UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = tl.sigmoid(gate_values)
    silu = gate_values * sigmoid
    silu_grad = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))

    tl.store(
        grad_gate
        + rows[:, None] * GRAD_GATE_ROW_STRIDE
        + cols[None, :] * GRAD_GATE_COL_STRIDE,
        grad_values * up_values * silu_grad,
        mask=mask,
    )
    tl.store(
        grad_up
        + rows[:, None] * GRAD_UP_ROW_STRIDE
        + cols[None, :] * GRAD_UP_COL_STRIDE,
        grad_values * silu,
        mask=mask,
    )


def _make_copy_full_counts_to_peers_kernel(ep_size: int):
    kernel = _copy_full_counts_to_peers_kernel_cache.get(ep_size)
    if kernel is not None:
        return kernel

    dst_params = ",\n    ".join(f"dst{peer}" for peer in range(ep_size))
    copy_blocks = []
    for peer in range(ep_size):
        copy_blocks.append(
            f"""
    values_{peer} = tl.load(counts + offsets, mask=mask, other=0)
    tl.store(
        dst{peer} + RANK * DST_ROW_STRIDE + offsets,
        values_{peer},
        mask=mask,
    )
"""
        )

    source = f"""
@triton.jit
def _copy_full_counts_to_peers_kernel_ep{ep_size}(
    counts,
    {dst_params},
    RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    DST_ROW_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_EXPERTS
{''.join(copy_blocks)}
"""
    filename = f"<torchtitan_minimal_async_ep_copy_full_counts_to_peers_ep{ep_size}>"
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    namespace = {
        "triton": triton,
        "tl": tl,
    }
    exec(compile(source, filename, "exec"), namespace)
    kernel = namespace[f"_copy_full_counts_to_peers_kernel_ep{ep_size}"]
    _copy_full_counts_to_peers_kernel_cache[ep_size] = kernel
    return kernel


def _make_copy_rows_to_peers_kernel(ep_size: int):
    kernel = _copy_rows_to_peers_kernel_cache.get(ep_size)
    if kernel is not None:
        return kernel

    dst_params = ",\n    ".join(f"dst{peer}" for peer in range(ep_size))
    dst_ptr_init = "dst0 + dst_row[:, None] * DST_ROW_STRIDE + col[None, :]"
    dst_ptr_select = []
    for peer in range(1, ep_size):
        dst_ptr_select.append(
            f"""
    dst_ptr = tl.where(
        dst_rank[:, None] == {peer},
        dst{peer} + dst_row[:, None] * DST_ROW_STRIDE + col[None, :],
        dst_ptr,
    )
"""
        )

    source = f"""
@triton.jit
def _copy_rows_to_peers_kernel_ep{ep_size}(
    src,
    {dst_params},
    dst_ranks,
    dst_rows,
    valid_rows,
    num_valid_rows,
    src_rows,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    SRC_ROW_STRIDE: tl.constexpr,
    SRC_COL_STRIDE: tl.constexpr,
    DST_ROW_STRIDE: tl.constexpr,
    HAS_VALID_ROWS: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
    HAS_SRC_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
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
    row_valid = row_mask
    if HAS_VALID_ROWS:
        row_valid = tl.load(valid_rows + row, mask=row_mask, other=0) != 0
    mask = row_valid[:, None] & col_mask[None, :]
    src_row = row
    if HAS_SRC_ROWS:
        src_row = tl.load(src_rows + row, mask=row_valid, other=0)
    dst_rank = tl.load(dst_ranks + row, mask=row_valid, other=-1)
    dst_row = tl.load(dst_rows + row, mask=row_valid, other=0)
    values = tl.load(
        src
        + src_row[:, None] * SRC_ROW_STRIDE
        + col[None, :] * SRC_COL_STRIDE,
        mask=mask,
    )
    dst_ptr = {dst_ptr_init}
{''.join(dst_ptr_select)}
    tl.store(dst_ptr, values, mask=mask & (dst_rank[:, None] >= 0))
"""
    filename = f"<torchtitan_minimal_async_ep_copy_rows_to_peers_ep{ep_size}>"
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    namespace = {
        "triton": triton,
        "tl": tl,
    }
    exec(compile(source, filename, "exec"), namespace)
    kernel = namespace[f"_copy_rows_to_peers_kernel_ep{ep_size}"]
    _copy_rows_to_peers_kernel_cache[ep_size] = kernel
    return kernel


def _make_copy_rows_to_peer_ptrs_kernel(dtype: torch.dtype):
    kernel = _copy_rows_to_peer_ptrs_kernel_cache.get(dtype)
    if kernel is not None:
        return kernel

    dtype_name = _TRITON_DTYPE_NAMES.get(dtype)
    if dtype_name is None:
        raise ValueError(f"Unsupported MinimalAsyncEP row-copy dtype: {dtype}.")
    element_size = torch.empty((), dtype=dtype).element_size()

    source = f"""
@triton.jit
def _copy_rows_to_peer_ptrs_kernel_{str(dtype).replace('.', '_')}(
    src,
    dst_ptrs,
    dst_ranks,
    dst_rows,
    valid_rows,
    num_valid_rows,
    src_rows,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    SRC_ROW_STRIDE: tl.constexpr,
    SRC_COL_STRIDE: tl.constexpr,
    DST_ROW_STRIDE: tl.constexpr,
    HAS_VALID_ROWS: tl.constexpr,
    HAS_NUM_VALID_ROWS: tl.constexpr,
    HAS_SRC_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
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
    row_valid = row_mask
    if HAS_VALID_ROWS:
        row_valid = tl.load(valid_rows + row, mask=row_mask, other=0) != 0
    mask = row_valid[:, None] & col_mask[None, :]
    src_row = row
    if HAS_SRC_ROWS:
        src_row = tl.load(src_rows + row, mask=row_valid, other=0)
    dst_rank = tl.load(dst_ranks + row, mask=row_valid, other=-1)
    dst_row = tl.load(dst_rows + row, mask=row_valid, other=0)
    values = tl.load(
        src
        + src_row[:, None] * SRC_ROW_STRIDE
        + col[None, :] * SRC_COL_STRIDE,
        mask=mask,
    )
    dst_base = tl.load(dst_ptrs + dst_rank, mask=row_valid, other=0)
    dst_byte_offset = (
        (dst_row[:, None] * DST_ROW_STRIDE + col[None, :]) * {element_size}
    )
    dst_ptr = (dst_base[:, None] + dst_byte_offset).to(tl.pointer_type({dtype_name}))
    tl.store(dst_ptr, values, mask=mask & (dst_rank[:, None] >= 0))
"""
    filename = f"<torchtitan_minimal_async_ep_copy_rows_to_peer_ptrs_{dtype}>"
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)
    namespace = {
        "triton": triton,
        "tl": tl,
    }
    exec(compile(source, filename, "exec"), namespace)
    kernel_name = f"_copy_rows_to_peer_ptrs_kernel_{str(dtype).replace('.', '_')}"
    kernel = namespace[kernel_name]
    _copy_rows_to_peer_ptrs_kernel_cache[dtype] = kernel
    return kernel


def expert_counting_sort(
    topk_expert_ids: torch.Tensor,
    *,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group flattened top-k slots by expert id without a global sort.

    Returns ``(counts, token_indices, flat_indices)``. Ordering within each
    expert segment is unspecified; callers must use ``flat_indices`` for the
    inverse mapping instead of relying on stable order.
    """
    if topk_expert_ids.ndim != 2:
        raise ValueError("expert_counting_sort expects a 2D topk_expert_ids tensor.")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}.")
    if topk_expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_counting_sort expects integer expert ids.")

    num_ids = topk_expert_ids.numel()
    top_k = topk_expert_ids.shape[1]
    flat_expert_ids = topk_expert_ids.reshape(-1)
    if not topk_expert_ids.is_cuda:
        flat_indices = torch.argsort(flat_expert_ids, stable=True)
        counts = torch.bincount(flat_expert_ids, minlength=num_experts).to(torch.int64)
        return counts, flat_indices // top_k, flat_indices

    counts = torch.zeros(
        num_experts,
        dtype=torch.int64,
        device=topk_expert_ids.device,
    )
    block_size = 256
    grid = (triton.cdiv(num_ids, block_size),)
    _count_expert_ids_kernel[grid](
        flat_expert_ids,
        counts,
        NUM_IDS=num_ids,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    starts = counts.cumsum(0) - counts
    write_offsets = torch.zeros_like(counts)
    token_indices = torch.empty(
        num_ids,
        dtype=torch.int64,
        device=topk_expert_ids.device,
    )
    flat_indices = torch.empty_like(token_indices)
    _scatter_expert_indices_kernel[grid](
        flat_expert_ids,
        starts,
        write_offsets,
        token_indices,
        flat_indices,
        NUM_IDS=num_ids,
        TOP_K=top_k,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return counts, token_indices, flat_indices


def copy_full_counts_to_peers(
    counts: torch.Tensor,
    dsts: list[torch.Tensor],
    *,
    rank: int,
    ep_size: int,
    num_experts: int,
    dst_ptrs: torch.Tensor | None = None,
) -> None:
    if counts.ndim != 1:
        raise ValueError("copy_full_counts_to_peers expects a 1D counts tensor.")
    if len(dsts) != ep_size:
        raise ValueError(f"Expected {ep_size} destination tensors, got {len(dsts)}.")
    if any(dst.ndim != 2 for dst in dsts):
        raise ValueError("copy_full_counts_to_peers expects 2D destination tensors.")
    if counts.numel() < num_experts:
        raise ValueError(
            "copy_full_counts_to_peers counts tensor is too small. Got "
            f"{counts.numel()} values for num_experts={num_experts}."
        )
    if counts.dtype != torch.int64:
        raise ValueError("copy_full_counts_to_peers expects int64 counts.")
    if any(dst.dtype != counts.dtype for dst in dsts):
        raise ValueError("copy_full_counts_to_peers destination dtype mismatch.")
    if dst_ptrs is not None:
        if dst_ptrs.ndim != 1:
            raise ValueError("copy_full_counts_to_peers dst_ptrs must be 1D.")
        if dst_ptrs.numel() < ep_size:
            raise ValueError("copy_full_counts_to_peers dst_ptrs tensor is too small.")
        if dst_ptrs.dtype != torch.int64:
            raise ValueError("copy_full_counts_to_peers dst_ptrs must be int64.")
        if dst_ptrs.device != counts.device:
            raise ValueError(
                "copy_full_counts_to_peers dst_ptrs must match counts device."
            )

    if not counts.is_cuda:
        for dst in dsts:
            dst[rank, :num_experts].copy_(counts[:num_experts])
        return

    use_pointer_table = ep_size > _POINTER_TABLE_EP_THRESHOLD
    if dst_ptrs is None and use_pointer_table:
        dst_ptrs = torch.tensor(
            [dst.data_ptr() for dst in dsts],
            dtype=torch.int64,
            device=counts.device,
        )
    block_size = 1024
    if use_pointer_table:
        assert dst_ptrs is not None
        grid = (ep_size, triton.cdiv(num_experts, block_size))
        _copy_full_counts_to_peer_ptrs_kernel[grid](
            counts,
            dst_ptrs,
            RANK=rank,
            EP_SIZE=ep_size,
            NUM_EXPERTS=num_experts,
            DST_ROW_STRIDE=dsts[0].stride(0),
            BLOCK_SIZE=block_size,
        )
    else:
        grid = (triton.cdiv(num_experts, block_size),)
        kernel = _make_copy_full_counts_to_peers_kernel(ep_size)
        kernel[grid](
            counts,
            *dsts,
            RANK=rank,
            NUM_EXPERTS=num_experts,
            DST_ROW_STRIDE=dsts[0].stride(0),
            BLOCK_SIZE=block_size,
        )


def _validate_active_swiglu_args(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> None:
    if gate.ndim != 2 or up.ndim != 2:
        raise ValueError("active_swiglu expects 2D gate and up tensors.")
    if gate.shape != up.shape:
        raise ValueError(
            "active_swiglu gate and up tensors must have the same shape. Got "
            f"{gate.shape} and {up.shape}."
        )
    if gate.device != up.device or gate.device != active_rows.device:
        raise ValueError("active_swiglu tensors must be on the same device.")
    if active_rows.ndim != 1 or active_rows.numel() < 1:
        raise ValueError("active_swiglu active_rows must be a non-empty 1D tensor.")
    if active_rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("active_swiglu active_rows must be int32 or int64.")


def active_swiglu_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute ``silu(gate) * up`` only for rows before ``active_rows[0]``.

    Rows at or after ``active_rows[0]`` in the returned tensor are unspecified.
    MinimalAsyncEP only consumes rows covered by the same grouped-mm offsets.
    """
    _validate_active_swiglu_args(gate, up, active_rows)
    out = torch.empty_like(gate)
    if not gate.is_cuda:
        out.copy_(torch.nn.functional.silu(gate) * up)
        return out

    block_m = 4
    block_n = min(2048, triton.next_power_of_2(gate.shape[1]))
    grid = (triton.cdiv(gate.shape[0], block_m), triton.cdiv(gate.shape[1], block_n))
    _active_swiglu_forward_kernel[grid](
        gate,
        up,
        out,
        active_rows,
        NUM_COLS=gate.shape[1],
        GATE_ROW_STRIDE=gate.stride(0),
        GATE_COL_STRIDE=gate.stride(1),
        UP_ROW_STRIDE=up.stride(0),
        UP_COL_STRIDE=up.stride(1),
        OUT_ROW_STRIDE=out.stride(0),
        OUT_COL_STRIDE=out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


def active_swiglu_backward(
    grad_out: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    active_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_active_swiglu_args(gate, up, active_rows)
    if grad_out.shape != gate.shape:
        raise ValueError(
            "active_swiglu grad_out must match gate/up shape. Got "
            f"{grad_out.shape} and {gate.shape}."
        )
    if grad_out.device != gate.device:
        raise ValueError("active_swiglu grad_out must be on the same device.")

    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    if not gate.is_cuda:
        with torch.enable_grad():
            gate_ref = gate.detach().requires_grad_(True)
            up_ref = up.detach().requires_grad_(True)
            out = torch.nn.functional.silu(gate_ref) * up_ref
            out.backward(grad_out)
        assert gate_ref.grad is not None
        assert up_ref.grad is not None
        grad_gate.copy_(gate_ref.grad)
        grad_up.copy_(up_ref.grad)
        return grad_gate, grad_up

    block_m = 4
    block_n = min(2048, triton.next_power_of_2(gate.shape[1]))
    grid = (triton.cdiv(gate.shape[0], block_m), triton.cdiv(gate.shape[1], block_n))
    _active_swiglu_backward_kernel[grid](
        grad_out,
        gate,
        up,
        grad_gate,
        grad_up,
        active_rows,
        NUM_COLS=gate.shape[1],
        GRAD_OUT_ROW_STRIDE=grad_out.stride(0),
        GRAD_OUT_COL_STRIDE=grad_out.stride(1),
        GATE_ROW_STRIDE=gate.stride(0),
        GATE_COL_STRIDE=gate.stride(1),
        UP_ROW_STRIDE=up.stride(0),
        UP_COL_STRIDE=up.stride(1),
        GRAD_GATE_ROW_STRIDE=grad_gate.stride(0),
        GRAD_GATE_COL_STRIDE=grad_gate.stride(1),
        GRAD_UP_ROW_STRIDE=grad_up.stride(0),
        GRAD_UP_COL_STRIDE=grad_up.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return grad_gate, grad_up


def copy_rows_to_peers(
    src: torch.Tensor,
    dsts: list[torch.Tensor],
    dst_ranks: torch.Tensor,
    dst_rows: torch.Tensor,
    valid_rows: torch.Tensor | None,
    *,
    ep_size: int,
    num_rows: int,
    num_cols: int,
    block_m: int = 1,
    num_warps: int = 4,
    src_rows: torch.Tensor | None = None,
    dst_ptrs: torch.Tensor | None = None,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    if src.ndim != 2:
        raise ValueError("copy_rows_to_peers expects a 2D source tensor.")
    if len(dsts) != ep_size:
        raise ValueError(f"Expected {ep_size} destination tensors, got {len(dsts)}.")
    if any(dst.ndim != 2 for dst in dsts):
        raise ValueError("copy_rows_to_peers expects 2D destination tensors.")
    if dst_ranks.ndim != 1 or dst_rows.ndim != 1:
        raise ValueError("copy_rows_to_peers expects 1D routing tensors.")
    if dst_ranks.device != src.device or dst_rows.device != src.device:
        raise ValueError("copy_rows_to_peers routing tensors must match src device.")
    if dst_ranks.dtype != torch.int64 or dst_rows.dtype != torch.int64:
        raise ValueError("copy_rows_to_peers routing tensors must be int64.")
    if valid_rows is not None:
        if valid_rows.ndim != 1:
            raise ValueError("copy_rows_to_peers valid_rows must be 1D.")
        if valid_rows.device != src.device:
            raise ValueError("copy_rows_to_peers valid_rows must match src device.")
    if num_valid_rows is not None:
        if num_valid_rows.shape != (1,):
            raise ValueError("copy_rows_to_peers num_valid_rows must have shape (1,).")
        if num_valid_rows.device != src.device:
            raise ValueError("copy_rows_to_peers num_valid_rows must match src device.")
        if num_valid_rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("copy_rows_to_peers num_valid_rows must be integer.")
    if src_rows is not None:
        if src_rows.ndim != 1:
            raise ValueError("copy_rows_to_peers src_rows must be 1D.")
        if src_rows.device != src.device:
            raise ValueError("copy_rows_to_peers src_rows must match src device.")
        if src_rows.dtype != torch.int64:
            raise ValueError("copy_rows_to_peers src_rows must be int64.")
    if dst_ptrs is not None:
        if dst_ptrs.ndim != 1:
            raise ValueError("copy_rows_to_peers dst_ptrs must be 1D.")
        if dst_ptrs.numel() < ep_size:
            raise ValueError("copy_rows_to_peers dst_ptrs tensor is too small.")
        if dst_ptrs.dtype != torch.int64:
            raise ValueError("copy_rows_to_peers dst_ptrs must be int64.")
        if dst_ptrs.device != src.device:
            raise ValueError("copy_rows_to_peers dst_ptrs must match src device.")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}.")
    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}.")
    if num_cols <= 0:
        raise ValueError(f"num_cols must be positive, got {num_cols}.")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}.")
    if num_warps <= 0:
        raise ValueError(f"num_warps must be positive, got {num_warps}.")
    if dst_ranks.numel() < num_rows or dst_rows.numel() < num_rows:
        raise ValueError("copy_rows_to_peers routing tensors are too small.")
    if valid_rows is not None and valid_rows.numel() < num_rows:
        raise ValueError("copy_rows_to_peers valid_rows tensor is too small.")
    if src_rows is not None and src_rows.numel() < num_rows:
        raise ValueError("copy_rows_to_peers src_rows tensor is too small.")

    if not src.is_cuda:
        row_limit = (
            num_rows
            if num_valid_rows is None
            else min(num_rows, int(num_valid_rows.item()))
        )
        valid = (
            torch.arange(num_rows) < row_limit
            if valid_rows is None
            else valid_rows[:num_rows].to(torch.bool).cpu()
        )
        for row in range(num_rows):
            if not bool(valid[row]):
                continue
            src_row = row if src_rows is None else int(src_rows[row])
            dsts[int(dst_ranks[row])][int(dst_rows[row]), :num_cols].copy_(
                src[src_row, :num_cols]
            )
        return

    use_pointer_table = ep_size > _POINTER_TABLE_EP_THRESHOLD
    if dst_ptrs is None and use_pointer_table:
        dst_ptrs = torch.tensor(
            [dst.data_ptr() for dst in dsts],
            dtype=torch.int64,
            device=src.device,
        )
    block_n = min(2048, triton.next_power_of_2(num_cols))
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(num_cols, block_n))
    if use_pointer_table:
        assert dst_ptrs is not None
        kernel = _make_copy_rows_to_peer_ptrs_kernel(src.dtype)
        kernel[grid](
            src,
            dst_ptrs,
            dst_ranks,
            dst_rows,
            valid_rows if valid_rows is not None else dst_rows,
            num_valid_rows if num_valid_rows is not None else dst_rows[:1],
            src_rows if src_rows is not None else dst_rows,
            NUM_ROWS=num_rows,
            NUM_COLS=num_cols,
            SRC_ROW_STRIDE=src.stride(0),
            SRC_COL_STRIDE=src.stride(1),
            DST_ROW_STRIDE=dsts[0].stride(0),
            HAS_VALID_ROWS=valid_rows is not None,
            HAS_NUM_VALID_ROWS=num_valid_rows is not None,
            HAS_SRC_ROWS=src_rows is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
    else:
        kernel = _make_copy_rows_to_peers_kernel(ep_size)
        kernel[grid](
            src,
            *dsts,
            dst_ranks,
            dst_rows,
            valid_rows if valid_rows is not None else dst_rows,
            num_valid_rows if num_valid_rows is not None else dst_rows[:1],
            src_rows if src_rows is not None else dst_rows,
            NUM_ROWS=num_rows,
            NUM_COLS=num_cols,
            SRC_ROW_STRIDE=src.stride(0),
            SRC_COL_STRIDE=src.stride(1),
            DST_ROW_STRIDE=dsts[0].stride(0),
            HAS_VALID_ROWS=valid_rows is not None,
            HAS_NUM_VALID_ROWS=num_valid_rows is not None,
            HAS_SRC_ROWS=src_rows is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )


def fill_dispatch_metadata(
    counts: torch.Tensor,
    local_dest_offsets: torch.Tensor,
    local_count_starts: torch.Tensor,
    *,
    num_routed_tokens: int,
    num_local_experts: int,
    max_tokens_per_segment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if counts.ndim != 1 or local_dest_offsets.ndim != 1 or local_count_starts.ndim != 1:
        raise ValueError("fill_dispatch_metadata expects 1D input tensors.")
    if not counts.is_cuda:
        positions = torch.arange(num_routed_tokens, device=counts.device)
        local_count_ends = local_count_starts + counts
        experts = torch.searchsorted(local_count_ends, positions, right=True)
        experts = experts.clamp(max=counts.numel() - 1)
        return (
            (experts // num_local_experts).to(torch.int64),
            (local_dest_offsets[experts] + positions - local_count_starts[experts]).to(
                torch.int64
            ),
        )

    dst_ranks = torch.empty(
        num_routed_tokens,
        device=counts.device,
        dtype=torch.int64,
    )
    dst_rows = torch.empty_like(dst_ranks)
    block_size = 256
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
        MAX_TOKENS_PER_SEGMENT=max_tokens_per_segment,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return dst_ranks, dst_rows


def fill_combine_metadata(
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
    if segment_lens.ndim != 1 or output_starts.ndim != 1:
        raise ValueError("fill_combine_metadata expects 1D segment tensors.")
    if source_input_starts.ndim != 2:
        raise ValueError("fill_combine_metadata expects 2D source_input_starts.")
    if not segment_lens.is_cuda:
        receive_positions = torch.arange(receive_capacity, device=segment_lens.device)
        output_ends = output_starts + segment_lens
        segment_ids = torch.searchsorted(output_ends, receive_positions, right=True)
        segment_ids = segment_ids.clamp(max=segment_lens.numel() - 1)
        dst_ranks = segment_ids % ep_size
        local_experts = segment_ids // ep_size
        within_segment = receive_positions - output_starts[segment_ids]
        global_experts = ep_rank * num_local_experts + local_experts
        dst_rows = (
            source_input_starts[dst_ranks, global_experts] + within_segment
        )
        return (
            dst_ranks.to(torch.int64),
            dst_rows.to(torch.int64),
            output_ends[-1:].to(torch.int64),
        )

    dst_ranks = torch.empty(
        receive_capacity,
        device=segment_lens.device,
        dtype=torch.int64,
    )
    dst_rows = torch.empty_like(dst_ranks)
    block_size = 256
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
        EP_RANK=ep_rank,
        EP_SIZE=ep_size,
        NUM_LOCAL_EXPERTS=num_local_experts,
        NUM_EXPERTS=ep_size * num_local_experts,
        MAX_TOKENS_PER_SEGMENT=max_tokens_per_segment,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return dst_ranks, dst_rows, (output_starts[-1:] + segment_lens[-1:]).to(torch.int64)


def invert_flat_indices(
    flat_indices: torch.Tensor,
    *,
    num_rows: int,
) -> torch.Tensor:
    if flat_indices.ndim != 1:
        raise ValueError("invert_flat_indices expects a 1D tensor.")
    if flat_indices.numel() != num_rows:
        raise ValueError(
            f"Expected {num_rows} flat indices, got {flat_indices.numel()}."
        )
    if flat_indices.dtype != torch.int64:
        raise ValueError("invert_flat_indices expects int64 indices.")

    slot_to_row = torch.empty_like(flat_indices)
    if not flat_indices.is_cuda:
        slot_to_row[flat_indices] = torch.arange(
            num_rows,
            device=flat_indices.device,
            dtype=flat_indices.dtype,
        )
        return slot_to_row

    block_size = 1024
    _invert_flat_indices_kernel[(triton.cdiv(num_rows, block_size),)](
        flat_indices,
        slot_to_row,
        NUM_ROWS=num_rows,
        BLOCK_SIZE=block_size,
    )
    return slot_to_row


def reduce_topk_slots(
    routed_output: torch.Tensor,
    slot_to_row: torch.Tensor,
    scores: torch.Tensor | None,
    *,
    num_tokens: int,
    top_k: int,
    scores_are_slot_ordered: bool = False,
) -> torch.Tensor:
    if routed_output.ndim != 2:
        raise ValueError("reduce_topk_slots expects a 2D routed_output tensor.")
    if slot_to_row.ndim != 1:
        raise ValueError("reduce_topk_slots expects a 1D slot_to_row tensor.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    num_rows, num_cols = routed_output.shape
    if num_tokens * top_k != num_rows:
        raise ValueError(
            "num_tokens * top_k must match routed rows. Got "
            f"{num_tokens} * {top_k} != {num_rows}."
        )
    if slot_to_row.numel() != num_rows:
        raise ValueError("slot_to_row length must match routed rows.")
    if slot_to_row.device != routed_output.device:
        raise ValueError("slot_to_row must be on the routed_output device.")
    if slot_to_row.dtype != torch.int64:
        raise ValueError("slot_to_row must be int64.")
    if scores is not None:
        if scores.ndim != 1:
            raise ValueError("scores must be 1D.")
        if scores.numel() != num_rows:
            raise ValueError("scores length must match routed rows.")
        if scores.device != routed_output.device:
            raise ValueError("scores must be on the routed_output device.")

    if not routed_output.is_cuda:
        values = routed_output[slot_to_row].reshape(num_tokens, top_k, num_cols)
        if scores is not None:
            if scores_are_slot_ordered:
                values = values * scores.reshape(num_tokens, top_k, 1)
            else:
                values = values * scores[slot_to_row].reshape(num_tokens, top_k, 1)
        return values.sum(dim=1)

    out = torch.empty(
        num_tokens,
        num_cols,
        device=routed_output.device,
        dtype=routed_output.dtype,
    )
    block_n = min(2048, triton.next_power_of_2(num_cols))
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


def expand_topk_grad(
    grad_out: torch.Tensor,
    flat_indices: torch.Tensor,
    scores: torch.Tensor | None,
    *,
    top_k: int,
    dtype: torch.dtype,
    scores_are_slot_ordered: bool = False,
) -> torch.Tensor:
    if grad_out.ndim != 2:
        raise ValueError("expand_topk_grad expects a 2D grad_out tensor.")
    if flat_indices.ndim != 1:
        raise ValueError("expand_topk_grad expects 1D flat_indices.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    if flat_indices.dtype != torch.int64:
        raise ValueError("flat_indices must be int64.")
    if flat_indices.device != grad_out.device:
        raise ValueError("flat_indices must be on the grad_out device.")
    if scores is not None and scores.device != grad_out.device:
        raise ValueError("scores must be on the grad_out device.")

    num_rows = flat_indices.numel()
    num_cols = grad_out.shape[1]
    if not grad_out.is_cuda:
        grad_routed = grad_out[flat_indices // top_k].to(dtype)
        if scores is not None:
            score_values = scores[flat_indices] if scores_are_slot_ordered else scores
            grad_routed = grad_routed * score_values.reshape(-1, 1)
        return grad_routed

    grad_routed = torch.empty(
        num_rows,
        num_cols,
        device=grad_out.device,
        dtype=dtype,
    )
    block_n = min(2048, triton.next_power_of_2(num_cols))
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


def topk_scores_grad(
    routed_output: torch.Tensor,
    grad_out: torch.Tensor,
    flat_indices: torch.Tensor,
    *,
    top_k: int,
    dtype: torch.dtype,
    scores_are_slot_ordered: bool = False,
) -> torch.Tensor:
    if routed_output.ndim != 2 or grad_out.ndim != 2:
        raise ValueError("topk_scores_grad expects 2D tensors.")
    if flat_indices.ndim != 1:
        raise ValueError("topk_scores_grad expects 1D flat_indices.")
    if routed_output.shape[1] != grad_out.shape[1]:
        raise ValueError("routed_output and grad_out hidden dimensions must match.")
    if flat_indices.numel() != routed_output.shape[0]:
        raise ValueError("flat_indices length must match routed rows.")
    if flat_indices.device != routed_output.device:
        raise ValueError("flat_indices must be on the routed_output device.")
    if grad_out.device != routed_output.device:
        raise ValueError("grad_out must be on the routed_output device.")

    if not routed_output.is_cuda:
        grad_scores_by_row = (
            routed_output.to(torch.float32)
            * grad_out[flat_indices // top_k].to(torch.float32)
        ).sum(dim=1)
        if not scores_are_slot_ordered:
            return grad_scores_by_row.to(dtype)

        grad_scores = torch.empty_like(flat_indices, dtype=dtype)
        grad_scores[flat_indices] = grad_scores_by_row.to(dtype)
        return grad_scores

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
