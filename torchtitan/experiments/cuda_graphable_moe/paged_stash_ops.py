# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Paged stash buffer, Triton kernels, and custom ops for MoE activation storage.

Instead of recomputing activations (standard AC), stores them in a pre-allocated
paged buffer managed by Triton kernels. This avoids recompute cost while reducing
memory fragmentation for MoE expert layers with dynamic token counts.

Custom ops are registered via torch.library so they can be inserted into FX
graphs by the graph-based paged SAC pass.

3-level overflow defense (mirrors Megatron):

  Level 1 (Host spillover): When CUDA pages exhausted, the Triton copy kernel
  falls back to a pinned host buffer. The pop kernel reads ``spilled_to_host``
  to select the source. All branching is data-dependent inside the kernel —
  CUDA-graph compatible.

  Level 2 (Cross-rank detection): After each step, ``all_reduce`` of 3 flags
  (stash overflow, HybridEP over-budget, host spill) ensures all ranks agree.

  Level 3 (Retry): On full overflow (both CUDA and host exhausted) or
  over-budget, zero grads, grow buffers, reset CUDA graphs, rerun fwd/bwd.

page_record format: [num_tokens, spilled_to_host, page_id_0, page_id_1, ...]
This encodes num_tokens and the spill flag in the first two elements, allowing
both to travel through the fwd→bwd boundary without extra saved tensors.

Buffer sizing: when ``capacity_factor`` is provided to ``create_paged_buffers``,
buffers are sized to ``max_tokens / capacity_factor`` (the balanced estimate
under uniform routing) instead of worst-case ``max_tokens``.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._functorch._activation_offloading.offload_ops import (
    _get_or_create_transfer_stream,
    _register_wait,
)

from torchtitan.tools.logging import logger

GLOBAL_BLOCK_SIZE = 1024


# ---------------------------------------------------------------------------
# Module-level buffer registry — accessed by custom ops at runtime
# ---------------------------------------------------------------------------
# Following the ao::offload pattern (which uses module-level _transfer_streams
# and _wait_registry dicts), paged stash buffers are accessed through this
# registry keyed by buffer ID.  The graph only carries the buffer_id as an
# integer constant — no buffer tensors appear as graph arguments, get_attr
# nodes, or placeholders.  This avoids deepcopy and DTensor issues entirely.

_PAGED_STASH_REGISTRY: dict[int, "PagedStashBuffer"] = {}


def register_paged_stash_buffer(buf: "PagedStashBuffer") -> int:
    """Register a buffer and return its ID for use in graph ops."""
    buf_id = id(buf)
    _PAGED_STASH_REGISTRY[buf_id] = buf
    return buf_id


def unregister_paged_stash_buffer(buf_id: int) -> None:
    """Remove a buffer from the registry."""
    _PAGED_STASH_REGISTRY.pop(buf_id, None)


# ---------------------------------------------------------------------------
# Triton kernels — dual CUDA/host buffer support
# ---------------------------------------------------------------------------


@triton.jit
def _paged_stash_copy_kernel(
    src_ptr,
    cuda_dst_ptr,
    host_dst_ptr,
    num_tokens_ptr,
    free_list_cuda_ptr,
    free_list_host_ptr,
    free_list_head_ptr,  # shape (2,): [cuda_head, host_head]
    free_list_tail_ptr,  # shape (2,): [cuda_tail, host_tail]
    free_list_capacity_ptr,  # shape (2,): [cuda_cap, host_cap]
    page_record_ptr,
    overflow_ptr,
    host_spill_global_ptr,
    spilled_to_host_ptr,  # output: 0 = CUDA, 1 = host
    new_free_list_head_ptr,  # output: shape (2,) updated heads
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_HOST_BUFFER: tl.constexpr,
):
    """Copy tokens to paged stash: try CUDA first (fast path), then host if CUDA full."""
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)

    # Load overflow first — if already set, skip entirely
    overflow = tl.load(overflow_ptr)

    num_tokens = tl.load(num_tokens_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)

    # Load CUDA state
    head_cuda = tl.load(free_list_head_ptr)
    head_host = tl.load(free_list_head_ptr + 1)
    tail_cuda = tl.load(free_list_tail_ptr)
    cap_cuda = tl.load(free_list_capacity_ptr)

    avail_cuda = tail_cuda - head_cuda
    use_cuda = avail_cuda >= required_pages

    # Assume CUDA path
    spill = 0
    dst_ptr = cuda_dst_ptr
    free_list_ptr = free_list_cuda_ptr
    head = head_cuda
    cap = cap_cuda
    new_head_cuda = head_cuda + required_pages
    new_head_host = head_host

    if overflow == 1:
        # Already overflowed — preserve heads, don't copy
        if pid == 0:
            tl.store(new_free_list_head_ptr, head_cuda)
            tl.store(new_free_list_head_ptr + 1, head_host)
        return

    # When CUDA is full: try host
    if not use_cuda:
        tail_host = tl.load(free_list_tail_ptr + 1)
        cap_host = tl.load(free_list_capacity_ptr + 1)
        use_host = HAS_HOST_BUFFER == 1 and (tail_host - head_host) >= required_pages
        if use_host:
            spill = 1
            dst_ptr = host_dst_ptr
            free_list_ptr = free_list_host_ptr
            head = head_host
            cap = cap_host
            new_head_cuda = head_cuda
            new_head_host = head_host + required_pages
        else:
            # Both CUDA and host exhausted — overflow
            if pid == 0:
                tl.store(overflow_ptr, 1)
                tl.store(spilled_to_host_ptr, 1)
                tl.store(new_free_list_head_ptr, head_cuda)
                tl.store(new_free_list_head_ptr + 1, head_host)
            return

    if pid == 0:
        tl.store(spilled_to_host_ptr, spill)
        if spill == 1:
            tl.store(host_spill_global_ptr, 1)

    # Copy loop: strided over tokens
    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        free_list_idx = (head + page_slot) % cap
        page_id = tl.load(free_list_ptr + free_list_idx)
        if token_in_page == 0:
            tl.store(page_record_ptr + page_slot, page_id)
        dst_token_idx = page_id * PAGE_SIZE + token_in_page
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        token_idx_i64 = token_idx.to(tl.int64)
        dst_token_idx_i64 = dst_token_idx.to(tl.int64)
        src_base = src_ptr + token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + dst_token_idx_i64 * HIDDEN_SIZE
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        token_idx += num_blocks

    if pid == 0:
        tl.store(new_free_list_head_ptr, new_head_cuda)
        tl.store(new_free_list_head_ptr + 1, new_head_host)


@triton.jit
def _paged_stash_pop_kernel(
    cuda_src_ptr,
    host_src_ptr,
    dst_ptr,
    num_tokens_ptr,
    page_record_ptr,
    spilled_to_host_ptr,
    overflow_ptr,
    free_list_cuda_ptr,
    free_list_host_ptr,
    free_list_tail_ptr,  # shape (2,): [cuda_tail, host_tail]
    free_list_capacity_ptr,  # shape (2,)
    new_free_list_tail_ptr,  # output: shape (2,) updated tails
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reload tokens from paged stash; reads spilled_to_host to select source."""
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)

    # If overflow was set, no valid data — preserve tails and return
    overflow = tl.load(overflow_ptr)
    num_tokens = tl.load(num_tokens_ptr)
    spill = tl.load(spilled_to_host_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)

    # Load both tails
    tail_cuda = tl.load(free_list_tail_ptr)
    tail_host = tl.load(free_list_tail_ptr + 1)
    cap_cuda = tl.load(free_list_capacity_ptr)

    if overflow == 1:
        if pid == 0:
            tl.store(new_free_list_tail_ptr, tail_cuda)
            tl.store(new_free_list_tail_ptr + 1, tail_host)
        return

    # Assume CUDA path
    src_ptr = cuda_src_ptr
    free_list_ptr = free_list_cuda_ptr
    tail = tail_cuda
    cap = cap_cuda
    new_tail_cuda = tail_cuda + required_pages
    new_tail_host = tail_host

    # Switch to host path if spilled
    if spill == 1:
        cap_host = tl.load(free_list_capacity_ptr + 1)
        src_ptr = host_src_ptr
        free_list_ptr = free_list_host_ptr
        tail = tail_host
        cap = cap_host
        new_tail_cuda = tail_cuda
        new_tail_host = tail_host + required_pages

    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        page_id = tl.load(page_record_ptr + page_slot)
        src_token_idx = page_id * PAGE_SIZE + token_in_page
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        src_token_idx_i64 = src_token_idx.to(tl.int64)
        token_idx_i64 = token_idx.to(tl.int64)
        src_base = src_ptr + src_token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + token_idx_i64 * HIDDEN_SIZE
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        if token_in_page == 0:
            write_idx = (tail + page_slot) % cap
            tl.store(free_list_ptr + write_idx, page_id)
        token_idx += num_blocks

    if pid == 0:
        tl.store(new_free_list_tail_ptr, new_tail_cuda)
        tl.store(new_free_list_tail_ptr + 1, new_tail_host)


# ---------------------------------------------------------------------------
# PagedStashBuffer — pre-allocated paged memory pool with optional host buffer
# ---------------------------------------------------------------------------


class PagedStashBuffer:
    """Pre-allocated paged memory pool for stashing activations.

    Supports both CUDA and optional pinned host buffer for overflow fallback.
    Uses per-buffer free lists (circular buffer) tracked as two-element state:
    index 0 = CUDA, index 1 = host.

    Args:
        num_tokens: Maximum tokens the CUDA buffer can hold.
        hidden_size: Size of the hidden dimension.
        page_size: Number of tokens per page.
        device: Device for the buffer.
        overflow: Shared int64 GPU tensor, set to 1 when both CUDA and host full.
        host_spill: Shared int64 GPU tensor, set to 1 if any activation spills to host.
        dtype: Data type for the buffer.
        num_tokens_host: If > 0, allocate pinned host buffer with this many tokens.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        page_size: int,
        device: str | torch.device,
        overflow: torch.Tensor,
        host_spill: torch.Tensor,
        dtype: torch.dtype,
        num_tokens_host: int = 0,
    ):
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.overflow = overflow  # shared across buffers
        self.host_spill = host_spill  # shared across buffers

        # CUDA buffer
        self.num_cuda_pages = (num_tokens + page_size - 1) // page_size
        self.total_cuda_tokens = self.num_cuda_pages * page_size
        self.cuda_buffer = torch.empty(
            (self.total_cuda_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )

        # Host buffer (pinned), optional
        self.num_host_pages = (
            (num_tokens_host + page_size - 1) // page_size if num_tokens_host > 0 else 0
        )
        self.total_host_tokens = (
            self.num_host_pages * page_size if self.num_host_pages > 0 else 0
        )
        if self.num_host_pages > 0:
            self.host_buffer = torch.empty(
                (self.total_host_tokens, hidden_size),
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            self.host_buffer = None

        # Free list state: shape (2,) — index 0 = CUDA, index 1 = host
        # All on GPU for kernel access
        self.free_list_head = torch.zeros(2, dtype=torch.int64, device=device)
        self.free_list_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages],
            dtype=torch.int64,
            device=device,
        )
        self.free_list_capacity = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages],
            dtype=torch.int64,
            device=device,
        )

        # Free list arrays (device memory): page IDs for each buffer
        self.free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=device
        )
        if self.num_host_pages > 0:
            self.free_list_host = torch.arange(
                self.num_host_pages, dtype=torch.int64, device=device
            )
        else:
            self.free_list_host = torch.empty(0, dtype=torch.int64, device=device)

        # Pre-allocated reset values (CUDA graph safe: no allocation in reset())
        self._reset_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages],
            dtype=torch.int64,
            device=device,
        )
        self._reset_free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=device
        )
        if self.num_host_pages > 0:
            self._reset_free_list_host = torch.arange(
                self.num_host_pages, dtype=torch.int64, device=device
            )
        else:
            self._reset_free_list_host = None

    def __repr__(self):
        return (
            f"PagedStashBuffer(cuda_pages={self.num_cuda_pages}, "
            f"host_pages={self.num_host_pages}, page_size={self.page_size}, "
            f"hidden_size={self.hidden_size}, device={self.device}, dtype={self.dtype})"
        )

    def reset(self):
        """Reset both CUDA and host free lists. CUDA graph safe (no allocations)."""
        self.free_list_cuda.copy_(self._reset_free_list_cuda)
        self.free_list_head.zero_()
        self.free_list_tail.copy_(self._reset_tail)
        if self._reset_free_list_host is not None:
            self.free_list_host.copy_(self._reset_free_list_host)

    def grow(self, factor: float = 2.0) -> None:
        """Grow the CUDA buffer by the given factor.

        Invalidates any captured CUDA graphs — callers must reset the
        CUDAGraphWrapper after calling this.  The host buffer is not grown
        (it's a spillover safety net, not a primary store).
        """
        new_num_cuda_pages = int(self.num_cuda_pages * factor)
        if new_num_cuda_pages <= self.num_cuda_pages:
            return

        old_num_cuda_pages = self.num_cuda_pages
        self.num_cuda_pages = new_num_cuda_pages
        self.total_cuda_tokens = self.num_cuda_pages * self.page_size

        # Reallocate CUDA buffer and free list
        self.cuda_buffer = torch.empty(
            (self.total_cuda_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=self.device
        )

        # Update shared state tensors
        self.free_list_head = torch.zeros(2, dtype=torch.int64, device=self.device)
        self.free_list_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages],
            dtype=torch.int64,
            device=self.device,
        )
        self.free_list_capacity = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages],
            dtype=torch.int64,
            device=self.device,
        )

        # Update reset templates
        self._reset_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages],
            dtype=torch.int64,
            device=self.device,
        )
        self._reset_free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=self.device
        )

        # With the module-level registry, the buffer object itself IS the
        # registry entry.  After grow() reallocates self.cuda_buffer etc., the
        # registry entry reflects the new tensors automatically.  Callers must
        # reset CUDA graphs so the next step re-captures with new pointers.

        logger.warning(
            "PagedStashBuffer grown: %d -> %d CUDA pages " "(hidden_size=%d, dtype=%s)",
            old_num_cuda_pages,
            self.num_cuda_pages,
            self.hidden_size,
            self.dtype,
        )


# ---------------------------------------------------------------------------
# create_paged_buffers — factory
# ---------------------------------------------------------------------------


def create_paged_buffers(
    model,
    *,
    max_tokens,
    capacity_factor=None,
    page_size=64,
    buffer_size_factor=1.1,
    host_buffer_size_factor=0.0,
    buffer_device="cuda",
):
    """Create paged stash buffers for MoE expert activations.

    Scans the model for GroupedExperts modules, counts stash ops per
    (dtype, hidden_size) key, and creates one PagedStashBuffer per key.

    Under HybridEP with capacity-factor padding, ``max_tokens`` includes the
    padding.  When ``capacity_factor`` is provided, the CUDA buffer is sized to
    ``max_tokens / capacity_factor`` (the balanced estimate).

    When ``host_buffer_size_factor > 0``, an additional pinned host buffer is
    allocated for spillover when the CUDA buffer is full.

    Args:
        model: The transformer model to scan for GroupedExperts.
        max_tokens: Upper bound on tokens routed to experts per step.
        capacity_factor: HybridEP capacity factor.  When set, CUDA buffers are
            sized to ``max_tokens / capacity_factor``.
        page_size: Number of tokens per page.
        buffer_size_factor: Factor to scale estimated_tokens for CUDA buffer.
        host_buffer_size_factor: Factor for host spillover buffer sizing.
            0 means no host buffer.
        buffer_device: Device for the paged stash buffer ("cuda" or "cpu").

    Returns:
        Tuple of (buffers, overflow, host_spill) where buffers is a dict mapping
        (dtype, hidden_size) to PagedStashBuffer, overflow and host_spill are
        shared int64 GPU scalars.  Returns (None, None, None) if no
        GroupedExperts found.
    """
    from collections import defaultdict

    from torchtitan.models.common.moe import GroupedExperts

    device = buffer_device

    # Count stash ops per (dtype, hidden_size) key across all GroupedExperts modules.
    ops_per_key: dict[tuple[torch.dtype, int], int] = defaultdict(int)
    num_expert_modules = 0
    for _fqn, mod in model.named_modules():
        if isinstance(mod, GroupedExperts):
            num_expert_modules += 1
            ops_per_key[(mod.w1.dtype, mod.w1.shape[-2])] += 4
            ops_per_key[(mod.w1.dtype, mod.w1.shape[-1])] += 1

    if not ops_per_key:
        logger.warning("No GroupedExperts found; no paged stash buffers created.")
        return None, None, None

    # When capacity_factor is provided, size to the balanced estimate
    estimated_tokens = max_tokens
    if capacity_factor is not None and capacity_factor > 0:
        estimated_tokens = int(max_tokens / capacity_factor)

    # Shared overflow and host_spill flags
    overflow = torch.zeros(1, dtype=torch.int64, device=device)
    host_spill = torch.zeros(1, dtype=torch.int64, device=device)
    buffers = {}
    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    if _rank0:
        logger.debug(
            "create_paged_buffers: max_tokens=%d, estimated_tokens=%d, "
            "capacity_factor=%s, buffer_size_factor=%.2f, "
            "host_buffer_size_factor=%.2f",
            max_tokens,
            estimated_tokens,
            capacity_factor,
            buffer_size_factor,
            host_buffer_size_factor,
        )
    for (dtype, hidden_size), num_ops in ops_per_key.items():
        scaled_cuda = int(estimated_tokens * buffer_size_factor * num_ops)
        scaled_host = (
            int(estimated_tokens * host_buffer_size_factor * num_ops)
            if host_buffer_size_factor > 0
            else 0
        )
        buffers[dtype, hidden_size] = PagedStashBuffer(
            scaled_cuda,
            hidden_size,
            page_size,
            device,
            overflow,
            host_spill,
            dtype,
            num_tokens_host=scaled_host,
        )
        buf = buffers[dtype, hidden_size]
        msg = f"  key=({dtype}, {hidden_size}): cuda={buf.num_cuda_pages} pages"
        if buf.host_buffer is not None:
            msg += f", host={buf.num_host_pages} pages"
        if _rank0:
            logger.debug(msg)

    logger.info(
        "Created %d paged stash buffers (max_tokens=%d, num_expert_modules=%d, "
        "ops_per_key=%s, page_size=%d, device=%s, host_buffer=%s)",
        len(buffers),
        max_tokens,
        num_expert_modules,
        dict(ops_per_key),
        page_size,
        device,
        "yes" if host_buffer_size_factor > 0 else "no",
    )

    return buffers, overflow, host_spill


def _block_size(hidden_size: int) -> int:
    """Compute the block size for Triton kernels, capped at hidden_size and rounded to power of 2."""
    return min(GLOBAL_BLOCK_SIZE, triton.next_power_of_2(hidden_size))


# ---------------------------------------------------------------------------
# paged_stash::copy — pack a tensor into the paged buffer (CUDA or host)
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "paged_stash::copy",
    mutates_args=(),
)
def paged_stash_copy(
    tensor: Tensor,
    page_size: int,
    hidden_size: int,
    num_tokens_tensor: Tensor,
    buffer_id: int,
) -> tuple[Tensor, Tensor]:
    """Pack tensor into paged buffer using actual token count.

    Looks up the ``PagedStashBuffer`` from the module-level registry by
    ``buffer_id``.  Tries CUDA buffer first; falls back to pinned host if
    CUDA pages exhausted; sets overflow flag if both exhausted.

    The Triton kernel and free-list update run on a dedicated transfer
    stream (from ao's ``_get_or_create_transfer_stream``), allowing
    overlap with compute on the main stream.  A completion event is
    registered in ao's ``_wait_registry`` keyed by the page_record's
    ``data_ptr()``; the graph pass inserts ``ao.wait_tensor`` to
    synchronize before the page_record is read in the backward pass.

    Returns (page_record, new_head).
    page_record format: [num_tokens, spilled_to_host, page_id_0, ..., page_id_{N-1}]
    Sized to worst-case for CUDA graph static shapes.
    """
    buf = _PAGED_STASH_REGISTRY[buffer_id]
    has_host_buffer = 1 if buf.host_buffer is not None else 0
    host_buffer = buf.host_buffer if buf.host_buffer is not None else buf.cuda_buffer

    flat = tensor.reshape(-1, hidden_size).contiguous()
    max_num_tokens = flat.shape[0]
    max_num_pages = (max_num_tokens + page_size - 1) // page_size

    # page_record: [num_tokens, spilled_to_host, page_ids...]
    # Zero-initialized so that on overflow (copy kernel skips writing page IDs),
    # the pop kernel reads page_id=0 — a valid page in the buffer.  The data is
    # wrong but that's fine: overflow gradients are thrown away and the step is
    # retried.
    page_record = torch.zeros(max_num_pages + 2, dtype=torch.int64, device=flat.device)
    num_tokens_i64 = num_tokens_tensor.reshape(1).to(torch.int64)
    page_record[0:1].copy_(num_tokens_i64)
    page_record[1:2].zero_()  # spilled_to_host = 0 (kernel may overwrite)
    page_ids = page_record[2:]  # view into page IDs portion

    new_free_list_head = torch.empty(2, dtype=torch.int64, device=flat.device)

    # Switch to transfer stream for the Triton kernel + free-list update.
    # Matches ao::offload pattern: transfer_stream.wait_stream(current),
    # do work on transfer stream, record event, restore current stream.
    device = tensor.device
    current_stream = torch.accelerator.current_stream(device)
    transfer_stream = _get_or_create_transfer_stream(device)
    transfer_stream.wait_stream(current_stream)
    torch.accelerator.set_stream(transfer_stream)

    num_blocks = max(min(max_num_tokens, 2048), 1)
    grid = (num_blocks,)
    _paged_stash_copy_kernel[grid](
        flat,
        buf.cuda_buffer,
        host_buffer,
        num_tokens_i64,
        buf.free_list_cuda,
        buf.free_list_host,
        buf.free_list_head,
        buf.free_list_tail,
        buf.free_list_capacity,
        page_ids,
        buf.overflow,
        buf.host_spill,
        page_record[1:2],  # spilled_to_host_ptr
        new_free_list_head,
        PAGE_SIZE=page_size,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=_block_size(hidden_size),
        HAS_HOST_BUFFER=has_host_buffer,
    )
    # Update the head pointers in-place (on transfer stream, after kernel)
    buf.free_list_head.copy_(new_free_list_head)

    # Register completion event for ao::wait_tensor
    completion_event = _register_wait(page_record, device)
    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    return page_record, new_free_list_head


@paged_stash_copy.register_fake
def paged_stash_copy_fake(
    tensor: Tensor,
    page_size: int,
    hidden_size: int,
    num_tokens_tensor: Tensor,
    buffer_id: int,
) -> tuple[Tensor, Tensor]:
    flat = tensor.reshape(-1, hidden_size)
    max_num_tokens = flat.shape[0]
    max_num_pages = (max_num_tokens + page_size - 1) // page_size
    # page_record: max_num_pages + 2 (num_tokens + spilled_to_host + page_ids)
    page_record = tensor.new_empty(max_num_pages + 2, dtype=torch.int64)
    new_head = tensor.new_empty(2, dtype=torch.int64)
    return page_record, new_head


# ---------------------------------------------------------------------------
# paged_stash::pop — restore a tensor from the paged buffer
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "paged_stash::pop",
    mutates_args=(),
)
def paged_stash_pop(
    page_record: Tensor,
    page_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    buffer_id: int,
) -> Tensor:
    """Pop tensor from paged buffer. Returns reconstructed 2D tensor.

    Looks up the ``PagedStashBuffer`` from the module-level registry by
    ``buffer_id``.  Reads ``spilled_to_host`` from the page_record to select
    CUDA or host source buffer.

    The output tensor is allocated on the compute stream (matching
    ao::reload's pattern for correct allocator ownership), then the
    Triton kernel and free-list update run on the transfer stream.
    A completion event is registered for ``ao.wait_tensor``.

    page_record format: [num_tokens, spilled_to_host, page_id_0, ..., page_id_{N-1}]
    """
    buf = _PAGED_STASH_REGISTRY[buffer_id]
    host_buffer = buf.host_buffer if buf.host_buffer is not None else buf.cuda_buffer

    num_pages = page_record.shape[0] - 2
    num_tokens = num_pages * page_size  # upper bound
    page_ids = page_record[2:]

    # Allocate output on compute stream (like ao::reload) so the
    # allocator tracks ownership correctly.
    device = buf.cuda_buffer.device
    flat_out = torch.zeros((num_tokens, hidden_size), dtype=dtype, device=device)
    num_tokens_tensor = page_record[0:1]
    spilled_to_host = page_record[1:2]

    new_free_list_tail = torch.empty(2, dtype=torch.int64, device=device)

    # Switch to transfer stream for the Triton kernel + free-list update.
    current_stream = torch.accelerator.current_stream(device)
    transfer_stream = _get_or_create_transfer_stream(device)
    completion_event = _register_wait(flat_out, device)
    transfer_stream.wait_stream(current_stream)
    torch.accelerator.set_stream(transfer_stream)

    num_blocks = max(min(num_tokens, 2048), 1)
    grid = (num_blocks,)

    _paged_stash_pop_kernel[grid](
        buf.cuda_buffer,
        host_buffer,
        flat_out,
        num_tokens_tensor,
        page_ids,
        spilled_to_host,
        buf.overflow,
        buf.free_list_cuda,
        buf.free_list_host,
        buf.free_list_tail,
        buf.free_list_capacity,
        new_free_list_tail,
        PAGE_SIZE=page_size,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=_block_size(hidden_size),
    )
    # Update the tail pointers in-place (on transfer stream, after kernel)
    buf.free_list_tail.copy_(new_free_list_tail)

    transfer_stream.record_event(completion_event)
    torch.accelerator.set_stream(current_stream)

    return flat_out


@paged_stash_pop.register_fake
def paged_stash_pop_fake(
    page_record: Tensor,
    page_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    buffer_id: int,
) -> Tensor:
    # num_tokens is data-dependent (stored in page_record[0]).
    # Use create_unbacked_symint for the dynamic first dimension.
    ctx = torch.library.get_ctx()
    num_tokens = ctx.create_unbacked_symint()
    return page_record.new_empty((num_tokens, hidden_size), dtype=dtype)
