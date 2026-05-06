# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paged stash memory policy for CUDA-graphable MoE training."""

import functools
import operator

import torch
import triton
import triton.language as tl

# Side-effect import: registers ao::wait_tensor custom op
import torch._functorch._activation_offloading.offload_ops as offload_ops  # noqa: F401
import torch.fx as fx
from torch import Tensor
from torch._functorch._activation_offloading.offload_ops import (
    _get_or_create_transfer_stream,
    _register_wait,
)
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_checkpoint import _get_save_ops
from torchtitan.experiments.graph_trainer.cpu_offload import _classify_forward_backward
from torchtitan.tools.logging import logger

from dataclasses import dataclass
from .configs import (
    register_memory_policy_config,
)
from .passes import (
    apply_sac_pass,
    construct_default_graph_passes,
    register_memory_policy,
    register_pass_pipeline,
    register_post_init_hook,
    register_pre_train_step_hook,
    selective_activation_remat_pass,
)


# ---------------------------------------------------------------------------
# Paged stash memory policy configs
# ---------------------------------------------------------------------------


@register_memory_policy_config("paged_stash")
@dataclass(kw_only=True, slots=True)
class PagedStashMemoryPolicy:
    """Paged stash: store MoE activations in a shared paged buffer.

    Replaces per-layer worst-case buffers with a single shared paged buffer,
    reducing memory from O(layers x worst_case) to O(worst_case + actual_usage).
    """

    buffer_size_factor: float = 1.1
    """Factor to scale estimated_tokens for CUDA buffer over-provisioning."""

    host_buffer_size_factor: float = 0.0
    """Factor for host (pinned CPU) spillover buffer. 0 = no host buffer."""

    page_size: int = 64
    """Number of tokens per page in the paged stash buffer."""


# ---------------------------------------------------------------------------
# Module-level state for pass pipeline access
# ---------------------------------------------------------------------------

_PAGED_STASH_BUFFERS_DICT: dict | None = None

# ===========================================================================
# Paged stash ops (buffer, Triton kernels, custom ops)
# ===========================================================================

GLOBAL_BLOCK_SIZE = 1024


# ---------------------------------------------------------------------------
# Module-level buffer registry -- accessed by custom ops at runtime
# ---------------------------------------------------------------------------
# Following the ao::offload pattern (which uses module-level _transfer_streams
# and _wait_registry dicts), paged stash buffers are accessed through this
# registry keyed by buffer ID.  The graph only carries the buffer_id as an
# integer constant -- no buffer tensors appear as graph arguments, get_attr
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
# Triton kernels -- dual CUDA/host buffer support
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

    # Load overflow first -- if already set, skip entirely
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
        # Already overflowed -- preserve heads, don't copy
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
            # Both CUDA and host exhausted -- overflow
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

    # If overflow was set, no valid data -- preserve tails and return
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
# PagedStashBuffer -- pre-allocated paged memory pool with optional host buffer
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

        # Free list state: shape (2,) -- index 0 = CUDA, index 1 = host
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


# ---------------------------------------------------------------------------
# create_paged_buffers -- factory
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
# paged_stash::copy -- pack a tensor into the paged buffer (CUDA or host)
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
    # the pop kernel reads page_id=0 -- a valid page in the buffer.  The data is
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
# paged_stash::pop -- restore a tensor from the paged buffer
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


# ===========================================================================
# Paged stash graph pass
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_moe_fc1_grouped_mm(node: fx.Node) -> bool:
    """Check if a node is a MoE fc1 (gate or up) ``_grouped_mm`` output.

    Matches ``aten._grouped_mm.default`` nodes that are part of the SwiGLU
    fc1 computation:
    - **fc1_1 (gate)**: output feeds ``aten.silu`` directly.
    - **fc1_2 (up)**: output feeds ``aten.mul`` where the *other* operand
      comes from ``aten.silu`` (i.e., the gated activation ``silu(fc1_1) * fc1_2``).

    The fc2 down projection also feeds ``aten.mul`` (combine with shared
    experts), but its ``mul`` partner is not a ``silu`` output -- so it is
    excluded.
    """
    if node.op != "call_function":
        return False
    if node.target != torch.ops.aten._grouped_mm.default:
        return False
    for user in node.users:
        if user.op != "call_function":
            continue
        if user.target == torch.ops.aten.silu.default:
            return True
        if user.target == torch.ops.aten.mul.Tensor:
            for mul_input in user.all_input_nodes:
                if mul_input is not node and _is_silu_output(mul_input):
                    return True
    return False


def _is_silu_output(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target == torch.ops.aten.silu.default


def _has_dynamic_first_dim(node: fx.Node) -> bool:
    """Check if a node's first dimension is dynamic (SymInt)."""
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return False
    return isinstance(val.shape[0], torch.SymInt)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper: find actual token count for _grouped_mm outputs
# ---------------------------------------------------------------------------


def _find_num_tokens_node(
    gm: fx.GraphModule,
    fwd_node: fx.Node,
    insert_before: fx.Node,
    *,
    _cache: dict[int, fx.Node] | None = None,
) -> fx.Node | None:
    """Find a graph node representing the actual token count for a saved tensor.

    For ``_grouped_mm`` outputs, the offsets tensor (from ``cumsum``) is
    available via ``kwargs["offs"]``.  ``offsets[-1]`` equals
    ``tokens_per_expert.sum()`` -- the actual (non-padded) token count.

    For derived tensors (transposes, casts, etc.), walks backward through
    the producer chain to find the originating ``_grouped_mm`` node.

    Returns an int64 scalar node, or None if not found.
    """
    if _cache is None:
        _cache = {}

    if (
        fwd_node.op == "call_function"
        and fwd_node.target == torch.ops.aten._grouped_mm.default
    ):
        offsets_node = fwd_node.kwargs.get("offs")
        if offsets_node is None and len(fwd_node.args) > 2:
            offsets_node = fwd_node.args[2]
        if offsets_node is None or not isinstance(offsets_node, fx.Node):
            return None

        cache_key = id(offsets_node)
        if cache_key in _cache:
            return _cache[cache_key]

        with gm.graph.inserting_before(insert_before):
            total_node = gm.graph.call_function(
                torch.ops.aten.select.int,
                args=(offsets_node, 0, -1),
            )
            offsets_val = offsets_node.meta.get("val")
            if offsets_val is not None:
                total_node.meta["val"] = offsets_val.select(0, -1)
            total_i64 = gm.graph.call_function(
                torch.ops.aten.to.dtype,
                args=(total_node, torch.int64),
            )
            if "val" in total_node.meta:
                total_i64.meta["val"] = total_node.meta["val"].to(torch.int64)
        _cache[cache_key] = total_i64
        return total_i64

    for arg in fwd_node.all_input_nodes:
        result = _find_num_tokens_node(gm, arg, insert_before, _cache=_cache)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Joint-graph pass: eligibility check
# ---------------------------------------------------------------------------


def _is_paged_stash_eligible(
    node: fx.Node,
    forward_nodes: set[fx.Node],
    backward_nodes: set[fx.Node],
    paged_buffers: dict[tuple[torch.dtype, int], "PagedStashBuffer"],
) -> tuple[list[fx.Node], "PagedStashBuffer"] | None:
    """Check if a forward node is eligible for paged stash.

    Returns ``(bwd_consumers, buffer)`` if eligible, or ``None``.

    A node is eligible when all of:
    1. It is a forward ``call_function`` node.
    2. It is an ``aten._grouped_mm`` whose consumer is ``silu`` or ``mul``
       (MoE fc1 gate/up projection, identified by op-target + consumer pattern).
    3. It has real (non-sym) backward consumers.
    4. It has a dynamic SymInt first dimension.
    5. Its ``(dtype, shape[-1])`` matches a pre-allocated paged buffer.
    """
    if node.op != "call_function":
        return None
    if node not in forward_nodes:
        return None
    if not _is_moe_fc1_grouped_mm(node):
        return None

    bwd_consumers = [u for u in node.users if u in backward_nodes]
    if not bwd_consumers:
        return None
    if all(is_sym_node(u) for u in bwd_consumers):
        return None
    if not _has_dynamic_first_dim(node):
        return None

    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return None
    key = (val.dtype, val.shape[-1])
    buf = paged_buffers.get(key)
    if buf is None:
        return None

    return bwd_consumers, buf


# ---------------------------------------------------------------------------
# Joint-graph pass: main entry point
# ---------------------------------------------------------------------------


def apply_paged_stash_pass(
    gm: torch.fx.GraphModule,
    paged_buffers: dict[tuple[torch.dtype, int], "PagedStashBuffer"],
) -> torch.fx.GraphModule:
    """Insert paged_stash.copy/pop + ao.wait_tensor into the joint graph.

    1. Classify forward vs backward nodes using ``autograd_backward``.
    2. For each eligible forward node (fc1 _grouped_mm, dynamic, has bwd consumers):
       a. Insert ``paged_stash.copy`` + ``ao.wait_tensor`` after the fwd node.
       b. Insert ``paged_stash.pop`` + ``ao.wait_tensor`` before bwd consumers.
       c. Redirect backward consumers to read from the pop output.
    3. After this pass, SAC's remat sees ``page_record`` (small int64) crossing
       the boundary, not the large activation.

    Args:
        gm: The joint forward-backward graph module.
        paged_buffers: Dict mapping ``(dtype, hidden_size)`` to
            ``PagedStashBuffer``.

    Returns:
        The modified graph module.
    """
    forward_nodes, backward_nodes = _classify_forward_backward(gm)

    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

    # Ensure all buffers are registered in the module-level registry.
    # register_paged_stash_buffer is idempotent for the same buffer object.
    buffer_ids: dict[int, int] = {}
    for buf in paged_buffers.values():
        buf_key = id(buf)
        if buf_key not in buffer_ids:
            buffer_ids[buf_key] = register_paged_stash_buffer(buf)

    # Get FakeTensorMode from the joint graph for creating proper metadata
    # on inserted nodes (the partitioner requires FakeTensor, not meta tensors).
    fake_mode = None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            v = node.meta["val"]
            if hasattr(v, "fake_mode"):
                fake_mode = v.fake_mode
                break

    def _make_fake(shape, dtype=torch.int64):
        """Create a FakeTensor with the correct mode for the partitioner."""
        if fake_mode is not None:
            with fake_mode:
                return torch.empty(shape, dtype=dtype, device="cuda")
        return torch.empty(shape, dtype=dtype, device="meta")

    # Collect eligible nodes in topological order
    eligible: list[tuple[fx.Node, list[fx.Node], PagedStashBuffer]] = []
    for node in gm.graph.nodes:
        result = _is_paged_stash_eligible(
            node, forward_nodes, backward_nodes, paged_buffers
        )
        if result is not None:
            bwd_consumers, buf = result
            eligible.append((node, bwd_consumers, buf))
            # SAC must have already annotated these nodes. They should be
            # PREFER_RECOMPUTE (expert activations are not in SAC's save
            # list), meaning SAC would recompute them -- paged stash saves
            # them via paging instead. If SAC marked them MUST_SAVE, the
            # paged stash is redundant; if SAC hasn't run, something is
            # wrong with pass ordering.
            sac_policy = node.meta.get("recompute")
            if sac_policy is None:
                if _rank0:
                    logger.debug(
                        "Paged stash node %s has no SAC annotation "
                        "(SAC may not have run yet).",
                        node.name,
                    )
            elif _rank0 and sac_policy != CheckpointPolicy.PREFER_RECOMPUTE:
                logger.warning(
                    "Paged stash node %s has SAC policy %s (expected "
                    "PREFER_RECOMPUTE). Paged stash will override.",
                    node.name,
                    sac_policy,
                )

    if not eligible:
        logger.info("apply_paged_stash_pass: no eligible nodes found")
        return gm

    node_to_index = {n: i for i, n in enumerate(gm.graph.nodes)}
    num_tokens_cache: dict[int, fx.Node] = {}

    for fwd_node, bwd_consumers, buf in eligible:
        buffer_id = buffer_ids[id(buf)]
        val = fwd_node.meta["val"]

        # --- Forward: insert copy + wait_tensor after fwd_node ---

        first_bwd_consumer = min(bwd_consumers, key=lambda n: node_to_index[n])

        # Find actual token count (offsets[-1] from _grouped_mm kwargs).
        # Helper nodes (select, to.dtype) are inserted right after fwd_node.
        num_tokens_node = _find_num_tokens_node(
            gm, fwd_node, fwd_node.next, _cache=num_tokens_cache
        )
        if _rank0:
            found_str = (
                "offsets[-1]"
                if num_tokens_node is not None
                else "tensor shape fallback"
            )
            logger.debug("  num_tokens for %s: %s", fwd_node.name, found_str)

        # Walk past any helper nodes that _find_num_tokens_node just created
        # so that our copy node is inserted after them (topological order).
        insert_pt = fwd_node
        cursor = fwd_node.next
        while (
            cursor is not None
            and cursor.op == "call_function"
            and cursor.target
            in (
                torch.ops.aten.select.int,
                torch.ops.aten.to.dtype,
            )
        ):
            insert_pt = cursor
            cursor = cursor.next

        if num_tokens_node is not None:
            actual_num_tokens = num_tokens_node
        else:
            with gm.graph.inserting_after(fwd_node):
                num_tokens_fallback = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=([1], val.shape[0]),
                    kwargs={"dtype": torch.int64, "device": val.device},
                )
                num_tokens_fallback.meta["val"] = _make_fake(1, torch.int64)
            insert_pt = num_tokens_fallback
            actual_num_tokens = num_tokens_fallback

        # Compute fake metadata for the inserted nodes
        flat_shape = val.reshape(-1, buf.hidden_size)
        max_num_tokens = flat_shape.shape[0]
        max_num_pages = (max_num_tokens + buf.page_size - 1) // buf.page_size

        page_record_fake = _make_fake(max_num_pages + 2, torch.int64)
        new_head_fake = _make_fake(2, torch.int64)
        pop_fake = _make_fake((max_num_tokens, buf.hidden_size), val.dtype)

        with gm.graph.inserting_after(insert_pt):
            copy_node = gm.graph.call_function(
                torch.ops.paged_stash.copy,
                args=(
                    fwd_node,
                    buf.page_size,
                    buf.hidden_size,
                    actual_num_tokens,
                    buffer_id,
                ),
            )
            copy_node.meta["val"] = (page_record_fake, new_head_fake)
            # MUST_SAVE so the partitioner doesn't treat it as impure
            copy_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            copy_node.meta["ac_graph_id"] = 0
            copy_node.meta["autograd_backward"] = False

        with gm.graph.inserting_after(copy_node):
            page_record_node = gm.graph.call_function(
                operator.getitem,
                args=(copy_node, 0),
            )
            page_record_node.meta["val"] = page_record_fake
            page_record_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            page_record_node.meta["ac_graph_id"] = 0
            page_record_node.meta["autograd_backward"] = False

        with gm.graph.inserting_after(page_record_node):
            # keepalive=fwd_node extends the activation's lifetime past the
            # async Triton copy on the transfer stream (same pattern as
            # ao.wait_tensor(offload_result, gpu_tensor) in cpu_offload_pass)
            wait_copy_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(page_record_node, fwd_node),
            )
            wait_copy_node.meta["val"] = page_record_fake
            # ao.wait_tensor has has_side_effect, which the partitioner treats
            # as impure. MUST_SAVE bypasses the impure-op assertion.
            wait_copy_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            wait_copy_node.meta["ac_graph_id"] = 0
            wait_copy_node.meta["autograd_backward"] = False

        # --- Backward: insert pop + wait_tensor before first bwd consumer ---

        with gm.graph.inserting_before(first_bwd_consumer):
            pop_node = gm.graph.call_function(
                torch.ops.paged_stash.pop,
                args=(
                    wait_copy_node,
                    buf.page_size,
                    buf.hidden_size,
                    val.dtype,
                    buffer_id,
                ),
            )
            pop_node.meta["val"] = pop_fake
            pop_node.meta["autograd_backward"] = True

        with gm.graph.inserting_after(pop_node):
            wait_pop_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(pop_node,),
            )
            wait_pop_node.meta["val"] = pop_fake
            wait_pop_node.meta["autograd_backward"] = True

        if len(val.shape) > 2:
            with gm.graph.inserting_after(wait_pop_node):
                restore_node = gm.graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(wait_pop_node, list(val.shape)),
                )
                restore_node.meta["val"] = val
                restore_node.meta["autograd_backward"] = True
        else:
            restore_node = wait_pop_node

        # Redirect backward consumers from fwd_node to the restored tensor
        for bwd_user in bwd_consumers:
            bwd_user.replace_input_with(fwd_node, restore_node)

    # TODO: Add scheduling optimizations (defer copy waits, prefetch pops)
    # for compute-copy overlap. Both this pass and PR #2879's cpu_offload_pass
    # place ao.wait_tensor adjacent to the copy/offload -- the compute stream
    # blocks immediately. Deferring the wait to a later point in the forward
    # graph (post-partition) would allow the Triton copy kernel to overlap
    # with subsequent compute. See paged_stashing_guide.md for details.

    gm.graph.lint()
    gm.recompile()

    logger.info(
        "Inserted paged stash ops: %d copy + wait in fwd, %d pop + wait in bwd",
        len(eligible),
        len(eligible),
    )

    # Per-layer breakdown
    layer_counts: dict[str, int] = {}
    for fwd_node, _, _ in eligible:
        fqn = fwd_node.meta.get("custom", {}).get("module_fqn", "")
        parts = fqn.split(".")
        layer_label = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else fqn or "unknown"
        layer_counts[layer_label] = layer_counts.get(layer_label, 0) + 1
    for layer_label in sorted(layer_counts):
        logger.info(
            "  %s: %d stashed activations", layer_label, layer_counts[layer_label]
        )

    return gm


# ---------------------------------------------------------------------------
# aot_fx_trace-compatible pass wrapper
# ---------------------------------------------------------------------------


def paged_stash_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    paged_buffers: dict[tuple[torch.dtype, int], "PagedStashBuffer"],
) -> torch.fx.GraphModule:
    """Paged stash pass for the ``aot_fx_trace`` compile-time pass list.

    Inserts ``paged_stash.copy/pop`` + ``ao.wait_tensor`` ops for eligible
    activations.  Must run **before** ``selective_activation_remat_pass`` so
    that backward consumers are already redirected when SAC + remat runs --
    the original activations have no backward users and are naturally ignored
    by rematerialization.
    """
    return apply_paged_stash_pass(gm, paged_buffers)


# ---------------------------------------------------------------------------
# Memory policies (registered into AVAILABLE_MEMORY_POLICIES)
# ---------------------------------------------------------------------------


def _make_paged_stash_policy(
    must_stash_action: CheckpointPolicy,
) -> callable:
    """Build a SAC policy function that handles MoE fc1 ``_grouped_mm`` nodes.

    Args:
        must_stash_action: What to do with MoE fc1 ``_grouped_mm`` nodes.
            ``PREFER_RECOMPUTE`` when the paged stash pass will redirect
            backward consumers (the activation has no backward users after
            surgery, so remat DCEs it).
            ``MUST_SAVE`` for the baseline experiment that saves expert
            activations as regular tensors (no paged stash).
    """
    save_ops = _get_save_ops()
    save_ops.add(torch.ops._c10d_functional.all_gather_into_tensor.default)

    def policy_fn(node: fx.Node) -> CheckpointPolicy:
        if _is_moe_fc1_grouped_mm(node):
            return must_stash_action
        if node.target in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


@register_memory_policy(PagedStashMemoryPolicy)
def paged_stash_tag_policy(
    gm: torch.fx.GraphModule,
    *,
    config,
) -> torch.fx.GraphModule:
    """Tag nodes for paged activation stashing.

    MoE fc1 ``_grouped_mm`` nodes (identified by op-target + consumer pattern)
    get ``PREFER_RECOMPUTE`` -- the paged stash execution pass will redirect
    their backward consumers through paged buffers, so remat naturally DCEs
    the original activations.
    """
    apply_sac_pass(
        gm, policy_fn=_make_paged_stash_policy(CheckpointPolicy.PREFER_RECOMPUTE)
    )
    return gm


# ---------------------------------------------------------------------------
# Registry registrations
# ---------------------------------------------------------------------------


@register_pass_pipeline(PagedStashMemoryPolicy)
def paged_stash_pass_pipeline(traced_result, config):
    """Default passes with paged_stash_pass inserted between tagging and remat."""
    passes = construct_default_graph_passes(traced_result, config)
    if _PAGED_STASH_BUFFERS_DICT is not None:
        bound_pass = functools.partial(
            paged_stash_pass, paged_buffers=_PAGED_STASH_BUFFERS_DICT
        )
        new_passes = []
        for p in passes:
            _fn = p.func if isinstance(p, functools.partial) else p
            if _fn is selective_activation_remat_pass:
                new_passes.append(bound_pass)
            new_passes.append(p)
        passes = new_passes
    return passes


@register_post_init_hook(PagedStashMemoryPolicy)
def setup_paged_stash_buffers(trainer):
    """Create paged stash buffers and attach them to the model."""
    global _PAGED_STASH_BUFFERS_DICT

    ps_config = trainer.config.compile.memory_policy

    model = trainer.model_parts[0]
    training = trainer.config.training
    parallel_dims = trainer.parallel_dims

    moe_config = next(
        layer.moe for layer in model.config.layers if layer.moe is not None
    )
    num_experts = moe_config.num_experts
    top_k = moe_config.router.top_k
    base_tokens = training.local_batch_size * training.seq_len
    dispatcher_cfg = moe_config.experts.token_dispatcher
    cf = getattr(dispatcher_cfg, "non_blocking_capacity_factor", None)
    comm_backend = getattr(dispatcher_cfg, "comm_backend", "standard")
    if comm_backend == "hybridep" and cf is not None and parallel_dims.ep_enabled:
        ep_size = parallel_dims.ep
        num_local_experts = num_experts // ep_size
        max_tokens = int(base_tokens * ep_size * min(num_local_experts, top_k) * cf)
    else:
        cf = None
        max_tokens = base_tokens * top_k

    buffers, overflow, host_spill = create_paged_buffers(
        model,
        max_tokens=max_tokens,
        capacity_factor=cf,
        page_size=ps_config.page_size,
        buffer_size_factor=ps_config.buffer_size_factor,
        host_buffer_size_factor=ps_config.host_buffer_size_factor,
        buffer_device="cuda",
    )

    if buffers is not None:
        model._paged_stash_buffers = list(buffers.values())
        model._paged_stash_overflow = overflow
        model._paged_stash_host_spill = host_spill
        model._paged_stash_paged_buffers_dict = buffers
        _PAGED_STASH_BUFFERS_DICT = buffers
        logger.info("Graph-based paged SAC enabled")
        logger.info(
            "aot_fx_trace mode: paged stash pass will be applied at trace time"
        )


@register_pre_train_step_hook(PagedStashMemoryPolicy)
def reset_paged_stash_buffers(trainer):
    """Check previous step's overflow flag, then reset buffers for this step."""
    for model_part in trainer.model_parts:
        overflow = getattr(model_part, "_paged_stash_overflow", None)
        if overflow is not None and overflow.item() != 0:
            raise RuntimeError(
                "Paged stash buffer overflow detected. "
                "The CUDA paged stash buffer is too small for the current "
                "routing pattern.\n\n"
                "To fix, try one of:\n"
                "  1. Enable host spillover: "
                "compile.memory_policy:paged_stash "
                "--compile.memory_policy.host_buffer_size_factor 1.0\n"
                "  2. Increase buffer size: "
                "--compile.memory_policy.buffer_size_factor 2.0\n"
                "  3. Disable paged stash: "
                "compile.memory_policy:default"
            )
        buffers = getattr(model_part, "_paged_stash_buffers", None)
        if buffers:
            for buf in buffers:
                buf.reset()
        if overflow is not None:
            overflow.zero_()
        host_spill = getattr(model_part, "_paged_stash_host_spill", None)
        if host_spill is not None:
            host_spill.zero_()
