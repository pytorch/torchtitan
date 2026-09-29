# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Paged stash memory policy for CUDA-graphable MoE training.

A port of Megatron-Core 0.19.0's paged stashing
(``megatron.core.transformer.moe.paged_stash``) onto GraphTrainer's
traced-graph flow.

Why Paged Stashing
------------------

When a MoE token dispatcher runs with a *static* capacity factor, every
permuted activation is allocated for the worst case while only
``sum(tokens_per_expert)`` rows hold data. With CUDA graphs on, every layer that
saves such an activation also needs its own capacity-padded buffer resident in
the graph pool, so the un-paged cost is ``num_layers x padded``.

Paged stashing reclaims that gap: once the forward consumers are done, the live
rows are copied into a shared pool of fixed-size pages and the padded tensor is
freed; backward pages it back in. The pool is shared by every layer -- one per
``(dtype, hidden_size)`` -- and sized from measured live rows rather than the
padded worst case. The saving grows with pipeline depth, because with PP those
activations otherwise sit idle through the whole bubble, once per in-flight
microbatch.

Usage
-----

Enable it with the ``sac_and_paged_stash`` memory policy on a model whose
dispatcher has a static capacity (HybridEP with ``non_blocking_capacity_factor``)::

    NGPU=4 MODULE=graph_trainer.deepseek_v3 \
        CONFIG=graph_trainer_deepseek_v3_debugmodel_hybridep \
        ./run_train.sh \
        --compile.mode aot_fx_trace \
        --compile.memory_policy sac_and_paged_stash \
        --parallelism.data_parallel_shard_degree 4 \
        --parallelism.expert_parallel_degree 2

With pipeline parallelism, where the feature pays for itself::

    NGPU=4 MODULE=graph_trainer.deepseek_v3 \
        CONFIG=graph_trainer_deepseek_v3_debugmodel_hybridep \
        ./run_train.sh \
        --training.disable_cuda_graphs \
        --compile.mode aot_fx_trace \
        --compile.memory_policy sac_and_paged_stash \
        --parallelism.pipeline_parallel_degree 2 \
        --parallelism.num_pp_microbatches 4 \
        --parallelism.pipeline_parallel_schedule Interleaved1F1B \
        --parallelism.data_parallel_shard_degree 2 \
        --parallelism.expert_parallel_degree 2

Options
-------

``--compile.paged_stash_page_size``
    Tokens per page (default 64). Smaller pages waste less on stashes that do
    not fill a page.
``--compile.paged_stash_buffer_size_factor_cuda``
    Headroom over the pages the measured step and the pipeline schedule
    require (default 1.10).
``--compile.paged_stash_buffer_size_factor_cpu``
    Headroom for an optional pinned-host spill buffer (default 0 = no host
    spill).
``--compile.paged_stash_prefetch_n_layers``
    Issue reloads this many backward layers early to overlap page reads with
    compute.
``--compile.paged_stash_module_fqn``
    Module-FQN prefix selecting eligible activations (default
    ``layers.*.moe.routed_experts``).
``--compile.paged_stash_skip_immediate_backward``
    Keep the last paged layer resident instead of stashing it when the schedule
    runs that microbatch's backward next (default on). Matches Megatron's
    ``remove_paged_tensor_from_stash``; switches itself off under CUDA graph
    capture.
``--compile.paged_stash_page_recomputed``
    Also page declared activations SAC would otherwise recompute, trading
    recompute for a stash round trip (default off).
``--compile.paged_stash_overflow_check``
    ``assert`` (default): device-side assertion, no host sync, fatal on
    overflow. ``blocking``: Megatron's per-step read, reruns an overflowing
    step with larger buffers. ``deferred``: non-blocking pinned-mirror read;
    the only mode that can apply one invalid step before growing the buffers.

How It Works
------------

1. **Annotation, not a context manager.** Megatron opens ``saved_tensors_hooks``
   around the fused grouped MLP and pages whatever Transformer Engine tagged
   inside it. There is no such window in a traced graph, so
   ``tag_paged_stash_activations`` annotates nodes -- following the same
   ``MUST_SAVE`` / ``MUST_CPU_OFFLOAD`` convention as the rest of the
   memory-policy framework -- and ``apply_paged_stash_pass`` rewrites them.
   It runs after selective activation remat, whose recomputed duplicates would
   otherwise keep reading the padded original.

2. **The forward stash is a mutation.** GraphPP defines its forward graph as
   the dependency closure of the forward user outputs. A value-returning stash
   reaches no forward output, so the whole chain would be classified as
   backward and the activation would still be saved. Writing the stash as a
   mutation of a small ``handle`` tensor routes it through GraphPP's existing
   forward-mutation path, so the handle -- not the padded tensor -- is what
   crosses into backward.

3. **Lifecycle.** The first step measures: ``stash_`` reads the real token
   count and parks the activation, so it is numerically identical to a run
   without the feature. The second allocates the buffers, primes the Triton
   kernels (Triton cannot compile inside a CUDA graph capture), and pages for
   real; CUDA graph capture happens on it. Later steps only reset freelists
   and flags, which allocates nothing.

4. **GraphPP replaces the virtual-pipeline estimate.** Megatron reconstructs
   its schedule to work out how many stashes are live at once.
   ``schedule_page_peak`` replays the real PP action list over a per-stage
   page footprint instead, which is both simpler and tighter.

5. **Sizing is in pages, not tokens.** Megatron accumulates raw token counts,
   which under-allocates when individual stashes are small relative to
   ``page_size``: four concurrent 16-token stashes need four pages, not one.

6. **Overflow grows the buffers instead of disabling the feature.** Detection
   is identical to Megatron -- a device-side flag from the Triton kernels, one
   all-reduce over ``(stash_overflow, overbudget, host_spill)``, where
   ``overbudget`` is HybridEP's token-drop flag accumulated over every
   dispatch of the step. Megatron then reruns the step with paged stash
   switched off, which it can do because the stash is an eager context
   manager. Here the stash ops are nodes in a compiled graph, so the retry
   releases the pages, scales the sizing factor, tears down the CUDA graph,
   discards the corrupted gradients, and reruns. A token drop is reported but
   never rerun: it comes from the capacity factor, which a rerun cannot change.

The Triton kernels and the ``PagedStashBuffer`` layout are copied from
Megatron-Core unchanged, so the page allocator, the pinned-host spill path, and
the overflow protocol behave identically.

Limitations
-----------

- Requires a token dispatcher with a static capacity factor. With a
  blocking/dynamic dispatcher the permuted buffer is already exactly sized and
  the pass tags nothing.
- Only activations SAC decided to keep are paged, plus -- with
  ``--compile.paged_stash_page_recomputed`` -- declared activations SAC would
  have recomputed.
- PP and CUDA graphs cannot be combined yet (a torchtitan-wide limitation), so
  a single run exercises either the CUDA-graph path or the GraphPP path, not
  both.
"""

from __future__ import annotations

import operator
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import torch

import triton
import triton.language as tl
from torch._library.custom_ops import custom_op
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineScheduleRuntime,
    BACKWARD_INPUT,
    FORWARD,
    FULL_BACKWARD,
)
from torch.fx import has_side_effect, Node
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.cudagraph import cudagraph_teardown
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _is_backward_node,
    _NOT_IN_LAYERS,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _collect_view_replay_info,
    _find_last_tensor_consumer,
    _get_storage_chain,
    _is_view,
    _tensor_bytes,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _default_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.registry import register_memory_policy
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


# ===========================================================================
# Triton kernels for MoE paged stash
# ===========================================================================
#
# Copied verbatim from Megatron-Core 0.19.0
# (``megatron/core/transformer/moe/ops/paged_stash.py``,
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES) so that GraphTrainer's
# paged stash is bit-for-bit the same page allocator, overflow protocol, and
# copy loop as the Megatron reference. Do not "clean up" these kernels: any
# divergence changes the freelist/overflow semantics the rest of this module
# and the Megatron reference both assume.


GLOBAL_BLOCK_SIZE = 1024


@triton.jit
def paged_stash_copy_kernel(
    src_ptr,
    cuda_dst_ptr,
    host_dst_ptr,
    num_tokens_ptr,
    free_list_cuda_ptr,
    free_list_host_ptr,
    free_list_head_ptr,  # shape (2,): [cuda_head, host_head]
    free_list_tail_ptr,  # shape (2,)
    free_list_capacity_ptr,
    page_record_ptr,
    overflow_ptr,
    host_spill_global_ptr,  # 1 if any successful host spill (not set on overflow path)
    spilled_to_host_ptr,  # Output: 0 = stored in CUDA, 1 = stored in host or overflow
    new_free_list_head_ptr,  # Output: shape (2,) updated heads
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_HOST_BUFFER: tl.constexpr,
):
    """Stash variable-length MoE activations into a paged buffer (CUDA, or pinned host).

    Uses a custom Triton kernel because the token count is only known at runtime and
    lives on device. Page allocation from the circular freelist, page_record metadata,
    and the activation copy are fused in one GPU launch to avoid host sync and keep
    stash CUDA-graph friendly. Fixed-size pages reduce fragmentation vs oversized
    static expert buffers.

    Per launch (program 0 handles metadata; all programs run the copy):
        1. If overflow is already set, restore freelist heads and return.
        2. Compute pages needed from num_tokens. Try the CUDA freelist; if full, try
           the host freelist when available; otherwise set overflow and return.
        3. Copy tokens in parallel: resolve page_id per token, record page_ids in
           page_record, write hidden vectors into the chosen CUDA or host pages.
        4. Program 0 writes updated freelist heads for the caller to copy_ back.
    """
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)

    # Load overflow first (get in flight early); branch on it only before any write
    overflow = tl.load(overflow_ptr)

    num_tokens = tl.load(num_tokens_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)

    # Common case: load only CUDA state (and head_host for output when use_cuda)
    head_cuda = tl.load(free_list_head_ptr)
    head_host = tl.load(free_list_head_ptr + 1)
    tail_cuda = tl.load(free_list_tail_ptr)
    cap_cuda = tl.load(free_list_capacity_ptr)

    avail_cuda = tail_cuda - head_cuda
    use_cuda = avail_cuda >= required_pages

    # Assume CUDA path: set everything for GPU stash
    spill = 0
    dst_ptr = cuda_dst_ptr
    free_list_ptr = free_list_cuda_ptr
    head = head_cuda
    cap = cap_cuda
    new_head_cuda = head_cuda + required_pages
    new_head_host = head_host

    if overflow == 1:
        # No stash; preserve heads so Python copy_ does not write garbage into the buffer.
        if pid == 0:
            tl.store(new_free_list_head_ptr, head_cuda)
            tl.store(new_free_list_head_ptr + 1, head_host)
        return

    # Only when CUDA is full: load host state and maybe switch to host
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
def paged_stash_pop_kernel(
    cuda_src_ptr,
    host_src_ptr,
    dst_ptr,
    num_tokens_ptr,
    page_record_ptr,
    spilled_to_host_ptr,  # 0 = read from CUDA, 1 = read from host
    overflow_ptr,
    free_list_cuda_ptr,
    free_list_host_ptr,
    free_list_tail_ptr,  # shape (2,)
    free_list_capacity_ptr,
    new_free_list_tail_ptr,  # Output: shape (2,) updated tails
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Restore variable-length MoE activations from a paged buffer (CUDA, or pinned host).

    Inverse of paged_stash_copy_kernel. Uses a custom Triton kernel for the same
    reasons: runtime token count and stash metadata live on device, so the reload,
    page_record lookup, and freelist recycle must fuse on-GPU without host sync.

    Per launch (program 0 handles metadata; all programs run the copy):
        1. If overflow is already set, restore freelist tails and return.
        2. Read spilled_to_host from the matching stash: CUDA buffer by default, host
           buffer when the forward stash spilled to pinned memory.
        3. Copy tokens in parallel: look up page_id from page_record, read hidden
           vectors from the stash pages into dst, return each page_id to the freelist.
        4. Program 0 writes updated freelist tails for the caller to copy_ back.
    """
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)

    # Load overflow first (get in flight early); branch on it only before any write
    overflow = tl.load(overflow_ptr)

    num_tokens = tl.load(num_tokens_ptr)
    spill = tl.load(spilled_to_host_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)

    # Common case: load only CUDA state (and tail_host for output when spill=0)
    tail_cuda = tl.load(free_list_tail_ptr)
    tail_host = tl.load(free_list_tail_ptr + 1)
    cap_cuda = tl.load(free_list_capacity_ptr)

    if overflow == 1:
        # No pop; preserve tails so Python copy_ does not write garbage into the buffer.
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

    # Only when spilled to host: load host state and switch
    if spill == 1:
        cap_host = tl.load(free_list_capacity_ptr + 1)
        if cap_host == 0:
            # Cannot pop from host; preserve tails (no-op for free-list state).
            if pid == 0:
                tl.store(new_free_list_tail_ptr, tail_cuda)
                tl.store(new_free_list_tail_ptr + 1, tail_host)
            return
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


# ===========================================================================
# Paged stash buffers and the process-wide stash manager
# ===========================================================================
#
# ``PagedStashBuffer`` is a structural port of Megatron-Core 0.19.0's
# ``megatron.core.transformer.moe.paged_stash.PagedStashBuffer``: the same
# ``[num_pages * page_size, hidden]`` layout, the same two-element
# ``[cuda, host]`` freelist state, and the same preallocated reset values so
# ``reset()`` never allocates (required inside a CUDA graph).
#
# ``PagedStashManager`` replaces Megatron's manager. Megatron's version also
# owned the pipeline schedule and drove stash/reload from autograd functions;
# here the FX graph owns ordering (see ``apply_paged_stash_pass``) and GraphPP owns the
# schedule (see ``schedule_page_peak``), so the manager keeps only what remains
# process-wide: the buffers, the shared overflow/host-spill flags, the
# pack/unpack streams, and the capture-phase token accounting.


# Megatron pages columnwise MXFP8 scale-inverse tensors at 1/32 the token
# granularity of their data tensor. GraphTrainer's MoE saves plain tensors, so
# the manager tracks a per-slot ``token_divisor`` instead of a boolean, with 1
# meaning "one row per token". Kept as a named constant so an MXFP8 GraphTrainer
# MoE can reuse the Megatron value directly.
SCALE_INV_BLOCK_SIZE = 32


class PagedStashBuffer:
    """A paged stash buffer with page-level memory management.

    Supports both CUDA and an optional pinned host buffer for overflow fallback.
    Buffers are organized as ``[num_pages * page_size, hidden_size]`` and use
    per-buffer circular freelists tracked as two-element state:
    ``[0] = CUDA``, ``[1] = host``.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        page_size: int,
        device: torch.device,
        overflow: torch.Tensor,
        host_spill: torch.Tensor,
        dtype: torch.dtype,
        num_tokens_host: int = 0,
    ) -> None:
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.overflow = overflow  # GPU flag (shared across all buffers)
        self.host_spill = host_spill  # GPU flag (shared across all buffers)

        # CUDA buffer
        self.num_cuda_pages = (num_tokens + page_size - 1) // page_size
        self.total_cuda_tokens = self.num_cuda_pages * page_size
        self.cuda_buffer = torch.empty(
            (self.total_cuda_tokens, hidden_size), dtype=dtype, device=device
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

        # Freelist state: shape (2,), index 0 = CUDA, 1 = host. Device memory so
        # the Triton kernels can read/update it without a host round trip.
        self.free_list_head = torch.zeros(2, dtype=torch.int64, device=device)
        self.free_list_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
        )
        self.free_list_capacity = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
        )

        # Freelist arrays (device memory): page IDs for each buffer.
        self.free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=device
        )
        if self.num_host_pages > 0:
            self.free_list_host = torch.arange(
                self.num_host_pages, dtype=torch.int64, device=device
            )
        else:
            self.free_list_host = torch.empty(0, dtype=torch.int64, device=device)

        # Preallocated reset values (CUDA graph safe: no allocation in reset()).
        self._reset_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
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

    def reset(self) -> None:
        """Reset both CUDA and host freelists (CUDA graph safe: no allocations)."""
        self.free_list_cuda.copy_(self._reset_free_list_cuda)
        self.free_list_head.zero_()
        self.free_list_tail.copy_(self._reset_tail)
        if self._reset_free_list_host is not None:
            self.free_list_host.copy_(self._reset_free_list_host)

    def nbytes(self) -> int:
        """Bytes of CUDA page memory held by this buffer."""
        return self.cuda_buffer.nelement() * self.cuda_buffer.element_size()

    def __repr__(self) -> str:
        return (
            f"PagedStashBuffer(num_cuda_pages={self.num_cuda_pages}, "
            f"num_host_pages={self.num_host_pages}, "
            f"page_size={self.page_size}, hidden_size={self.hidden_size}, "
            f"device={self.device}, dtype={self.dtype})"
        )


@dataclass(slots=True)
class StashSlot:
    """Static, per-graph-node description of one stashed activation.

    The FX pass assigns each tagged node a slot id and bakes it into the
    ``paged_stash`` op call, so the runtime can look up everything it needs
    without inspecting the tensor. ``max_num_tokens`` is the padded
    (capacity-factor sized) leading dimension; ``token_divisor`` is 1 for plain
    activations and ``SCALE_INV_BLOCK_SIZE`` for MXFP8 columnwise scale
    inverses, matching Megatron's ``is_columnwise_scale_inv`` handling.
    """

    slot_id: int
    dtype: torch.dtype
    hidden_size: int
    max_num_tokens: int
    token_divisor: int = 1
    layer_id: int = -1
    node_name: str = ""
    # Pipeline stage that owns this slot. Sizing needs it because a stashed
    # activation is live exactly as long as its stage's microbatch is in flight,
    # and different stages hold different numbers of microbatches.
    stage_index: int = 0

    @property
    def key(self) -> tuple[torch.dtype, int]:
        return (self.dtype, self.hidden_size)

    @property
    def max_stash_tokens(self) -> int:
        return self.max_num_tokens // self.token_divisor


# Megatron stores 1-byte float8 activations in a uint8 page buffer because
# Triton cannot address float8_e4m3fn/float8_e8m0fnu element types directly.
_BUFFER_DTYPE_OVERRIDE = {
    torch.float8_e4m3fn: torch.uint8,
    torch.float8_e8m0fnu: torch.uint8,
}


def buffer_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Storage dtype used for a stash buffer holding ``dtype`` activations."""
    return _BUFFER_DTYPE_OVERRIDE.get(dtype, dtype)


class PagedStashManager:
    """Process-wide owner of paged stash buffers, streams, and overflow flags.

    Lifecycle mirrors Megatron's ``begin -> capture -> captured`` state machine:

    ``begin``
        Buffers do not exist. The ``paged_stash`` ops run in measure mode: they
        keep the activation alive in Python and record its real (device-side)
        token count, exactly like Megatron's capture iteration, which reads
        ``num_tokens_tensor.item()`` before any buffer exists.
    ``capture``
        One full measured step has been observed. ``allocate_stash_buffers``
        turns the recorded peaks into real page buffers.
    ``captured``
        Steady state. The ops page activations through the buffers.

    GraphTrainer differs from Megatron in *where* the peak comes from: Megatron
    accumulates it while walking its virtual-pipeline schedule, while here the
    per-step peak is measured once and then multiplied by the pipeline
    concurrency that GraphPP reports (see ``schedule_page_peak``).
    """

    _INSTANCE: PagedStashManager | None = None

    @classmethod
    def get_instance(cls) -> PagedStashManager:
        if cls._INSTANCE is None:
            cls._INSTANCE = PagedStashManager()
        return cls._INSTANCE

    @classmethod
    def reset_instance(cls) -> None:
        cls._INSTANCE = None

    def __init__(self) -> None:
        self.enabled = False
        self.status = "begin"  # begin, capture, captured
        self.device: torch.device | None = None

        # Megatron uses one stream for both pack and unpack because paged
        # stashing is not stream-safe (the freelist is shared mutable state).
        # Keep that invariant.
        self._pack_stream: torch.cuda.Stream | None = None

        self.page_size = 64
        self.buffer_size_factor_cuda = 1.10
        self.buffer_size_factor_cpu = 0.0
        # Pipeline stage whose graph is currently being rewritten, stamped onto
        # each slot the pass registers. GraphPP sets it per stage; the non-PP
        # joint graph leaves it at 0.
        self.current_stage_index = 0
        # Pages per (dtype, hidden_size) that the GraphPP schedule says must be
        # resident at the pipeline's peak. Empty without pipeline parallelism.
        self.schedule_estimate: dict[tuple[torch.dtype, int], int] = {}

        self.stash_buffers: dict[torch.dtype, dict[int, PagedStashBuffer]] | None = None
        # Set whenever buffers are (re)allocated: the Triton kernels must be
        # JIT-compiled outside CUDA graph capture, and a fresh set of buffers
        # can introduce constexpr combinations that were never compiled.
        self.needs_kernel_warmup = False
        self.overflow: torch.Tensor | None = None
        self.host_spill: torch.Tensor | None = None

        # slot_id -> StashSlot, registered by the graph pass.
        self.slots: dict[int, StashSlot] = {}
        self._next_slot_id = 0

        # Capture-phase accounting, mirroring Megatron's
        # temp_/max_tokens_across_vp_stages pair: ``temp`` is the currently live
        # token count per (dtype, hidden_size), ``max`` its running peak.
        self._live_tokens: dict[tuple[torch.dtype, int], int] = {}
        self._peak_tokens: dict[tuple[torch.dtype, int], int] = {}
        # Same accounting in whole pages, so buffer sizing accounts for the fact
        # that every individual stash rounds up to a page boundary.
        self._live_pages: dict[tuple[torch.dtype, int], int] = {}
        self._peak_pages: dict[tuple[torch.dtype, int], int] = {}

        # Largest live token/page count observed for a *single* stash of each
        # slot. Together with the GraphPP pipeline concurrency this reproduces
        # Megatron's per-schedule-entry accumulation without needing to model
        # the virtual pipeline by hand.
        self._slot_tokens: dict[int, int] = {}

        # Measure-mode activation parking: slot_id -> queued tensors. Megatron
        # keeps the real tensor on the PagedTensor during its capture iteration;
        # this is the same trick, keyed by the static slot id. A queue is needed
        # because with pipeline parallelism the same slot is stashed once per
        # in-flight microbatch before the first one is reloaded, and 1F1B
        # reloads them in stash order.
        self._measured_tensors: dict[
            int, deque[tuple[torch.Tensor, int]]
        ] = defaultdict(deque)

        # Skip-stash (Megatron's `current_schedule_layer != -next_schedule_layer`
        # test). `_skip_plan` is one flag per forward of this rank's schedule,
        # `_forward_ordinal` indexes into it, and `_skipped` records per slot,
        # in stash order, whether that stash was skipped so the paired reload
        # knows whether to pop pages or hand back the resident tensor. The
        # queues mirror `_measured_tensors`: with pipeline parallelism a slot is
        # stashed once per in-flight microbatch and reloaded in stash order.
        self.skip_immediate_backward = False
        self._skip_static: bool | None = None
        self._skip_plan: list[bool] | None = None
        self._forward_ordinal = -1
        self._skipped: dict[int, deque[bool]] = defaultdict(deque)
        self._resident: dict[int, deque[torch.Tensor]] = defaultdict(deque)
        self._warned_capture_skip = False

        self.step = 0

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------
    @property
    def pack_stream(self) -> torch.cuda.Stream:
        """Stream used for both stash and reload copies."""
        if self._pack_stream is None:
            self._pack_stream = torch.cuda.Stream()
        return self._pack_stream

    @property
    def unpack_stream(self) -> torch.cuda.Stream:
        # Paged stashing is not stream-safe (shared freelist), so pack and
        # unpack share one stream. Same as Megatron.
        return self.pack_stream

    # ------------------------------------------------------------------
    # Slot registration (called by the graph pass at compile time)
    # ------------------------------------------------------------------
    def next_slot_id(self) -> int:
        """Allocate a slot id. Ids are process-global so GraphPP stage graphs,
        which are traced and rewritten independently, never collide."""
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        return slot_id

    def register_slot(self, slot: StashSlot) -> None:
        existing = self.slots.get(slot.slot_id)
        if existing is not None and existing != slot:
            raise ValueError(
                f"paged stash slot id {slot.slot_id} reused with different "
                f"metadata: {existing} vs {slot}"
            )
        self.slots[slot.slot_id] = slot

    def clear_slots(self) -> None:
        self.slots.clear()

    # ------------------------------------------------------------------
    # Capture-phase accounting
    # ------------------------------------------------------------------
    def record_stash(self, slot: StashSlot, num_tokens: int) -> None:
        """Record one measured stash and update the per-key live/peak counters."""
        key = slot.key
        pages = (num_tokens + self.page_size - 1) // self.page_size
        self._live_tokens[key] = self._live_tokens.get(key, 0) + num_tokens
        self._live_pages[key] = self._live_pages.get(key, 0) + pages
        self._peak_tokens[key] = max(
            self._peak_tokens.get(key, 0), self._live_tokens[key]
        )
        self._peak_pages[key] = max(self._peak_pages.get(key, 0), self._live_pages[key])
        self._slot_tokens[slot.slot_id] = max(
            self._slot_tokens.get(slot.slot_id, 0), num_tokens
        )

    def record_reload(self, slot: StashSlot, num_tokens: int) -> None:
        """Release one measured stash from the live counters."""
        key = slot.key
        pages = (num_tokens + self.page_size - 1) // self.page_size
        if key in self._live_tokens:
            self._live_tokens[key] -= num_tokens
            self._live_pages[key] -= pages

    def park_measured_tensor(
        self, slot_id: int, tensor: torch.Tensor, num_tokens: int
    ) -> None:
        """Park an activation and its measured token count for the reload.

        The count travels with the tensor so the reload can release it from the
        live counters without a second device-to-host read.
        """
        self._measured_tensors[slot_id].append((tensor, num_tokens))

    def pop_measured_tensor(self, slot_id: int) -> tuple[torch.Tensor, int]:
        queue = self._measured_tensors.get(slot_id)
        if not queue:
            raise RuntimeError(
                f"paged stash measure mode: no parked activation for slot {slot_id}. "
                "Every paged_stash.reload must follow its paged_stash.stash."
            )
        return queue.popleft()

    # ------------------------------------------------------------------
    # Skip stash on immediate backward
    # ------------------------------------------------------------------
    @property
    def skip_enabled(self) -> bool:
        """Whether any stash may be skipped this step."""
        return self.skip_immediate_backward and (
            self._skip_static is not None or self._skip_plan is not None
        )

    def set_static_skip(self, skip: bool) -> None:
        """Install a decision that is the same for every forward of every step.

        Only correct where the schedule cannot vary between executions of the
        compiled graph, which is why the caller -- not this class -- decides
        when it applies. Being invariant is also what makes it safe under CUDA
        graph capture, unlike :meth:`set_skip_plan`.
        """
        self._skip_static = skip

    def set_skip_plan(self, plan: list[bool]) -> None:
        """Install one skip flag per forward, in this rank's execution order."""
        self._skip_plan = plan

    def begin_forward(self) -> None:
        """Advance the schedule cursor. Called once per forward that stashes."""
        self._forward_ordinal += 1

    def should_skip_stash(self) -> bool:
        """Whether the current forward's stash can be kept resident instead.

        Returns False -- i.e. stash normally -- whenever the answer is not
        certain: no plan, cursor past the end of the plan, or a plan that would
        have to be re-read on every CUDA graph replay. The last case is the
        important one. A captured graph freezes host-side Python at capture
        time, so a per-microbatch decision read here would be baked in and
        replayed against microbatches it was not computed for. Stashing is
        correct under every schedule, so that is the safe direction to fail;
        the optimization simply switches off. Making it survive capture needs
        two captured variants of the stage graph, one stashing and one not,
        selected by the GraphPP runtime -- a runtime change, not a change here.
        """
        if not self.skip_enabled:
            return False
        if self._skip_static is not None:
            return self._skip_static
        if torch.cuda.is_current_stream_capturing():
            if not self._warned_capture_skip:
                self._warned_capture_skip = True
                logger.warning(
                    "Paged stash: skip-on-immediate-backward disabled under CUDA "
                    "graph capture; the per-microbatch decision cannot be replayed. "
                    "Stashing every forward instead (correct, just not optimized)."
                )
            return False
        plan = self._skip_plan
        assert plan is not None  # guarded by skip_enabled
        if not 0 <= self._forward_ordinal < len(plan):
            return False
        return plan[self._forward_ordinal]

    def note_stash(self, slot_id: int, skipped: bool) -> None:
        """Record, in stash order, whether this slot's stash was skipped."""
        self._skipped[slot_id].append(skipped)

    def take_stash_skipped(self, slot_id: int) -> bool:
        """Pop the oldest outstanding stash flag for ``slot_id``."""
        queue = self._skipped.get(slot_id)
        if not queue:
            return False
        return queue.popleft()

    def keep_resident(self, slot_id: int, tensor: torch.Tensor) -> None:
        """Hold a skipped activation so the allocator cannot reuse it.

        Megatron gets this for free: skipping means never running
        ``offload_to_stash``, which is what clears ``PagedTensor._tensor``. Our
        stash and reload are separate graph nodes, so the reference has to live
        somewhere between them, and that is here.
        """
        self._resident[slot_id].append(tensor)

    def pop_resident(self, slot_id: int) -> torch.Tensor:
        queue = self._resident.get(slot_id)
        if not queue:
            raise RuntimeError(
                f"paged stash: slot {slot_id} was recorded as skipped but no "
                "resident activation is held for it. stash_ and reload must "
                "pair one-to-one and reload in stash order."
            )
        return queue.popleft()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------
    def stage_page_footprint(self) -> dict[int, dict[tuple[torch.dtype, int], int]]:
        """Pages one in-flight microbatch of each pipeline stage keeps stashed.

        Keyed by stage, then by ``(dtype, hidden_size)``. ``schedule_page_peak`` replays
        the GraphPP schedule over this table to find the pipeline-wide peak,
        which is the quantity Megatron reconstructs by hand from its
        virtual-pipeline order.
        """
        footprint: dict[int, dict[tuple[torch.dtype, int], int]] = {}
        for slot_id, tokens in self._slot_tokens.items():
            slot = self.slots.get(slot_id)
            if slot is None:
                continue
            pages = (tokens + self.page_size - 1) // self.page_size
            stage = footprint.setdefault(slot.stage_index, {})
            stage[slot.key] = stage.get(slot.key, 0) + pages
        return footprint

    def allocate_stash_buffers(self) -> None:
        """Allocate page buffers from the measured peak and the schedule estimate.

        Sizing follows Megatron -- peak concurrent usage per ``(dtype,
        hidden_size)`` scaled by ``buffer_size_factor_cuda``, plus an optional
        pinned host buffer scaled by ``..._cpu`` -- with two changes the graph
        representation makes possible:

        * Sizing is in **pages**, not tokens. Megatron accumulates raw token
          counts, which under-allocates whenever an individual stash is small
          relative to ``page_size``: four concurrent 16-token stashes need four
          pages, but their 64-token sum fits in one.
        * The measured peak is combined with a **schedule estimate** obtained
          by replaying the GraphPP pipeline schedule over the per-stage
          footprint (see ``stage_page_footprint`` and ``schedule_page_peak``) instead of
          trusting a single observed step, so buffers are sized for the
          steady-state pipeline depth even if the measure step was shallower.
        """
        if self.device is None:
            raise RuntimeError("paged stash: device is unset; nothing was stashed.")

        if self.overflow is None:
            self.overflow = torch.zeros(1, dtype=torch.int64, device=self.device)
        else:
            self.overflow.zero_()
        if self.host_spill is None:
            self.host_spill = torch.zeros(1, dtype=torch.int64, device=self.device)
        else:
            self.host_spill.zero_()

        if not self._peak_pages:
            logger.info(
                "Paged stash: no activations were stashed during the measure step; "
                "skipping stash buffer allocation."
            )
            self.stash_buffers = {}
            return

        schedule_estimate = self.schedule_estimate
        self.needs_kernel_warmup = True

        self.stash_buffers = {}
        total_bytes = 0
        for key, measured_pages in self._peak_pages.items():
            dtype, hidden_size = key
            required_pages = max(measured_pages, schedule_estimate.get(key, 0))
            num_tokens = (
                int(required_pages * self.buffer_size_factor_cuda) * self.page_size
            )
            num_tokens_host = (
                int(required_pages * self.buffer_size_factor_cpu) * self.page_size
                if self.buffer_size_factor_cpu > 0
                else 0
            )
            logger.debug(
                "  paged stash sizing %s/%d: measured=%d pages, schedule=%d pages",
                dtype,
                hidden_size,
                measured_pages,
                schedule_estimate.get(key, 0),
            )

            buffer = PagedStashBuffer(
                num_tokens,
                hidden_size,
                self.page_size,
                self.device,
                self.overflow,
                self.host_spill,
                buffer_dtype_for(dtype),
                num_tokens_host=num_tokens_host,
            )
            self.stash_buffers.setdefault(dtype, {})[hidden_size] = buffer
            total_bytes += buffer.nbytes()
            message = (
                f"Paged stash: allocated cuda {tuple(buffer.cuda_buffer.shape)} "
                f"pages={buffer.num_cuda_pages}"
            )
            if buffer.host_buffer is not None:
                message += f" host {tuple(buffer.host_buffer.shape)}"
            message += f" dtype={buffer.dtype} ({dtype})"
            logger.info(message)

        logger.info(
            "Paged stash: %d buffer(s), %.2f MiB of CUDA pages "
            "(page_size=%d, factor=%.2f)",
            sum(len(v) for v in self.stash_buffers.values()),
            total_bytes / (1024 * 1024),
            self.page_size,
            self.buffer_size_factor_cuda,
        )

    def release_stash_buffers(self) -> None:
        """Drop the large page buffers, keeping the small shared flags.

        Called on the fallback path after the CUDA graph that references these
        pointers has been torn down, matching Megatron's
        ``PagedStashManager.release_stash_buffers``.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stash_buffers = None
        logger.info(
            "Paged stash: released stash page buffers after fallback "
            "(reallocated on the next stash reset)."
        )

    def get_buffer(self, dtype: torch.dtype, hidden_size: int) -> PagedStashBuffer:
        if self.stash_buffers is None:
            raise RuntimeError("paged stash: buffers have not been allocated yet.")
        try:
            return self.stash_buffers[dtype][hidden_size]
        except KeyError:
            raise RuntimeError(
                f"paged stash: no buffer for dtype={dtype} hidden_size={hidden_size}. "
                f"Known: {[(d, list(h)) for d, h in self.stash_buffers.items()]}"
            ) from None

    # ------------------------------------------------------------------
    # Per-step lifecycle
    # ------------------------------------------------------------------
    def begin_step(self) -> None:
        """Advance the state machine and reset per-step state.

        Mirrors Megatron's ``paged_stash_reset``, which runs at the top of every
        ``forward_backward_func`` call.
        """
        if not self.enabled:
            return
        self.step += 1
        self._measured_tensors.clear()
        self._skipped.clear()
        self._resident.clear()
        self._forward_ordinal = -1
        self._live_tokens = {key: 0 for key in self._live_tokens}
        self._live_pages = {key: 0 for key in self._live_pages}

        if self.status == "begin":
            self.status = "capture"
        elif self.status == "capture":
            self.status = "captured"
            self.allocate_stash_buffers()
        elif self.status == "captured" and self.stash_buffers is None:
            # Buffers were released by a fallback; reallocate from the same
            # measured peaks.
            self.allocate_stash_buffers()

        if self.status == "captured" and self.stash_buffers:
            for buffers in self.stash_buffers.values():
                for buffer in buffers.values():
                    buffer.reset()
            assert self.overflow is not None and self.host_spill is not None
            self.overflow.zero_()
            self.host_spill.zero_()

    @property
    def active(self) -> bool:
        """True when the ops should page through real buffers."""
        return self.enabled and self.status == "captured" and bool(self.stash_buffers)

    def summary(self) -> str:
        if not self.stash_buffers:
            return "paged stash: no buffers"
        parts = []
        for dtype, buffers in self.stash_buffers.items():
            for hidden_size, buffer in buffers.items():
                parts.append(
                    f"{dtype}/{hidden_size}: {buffer.num_cuda_pages} pages "
                    f"({buffer.nbytes() / (1024 * 1024):.1f} MiB)"
                )
        return "paged stash buffers: " + ", ".join(parts)


def check_paged_stash_overflow() -> torch.Tensor:
    """Device-side flag: did any stash fail to find a free page this step?"""
    manager = PagedStashManager.get_instance()
    if not manager.enabled or manager.overflow is None:
        device = manager.device or torch.device("cuda")
        return torch.zeros(1, dtype=torch.bool, device=device)
    return manager.overflow.ne(0)


def check_paged_stash_host_spill() -> torch.Tensor:
    """Device-side flag: did any activation spill to pinned host memory?"""
    manager = PagedStashManager.get_instance()
    if not manager.enabled or manager.host_spill is None:
        device = manager.device or torch.device("cuda")
        return torch.zeros(1, dtype=torch.bool, device=device)
    return manager.host_spill.ne(0)


# ===========================================================================
# Custom ops that page MoE activations in and out of the stash buffers
# ===========================================================================
#
# These are the FX-visible form of Megatron's ``PagedTensor.offload_to_stash`` /
# ``reload_from_stash``. Megatron drives those from ``saved_tensors_hooks`` plus a
# pair of autograd functions that walk its pipeline schedule; GraphTrainer has no
# autograd tape at graph-execution time, so the same two kernels are exposed as
# ops and ``apply_paged_stash_pass`` places them.
#
# The resulting IR is::
#
#     Forward:  handle = aten.empty([max_pages + 1], int64)
#               paged_stash.stash_(handle, act, num_tokens, slot)
#               paged_stash.wait_stash_(handle, act, last_fwd_use)
#
#     Backward: act = paged_stash.reload(handle, num_tokens, slot, size, dtype)
#               act = paged_stash.wait(act)
#
# ``handle`` is one small int64 tensor: ``[:max_pages]`` records which page backs
# each chunk of the activation, and the last element is the spill flag. It is the
# only thing that crosses into backward, in place of the padded activation.
#
# The two forward ops are *mutations* of ``handle`` rather than value-returning
# ops, and that is load-bearing rather than cosmetic. GraphPP defines its forward
# graph as the dependency closure of the forward user outputs; a value-returning
# stash reaches no forward output, so the whole chain would be classified as
# backward, the activation would still be saved, and the pass would achieve
# nothing. Expressing the stash as a forward-side mutation puts it through
# GraphPP's existing ``_forward_mutations_to_materialize`` path, which runs the
# mutation in the forward callable and saves the mutated base -- the handle --
# for backward.
#
# ``paged_stash.wait_stash_`` is what makes the memory saving real: it joins the
# pack stream back into the compute stream, after which the padded activation has
# no remaining users and FX frees it. That is the same moment Megatron drops
# ``PagedTensor._original_tensor`` inside ``wait_for_stash_to_complete``.
#
# All stash and reload copies run on a single side stream. Megatron shares one
# stream between pack and unpack because the freelist is unsynchronized mutable
# state; that invariant is preserved here.


# Megatron caps the copy grid at 2048 programs; the kernels stride over tokens
# so this only bounds parallelism, not correctness.
_MAX_BLOCKS = 2048

# --- Wait registry: data_ptr() -> (completion event, device, retained inputs).
#
# Same shape as ao's registry, plus the ``retain`` tuple, which exists because
# the stash and reload kernels run on a side stream while their *inputs* are
# ordinary graph values owned by the compute stream. FX emits a ``del`` for a
# value right after its last graph use, so without an extra reference the
# caching allocator would hand ``page_record`` / ``spilled`` / ``num_tokens``
# to the next compute-stream allocation while the pop kernel is still reading
# them. Megatron never hits this because its ``PagedTensor`` keeps those
# tensors alive for the whole iteration; retaining them until the matching
# ``wait`` reproduces that lifetime.
_wait_registry: dict[
    int, tuple[torch.Event, torch.device, tuple[torch.Tensor, ...]]
] = {}


def _register_wait(
    tensor: torch.Tensor,
    device: torch.device,
    retain: tuple[torch.Tensor, ...] = (),
) -> torch.Event:
    """Record the event ``tensor``'s pending copy will signal, and what it reads.

    ``tensor`` is retained alongside ``retain`` -- not for its own sake, but to
    keep the key valid. The registry is addressed by device pointer, which is
    only unique while that address stays claimed: drop the last reference and
    the caching allocator hands the same address to the next allocation, whose
    registration then overwrites a live entry and releases its retained inputs
    while the copy kernel is still reading them. Holding the keyed tensor makes
    "a key is the address of a tensor the registry keeps alive" an invariant
    rather than an assumption.
    """
    key = tensor.data_ptr()
    if key in _wait_registry:
        raise RuntimeError(
            f"paged stash: a wait is already registered at address {key:#x}. "
            "Every stash_/reload must be paired with exactly one "
            "wait_stash_/wait before the next use of that buffer."
        )
    event = torch.Event()
    _wait_registry[key] = (event, device, retain + (tensor,))
    return event


def _pop_wait(
    tensor: torch.Tensor,
) -> tuple[torch.Event, torch.device, tuple[torch.Tensor, ...]] | None:
    return _wait_registry.pop(tensor.data_ptr(), None)


def clear_wait_registry() -> None:
    _wait_registry.clear()


def _slot(slot_id: int) -> StashSlot:
    manager = PagedStashManager.get_instance()
    try:
        return manager.slots[slot_id]
    except KeyError:
        raise RuntimeError(
            f"paged stash: unknown slot id {slot_id}. The graph pass must "
            "register every slot before the graph runs."
        ) from None


def _check_slot_dims(activation: torch.Tensor, slot: StashSlot) -> int:
    """Validate that ``activation`` really has the slot's row layout.

    The kernels address the activation as ``view(-1, hidden_size)`` and index a
    prefix of at most ``max_stash_tokens`` rows. If the slot's row layout does not
    describe this tensor, both of those are silently wrong and the copy walks
    off the end of the buffer -- an illegal access far from its cause. Element
    counts alone are ambiguous, so check them here where the real tensor is in
    hand and the numbers can be named in the error.
    """
    numel = activation.numel()
    if numel % slot.hidden_size:
        raise RuntimeError(
            f"paged stash slot {slot.slot_id} ({slot.node_name}): tensor with "
            f"{numel} elements is not a whole number of {slot.hidden_size}-wide "
            "rows; the slot's row layout does not describe this tensor."
        )
    rows = numel // slot.hidden_size
    if rows > slot.max_stash_tokens:
        raise RuntimeError(
            f"paged stash slot {slot.slot_id} ({slot.node_name}): tensor has "
            f"{rows} rows of {slot.hidden_size} but the slot was sized for at "
            f"most {slot.max_stash_tokens} (max_num_tokens="
            f"{slot.max_num_tokens}, token_divisor={slot.token_divisor}). "
            "Paging it would write past the page record."
        )
    return rows


def _num_tokens_arg(num_tokens: torch.Tensor, slot: StashSlot) -> torch.Tensor:
    """Shape the device-side token count the way the Triton kernels expect.

    Megatron reshapes a 0-dim count to ``(1,)`` and divides by
    ``SCALE_INV_BLOCK_SIZE`` for columnwise scale inverses; ``token_divisor``
    generalizes that second step.
    """
    if num_tokens.dim() == 0:
        num_tokens = num_tokens.reshape(1)
    if slot.token_divisor != 1:
        num_tokens = num_tokens // slot.token_divisor
    return num_tokens


def _handle_views(
    handle: torch.Tensor, max_pages: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the handle into its page-record and spill-flag views.

    Megatron keeps these as two tensors on the ``PagedTensor``. One tensor is
    used here so the forward stash has a single mutated base for GraphPP to
    save, and so backward carries one saved value per activation instead of two.
    """
    return handle[:max_pages], handle[max_pages : max_pages + 1]


@custom_op("paged_stash::stash_", mutates_args=("handle",))
def stash_(
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    group_index: int = -1,
    skippable: bool = False,
) -> None:
    """Copy the live rows of ``activation`` into the paged stash, into ``handle``.

    ``handle`` must be an ``int64`` tensor of ``max_pages + 1`` elements. On
    return (after the paired ``wait_stash_``) it holds the page id backing each
    ``page_size``-token chunk, plus a trailing flag recording whether the stash
    landed in the pinned host buffer.

    Callers MUST pair this with ``paged_stash.wait_stash_(handle, activation)``
    so the compute stream joins the pack stream before ``activation`` is freed.

    ``group_index`` is 0 on exactly one stash per forward -- the first in graph
    order -- which is what advances the schedule cursor, and -1 everywhere else.
    ``skippable`` marks every tensor of the last tagged layer, the only layer
    whose backward can be the next thing the schedule runs, so the only one
    eligible to stay resident instead of making a stash round trip. Megatron
    likewise skips a layer's tensors as a group: ``remove_paged_tensor_from_stash``
    clears the whole pending list.
    """
    manager = PagedStashManager.get_instance()
    slot = _slot(slot_id)
    device = activation.device
    max_pages = handle.numel() - 1

    if not manager.active:
        # Measure phase. Megatron's capture iteration reads the real token count
        # off the device and keeps the activation alive; do the same, and leave
        # the handle zeroed so the graph shape never changes between phases.
        manager.device = device
        count = int(num_tokens.reshape(-1)[0].item()) // slot.token_divisor
        manager.record_stash(slot, count)
        manager.park_measured_tensor(slot_id, activation, count)
        handle.zero_()
        return

    _check_slot_dims(activation, slot)

    if manager.skip_enabled:
        if group_index == 0:
            # The one stash the pass marked as this forward's cursor tick. It
            # runs before any skip decision is taken, so the cursor advances
            # even in a graph whose only tagged layer is the skippable one.
            manager.begin_forward()
        if skippable and manager.should_skip_stash():
            # Megatron's `remove_paged_tensor_from_stash`: no copy, no pages,
            # the activation just stays where it is. Zero the handle so the
            # graph shape is unchanged and nothing reads a stale page record.
            handle.zero_()
            manager.keep_resident(slot_id, activation)
            manager.note_stash(slot_id, True)
            return
        manager.note_stash(slot_id, False)

    buffer = manager.get_buffer(slot.dtype, slot.hidden_size)
    current_stream = torch.cuda.current_stream(device)
    page_record, spilled = _handle_views(handle, max_pages)

    # Derive the token count *before* the stream join. For a scale slot
    # ``token_divisor != 1``, so this enqueues a division on the compute
    # stream, and ``wait_stream`` only covers work already enqueued when it
    # runs -- ordering it afterwards lets the copy kernel read the count before
    # the division has written it. Data slots divide by 1 and launch nothing,
    # which is why only the MXFP8 scale slots ever raced.
    tokens = _num_tokens_arg(num_tokens, slot)

    pack_stream = manager.pack_stream
    pack_stream.wait_stream(current_stream)

    # Keep every tensor the copy kernel reads alive until the matching wait.
    completion_event = _register_wait(
        handle, device, retain=(activation, num_tokens, tokens)
    )
    num_blocks = min(slot.max_stash_tokens, _MAX_BLOCKS)
    has_host = 1 if buffer.host_buffer is not None else 0
    host_dst = (
        buffer.host_buffer if buffer.host_buffer is not None else buffer.cuda_buffer
    )

    with torch.cuda.stream(pack_stream):
        # Freelist scratch never leaves the pack stream, so it must be
        # allocated here: allocating on the compute stream and then freeing it
        # at op exit would let the allocator hand the block to another
        # compute-stream tensor while the copy kernel is still reading it.
        new_free_list_head = torch.empty(2, dtype=torch.int64, device=device)
        handle.zero_()
        source = activation.contiguous().view(-1, slot.hidden_size)
        paged_stash_copy_kernel[(num_blocks,)](
            source.view(buffer.cuda_buffer.dtype),
            buffer.cuda_buffer,
            host_dst,
            tokens,
            buffer.free_list_cuda,
            buffer.free_list_host,
            buffer.free_list_head,
            buffer.free_list_tail,
            buffer.free_list_capacity,
            page_record,
            buffer.overflow,
            buffer.host_spill,
            spilled,
            new_free_list_head,
            PAGE_SIZE=buffer.page_size,
            HIDDEN_SIZE=slot.hidden_size,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
            HAS_HOST_BUFFER=has_host,
        )
        buffer.free_list_head.copy_(new_free_list_head)
        pack_stream.record_event(completion_event)


@stash_.register_fake
def _stash_fake(
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    group_index: int = -1,
    skippable: bool = False,
) -> None:
    return None


@custom_op("paged_stash::wait_stash_", mutates_args=("handle",))
def wait_stash_(
    handle: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> None:
    """Complete the asynchronous write to ``handle`` started by ``stash_``.

    Declared as a mutation because that is exactly what it is from the compute
    stream's point of view: ``handle``'s contents only become visible here.
    Being a mutation is also what makes GraphPP schedule it into the forward
    callable (see the module docstring).

    ``keepalive`` is the padded activation. Passing it as an argument pins it
    across the async copy; this call is its last use in the graph, so FX frees
    it immediately afterwards. ``last_use_of_storage`` carries no data and only
    orders an aliasing forward consumer before this join.
    """
    registered = _pop_wait(handle)
    if registered is None:
        # Measure phase: nothing was stashed, so there is nothing to join.
        return

    completion_event, device, retained = registered
    torch.cuda.current_stream(device).wait_event(completion_event)
    # The compute stream is now ordered after the copy kernel, so the retained
    # inputs may go back to the allocator.
    del retained, keepalive, last_use_of_storage


@wait_stash_.register_fake
def _wait_stash_fake(
    handle: torch.Tensor,
    keepalive: torch.Tensor | None = None,
    last_use_of_storage: torch.Tensor | None = None,
) -> None:
    return None


@custom_op("paged_stash::reload", mutates_args=())
def reload(
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore a stashed activation from its pages on the unpack stream.

    The destination is allocated on the compute stream (so the caching allocator
    owns it there) and filled on the unpack stream, matching both Megatron's
    ``reload_paged_tensors`` and upstream ``ao::reload``.
    """
    manager = PagedStashManager.get_instance()
    slot = _slot(slot_id)
    device = handle.device

    if not manager.active:
        # Measure phase: hand back the activation the stash parked, so warmup
        # steps are numerically identical to a run without paged stash, and
        # release its pages from the live counters. Without the release the
        # "peak" would be the sum of every stash in the step rather than the
        # largest set alive at one time -- which with pipeline parallelism
        # over-sizes the buffers by the microbatch count.
        activation, count = manager.pop_measured_tensor(slot_id)
        manager.record_reload(slot, count)
        return activation

    if manager.skip_enabled and manager.take_stash_skipped(slot_id):
        # The paired stash kept this activation resident rather than paging it,
        # so there is nothing to pop; hand the original straight back. Reloads
        # arrive in stash order, which is what keeps the two queues aligned.
        return manager.pop_resident(slot_id)

    buffer = manager.get_buffer(slot.dtype, slot.hidden_size)
    current_stream = torch.cuda.current_stream(device)
    max_pages = handle.numel() - 1
    page_record, spilled = _handle_views(handle, max_pages)

    result = torch.empty(size, dtype=dtype, device=device)

    # Before the join, for the reason given in ``stash_``.
    tokens = _num_tokens_arg(num_tokens, slot)

    unpack_stream = manager.unpack_stream
    unpack_stream.wait_stream(current_stream)

    # ``handle`` and the token count are dead in the graph the moment this op
    # returns, but the pop kernel has not run yet.
    completion_event = _register_wait(
        result, device, retain=(handle, page_record, spilled, num_tokens, tokens)
    )
    num_blocks = min(slot.max_stash_tokens, _MAX_BLOCKS)
    host_src = (
        buffer.host_buffer if buffer.host_buffer is not None else buffer.cuda_buffer
    )

    with torch.cuda.stream(unpack_stream):
        # Unpack-stream-local scratch; see the note in ``stash_``.
        new_free_list_tail = torch.empty(2, dtype=torch.int64, device=device)
        paged_stash_pop_kernel[(num_blocks,)](
            buffer.cuda_buffer,
            host_src,
            result.view(-1, slot.hidden_size).view(buffer.cuda_buffer.dtype),
            tokens,
            page_record,
            spilled,
            buffer.overflow,
            buffer.free_list_cuda,
            buffer.free_list_host,
            buffer.free_list_tail,
            buffer.free_list_capacity,
            new_free_list_tail,
            PAGE_SIZE=buffer.page_size,
            HIDDEN_SIZE=slot.hidden_size,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
        )
        buffer.free_list_tail.copy_(new_free_list_tail)
        unpack_stream.record_event(completion_event)

    return result


@reload.register_fake
def _reload_fake(
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(size, dtype=dtype, device=handle.device)


# ``paged_stash::wait`` needs an aliasing schema (the output aliases the input),
# which ``custom_op`` forbids, so it is defined through torch.library directly --
# the same reason ``ao::wait_tensor`` is. Unlike the forward half this one has to
# return a value: its result is what backward consumes, which is also what keeps
# it inside GraphPP's backward graph.
_lib = torch.library.Library("paged_stash", "FRAGMENT")
_lib.define("wait(Tensor(a) tensor) -> Tensor(a)")


@torch.library.impl("paged_stash::wait", "CompositeExplicitAutograd")
def _paged_stash_wait(tensor: torch.Tensor) -> torch.Tensor:
    """Join the unpack stream into compute before backward reads the reload."""
    registered = _pop_wait(tensor)
    if registered is None:
        # Measure phase: ``reload`` handed back the parked activation, which
        # needs no synchronization.
        return tensor
    completion_event, device, retained = registered
    torch.cuda.current_stream(device).wait_event(completion_event)
    del retained
    return tensor


@torch.library.register_fake("paged_stash::wait")
def _paged_stash_wait_fake(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


has_side_effect(torch.ops.paged_stash.wait.default)


# ``paged_stash::mark_pageable`` is the declaration half of the stash contract:
# the producer of an activation states how it is laid out, instead of the pass
# inferring it. Transformer Engine does the same thing for Megatron with
# ``mark_grouped_tensor``, which stamps ``grouped_tensor_scale_inv`` on the
# tensors its grouped MLP saves; Megatron then pages exactly what is stamped and
# never has to guess. We cannot stamp a tensor attribute because the value the
# pass sees is an FX node, not a tensor, so the declaration has to be an op that
# survives tracing. It is an identity: ``apply_paged_stash_pass`` reads the
# declaration off the node and erases the call, so nothing survives into the
# executed graph.
_lib.define(
    "mark_pageable(Tensor(a) tensor, int token_divisor, int hidden_size) -> Tensor(a)"
)


@torch.library.impl("paged_stash::mark_pageable", "CompositeExplicitAutograd")
def _mark_pageable(
    tensor: torch.Tensor, token_divisor: int, hidden_size: int
) -> torch.Tensor:
    """Identity. The declaration is carried by the call, not by the value."""
    del token_divisor, hidden_size
    return tensor


@torch.library.register_fake("paged_stash::mark_pageable")
def _mark_pageable_fake(
    tensor: torch.Tensor, token_divisor: int, hidden_size: int
) -> torch.Tensor:
    del token_divisor, hidden_size
    return tensor


_MARKING_ENABLED = False


def set_marking_enabled(enabled: bool) -> None:
    """Turn declarations on. Off by default so a run without paged stashing
    traces no ``mark_pageable`` calls at all, rather than identity ops the pass
    would have to clean up in a graph it never sees."""
    global _MARKING_ENABLED
    _MARKING_ENABLED = enabled


def mark_pageable(
    tensor: torch.Tensor, *, token_divisor: int = 1, hidden_size: int | None = None
) -> torch.Tensor:
    """Declare ``tensor`` pageable, stating how the stash should address it.

    ``hidden_size`` is the row width the copy kernel addresses and
    ``token_divisor`` how many tokens share a row -- 1 for data, 32 for an MXFP8
    block-scale buffer, matching Megatron's ``SCALE_INV_BLOCK_SIZE`` handling.
    Defaults to the tensor's last dimension, which is right for any token-major
    2-D activation.
    """
    if not _MARKING_ENABLED:
        return tensor
    if hidden_size is None:
        hidden_size = int(tensor.shape[-1])
    return torch.ops.paged_stash.mark_pageable(tensor, token_divisor, hidden_size)


def wait(tensor: torch.Tensor) -> torch.Tensor:
    """Importable wrapper so the op can be referenced by name."""
    return torch.ops.paged_stash.wait.default(tensor)


PAGED_STASH_OPS = {
    torch.ops.paged_stash.stash_.default,
    torch.ops.paged_stash.wait_stash_.default,
    torch.ops.paged_stash.reload.default,
    torch.ops.paged_stash.wait.default,
}


def warmup_paged_stash_kernels() -> None:
    """JIT-compile both Triton kernels for every live constexpr combination.

    Triton compiles on first launch, which is illegal inside CUDA graph
    capture. Megatron never hits this because its stash runs for real during the
    eager warmup iterations that precede capture; GraphTrainer's warmup step is
    the *measure* step, which launches no kernels at all, so the first real
    launch would land inside the capture. Priming the cache here removes that
    hazard.

    Each launch passes ``num_tokens=0``, so the kernels compile and then exit
    without touching pages or freelists.
    """
    manager = PagedStashManager.get_instance()
    if not manager.needs_kernel_warmup or not manager.stash_buffers:
        return

    device = manager.device
    zero_tokens = torch.zeros(1, dtype=torch.int64, device=device)
    compiled = 0
    for buffers in manager.stash_buffers.values():
        for hidden_size, buffer in buffers.items():
            has_host = 1 if buffer.host_buffer is not None else 0
            host_buf = (
                buffer.host_buffer
                if buffer.host_buffer is not None
                else buffer.cuda_buffer
            )
            scratch = torch.zeros(
                (buffer.page_size, hidden_size), dtype=buffer.dtype, device=device
            )
            page_record = torch.zeros(1, dtype=torch.int64, device=device)
            spilled = torch.zeros(1, dtype=torch.int64, device=device)
            new_head = torch.zeros(2, dtype=torch.int64, device=device)
            new_tail = torch.zeros(2, dtype=torch.int64, device=device)
            paged_stash_copy_kernel[(1,)](
                scratch,
                buffer.cuda_buffer,
                host_buf,
                zero_tokens,
                buffer.free_list_cuda,
                buffer.free_list_host,
                buffer.free_list_head,
                buffer.free_list_tail,
                buffer.free_list_capacity,
                page_record,
                buffer.overflow,
                buffer.host_spill,
                spilled,
                new_head,
                PAGE_SIZE=buffer.page_size,
                HIDDEN_SIZE=hidden_size,
                BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
                HAS_HOST_BUFFER=has_host,
            )
            paged_stash_pop_kernel[(1,)](
                buffer.cuda_buffer,
                host_buf,
                scratch,
                zero_tokens,
                page_record,
                spilled,
                buffer.overflow,
                buffer.free_list_cuda,
                buffer.free_list_host,
                buffer.free_list_tail,
                buffer.free_list_capacity,
                new_tail,
                PAGE_SIZE=buffer.page_size,
                HIDDEN_SIZE=hidden_size,
                BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
            )
            compiled += 1

    # The warmup launches are no-ops, but reset anyway so the step starts from
    # a pristine freelist regardless of kernel-side bookkeeping.
    for buffers in manager.stash_buffers.values():
        for buffer in buffers.values():
            buffer.reset()
    assert manager.overflow is not None and manager.host_spill is not None
    manager.overflow.zero_()
    manager.host_spill.zero_()
    torch.cuda.synchronize(device)
    manager.needs_kernel_warmup = False
    logger.info(
        "Paged stash: compiled Triton kernels for %d buffer configuration(s).",
        compiled,
    )


# ===========================================================================
# Annotation pass: which activations should be paged
# ===========================================================================
#
# Megatron marks stash candidates imperatively -- ``get_paged_stash_context()``
# opens ``saved_tensors_hooks`` around the fused grouped MLP, and every tensor
# Transformer Engine tagged with ``grouped_tensor_scale_inv`` inside that context
# gets paged. GraphTrainer has no such dynamic context, so the same selection is
# expressed as node annotations on the traced graph, following the existing
# memory-policy convention in ``memory_policy.py`` (``MUST_SAVE`` /
# ``MUST_RECOMPUTE`` / ``MUST_CPU_OFFLOAD``).
#
# The selection rule is the graph equivalent of Megatron's: an activation is
# pageable when it is produced inside the routed-expert region, its leading
# dimension is the *capacity-padded* permuted-token dimension, and only the first
# ``sum(tokens_per_expert)`` rows of it are live. That padded-vs-live gap is
# exactly what paged stashing reclaims, and it is why the token dispatcher must be
# running with a static capacity factor.


# node.meta key set by this pass and consumed by apply_paged_stash_pass.
PAGED_STASH_TAG = "paged_stash"

# Default module-FQN pattern for the routed-expert region. Matched against the
# leading FQN components, so it selects the whole ``layers.<n>.moe.
# routed_experts`` subtree -- the dispatcher output and everything inside
# ``inner_experts``. That is the same set Megatron pages: the fused grouped
# MLP's fc1 input, activation input, scales, and fc2 input.
DEFAULT_MODULE_FQN = "layers.*.moe.routed_experts"

# Paging a tensor costs one full round trip through the stash, so tiny tensors
# are never worth it. Same rationale as cpu_offload's ``_MIN_OFFLOAD_BYTES``.
_MIN_STASH_BYTES = 1 << 20  # 1 MiB


def matches_module_subtree(pattern: str, fqn: str) -> bool:
    """Component-wise match of ``pattern`` against the *leading* parts of ``fqn``.

    ``matches_module_fqn_pattern`` requires the same number of components, which
    would select ``layers.1.moe.routed_experts`` but not
    ``layers.1.moe.routed_experts.inner_experts``. Paged stashing wants the whole
    subtree, so the pattern is matched as a prefix.
    """
    parts = fqn.split(".")
    pattern_parts = pattern.split(".")
    if len(parts) < len(pattern_parts):
        return False
    return matches_module_fqn_pattern(pattern, ".".join(parts[: len(pattern_parts)]))


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Where a layer's padded and live token counts come from.

    ``padded`` is the static leading dimension of every permuted activation in
    the layer -- the capacity-factor-sized buffer the dispatcher writes into.
    ``per_expert_node`` produces the device-side per-local-expert token counts;
    summing it gives the live row count, which is what the stash actually
    copies. This pair is the GraphTrainer analogue of Megatron's
    ``(max_num_tokens, num_tokens_tensor)``.
    """

    layer_id: int
    padded: int
    per_expert_node: Node


def _dispatch_token_count_output(node: Node) -> int | None:
    """Return the output index carrying per-local-expert token counts.

    Token dispatchers that expose a *static* permuted-token capacity are the
    only ones paged stashing applies to: a dynamically sized permuted buffer is
    already exactly as large as the live tokens, so there is nothing to reclaim.
    """
    target = node.target
    if target is torch.ops.hybridep.dispatch.default:
        # (hidden, scores, tokens_per_expert, handle)
        return 2
    return None


def find_layer_token_counts(gm: torch.fx.GraphModule) -> dict[int, TokenCounts]:
    """Map each MoE layer to its padded capacity and live-token-count node."""
    counts: dict[int, TokenCounts] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        count_index = _dispatch_token_count_output(node)
        if count_index is None:
            continue
        layer_id = _get_layer_id(node)
        if layer_id == _NOT_IN_LAYERS:
            continue

        hidden_node: Node | None = None
        per_expert_node: Node | None = None
        for user in node.users:
            if user.target is not operator.getitem:
                continue
            if user.args[1] == 0:
                hidden_node = user
            elif user.args[1] == count_index:
                per_expert_node = user
        if hidden_node is None or per_expert_node is None:
            continue
        hidden_val = hidden_node.meta.get("val")
        if not isinstance(hidden_val, torch.Tensor) or hidden_val.dim() < 1:
            continue
        padded = hidden_val.shape[0]
        if isinstance(padded, torch.SymInt):
            # A symbolic permuted dimension means the dispatcher is in blocking
            # mode and already sizes exactly; nothing to page.
            continue
        counts[layer_id] = TokenCounts(
            layer_id=layer_id,
            padded=int(padded),
            per_expert_node=per_expert_node,
        )
    return counts


# MXFP8 block-scale exponents. Megatron learns that a saved tensor is a scale
# inverse from the ``grouped_tensor_scale_inv`` flag Transformer Engine stamps
# on it; nothing marks ours, but e8m0 is only ever used for block scales, so the
# dtype carries the same information.


def declared_slot_dims(node: Node) -> tuple[int, int] | None:
    """``(token_divisor, hidden_size)`` a producer declared for ``node``.

    These are the two numbers the copy kernel needs: how wide a row is, and how
    many tokens share one. A tensor's own shape gives neither -- a flat
    ``(786432,)`` scale buffer is 3072 rows of 256, and nothing in the shape
    says so.

    The counterpart of Transformer Engine's ``mark_grouped_tensor``: whoever
    built the activation says how it is laid out, so nothing downstream has to
    work it out from an element count.
    """
    if node.op != "call_function":
        return None
    if not str(node.target).startswith("paged_stash.mark_pageable"):
        return None
    return int(node.args[1]), int(node.args[2])


def infer_slot_dims(val: torch.Tensor, padded: int) -> tuple[int, int] | None:
    """``(token_divisor, hidden_size)`` -- how the stash addresses this tensor as
    rows -- or ``None`` if it cannot be paged.

    Megatron does not require a saved tensor to be two-dimensional. It flattens
    it and divides the element count by the padded row count -- ``max_num_tokens``
    for data, ``max_num_tokens // 32`` for a columnwise scale inverse -- and the
    copy kernel then addresses a prefix of those rows. That is what lets it page
    Transformer Engine's scale buffers, which are flat 1-D arrays with no token
    axis to test.

    Data tensors are held to the stricter rule: their leading dimension must be
    the token count. A tensor whose rows are not tokens has its live values
    strided through the buffer rather than sitting in a prefix, which is
    precisely the case the copy kernel cannot express.
    """
    if val.dtype is torch.float8_e8m0fnu:
        # 32 tokens share one scale row, so the row count is padded // 32.
        if padded % SCALE_INV_BLOCK_SIZE:
            return None
        divisor, rows = SCALE_INV_BLOCK_SIZE, padded // SCALE_INV_BLOCK_SIZE
        if val.dim() > 1 and int(val.shape[0]) != rows:
            # Element counts are ambiguous: 786432 is 3072x256 and 1536x512
            # alike, and picking the wrong split silently reads off the end of
            # the buffer. Megatron never has to choose because Transformer
            # Engine hands it flat buffers whose only possible reading is the
            # token one. A tensor that carries its own shape has already
            # answered the question, so believe it rather than refactoring its
            # element count; only genuinely flat buffers get the derived split.
            return None
    else:
        if int(val.shape[0]) != padded:
            return None
        divisor, rows = 1, padded
    if rows <= 0 or val.numel() % rows:
        return None
    return divisor, val.numel() // rows


def _is_pageable(node: Node, padded: int) -> bool:
    """Whether ``node``'s output is a permuted activation worth paging."""
    if node.op != "call_function" or _is_backward_node(node):
        return False
    if _is_view(node):
        # Views alias their base, so paging one frees nothing.
        return False
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return False
    if val.device.type != "cuda":
        return False
    if val.dim() == 0 or any(isinstance(s, torch.SymInt) for s in val.shape):
        return False
    declared = declared_slot_dims(node)
    dims = declared if declared is not None else infer_slot_dims(val, padded)
    if dims is None:
        return False
    divisor, hidden = dims
    # A declaration is trusted for layout but still has to be arithmetically
    # possible: a wrong one would index off the end of the buffer just as an
    # inferred one would, and this is the cheapest place to catch it.
    rows = padded // divisor
    if divisor <= 0 or padded % divisor or rows <= 0 or val.numel() % hidden:
        return False
    if val.numel() // hidden > rows:
        return False
    try:
        if not val.is_contiguous():
            return False
    except Exception:
        return False
    # A block-scale buffer is 1/32 of the tensor it describes and is only ever
    # paged alongside it -- Transformer Engine marks the pair together and
    # Megatron stashes a layer's tensors as one list. Judging it on its own
    # bytes would reject it for being small when the round trip it shares is
    # already paid for, so weigh it by the data it scales.
    weight = divisor if divisor != 1 else 1
    if _tensor_bytes(val) * weight < _MIN_STASH_BYTES:
        return False
    return True


# MXFP8 element types. A tensor in one of these that reaches backward is
# already the quantized form backward consumes.
_LOW_PRECISION_DTYPES = (torch.float8_e4m3fn, torch.float8_e8m0fnu)


def promote_low_precision_saves(
    gm: torch.fx.GraphModule,
    *,
    module_fqn_pattern: str = DEFAULT_MODULE_FQN,
) -> int:
    """Save routed-expert fp8 activations that reach backward, don't recompute them.

    A fused MXFP8 expert MLP emits its quantized operands once and its backward
    consumes them directly, so an fp8 tensor that crosses the boundary is
    already the exact value backward needs. SAC still marks it
    ``PREFER_RECOMPUTE``, which makes remat re-run the whole fused GEMM to
    rebuild it -- and a tensor that never crosses cannot be paged at all.
    Promoting it is both cheaper and what makes it pageable; it is the state
    Transformer Engine hands Megatron, whose stash pages fp8 for the same
    reason.

    Deliberately keyed on dtype and on having backward users rather than on a
    list of producing ops: the property that matters is "quantized value live
    across the boundary", which no op list would track. Returns the count.
    """
    token_counts = find_layer_token_counts(gm)
    if not token_counts:
        return 0

    promoted = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype not in _LOW_PRECISION_DTYPES:
            continue
        if node.meta.get("recompute") is not CheckpointPolicy.PREFER_RECOMPUTE:
            continue
        if not matches_module_subtree(module_fqn_pattern, _get_module_fqn(node)):
            continue
        counts = token_counts.get(_get_layer_id(node))
        # Promote only what the stash can actually take. Saving a tensor the
        # tagger will then reject -- a block scale, or anything not shaped like
        # a permuted activation -- costs memory and buys nothing, since the
        # point of the promotion is to make paging possible.
        if counts is None or not _is_pageable(node, counts.padded):
            continue
        _, has_backward = _get_storage_chain(node)
        if not has_backward and not any(_is_backward_node(u) for u in node.users):
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        promoted += 1

    if promoted:
        logger.info(
            "Paged stash: promoted %d low-precision routed-expert activation(s) "
            "to MUST_SAVE so they cross into backward and can be paged.",
            promoted,
        )
    return promoted


def promote_declared_saves(
    gm: torch.fx.GraphModule,
    *,
    module_fqn_pattern: str = DEFAULT_MODULE_FQN,
) -> int:
    """Save declared activations that SAC would otherwise recompute.

    Megatron pages the fused MLP's ``activation_in`` -- the BF16 FC1 output --
    because Transformer Engine saves it and the stash reclaims it across the
    bubble. Our SAC reaches the opposite conclusion for the same tensor and
    rebuilds it in backward, so it never crosses into backward and cannot be
    paged. Which is better is a real trade rather than a bug: recompute costs a
    grouped GEMM, paging costs a round trip plus the pages to hold it, and the
    FC1 output is several times larger than the quantized operands beside it.

    This promotes any activation a producer declared (:func:`declared_slot_dims`)
    and that ``_is_pageable`` accepts, whatever its dtype -- which is what makes
    Megatron's choice available. Off unless asked for; see
    ``--compile.paged_stash_page_recomputed``. Returns the count.
    """
    token_counts = find_layer_token_counts(gm)
    if not token_counts:
        return 0

    promoted = 0
    for node in gm.graph.nodes:
        if declared_slot_dims(node) is None:
            continue
        if not matches_module_subtree(module_fqn_pattern, _get_module_fqn(node)):
            continue
        counts = token_counts.get(_get_layer_id(node))
        if counts is None or not _is_pageable(node, counts.padded):
            continue
        if node.meta.get("recompute") not in (
            CheckpointPolicy.PREFER_RECOMPUTE,
            CheckpointPolicy.MUST_RECOMPUTE,
        ):
            continue
        _, has_backward = _get_storage_chain(node)
        if not has_backward and not any(_is_backward_node(u) for u in node.users):
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        promoted += 1

    if promoted:
        logger.info(
            "Paged stash: promoted %d declared routed-expert activation(s) from "
            "recompute to MUST_SAVE so the stash can reclaim them.",
            promoted,
        )
    return promoted


def tag_paged_stash_activations(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    module_fqn_pattern: str = DEFAULT_MODULE_FQN,
) -> torch.fx.GraphModule:
    """Tag capacity-padded routed-expert activations for paged stashing.

    Runs after SAC tagging, so it only considers activations SAC decided to keep
    (``MUST_SAVE``, or untagged-but-consumed-by-backward). Recomputed
    activations are skipped: they are never live across the pipeline bubble, so
    there is nothing for the stash to reclaim.
    """
    token_counts = find_layer_token_counts(gm)
    if not token_counts:
        logger.info(
            "Paged stash: no static-capacity token dispatch found; nothing tagged."
        )
        return gm

    tagged = 0
    tagged_bytes = 0
    per_layer: dict[int, int] = {}
    for node in gm.graph.nodes:
        layer_id = _get_layer_id(node)
        counts = token_counts.get(layer_id)
        if counts is None:
            continue
        if not matches_module_subtree(module_fqn_pattern, _get_module_fqn(node)):
            continue
        if not _is_pageable(node, counts.padded):
            continue

        policy = node.meta.get("recompute")
        if policy in (
            CheckpointPolicy.MUST_RECOMPUTE,
            CheckpointPolicy.PREFER_RECOMPUTE,
        ):
            continue
        if policy == CheckpointPolicy.MUST_CPU_OFFLOAD:
            # Already claimed by the CPU offload pass; do not page it twice.
            continue

        # Only activations that survive into backward are worth paging.
        _, has_backward = _get_storage_chain(node)
        if not has_backward and not any(_is_backward_node(u) for u in node.users):
            continue

        node.meta[PAGED_STASH_TAG] = True
        tagged += 1
        tagged_bytes += _tensor_bytes(node.meta["val"])
        per_layer[layer_id] = per_layer.get(layer_id, 0) + 1

    if tagged:
        logger.info(
            "Paged stash: tagged %d activation(s) across %d layer(s) "
            "(%.2f MiB padded, pattern %r)",
            tagged,
            len(per_layer),
            tagged_bytes / (1024 * 1024),
            module_fqn_pattern,
        )
        for layer_id in sorted(per_layer):
            logger.debug(
                "  layer %d: %d tagged, padded tokens %d",
                layer_id,
                per_layer[layer_id],
                token_counts[layer_id].padded,
            )
    else:
        logger.info("Paged stash: no activations matched %r", module_fqn_pattern)
    return gm


# ===========================================================================
# Insert paged stash / reload ops around tagged activations
# ===========================================================================
#
# This is the structural counterpart of ``cpu_offload.apply_cpu_offload_pass``,
# and of Megatron's ``PipelinePreScheduleFunction`` /
# ``PipelinePostScheduleFunction`` pair. Megatron needs those autograd functions
# because it discovers stash and reload points while the tape runs; here the
# joint graph already contains both halves, so a single pass can place all four
# ops and rewire backward directly.
#
# Rewrite performed for each node tagged by ``policy.tag_paged_stash_activations``::
#
#     forward:
#         act = <producer>
#         handle = aten.empty([max_pages + 1], int64)
#         paged_stash.stash_(handle, act, num_tokens, slot)
#         ... remaining forward consumers of act ...
#         paged_stash.wait_stash_(handle, act, last_forward_use)
#
#     backward:
#         act' = paged_stash.reload(handle, num_tokens, slot, size, dtype)
#         act' = paged_stash.wait(act')
#         ... backward consumers rewired from act to act' ...
#
# After the rewrite the padded activation has no backward users and no forward
# users past the wait, so it is freed there; ``handle`` and ``num_tokens`` -- a
# few kilobytes -- are what cross into backward. Under GraphPP that means the
# stage's ``saved_values_for_backward`` carries the handle instead of the padded
# tensor, which is where the memory is actually recovered.


aten = torch.ops.aten
paged_stash_ops = torch.ops.paged_stash


@dataclass(slots=True)
class _StashInfo:
    node: Node
    slot: StashSlot
    num_tokens_node: Node
    max_pages: int
    direct_bwd_users: list[Node]
    replay_views: list[Node] = field(default_factory=list)
    view_bwd_redirects: list[tuple[Node, Node]] = field(default_factory=list)
    first_bwd_consumer: Node | None = None
    last_fwd_consumer: Node | None = None


def _make_num_tokens_node(
    gm: torch.fx.GraphModule,
    per_expert_node: Node,
) -> Node:
    """Insert ``sum(tokens_per_expert)`` -- the live row count for a layer.

    Megatron passes ``tokens_per_expert.sum()`` into
    ``get_paged_stash_context(num_tokens_tensor=...)``; the same value is
    materialized here as a graph node so it stays on device and never forces a
    host sync, which is what keeps the stash CUDA-graph safe.
    """
    val = per_expert_node.meta.get("val")
    with gm.graph.inserting_after(per_expert_node):
        node = gm.graph.call_function(aten.sum.default, args=(per_expert_node,))
    node.meta.update(
        {k: v for k, v in per_expert_node.meta.items() if k not in ("val", "recompute")}
    )
    if isinstance(val, torch.Tensor):
        node.meta["val"] = val.sum()
    # This is a forward-side value even though backward reads it.
    node.meta.pop("autograd_backward", None)
    return node


def apply_paged_stash_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    page_size: int = 64,
    prefetch_lookahead: int = 1,
) -> torch.fx.GraphModule:
    """Insert paged stash/reload ops for nodes tagged ``paged_stash``.

    Args:
        gm: Joint forward+backward graph module.
        example_inputs: Unused; required by the graph pass interface.
        page_size: Tokens per page. Must match the value the manager allocates
            buffers with, so it is written back onto the manager here.
        prefetch_lookahead: Move each ``paged_stash.reload`` this many backward
            layers earlier to overlap the page reads with backward compute.

    Returns:
        The transformed GraphModule.
    """
    tagged = [n for n in gm.graph.nodes if n.meta.get(PAGED_STASH_TAG)]
    if not tagged:
        return gm

    manager = PagedStashManager.get_instance()
    manager.page_size = page_size

    token_counts = find_layer_token_counts(gm)
    node_to_index: dict[Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # 1. One live-token-count node per MoE layer, shared by every stash in it.
    num_tokens_nodes: dict[int, Node] = {}
    for layer_id in sorted({_get_layer_id(n) for n in tagged}):
        counts = token_counts.get(layer_id)
        if counts is None:
            raise RuntimeError(
                f"paged stash: layer {layer_id} has tagged activations but no "
                "token-count node; tag_paged_stash_activations and "
                "apply_paged_stash_pass disagree about the dispatcher."
            )
        num_tokens_nodes[layer_id] = _make_num_tokens_node(gm, counts.per_expert_node)

    # The inserted count nodes shift positions, so re-index before any ordering
    # decision below depends on it.
    node_to_index = {n: i for i, n in enumerate(gm.graph.nodes)}

    # 2. Collect the rewrite plan before mutating the graph.
    infos: list[_StashInfo] = []
    for node in tagged:
        val = node.meta["val"]
        layer_id = _get_layer_id(node)
        counts = token_counts.get(layer_id)
        declared = declared_slot_dims(node)
        dims = (
            declared
            if declared is not None
            else (infer_slot_dims(val, counts.padded) if counts is not None else None)
        )
        if dims is None:
            # tag_paged_stash_activations only tags nodes infer_slot_dims
            # accepts, so this means the two disagree about the token count.
            raise RuntimeError(
                f"paged stash: tagged node {node.name} has no slot dims "
                f"for layer {layer_id}; tagging and rewriting disagree."
            )
        token_divisor, hidden_size = dims
        max_num_tokens = counts.padded

        direct_bwd_users = [u for u in node.users if _is_backward_node(u)]
        replay_views, view_bwd_redirects = _collect_view_replay_info(node)
        all_bwd_users = direct_bwd_users + [u for _, u in view_bwd_redirects]
        if not all_bwd_users:
            continue

        slot = StashSlot(
            slot_id=manager.next_slot_id(),
            dtype=val.dtype,
            hidden_size=int(hidden_size),
            max_num_tokens=max_num_tokens,
            token_divisor=token_divisor,
            layer_id=layer_id,
            node_name=node.name,
            stage_index=manager.current_stage_index,
        )
        manager.register_slot(slot)
        logger.debug(
            "Paged stash slot %d: %s shape=%s numel=%d dtype=%s -> "
            "hidden=%d divisor=%d max_tokens=%d rows=%d",
            slot.slot_id,
            node.name,
            tuple(val.shape),
            val.numel(),
            val.dtype,
            slot.hidden_size,
            slot.token_divisor,
            slot.max_num_tokens,
            slot.max_stash_tokens,
        )

        chain_nodes, _ = _get_storage_chain(node)
        last_fwd_consumer = node
        last_pos = node_to_index[node]
        for consumer in chain_nodes:
            pos = node_to_index.get(consumer)
            if pos is not None and pos > last_pos:
                last_fwd_consumer = consumer
                last_pos = pos

        infos.append(
            _StashInfo(
                node=node,
                slot=slot,
                num_tokens_node=num_tokens_nodes[layer_id],
                max_pages=(slot.max_stash_tokens + page_size - 1) // page_size,
                direct_bwd_users=direct_bwd_users,
                replay_views=replay_views,
                view_bwd_redirects=view_bwd_redirects,
                first_bwd_consumer=min(all_bwd_users, key=lambda n: node_to_index[n]),
                last_fwd_consumer=last_fwd_consumer,
            )
        )

    if not infos:
        return gm

    # Megatron skips the stash when the next schedule entry is this layer's own
    # backward. Only the *last* tagged layer can ever satisfy that: every
    # earlier layer is followed by the next layer's forward, never by its own
    # backward. So it is the only one marked skippable. The group index gives
    # the runtime a once-per-forward tick that does not depend on which layers
    # actually ended up stashing.
    ordered_layers = list(dict.fromkeys(info.slot.layer_id for info in infos))
    last_layer_id = ordered_layers[-1]
    # The cursor tick must fire once per forward, so it goes on a single node --
    # the first stash in graph order -- not on every node of the first layer.
    # A layer can hold several tensors (data plus its block scales) and they all
    # share a layer index.
    cursor_node = infos[0].node

    # 3. Rewrite.
    stashed_bytes = 0
    replay_count = 0
    for info in infos:
        node = info.node
        val = node.meta["val"]
        # Carry producer metadata onto the inserted nodes so tlparse dumps and
        # kernel annotations still attribute them to the right module.
        src_meta = {
            k: v
            for k, v in node.meta.items()
            if k not in ("val", "recompute", PAGED_STASH_TAG)
        }

        # The stash reads both the activation and the layer's live-token count.
        # The count node is emitted next to the dispatcher, which for the
        # dispatcher's own outputs can be *after* the activation, so anchor on
        # whichever of the two comes later.
        stash_anchor = max((node, info.num_tokens_node), key=lambda n: node_to_index[n])
        with gm.graph.inserting_after(stash_anchor):
            # One int64 tensor per stashed activation: page ids followed by the
            # spill flag. Keeping it as a single tensor gives the forward
            # mutation a single base for GraphPP to save.
            handle = gm.graph.call_function(
                aten.empty.memory_format,
                args=([info.max_pages + 1],),
                kwargs={
                    "dtype": torch.int64,
                    "device": val.device,
                    "pin_memory": False,
                },
            )
            handle.meta.update(src_meta)
            handle.meta["val"] = val.new_empty((info.max_pages + 1,), dtype=torch.int64)
        with gm.graph.inserting_after(handle):
            stash_node = gm.graph.call_function(
                paged_stash_ops.stash_.default,
                args=(
                    handle,
                    node,
                    info.num_tokens_node,
                    info.slot.slot_id,
                    0 if node is cursor_node else -1,
                    info.slot.layer_id == last_layer_id,
                ),
            )
            stash_node.meta.update(src_meta)
            stash_node.meta["val"] = None

        # The forward wait joins the stash stream and lets the padded
        # activation die, so it must sit after both the stash and the
        # activation's last forward use.
        wait_anchor = info.last_fwd_consumer
        if (
            wait_anchor is node
            or node_to_index.get(wait_anchor, -1) < node_to_index[stash_anchor]
        ):
            wait_anchor = stash_node
        # last_use_of_storage only exists to pin an aliasing consumer before the
        # wait; a consumer that would itself land after the wait cannot serve
        # that role, so drop it rather than create a cycle.
        last_use_arg = _find_last_tensor_consumer(info.last_fwd_consumer)
        if last_use_arg is not None and last_use_arg is not wait_anchor:
            if node_to_index.get(last_use_arg, -1) > node_to_index.get(wait_anchor, -1):
                last_use_arg = None
        wait_node = gm.graph.call_function(
            paged_stash_ops.wait_stash_.default,
            args=(handle, node, last_use_arg),
        )
        wait_node.meta.update(src_meta)
        wait_node.meta["val"] = None
        wait_anchor.append(wait_node)

        # Backward: page the activation back in just before its first consumer.
        with gm.graph.inserting_before(info.first_bwd_consumer):
            reload_node = gm.graph.call_function(
                paged_stash_ops.reload.default,
                args=(
                    handle,
                    info.num_tokens_node,
                    info.slot.slot_id,
                    list(val.size()),
                    val.dtype,
                ),
            )
            reload_node.meta.update(src_meta)
            reload_node.meta["val"] = val
            reload_node.meta["autograd_backward"] = True

            reload_wait = gm.graph.call_function(
                paged_stash_ops.wait.default,
                args=(reload_node,),
            )
            reload_wait.meta.update(src_meta)
            reload_wait.meta["val"] = val
            reload_wait.meta["autograd_backward"] = True

        for user in info.direct_bwd_users:
            user.replace_input_with(node, reload_wait)

        # Replay any view chain the backward consumers read through, so they see
        # views of the reloaded tensor rather than of the released one.
        if info.replay_views:
            replay_count += 1
            replay_map: dict[Node, Node] = {node: reload_wait}
            for view_node in sorted(info.replay_views, key=lambda n: node_to_index[n]):
                new_args = tuple(
                    replay_map.get(a, a) if isinstance(a, Node) else a
                    for a in view_node.args
                )
                new_kwargs = {
                    k: replay_map.get(v, v) if isinstance(v, Node) else v
                    for k, v in view_node.kwargs.items()
                }
                with gm.graph.inserting_before(info.first_bwd_consumer):
                    replayed = gm.graph.call_function(
                        view_node.target, args=new_args, kwargs=new_kwargs
                    )
                    replayed.meta.update(
                        {
                            k: v
                            for k, v in view_node.meta.items()
                            if k not in ("recompute", PAGED_STASH_TAG)
                        }
                    )
                    replayed.meta["autograd_backward"] = True
                replay_map[view_node] = replayed
            for consumed_node, bwd_user in info.view_bwd_redirects:
                replayed = replay_map.get(consumed_node)
                if replayed is not None:
                    bwd_user.replace_input_with(consumed_node, replayed)

        stashed_bytes += _tensor_bytes(val)
        logger.debug(
            "Paged stash: slot %d <- %s (%s %s, %.2f MiB padded)",
            info.slot.slot_id,
            node.name,
            val.dtype,
            tuple(val.shape),
            _tensor_bytes(val) / (1024 * 1024),
        )

    moved = 0
    _erase_marks(gm)

    if prefetch_lookahead > 0:
        moved = _prefetch_reloads(gm, prefetch_lookahead)

    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Paged stash: paged %d activation(s) (%.2f MiB padded), "
        "%d with view replay, prefetched %d reload(s)",
        len(infos),
        stashed_bytes / (1024 * 1024),
        replay_count,
        moved,
    )
    return gm


def _erase_marks(gm: torch.fx.GraphModule) -> int:
    """Drop every ``paged_stash.mark_pageable`` call from the graph.

    The declaration exists to be read at compile time, not executed. Erasing
    unconditionally -- including marks on tensors that did not end up tagged --
    keeps a declaration free at runtime, so a producer can mark liberally
    without paying for tensors the policy declines to page.
    """
    erased = 0
    for node in list(gm.graph.nodes):
        if declared_slot_dims(node) is None:
            continue
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
        erased += 1
    return erased


def _prefetch_reloads(gm: torch.fx.GraphModule, n_layers: int) -> int:
    """Move ``paged_stash.reload`` nodes ``n_layers`` backward layers earlier.

    The paired ``paged_stash.wait`` stays put, so the compute stream still
    synchronizes immediately before the data is used; only the page read starts
    sooner. Same structure as ``cpu_offload.prefetch_reloads``.
    """
    bwd_anchors: list[Node] = []
    layer_to_bwd_idx: dict[int, int] = {}
    current_layer = None
    for node in gm.graph.nodes:
        if node.op != "call_function" or not _is_backward_node(node):
            continue
        if node.target in PAGED_STASH_OPS:
            continue
        layer_id = _get_layer_id(node)
        if layer_id != current_layer:
            bwd_anchors.append(node)
            if layer_id not in layer_to_bwd_idx:
                layer_to_bwd_idx[layer_id] = len(bwd_anchors) - 1
            current_layer = layer_id
    if not bwd_anchors:
        return 0

    bwd_layer_order = sorted(
        layer_to_bwd_idx.keys(), key=lambda lid: layer_to_bwd_idx[lid]
    )
    layer_pos = {lid: i for i, lid in enumerate(bwd_layer_order)}

    moved = 0
    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not paged_stash_ops.reload.default
        ):
            continue
        layer_id = _get_layer_id(node)
        if layer_id not in layer_pos:
            continue
        target_pos = layer_pos[layer_id] - n_layers
        target_idx = (
            0 if target_pos < 0 else layer_to_bwd_idx[bwd_layer_order[target_pos]]
        )
        if target_idx >= layer_to_bwd_idx[layer_id]:
            continue
        bwd_anchors[target_idx].prepend(node)
        moved += 1
    return moved


# ===========================================================================
# Derive paged stash buffer sizing from the GraphPP pipeline schedule
# ===========================================================================
#
# Megatron sizes its stash buffers by reconstructing its own virtual-pipeline
# order: ``PagedStashManager`` builds a ``_pp_schedule`` list while a capture
# iteration runs, encodes ``(vp_stage, layer, microbatch)`` triples into integers,
# and accumulates per-tensor token counts over the entries that are live at the
# same time. The peak of that accumulation is the buffer size.
#
# GraphPP does not need the reconstruction. The runtime PP schedule is already a
# concrete list of ``_Action`` s per rank, so the same accumulation can be run
# directly against it: walk the actions, add a stage's stash footprint when a
# microbatch enters forward, subtract it when the backward that consumes that
# stage's saved values runs, and keep the peak.
#
# A stashed activation becomes live at ``FORWARD(stage, microbatch)`` and is
# released by the backward that consumes the stage's saved values:
# ``FULL_BACKWARD`` or, for split schedules, ``BACKWARD_INPUT``.
# ``BACKWARD_WEIGHT`` consumes a different set of live values and does not release
# the stash, so it is not counted.


PageCounts = dict[tuple[torch.dtype, int], int]

_LOGGED_SKIP_PLAN: list[bool] = []


def _iter_compute_actions(action: _Action):
    """Yield ``action`` itself, or its sub-actions for a multiplexed action."""
    if action.sub_actions:
        yield from action.sub_actions
    else:
        yield action


def replay_schedule_page_peak(
    actions: list[_Action],
    stage_footprint: dict[int, PageCounts],
) -> PageCounts:
    """Peak stashed pages per ``(dtype, hidden_size)`` for one rank's schedule.

    Args:
        actions: The rank's ordered pipeline actions.
        stage_footprint: Pages one in-flight microbatch of each stage keeps
            stashed, from ``PagedStashManager.stage_page_footprint``.

    Returns:
        Peak concurrently stashed pages per key.
    """
    live: PageCounts = {}
    peak: PageCounts = {}

    def _apply(stage_index: int, sign: int) -> None:
        counts = stage_footprint.get(stage_index)
        if not counts:
            return
        for key, pages in counts.items():
            live[key] = live.get(key, 0) + sign * pages
            if sign > 0:
                peak[key] = max(peak.get(key, 0), live[key])

    for action in actions:
        for sub in _iter_compute_actions(action):
            kind = sub.computation_type
            if kind is FORWARD:
                _apply(sub.stage_index, +1)
            elif kind in (FULL_BACKWARD, BACKWARD_INPUT):
                _apply(sub.stage_index, -1)
    return peak


def max_live_stash_generations(actions: list[_Action]) -> int:
    """Peak number of concurrently stashed microbatches for one rank's schedule.

    Reported for logging; the actual sizing uses ``replay_schedule_page_peak``,
    which is per-stage and therefore tighter than this rank-wide count.
    """
    live = 0
    peak = 0
    for action in actions:
        for sub in _iter_compute_actions(action):
            kind = sub.computation_type
            if kind is FORWARD:
                live += 1
                peak = max(peak, live)
            elif kind in (FULL_BACKWARD, BACKWARD_INPUT):
                live = max(0, live - 1)
    return peak


def forward_skip_plan(actions: list[_Action]) -> list[bool]:
    """Per forward, whether that microbatch's own backward is the next compute.

    Megatron makes this call in ``PipelinePostScheduleFunction`` by comparing
    the current schedule entry against the negation of the next one -- its
    entries encode ``(vp_stage, layer, microbatch)`` and a backward is the
    arithmetic negative of its forward, so ``current != -next`` reads as "the
    next thing to run is not my own backward". When it *is*, the activation
    would be written to the stash and read straight back with no bubble in
    between, so Megatron keeps it resident instead
    (``remove_paged_tensor_from_stash``).

    GraphPP actions carry ``stage_index`` and ``microbatch_index`` directly, so
    the same test is a plain comparison. Only compute actions participate:
    sends, receives and unshard actions sit between a forward and its backward
    in ``pipeline_order_with_comms`` without changing which activation is
    consumed next.

    ``BACKWARD_WEIGHT`` is deliberately not a match. It consumes a different
    set of saved values and does not release the stash, the same reason
    :func:`replay_schedule_page_peak` ignores it.

    Returns one flag per forward, in execution order.
    """
    compute = [
        sub
        for action in actions
        for sub in _iter_compute_actions(action)
        if sub.computation_type in (FORWARD, FULL_BACKWARD, BACKWARD_INPUT)
    ]
    plan: list[bool] = []
    for i, action in enumerate(compute):
        if action.computation_type is not FORWARD:
            continue
        nxt = compute[i + 1] if i + 1 < len(compute) else None
        plan.append(
            nxt is not None
            and nxt.computation_type in (FULL_BACKWARD, BACKWARD_INPUT)
            and nxt.stage_index == action.stage_index
            and nxt.microbatch_index == action.microbatch_index
        )
    return plan


def schedule_page_peak(
    schedule: _PipelineScheduleRuntime,
    stage_footprint: dict[int, PageCounts],
) -> PageCounts:
    """Peak stashed pages across every rank of ``schedule``.

    Taking the max over ranks rather than only the local rank keeps buffer sizes
    uniform, which matters because the overflow fallback is collective: one rank
    overflowing forces every rank to rerun.
    """
    try:
        pipeline_order = schedule.pipeline_order_with_comms
    except AttributeError as exc:
        raise ValueError(
            "Paged stash sizing requires a runtime PP schedule exposing "
            "pipeline_order_with_comms."
        ) from exc
    if not pipeline_order:
        return {}

    combined: PageCounts = {}
    for actions in pipeline_order.values():
        for key, pages in replay_schedule_page_peak(actions, stage_footprint).items():
            combined[key] = max(combined.get(key, 0), pages)
    return combined


def apply_graph_pp_skip_plan(schedule: _PipelineScheduleRuntime) -> int:
    """Install this rank's forward skip plan on the manager.

    Sizing takes the max over ranks so buffers stay uniform; the skip plan is
    the opposite -- it must be *this* rank's action list, because whether a
    forward is followed by its own backward is a local property of the local
    schedule. Returns the number of forwards the plan will skip, for logging.
    """
    manager = PagedStashManager.get_instance()
    if not manager.enabled or not manager.skip_immediate_backward:
        return 0
    try:
        pipeline_order = schedule.pipeline_order_with_comms
    except AttributeError:
        return 0
    if not pipeline_order:
        return 0

    # ``pipeline_order`` is keyed by pipeline-group rank, which differs from the
    # global rank whenever another parallelism (e.g. data parallel) sits
    # alongside PP.
    rank = schedule.rank
    actions = pipeline_order.get(rank)
    if actions is None:
        return 0

    plan = forward_skip_plan(actions)
    manager.set_skip_plan(plan)
    if not _LOGGED_SKIP_PLAN:
        # Log even when nothing is skippable: a schedule where no forward is
        # immediately followed by its own backward is a real and common answer
        # (in 1F1B only the last stage sees that adjacency), and silence would
        # not distinguish it from a plan that failed to install.
        _LOGGED_SKIP_PLAN.append(True)
        per_rank = {
            r: sum(forward_skip_plan(a)) for r, a in sorted(pipeline_order.items())
        }
        logger.info(
            "Paged stash: skip-on-immediate-backward plan for rank %d covers %d "
            "forward(s), %d skippable; per-rank skippable %s",
            rank,
            len(plan),
            sum(plan),
            per_rank,
        )
    return sum(plan)


def apply_graph_pp_sizing(schedule: _PipelineScheduleRuntime) -> PageCounts:
    """Install the schedule-derived page peak on the manager.

    Called before every step so the estimate tracks the per-slot footprints as
    they are measured. It is a no-op until the measure step has populated those
    footprints, which is also when buffer allocation first needs it.
    """
    manager = PagedStashManager.get_instance()
    if not manager.enabled:
        return {}
    footprint = manager.stage_page_footprint()
    if not footprint:
        return {}

    estimate = schedule_page_peak(schedule, footprint)
    manager.schedule_estimate = estimate
    if estimate:
        generations = max(
            (
                max_live_stash_generations(actions)
                for actions in schedule.pipeline_order_with_comms.values()
            ),
            default=1,
        )
        logger.info(
            "Paged stash: GraphPP schedule holds up to %d stashed microbatch(es) "
            "per rank; peak pages %s",
            generations,
            {f"{dtype}/{hidden}": pages for (dtype, hidden), pages in estimate.items()},
        )
    return estimate


# ===========================================================================
# Overflow detection and recovery around one forward/backward step
# ===========================================================================
#
# Port of Megatron-Core's ``PagedStashRunner``. The detection half is identical:
# the Triton kernels raise a device-side ``overflow`` flag when neither the CUDA
# nor the pinned-host freelist can satisfy a stash, both kernels then become
# no-ops for the rest of the step so nothing is corrupted further, and the flags
# are combined across ranks with a single all-reduce so every rank agrees on
# whether the step is usable.
#
# Two things differ from Megatron, and they trade against each other.
#
# **Reading the verdict.** Megatron calls ``.item()`` on the all-reduced flags at
# every step boundary. That is a device sync on a value that is almost always
# zero: it forfeits CPU run-ahead and turns each step's CPU work into a GPU idle
# bubble. Two modes here avoid it. ``assert`` enqueues ``torch._assert_async`` on
# the flag, so the check is a kernel rather than a host read; ``deferred`` copies
# the flags into a pinned host mirror asynchronously and reads the mirror on a
# later step once ``Event.query()`` says the copy landed. ``blocking`` keeps
# Megatron's synchronous read.
#
# **Recovering.** Megatron reruns the step with ``moe_paged_stash`` switched off,
# which it can do because the stash is an eager context manager the next forward
# simply will not enter. In GraphTrainer the stash ops are *nodes in the compiled
# graph*, so "turn the feature off" would mean retracing and recompiling mid-step.
# Rerunning the same graph with the same buffers would also just overflow again,
# since routing is unchanged. So recovery here grows the buffers: release the
# pages, scale the sizing factor, tear down the CUDA graph that referenced the old
# pointers, and reallocate. That converges for the same reason Megatron's fallback
# does -- attempt two has more room than attempt one -- and it keeps paged
# stashing enabled for the rest of training rather than silently losing it.
#
# Overflow means backward read activations that were never written back, so the
# step's gradients are invalid. The three modes differ in what they guarantee:
#
# ``assert`` (default)
#     No per-step sync, and an invalid step is *never* applied: the assertion is
#     ordered ahead of every optimizer kernel ``train_step`` launches next, the
#     same construction ``Trainer.train_step`` uses for its non-finite
#     loss/grad-norm guard. A fired device assertion ends the job, so there is no
#     in-process recovery.
# ``blocking``
#     The verdict is known before the step returns, so an overflowing step is
#     rerun with larger buffers and an invalid step is never applied. Recovers in
#     process; costs one device sync per step.
# ``deferred``
#     No per-step sync and no job loss, but the verdict for step N arrives at step
#     N+1 or later, after step N's gradients were applied. The buffers are grown so
#     later steps are correct and the affected step is named in a warning. This is
#     the only mode that can train on invalid gradients, so it is opt-in. A
#     blocking drain at shutdown keeps an overflow on the final steps from exiting
#     cleanly with a checkpoint that absorbed them.


OverflowCheckMode = Literal["assert", "blocking", "deferred"]

# Extra device-side "this step dropped data" flags folded into the same
# all-reduce, matching the ``overbudget`` slot in Megatron's
# ``check_moe_overflow``. Megatron fills it from
# ``token_dispatcher.check_over_budget()``; here the HybridEP dispatcher's
# accumulated ``overflow_flag`` is registered below.
_OVERBUDGET_PROBES: list[Callable[[], torch.Tensor | None]] = []
_OVERBUDGET_RESETS: list[Callable[[], None]] = []


def register_overbudget_probe(
    probe: Callable[[], torch.Tensor | None],
    *,
    reset: Callable[[], None] | None = None,
) -> None:
    """Register a device-side flag reporting dropped tokens during dispatch.

    ``reset`` clears the flag; it runs at the start of every step, next to the
    stash flag reset, so each verdict covers exactly one step.
    """
    _OVERBUDGET_PROBES.append(probe)
    if reset is not None:
        _OVERBUDGET_RESETS.append(reset)


class PagedStashRunner:
    """Run one forward/backward step with paged-stash overflow recovery.

    Sits exactly where Megatron's ``PagedStashRunner`` sits: wrapped around the
    call that runs a whole schedule of microbatches, so a single reset and a
    single overflow verdict cover the microbatches that share the stash buffers.

    Args:
        model_parts: Model chunks whose gradients are discarded on a retry.
        optimizers: Optimizer container zeroed on a retry.
        max_attempts: Attempts per step in ``blocking`` mode, including the
            first; in ``deferred`` mode, one more than the number of buffer
            growths allowed before the overflow becomes fatal. Megatron asserts
            on a third attempt; the same bound applies here.
        overflow_growth: Multiplier applied to the buffer sizing factor before
            growing.
        process_group: Group for the flag all-reduce. Defaults to WORLD, which
            is what Megatron uses.
        overflow_check: ``assert`` for the device-side assertion, ``blocking``
            for Megatron's synchronous per-step read, ``deferred`` for the
            non-blocking pinned-mirror read.
    """

    def __init__(
        self,
        model_parts,
        optimizers,
        *,
        max_attempts: int = 2,
        overflow_growth: float = 1.25,
        process_group: torch.distributed.ProcessGroup | None = None,
        overflow_check: OverflowCheckMode = "assert",
    ) -> None:
        if overflow_check not in ("assert", "blocking", "deferred"):
            raise ValueError(
                "paged stash: overflow_check must be 'assert', 'blocking' or "
                f"'deferred', got {overflow_check!r}."
            )
        self.manager = PagedStashManager.get_instance()
        self.model_parts = model_parts
        self.optimizers = optimizers
        self.max_attempts = max_attempts
        self.overflow_growth = overflow_growth
        self._logged_skip_plan = False
        self.process_group = process_group
        self.overflow_check = overflow_check

        # Deferred-read state. Allocated up front because ``cudaHostAlloc``
        # synchronizes: doing it lazily on the first captured step would put one
        # surprise sync in the training path, which is exactly what this mode
        # exists to avoid. ``_pending_step`` is the step the in-flight copy
        # covers, which is what the overflow warning names.
        self._flags_host = torch.zeros(
            3, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._flags_event = torch.cuda.Event()
        # Device-side scratch for the all-reduced flags, bound on first use and
        # then reused. A fresh allocation of a size the caching allocator has
        # not seen calls ``cudaMalloc``, which synchronizes the device -- so
        # building this tensor per step would reintroduce the very stall the
        # deferred read removes.
        self._flags_device: torch.Tensor | None = None
        self._pending_step: int | None = None
        self._growths = 0
        if self.manager.device is not None:
            self._ensure_flag_scratch(self.manager.device)
            self._warmup_flag_path()

    def _warmup_flag_path(self) -> None:
        """Run the flag copies once so no training step pays their first launch.

        Lazy CUDA module loading resolves a kernel on its first launch, and that
        resolution synchronizes. Paying it here, before training starts, is what
        makes ``deferred`` mode sync-free for *every* step rather than every
        step after the first. The collective is deliberately skipped: it is not
        what stalls, and a constructor is the wrong place for one.
        """
        flags = self._fill_local_flags()
        self._flags_host.copy_(flags, non_blocking=True)
        self._flags_event.record()
        self._flags_event.synchronize()

    # ------------------------------------------------------------------
    # Flag plumbing
    # ------------------------------------------------------------------
    def _ensure_flag_scratch(self, device: torch.device) -> torch.Tensor:
        if self._flags_device is None:
            self._flags_device = torch.zeros(3, dtype=torch.int32, device=device)
        return self._flags_device

    def _fill_local_flags(self) -> torch.Tensor:
        """Write this rank's ``(overflow, overbudget, host_spill)`` into scratch.

        Allocates nothing: the kernels store exactly 0 or 1, so the raw flags
        can be copied straight in, whereas ``check_paged_stash_*`` would build a
        comparison temporary every step. Those helpers stay as the public API
        and as the fallback for a manager without allocated flags.
        """
        manager = self.manager
        if manager.overflow is not None and manager.host_spill is not None:
            stash_overflow = manager.overflow.view(-1)[0]
            host_spill = manager.host_spill.view(-1)[0]
        else:
            stash_overflow = check_paged_stash_overflow().view(-1)[0]
            host_spill = check_paged_stash_host_spill().view(-1)[0]
        flags = self._ensure_flag_scratch(stash_overflow.device)
        flags[0].copy_(stash_overflow)
        flags[1].zero_()
        flags[2].copy_(host_spill)
        for probe in _OVERBUDGET_PROBES:
            flag = probe()
            if flag is not None:
                flags[1].logical_or_(flag.view(-1)[0])
        return flags

    def _device_flags(self) -> torch.Tensor:
        """``(stash_overflow, overbudget, host_spill)`` rank sums, one all_reduce.

        Returns a device tensor. Allocation-free, so nothing here synchronizes.
        """
        flags = self._fill_local_flags()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                flags,
                op=torch.distributed.ReduceOp.SUM,
                group=self.process_group,
            )
        return flags

    def _enqueue_probe(self, flags: torch.Tensor | None = None) -> None:
        """Start an asynchronous read of this step's flags. Never blocks.

        The device flags are cleared by the next ``begin_step``, which is
        enqueued after this copy on the same stream, so the mirror always sees
        the step it was issued for.
        """
        if flags is None:
            flags = self._device_flags()
        self._flags_host.copy_(flags, non_blocking=True)
        self._flags_event.record()
        self._pending_step = self.manager.step

    def _take_verdict(
        self, *, blocking: bool
    ) -> tuple[int, tuple[int, int, int]] | None:
        """Consume the in-flight verdict if it has landed.

        Returns ``(step, (overflow_ranks, overbudget_ranks, host_spill_ranks))``
        or ``None`` when nothing is pending, or when the copy is still in flight
        and ``blocking`` is False.
        """
        if self._pending_step is None:
            return None
        if blocking:
            self._flags_event.synchronize()
        elif not self._flags_event.query():
            return None
        step = self._pending_step
        self._pending_step = None
        counts = self._flags_host.tolist()
        return step, (int(counts[0]), int(counts[1]), int(counts[2]))

    def check_moe_overflow(self) -> tuple[int, int, int]:
        """``(stash_overflow, overbudget, host_spill)`` rank sums, read now.

        Megatron's synchronous form, kept for ``blocking`` mode.
        """
        flags = self._device_flags()
        return int(flags[0].item()), int(flags[1].item()), int(flags[2].item())

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def _grow_buffers(self) -> None:
        """Scale the sizing factor and drop the pages the CUDA graph captured."""
        self.manager.buffer_size_factor_cuda *= self.overflow_growth
        if self.manager.buffer_size_factor_cpu > 0:
            self.manager.buffer_size_factor_cpu *= self.overflow_growth
        logger.info(
            "Paged stash: growing buffers to buffer_size_factor_cuda=%.3f.",
            self.manager.buffer_size_factor_cuda,
        )

        if self.manager.overflow is not None:
            self.manager.overflow.zero_()
        if self.manager.host_spill is not None:
            self.manager.host_spill.zero_()

        # Destroy the CUDA graphs before releasing the pages they captured
        # pointers to. Same ordering constraint as Megatron's
        # ``reset_cuda_graph`` before ``release_stash_buffers``.
        cudagraph_teardown()
        self.manager.release_stash_buffers()

    def prepare_for_rerun(self) -> None:
        """Discard the corrupted attempt and grow the stash for the retry."""
        logger.info("Paged stash: rerunning forward-backward with larger buffers.")
        for model_part in self.model_parts:
            model_part.zero_grad(set_to_none=True)
        if self.optimizers is not None:
            self.optimizers.zero_grad()
        self._grow_buffers()

    # ------------------------------------------------------------------
    # Verdict handling
    # ------------------------------------------------------------------
    @staticmethod
    def _log_overbudget(overbudget_ranks: int) -> None:
        logger.info(
            "Paged stash: token drop during MoE token dispatch (over budget) on "
            "%d rank(s). Consider increasing the token dispatcher capacity factor.",
            overbudget_ranks,
        )

    @staticmethod
    def _log_host_spill(host_spill_ranks: int) -> None:
        logger.info(
            "Paged stash: spilled activations to pinned host on %d rank(s) "
            "(CUDA stash full). Consider increasing "
            "paged_stash_buffer_size_factor_cuda for potentially better "
            "performance.",
            host_spill_ranks,
        )

    def _handle_deferred_verdict(
        self, verdict: tuple[int, tuple[int, int, int]] | None
    ) -> None:
        """Act on a verdict that belongs to an already-completed step."""
        if verdict is None:
            return
        step, (overflow_ranks, overbudget_ranks, host_spill_ranks) = verdict
        if overflow_ranks == 0 and overbudget_ranks == 0:
            if host_spill_ranks > 0:
                self._log_host_spill(host_spill_ranks)
            return

        if overbudget_ranks > 0:
            self._log_overbudget(overbudget_ranks)
        if overflow_ranks == 0:
            return

        self._growths += 1
        if self._growths >= self.max_attempts:
            raise RuntimeError(
                f"Paged stash: stashing buffer overflow on {overflow_ranks} "
                f"rank(s) at step {step}, after {self._growths - 1} buffer "
                "growth(s). Increase paged_stash_buffer_size_factor_cuda, "
                "enable paged_stash_buffer_size_factor_cpu for host spill, or "
                "disable the feature with --compile.memory_policy default."
            )
        logger.warning(
            "Paged stash: stashing buffer overflow on %d rank(s) at step %d. "
            "The deferred overflow check observes a step after its gradients "
            "have been applied, so step %d trained on stashed activations that "
            "were not written back. Growing the buffers so later steps are "
            "correct; use --compile.paged_stash_overflow_check blocking to "
            "rerun the offending step instead.",
            overflow_ranks,
            step,
            step,
        )
        self._grow_buffers()

    def final_overflow_check(self) -> None:
        """Drain the last in-flight verdict at shutdown, blocking.

        The deferred check needs a later step to observe an overflow, so an
        overflow on the final steps would otherwise escape and the job would
        exit cleanly with its checkpoint. One synchronizing read at shutdown
        closes that window at zero per-step cost.
        """
        if not self.manager.enabled or self.overflow_check == "blocking":
            return
        if self.overflow_check == "assert":
            # Overflow cannot reach the host un-asserted in this mode; the
            # mirror only carries spill / over-budget diagnostics.
            self._report_only(self._take_verdict(blocking=True))
            return
        verdict = self._take_verdict(blocking=True)
        if verdict is None:
            return
        step, (overflow_ranks, overbudget_ranks, host_spill_ranks) = verdict
        if host_spill_ranks > 0:
            self._log_host_spill(host_spill_ranks)
        if overbudget_ranks > 0:
            self._log_overbudget(overbudget_ranks)
        if overflow_ranks > 0:
            raise RuntimeError(
                f"Paged stash: stashing buffer overflow on {overflow_ranks} "
                f"rank(s) at step {step} (observed at shutdown). That step's "
                "gradients were computed from stashed activations that were "
                "not written back. Increase "
                "paged_stash_buffer_size_factor_cuda or enable "
                "paged_stash_buffer_size_factor_cpu."
            )

    # ------------------------------------------------------------------
    # Step entry points
    # ------------------------------------------------------------------
    def __call__(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run ``fwd_bwd_fn`` under the configured overflow policy."""
        if not self.manager.enabled:
            return fwd_bwd_fn()
        if self.overflow_check == "blocking":
            return self._call_blocking(fwd_bwd_fn)
        if self.overflow_check == "assert":
            return self._call_assert(fwd_bwd_fn)
        return self._call_deferred(fwd_bwd_fn)

    def _call_assert(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Fail the step on device rather than ever applying its gradients.

        ``torch._assert_async`` enqueues the check as a kernel, so it costs no
        host sync, and stream ordering puts it ahead of every optimizer kernel
        that ``train_step`` launches next. This is the same construction
        ``Trainer.train_step`` already uses for the non-finite loss/grad-norm
        guard immediately before ``optimizers.step()``.

        The asynchronous mirror is still filled, but only so host spill and
        over-budget dispatch can be *reported*; overflow never reaches it
        un-asserted.
        """
        self._report_only(self._take_verdict(blocking=False))

        self._begin_step()
        warmup_paged_stash_kernels()
        result = fwd_bwd_fn()

        if self.manager.active:
            flags = self._device_flags()
            # torch._assert_async caps the message at 255 characters; the full
            # remediation is in the paged_stash_overflow_check config docstring.
            torch._assert_async(
                flags[0].eq(0),
                f"Paged stash: buffer overflow at step {self.manager.step}; "
                "backward read activations that were never written back. "
                "Stopped before the optimizer update. Raise "
                "--compile.paged_stash_buffer_size_factor_cuda.",
            )
            self._enqueue_probe(flags)
        return result

    def _report_only(self, verdict: tuple[int, tuple[int, int, int]] | None) -> None:
        """Log host spill / over-budget from a landed verdict; never act on it."""
        if verdict is None:
            return
        _step, (_overflow, overbudget_ranks, host_spill_ranks) = verdict
        if host_spill_ranks > 0:
            self._log_host_spill(host_spill_ranks)
        if overbudget_ranks > 0:
            self._log_overbudget(overbudget_ranks)

    def _call_deferred(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        # Consume any verdict that has landed since the last step first, so a
        # buffer growth is applied before ``begin_step`` reallocates.
        self._handle_deferred_verdict(self._take_verdict(blocking=False))

        self._begin_step()
        # Triton compiles on first launch, which a CUDA graph capture cannot
        # tolerate; prime the cache while we are still eager.
        warmup_paged_stash_kernels()
        result = fwd_bwd_fn()

        if self.manager.active:
            self._enqueue_probe()
        return result

    def _call_blocking(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        attempts = 0
        while True:
            attempts += 1
            assert attempts <= self.max_attempts, (
                f"PagedStashRunner: {attempts} attempts exceeded the maximum "
                f"of {self.max_attempts}."
            )
            self._begin_step()
            warmup_paged_stash_kernels()
            result = fwd_bwd_fn()

            if not self.manager.active:
                # Measure step: no pages were used, so no verdict to reach.
                return result

            (
                overflow_ranks,
                overbudget_ranks,
                host_spill_ranks,
            ) = self.check_moe_overflow()
            # A token drop comes from the dispatcher's capacity factor, which a
            # rerun with larger stash buffers cannot change -- routing is
            # deterministic -- so it is reported, as in the other modes, and
            # only a stash overflow is rerun.
            if overbudget_ranks > 0:
                self._log_overbudget(overbudget_ranks)
            if overflow_ranks == 0:
                if host_spill_ranks > 0:
                    self._log_host_spill(host_spill_ranks)
                return result

            logger.info(
                "Paged stash: stashing buffer overflow on %d rank(s). "
                "Consider increasing paged_stash_buffer_size_factor_cuda "
                "or paged_stash_buffer_size_factor_cpu.",
                overflow_ranks,
            )
            self.prepare_for_rerun()

    def _begin_step(self) -> None:
        """Reset the stash and the over-budget probes for a new step."""
        self.manager.begin_step()
        for reset in _OVERBUDGET_RESETS:
            reset()

    def apply_pp_schedule(self, schedule) -> None:
        """Feed the GraphPP schedule into buffer sizing and the skip plan.

        This is the GraphTrainer replacement for Megatron's virtual-pipeline
        bookkeeping: instead of replaying the schedule to find how many stashes
        are live at once, read it off the runtime PP schedule. Called before
        every step, so the first allocation after the measure step already
        accounts for it.
        """
        # GraphPP wraps the runtime schedule in a GraphPipelineRuntime.
        schedule = getattr(schedule, "schedule", schedule)
        if schedule is None:
            return
        apply_graph_pp_sizing(schedule)
        skipped = apply_graph_pp_skip_plan(schedule)
        if skipped and not self._logged_skip_plan:
            self._logged_skip_plan = True
            logger.info(
                "Paged stash: %d forward(s) per step run their own backward next; "
                "their last paged layer stays resident instead of stashing.",
                skipped,
            )


# ===========================================================================
# Memory policy registration and trainer setup
# ===========================================================================


@register_memory_policy("sac_and_paged_stash")
def _sac_and_paged_stash_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: GraphTrainer.Config,
) -> torch.fx.GraphModule:
    """SAC + MoE paged stash: apply default SAC, then tag padded MoE activations."""
    _default_memory_policy_pass(gm, config=config)
    # A tensor that never crosses into backward cannot be paged, and SAC leaves
    # fused-MXFP8 quantized activations on recompute. No-op when the expert MLP
    # does not emit fp8 across the boundary.
    promote_low_precision_saves(
        gm,
        module_fqn_pattern=config.compile.paged_stash_module_fqn,
    )
    if config.compile.paged_stash_page_recomputed:
        # Megatron's choice for the FC1 output: stash it rather than rebuild it.
        promote_declared_saves(
            gm,
            module_fqn_pattern=config.compile.paged_stash_module_fqn,
        )
    tag_paged_stash_activations(
        gm,
        module_fqn_pattern=config.compile.paged_stash_module_fqn,
    )
    return gm


def build_paged_stash_runner(
    compile_config: GraphTrainerCompileConfig,
    *,
    device: torch.device,
    pp_enabled: bool,
    model_parts,
    optimizers,
) -> PagedStashRunner | None:
    """Enable MoE paged stashing and return its step runner, if configured.

    Buffers are not created here: their size comes from the first measured
    step. What is created up front is everything that must outlive a step --
    the manager, its sizing configuration, and the pack/unpack stream --
    matching Megatron, which also defers ``allocate_stash_buffers`` to the
    end of its capture iteration.
    """
    if compile_config.memory_policy != "sac_and_paged_stash":
        return None

    # Producers declare their row layout only when the feature is on,
    # so a run without paged stashing traces exactly the graph it did
    # before. Set before the first trace, which happens on the first step.
    set_marking_enabled(True)

    # Token drops in the HybridEP dispatcher fold into the same verdict as
    # stash overflow -- the ``overbudget`` slot of Megatron's check_moe_overflow.
    from torchtitan.distributed.deepep.hybridep import (
        check_hybridep_over_budget,
        reset_hybridep_over_budget,
    )

    if check_hybridep_over_budget not in _OVERBUDGET_PROBES:
        register_overbudget_probe(
            check_hybridep_over_budget, reset=reset_hybridep_over_budget
        )

    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = device
    manager.page_size = compile_config.paged_stash_page_size
    manager.buffer_size_factor_cuda = compile_config.paged_stash_buffer_size_factor_cuda
    manager.buffer_size_factor_cpu = compile_config.paged_stash_buffer_size_factor_cpu
    manager.skip_immediate_backward = compile_config.paged_stash_skip_immediate_backward
    if manager.skip_immediate_backward and not pp_enabled:
        # Without pipeline parallelism the joint graph runs every forward
        # then every backward, so the last paged layer's backward is always
        # the next thing to run -- the same test Megatron applies to its
        # single-stage schedule. The answer never varies between steps,
        # which is what makes it safe to freeze into a CUDA graph.
        manager.set_static_skip(True)
    logger.info(
        "Paged stash enabled: page_size=%d, factor_cuda=%.2f, factor_cpu=%.2f, "
        "overflow_check=%s",
        manager.page_size,
        manager.buffer_size_factor_cuda,
        manager.buffer_size_factor_cpu,
        compile_config.paged_stash_overflow_check,
    )
    return PagedStashRunner(
        model_parts,
        optimizers,
        overflow_check=compile_config.paged_stash_overflow_check,
    )


__all__ = [
    "GLOBAL_BLOCK_SIZE",
    "paged_stash_copy_kernel",
    "paged_stash_pop_kernel",
    "PagedStashBuffer",
    "PagedStashManager",
    "SCALE_INV_BLOCK_SIZE",
    "StashSlot",
    "buffer_dtype_for",
    "check_paged_stash_host_spill",
    "check_paged_stash_overflow",
    "PAGED_STASH_OPS",
    "clear_wait_registry",
    "reload",
    "stash_",
    "wait",
    "wait_stash_",
    "warmup_paged_stash_kernels",
    "promote_declared_saves",
    "DEFAULT_MODULE_FQN",
    "promote_low_precision_saves",
    "matches_module_subtree",
    "PAGED_STASH_TAG",
    "TokenCounts",
    "find_layer_token_counts",
    "tag_paged_stash_activations",
    "apply_paged_stash_pass",
    "apply_graph_pp_sizing",
    "apply_graph_pp_skip_plan",
    "forward_skip_plan",
    "max_live_stash_generations",
    "replay_schedule_page_peak",
    "schedule_page_peak",
    "OverflowCheckMode",
    "PagedStashRunner",
    "register_overbudget_probe",
    "build_paged_stash_runner",
    "mark_pageable",
    "set_marking_enabled",
]
