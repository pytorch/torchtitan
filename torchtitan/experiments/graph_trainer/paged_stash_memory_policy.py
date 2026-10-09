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
dispatcher has a static capacity (HybridEP with ``non_blocking_capacity_factor``),
in a recipe derived from one that builds such a model::

    import dataclasses

    from torchtitan.models.common.moe import (
        RoundRobinTokenChoiceTopKRouter,
        TokenChoiceTopKRouter,
    )
    from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
        graph_trainer_deepseek_v3_debugmodel_hybridep,
    )


    def paged_stash_debugmodel():
        config = graph_trainer_deepseek_v3_debugmodel_hybridep()
        config.compile.memory_policy = "sac_and_paged_stash"
        # GraphTrainer applies its own CUDA graph pass when the trainer-level
        # CUDA graph wrapper is off.
        config.training.disable_cuda_graphs = True
        config.parallelism.data_parallel_shard_degree = 4
        config.parallelism.expert_parallel_degree = 2
        # Route round-robin, with exact balance (see below).
        router_fields = dataclasses.fields(TokenChoiceTopKRouter.Config)
        for layer in config.model.layers:
            if getattr(layer, "moe", None) is not None:
                router = layer.moe.router
                layer.moe.router = RoundRobinTokenChoiceTopKRouter.Config(
                    **{f.name: getattr(router, f.name) for f in router_fields}
                )
        return config


    def paged_stash_debugmodel_pp():
        # With pipeline parallelism, where the feature pays for itself.
        config = paged_stash_debugmodel()
        config.parallelism.pipeline_parallel_degree = 2
        config.parallelism.num_pp_microbatches = 4
        config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
        config.parallelism.data_parallel_shard_degree = 2
        return config

Launch either from the repo root with the recipe module on the path::

    NGPU=4 MODULE=my_recipes CONFIG=paged_stash_debugmodel ./run_train.sh

Both route tokens round-robin across experts, with exact balance, as the
integration tests do. Buffers are sized from the first step's routing, and the
debug model's untrained router shifts enough over the next few steps to outgrow
them, which the default ``deferred`` overflow check treats as fatal.

Options
-------

``compile.paged_stash.page_size``
    Tokens per page (default 64). Smaller pages waste less on stashes that do
    not fill a page.
``compile.paged_stash.buffer_size_factor_cuda``
    Headroom over the pages the measured step and the pipeline schedule
    require (default 1.10).
``compile.paged_stash.buffer_size_factor_cpu``
    Headroom for an optional pinned-host spill buffer (default 0 = no host
    spill).
``compile.paged_stash.prefetch_n_layers``
    Issue a backward's reloads this many paged-layer schedule entries early,
    across pipeline actions, to overlap page reads with compute (default 1,
    Megatron's). 0 reloads an activation when backward first reads it.
``compile.paged_stash.module_fqn``
    Module-FQN pattern of the modules annotated as the paged-stash region
    (default ``layers.*.moe.routed_experts``).
``compile.paged_stash.skip_immediate_backward``
    Keep the last paged layer resident instead of stashing it when the schedule
    runs that microbatch's backward next (default on). Matches Megatron's
    ``remove_paged_tensor_from_stash``.
``compile.paged_stash.overflow_check``
    ``deferred`` (default): reads the overflow flag from a pinned host copy a
    step or two later, never syncing, skips the optimizer steps in between on
    the GPU, and fails the job. Requires fused Adam or AdamW, the optimizers
    that can skip a step on device.
    ``blocking``: Megatron's per-step read, reruns an overflowing step with
    paging disabled, as Megatron does. Works with any optimizer.

How It Works
------------

1. **Annotation, not a context manager.** Megatron opens ``saved_tensors_hooks``
   around the fused grouped MLP and pages whatever Transformer Engine tagged
   inside it. There is no such window in a traced graph, so
   ``tag_paged_stash_activations`` annotates nodes -- following the same
   ``MUST_SAVE`` / ``MUST_RECOMPUTE`` convention as the rest of the
   memory-policy framework -- and ``apply_paged_stash_pass`` rewrites them.
   The region it decides is annotated on the model before tracing, the way
   GraphTrainer annotates module FQNs and EP regions: every activation there
   that backward reads and the stash can hold is stashed, whatever SAC chose.
   The rewrite runs after selective activation remat, whose recomputed
   duplicates would otherwise keep reading the padded original.

2. **The forward stash is a mutation.** GraphPP defines its forward graph as
   the dependency closure of the forward user outputs. A value-returning stash
   reaches no forward output, so the whole chain would be classified as
   backward and the activation would still be saved. Writing the stash as a
   mutation of a small ``handle`` tensor routes it through GraphPP's existing
   forward-mutation path, so the handle -- not the padded tensor -- is what
   crosses into backward.

3. **Megatron's stash and reload points.** A layer's stashes launch together
   at the end of its expert MLP and are joined, freeing the padded
   activations, at the next paged layer's group start -- just before its
   expert MLP reads the dispatch output -- as Megatron's
   ``paged_stash_group_commit`` and ``paged_stash_group_start`` do. Reloads
   are issued one schedule entry ahead, as Megatron's are: the paged-layer
   schedule is read off GraphPP's actions, so a backward's first layer is
   reloaded at the end of whatever forward or backward precedes it on the
   rank, even in another pipeline action.

4. **Lifecycle.** The first step measures: ``stash_`` reads the real token
   count and parks the activation, so it is numerically identical to a run
   without the feature. The second allocates the buffers, primes the Triton
   kernels (Triton cannot compile inside a CUDA graph capture), and pages for
   real; CUDA graph capture happens on it. Later steps only reset freelists
   and flags, which allocates nothing.

5. **GraphPP replaces the virtual-pipeline estimate.** Megatron reconstructs
   its schedule to work out how many stashes are live at once.
   ``schedule_page_peak`` replays the real PP action list over a per-stage
   page footprint instead, which is both simpler and tighter.

6. **Sizing is in pages, not tokens.** Megatron accumulates raw token counts,
   which under-allocates when individual stashes are small relative to
   ``page_size``: four concurrent 16-token stashes need four pages, not one.

7. **Overflow is fatal, and checking for it never syncs.** Detection is
   identical to Megatron -- a device-side flag from the Triton kernels, next to
   ``host_spill`` and ``overbudget``, HybridEP's token-drop flag accumulated
   over every dispatch of the step. Megatron all-reduces the flags and reads
   them with ``.item()`` every step; here each step copies the rank's own flags
   to pinned host memory asynchronously and a later step reads the copy, and a
   rank that overflowed raises. The overflow flag is sticky -- ``begin_step``
   leaves it set -- so a check that reads it steps later cannot miss it, and a
   blocking read at shutdown covers the final steps. The steps run before the
   raise never apply their corrupted gradients: each optimizer step
   all-reduces the flag on device and passes it to fused Adam/AdamW as AMP's
   ``found_inf``, so every rank skips the update, and a skipped step's MoE
   expert-bias update is undone. ``blocking`` mode instead reads the
   all-reduced flags synchronously and reruns an overflowing step with paging
   disabled, as Megatron does. A token drop is reported but never fatal: it
   comes from the capacity factor, not the stash.

The Triton kernels and the ``PagedStashBuffer`` layout are copied from
Megatron-Core unchanged, so the page allocator, the pinned-host spill path, and
the overflow protocol behave identically.

Limitations
-----------

- Requires a token dispatcher with a static capacity factor. With a
  blocking/dynamic dispatcher the permuted buffer is already exactly sized and
  the pass tags nothing.
- Only capacity-padded activations in the annotated region are paged;
  elsewhere SAC decides as usual.
- The stash copies a tensor's live tokens as a prefix of its bytes, so tensors
  in the region must store tokens in order. The pass reads the row layout off
  a tensor's shape or any reshape of it -- a flat block-scale buffer reads as
  token-ordered, as Transformer Engine's do in Megatron. A tensor no shape shows
  token rows for is left to SAC; if backward then rebuilds it from a padded
  input, that input crosses into backward unpaged, and a warning names it.
- GraphTrainer's GraphPP runtime does not capture CUDA graphs yet, so a single
  run exercises either the CUDA-graph path or the GraphPP path, not both.
- With the default ``deferred`` overflow check, an overflow can surface as the
  trainer's "Loss or gradient norm is not finite" error rather than the paged
  stash one: the overflowing step's backward reads activations that were never
  written back, and if they make the gradient norm non-finite, that check stops
  the job first. Either way the step is never applied.
"""

from __future__ import annotations

import logging
import operator
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch
import torch.nn as nn

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
from torch.distributed.tensor import DTensor
from torch.fx import has_side_effect, Node
from torch.fx.traceback import annotate_fn
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _can_offload_node,
    _collect_offload_info,
    _get_aten_target,
    _get_storage_chain,
    _is_view,
    _OffloadInfo,
    _redirect_backward_consumers,
    _tensor_bytes,
    _tensor_is_contiguous,
)
from torchtitan.experiments.graph_trainer.cuda_graph import reset_cuda_graphs
from torchtitan.experiments.graph_trainer.memory_policy import (
    _default_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.registry import register_memory_policy

if TYPE_CHECKING:
    from torchtitan.components.optim.optimizer import OptimizersContainer
    from torchtitan.experiments.graph_trainer.configs import PagedStashConfig
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)


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
# recorded the pipeline schedule while it ran and drove stash/reload from
# autograd functions; here the FX graph places the ops (see
# ``apply_paged_stash_pass``) and the schedule is read off GraphPP's actions
# (see ``paged_layer_schedule``). The manager keeps what spans graph
# boundaries: the buffers, the shared overflow/host-spill flags, the side
# stream, the capture-phase token accounting, and the stashes in flight or
# awaiting their reload, which a later pipeline action joins or prefetches.


# Megatron pages columnwise MXFP8 scale-inverse tensors at 1/32 the token
# granularity of their data tensor. GraphTrainer's MoE saves plain tensors, so
# the manager tracks a per-slot ``token_divisor`` instead of a boolean, with 1
# meaning "one row per token". Kept as a named constant so an MXFP8 GraphTrainer
# MoE can reuse the Megatron value directly.
SCALE_INV_BLOCK_SIZE = 32


def _num_pages(num_tokens: int, page_size: int) -> int:
    """Pages needed to hold ``num_tokens`` rows."""
    return (num_tokens + page_size - 1) // page_size


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
        self.num_cuda_pages = _num_pages(num_tokens, page_size)
        self.total_cuda_tokens = self.num_cuda_pages * page_size
        self.cuda_buffer = torch.empty(
            (self.total_cuda_tokens, hidden_size), dtype=dtype, device=device
        )

        # Host buffer (pinned), optional
        self.num_host_pages = _num_pages(num_tokens_host, page_size)
        self.total_host_tokens = self.num_host_pages * page_size
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

    def launch_copy(
        self,
        source: torch.Tensor,
        num_tokens: torch.Tensor,
        page_record: torch.Tensor,
        spilled: torch.Tensor,
        new_free_list_head: torch.Tensor,
        *,
        hidden_size: int,
        num_blocks: int,
    ) -> None:
        """Launch ``paged_stash_copy_kernel`` from ``source`` into free pages."""
        paged_stash_copy_kernel[(num_blocks,)](
            source,
            self.cuda_buffer,
            self._host_or_cuda_buffer(),
            num_tokens,
            self.free_list_cuda,
            self.free_list_host,
            self.free_list_head,
            self.free_list_tail,
            self.free_list_capacity,
            page_record,
            self.overflow,
            self.host_spill,
            spilled,
            new_free_list_head,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=hidden_size,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
            HAS_HOST_BUFFER=1 if self.host_buffer is not None else 0,
        )

    def launch_pop(
        self,
        dest: torch.Tensor,
        num_tokens: torch.Tensor,
        page_record: torch.Tensor,
        spilled: torch.Tensor,
        new_free_list_tail: torch.Tensor,
        *,
        hidden_size: int,
        num_blocks: int,
    ) -> None:
        """Launch ``paged_stash_pop_kernel`` from the recorded pages into ``dest``."""
        paged_stash_pop_kernel[(num_blocks,)](
            self.cuda_buffer,
            self._host_or_cuda_buffer(),
            dest,
            num_tokens,
            page_record,
            spilled,
            self.overflow,
            self.free_list_cuda,
            self.free_list_host,
            self.free_list_tail,
            self.free_list_capacity,
            new_free_list_tail,
            PAGE_SIZE=self.page_size,
            HIDDEN_SIZE=hidden_size,
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,
        )

    def _host_or_cuda_buffer(self) -> torch.Tensor:
        # The kernels take a host pointer unconditionally; without a host buffer
        # it is never dereferenced, so any valid pointer will do.
        return self.host_buffer if self.host_buffer is not None else self.cuda_buffer

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


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    """One paged MoE layer's forward or backward for one microbatch.

    Megatron's schedule entry: it encodes ``(vp_stage, layer, microbatch)`` as
    one integer and negates it for the backward.
    """

    forward: bool
    stage_index: int
    layer_id: int
    microbatch: int


@dataclass(slots=True)
class _StashRecord:
    """One stash of one slot, held until the reload that consumes it.

    ``resident`` is the activation itself when the stash kept it instead of
    paging it; ``reloaded`` is the tensor a prefetch is filling, which the
    reload hands to backward instead of popping the pages itself.
    """

    microbatch: int
    handle: torch.Tensor
    num_tokens: torch.Tensor
    shape: list[int]
    dtype: torch.dtype
    resident: torch.Tensor | None = None
    reloaded: torch.Tensor | None = None


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
    measure step records each slot's live pages and GraphPP's schedule is
    replayed over them (see ``schedule_page_peak``).
    """

    _INSTANCE: PagedStashManager | None = None

    @classmethod
    def get_instance(cls) -> PagedStashManager:
        if cls._INSTANCE is None:
            cls._INSTANCE = PagedStashManager()
        return cls._INSTANCE

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the manager, if there is one, so the next ``get_instance``
        starts a fresh one.

        Its page buffers are freed here rather than with the last reference to
        it: the ``deferred`` runner's optimizer hooks hold one for as long as
        the optimizers live. As on the fallback path, this must run after the
        CUDA graphs that captured the buffers' pointers are torn down.
        """
        manager = cls._INSTANCE
        if manager is None:
            return
        cls._INSTANCE = None
        if manager.stash_buffers:
            manager.release_stash_buffers()

    def __init__(self) -> None:
        self.enabled = False
        self.status = "begin"  # begin, capture, captured
        # Set for a ``blocking`` rerun, which runs with paging disabled as
        # Megatron's does: the ops park activations instead of paging them.
        self.paging_disabled = False
        self.device: torch.device | None = None

        self._stream: torch.cuda.Stream | None = None

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

        # (dtype, hidden_size) -> buffer, shared by every slot with that key.
        self.stash_buffers: dict[
            tuple[torch.dtype, int], PagedStashBuffer
        ] | None = None
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
        # temp_/max_tokens_across_vp_stages pair but in whole pages, since every
        # individual stash rounds up to a page boundary: ``_live_pages`` is the
        # currently stashed page count per (dtype, hidden_size), ``_peak_pages``
        # its running peak.
        self._live_pages: dict[tuple[torch.dtype, int], int] = {}
        self._peak_pages: dict[tuple[torch.dtype, int], int] = {}

        # Largest live token count observed for a *single* stash of each slot.
        # Replaying the GraphPP schedule over these reproduces Megatron's
        # per-schedule-entry accumulation without modeling the virtual pipeline
        # by hand.
        self._slot_tokens: dict[int, int] = {}

        # Activations parked while not paging -- the measure step and a
        # ``blocking`` rerun: slot_id -> queued (tensor, measured token count,
        # or None outside the measure step). Megatron keeps the real tensor on
        # the PagedTensor during its capture iteration; this is the same trick,
        # keyed by the static slot id. A queue is needed because with pipeline
        # parallelism the same slot is stashed once per in-flight microbatch
        # before the first one is reloaded, and 1F1B reloads them in stash order.
        self._parked_tensors: dict[
            int, deque[tuple[torch.Tensor, int | None]]
        ] = defaultdict(deque)

        # The schedule: this rank's GraphPP actions, or None without pipeline
        # parallelism. ``paged_layer_schedule`` turns them into Megatron's
        # ``_pp_schedule``, which decides when a stash is skipped and which
        # reloads are prefetched; ``_schedule_cache`` holds it per set of paged
        # layers, which is only known once the graph pass has run.
        self.skip_immediate_backward = False
        self._pp_actions: list[_Action] | None = None
        self._schedule_cache: (
            tuple[
                dict[int, tuple[int, ...]],
                list[ScheduleEntry],
                dict[ScheduleEntry, int],
            ]
            | None
        ) = None
        self._warned_capture_schedule = False

        # Every stash of a slot, in stash order, until its reload: Megatron's
        # `paged_tensors_to_reload`. Records are per slot because with
        # pipeline parallelism a slot is stashed once per in-flight microbatch
        # and reloaded in stash order, so the count of a slot's stashes so far
        # is the microbatch its next stash belongs to. A stage's backward runs
        # the microbatch its latest reload was for.
        self._records: dict[int, deque[_StashRecord]] = defaultdict(deque)
        self._num_stashed: dict[int, int] = defaultdict(int)
        self._backward_microbatch: dict[int, int] = {}
        # Copies launched but not yet joined, with what they read: Megatron's
        # `paged_tensors_stash_in_progress`. Holding the activations here is
        # what keeps them alive until a later layer's group start, possibly in
        # a later pipeline action, as Megatron's `_original_tensor` does.
        self._stashes_in_flight: list[tuple[torch.Event, tuple[torch.Tensor, ...]]] = []

        self.step = 0

    @property
    def stream(self) -> torch.cuda.Stream:
        """Side stream for both the stash and the reload copies.

        One stream, as in Megatron: the freelists are unsynchronized mutable
        state, so packing and unpacking must never run concurrently.
        """
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

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

    # ------------------------------------------------------------------
    # Capture-phase accounting
    # ------------------------------------------------------------------
    def record_stash(self, slot: StashSlot, num_tokens: int) -> None:
        """Record one measured stash and update the per-key live/peak counters."""
        key = slot.key
        self._live_pages[key] = self._live_pages.get(key, 0) + _num_pages(
            num_tokens, self.page_size
        )
        self._peak_pages[key] = max(self._peak_pages.get(key, 0), self._live_pages[key])
        self._slot_tokens[slot.slot_id] = max(
            self._slot_tokens.get(slot.slot_id, 0), num_tokens
        )

    def record_reload(self, slot: StashSlot, num_tokens: int) -> None:
        """Release one measured stash from the live counters."""
        if slot.key in self._live_pages:
            self._live_pages[slot.key] -= _num_pages(num_tokens, self.page_size)

    def park_tensor(
        self, slot_id: int, tensor: torch.Tensor, num_tokens: int | None
    ) -> None:
        """Park an activation that is not paged, and its measured token count
        (``None`` outside the measure step), for the reload.

        The count travels with the tensor so the reload can release it from the
        live counters without a second device-to-host read.
        """
        self._parked_tensors[slot_id].append((tensor, num_tokens))

    def pop_parked_tensor(self, slot_id: int) -> tuple[torch.Tensor, int | None]:
        queue = self._parked_tensors.get(slot_id)
        if not queue:
            raise RuntimeError(
                f"paged stash: no parked activation for slot {slot_id}. "
                "Every paged_stash.reload must follow its paged_stash.stash."
            )
        return queue.popleft()

    # ------------------------------------------------------------------
    # The schedule: skipping, joining, and prefetching
    # ------------------------------------------------------------------
    def set_pipeline_actions(self, actions: list[_Action] | None) -> None:
        """Install this rank's GraphPP actions, or None without pipeline
        parallelism."""
        if actions != self._pp_actions:
            self._pp_actions = actions
            self._schedule_cache = None

    def _paged_layers(self) -> dict[int, tuple[int, ...]]:
        """Each stage's paged layers, in forward order."""
        layers: dict[int, set[int]] = defaultdict(set)
        for slot in self.slots.values():
            layers[slot.stage_index].add(slot.layer_id)
        return {stage: tuple(sorted(ids)) for stage, ids in layers.items()}

    def _schedule(
        self, microbatch: int
    ) -> tuple[list[ScheduleEntry], dict[ScheduleEntry, int]]:
        """This rank's schedule and each entry's position in it."""
        layers = self._paged_layers()
        if self._pp_actions is None:
            # Without pipeline parallelism a microbatch runs its whole forward,
            # then its whole backward.
            actions = [
                _Action(0, FORWARD, microbatch),
                _Action(0, FULL_BACKWARD, microbatch),
            ]
            schedule = paged_layer_schedule(actions, layers)
            return schedule, {entry: i for i, entry in enumerate(schedule)}
        if self._schedule_cache is None or self._schedule_cache[0] != layers:
            schedule = paged_layer_schedule(self._pp_actions, layers)
            self._schedule_cache = (
                layers,
                schedule,
                {entry: i for i, entry in enumerate(schedule)},
            )
            logger.info(
                "Paged stash: this rank's schedule has %d paged-layer entries "
                "over stages %s",
                len(schedule),
                sorted(layers),
            )
        return self._schedule_cache[1], self._schedule_cache[2]

    def next_entries(self, entry: ScheduleEntry, count: int = 1) -> list[ScheduleEntry]:
        """The ``count`` schedule entries that follow ``entry`` on this rank.

        Empty when the answer would not survive CUDA graph replay. A captured
        graph freezes host-side Python, so with pipeline parallelism a stage
        graph captured at one place in the schedule would replay that place's
        skips and prefetches -- and the stash handles they read -- for every
        microbatch. Without pipeline parallelism the step is one graph whose
        place never changes. Not skipping and not prefetching is correct under
        every schedule, so that is the safe direction to fail: each stash then
        makes the round trip and each reload pops its pages when backward
        first reads them.
        """
        if self._pp_actions is not None and torch.cuda.is_current_stream_capturing():
            if not self._warned_capture_schedule:
                self._warned_capture_schedule = True
                logger.warning(
                    "Paged stash: skipping stashes and prefetching reloads are "
                    "disabled under CUDA graph capture with pipeline parallelism; "
                    "a microbatch's place in the schedule cannot be replayed. "
                    "Paging every stash and reloading on demand instead "
                    "(correct, just not optimized)."
                )
            return []
        schedule, index = self._schedule(entry.microbatch)
        position = index.get(entry)
        if position is None:
            return []
        return schedule[position + 1 : position + 1 + count]

    def stash_entry(self, slot: StashSlot) -> ScheduleEntry:
        """The forward entry the next stash of ``slot`` belongs to."""
        return ScheduleEntry(
            True, slot.stage_index, slot.layer_id, self._num_stashed[slot.slot_id]
        )

    def backward_entry(self, slot: StashSlot) -> ScheduleEntry | None:
        """The entry of the running backward of ``slot``'s layer.

        Its microbatch is that of the stage's latest reload, not of ``slot``'s:
        ``reload_next`` runs once the expert MLP's backward has produced its
        input gradient, which reads some of the layer's activations but not
        necessarily all -- a weight gradient may read the rest later.
        """
        microbatch = self._backward_microbatch.get(slot.stage_index)
        if microbatch is None:
            return None
        return ScheduleEntry(False, slot.stage_index, slot.layer_id, microbatch)

    def should_skip_stash(self, entry: ScheduleEntry) -> bool:
        """Whether the next entry is ``entry``'s own backward.

        Megatron's ``current_schedule_layer != -next_schedule_layer`` test: if
        so, the activation would be written to the stash and read straight
        back with no bubble in between, so it stays resident instead
        (``remove_paged_tensor_from_stash``). Only a stage's last paged layer
        can pass it, and only on a forward its backward follows directly --
        every forward without pipeline parallelism, the last stage's in 1F1B.
        """
        if not self.skip_immediate_backward:
            return False
        own_backward = ScheduleEntry(
            False, entry.stage_index, entry.layer_id, entry.microbatch
        )
        return self.next_entries(entry) == [own_backward]

    def add_stash_record(self, slot_id: int, record: _StashRecord) -> None:
        self._num_stashed[slot_id] += 1
        self._records[slot_id].append(record)

    def take_stash_record(self, slot: StashSlot, handle: torch.Tensor) -> _StashRecord:
        """Pop the oldest stash of ``slot``, which the reload of ``handle``
        consumes."""
        queue = self._records.get(slot.slot_id)
        if not queue or queue[0].handle.data_ptr() != handle.data_ptr():
            raise RuntimeError(
                f"paged stash: the reload of slot {slot.slot_id} "
                f"({slot.node_name}) is not for its oldest stash. Every slot's "
                "stashes must be reloaded in stash order."
            )
        record = queue.popleft()
        self._backward_microbatch[slot.stage_index] = record.microbatch
        return record

    def add_stash_in_flight(
        self, event: torch.Event, retained: tuple[torch.Tensor, ...]
    ) -> None:
        """Hold what a launched copy reads until ``join_stashes``."""
        self._stashes_in_flight.append((event, retained))

    def join_stashes(self) -> None:
        """Megatron's ``wait_for_stash_to_complete``.

        Orders the compute stream after every copy launched so far and drops
        the activations they read, which frees the padded tensors. Runs at
        each paged layer's group start and backward start, whichever comes
        first, as in Megatron.
        """
        if not self._stashes_in_flight:
            return
        # The copies run in launch order on one stream, so the last one's
        # event covers them all.
        last_event = self._stashes_in_flight[-1][0]
        torch.cuda.current_stream(self.device).wait_event(last_event)
        self._stashes_in_flight.clear()

    def prefetch_after(self, entry: ScheduleEntry, count: int) -> None:
        """Start the reloads of every backward among the ``count`` schedule
        entries after ``entry``.

        Megatron's ``reload_paged_tensors(next_schedule_layer)``. The reloaded
        tensors wait in their stash records for the reload ops, which may run
        in another pipeline action's graph; a stash that stayed resident has
        nothing to reload.
        """
        if count <= 0 or not self.active:
            return
        for upcoming in self.next_entries(entry, count):
            if upcoming.forward:
                continue
            for slot in self.slots.values():
                if (slot.stage_index, slot.layer_id) != (
                    upcoming.stage_index,
                    upcoming.layer_id,
                ):
                    continue
                queue = self._records.get(slot.slot_id)
                if not queue:
                    continue
                record = queue[0]
                if (
                    record.microbatch != upcoming.microbatch
                    or record.resident is not None
                    or record.reloaded is not None
                ):
                    continue
                record.reloaded = _launch_reload(
                    slot, record.handle, record.num_tokens, record.shape, record.dtype
                )

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
            stage = footprint.setdefault(slot.stage_index, {})
            stage[slot.key] = stage.get(slot.key, 0) + _num_pages(
                tokens, self.page_size
            )
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

        self.needs_kernel_warmup = True
        self.stash_buffers = {}
        total_bytes = 0
        for key, measured_pages in self._peak_pages.items():
            dtype, hidden_size = key
            schedule_pages = self.schedule_estimate.get(key, 0)
            required_pages = max(measured_pages, schedule_pages)
            logger.debug(
                "  paged stash sizing %s/%d: measured=%d pages, schedule=%d pages",
                dtype,
                hidden_size,
                measured_pages,
                schedule_pages,
            )

            buffer = PagedStashBuffer(
                int(required_pages * self.buffer_size_factor_cuda) * self.page_size,
                hidden_size,
                self.page_size,
                self.device,
                self.overflow,
                self.host_spill,
                buffer_dtype_for(dtype),
                num_tokens_host=(
                    int(required_pages * self.buffer_size_factor_cpu) * self.page_size
                ),
            )
            self.stash_buffers[key] = buffer
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
            len(self.stash_buffers),
            total_bytes / (1024 * 1024),
            self.page_size,
            self.buffer_size_factor_cuda,
        )

    def release_stash_buffers(self) -> None:
        """Drop the large page buffers, keeping the small shared flags.

        Called on the fallback path, matching Megatron's
        ``PagedStashManager.release_stash_buffers``, and by ``reset_instance``;
        both after the CUDA graphs that reference these pointers are torn down.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stash_buffers = None
        logger.info("Paged stash: released stash page buffers.")

    def get_buffer(self, dtype: torch.dtype, hidden_size: int) -> PagedStashBuffer:
        if self.stash_buffers is None:
            raise RuntimeError("paged stash: buffers have not been allocated yet.")
        try:
            return self.stash_buffers[(dtype, hidden_size)]
        except KeyError:
            raise RuntimeError(
                f"paged stash: no buffer for dtype={dtype} hidden_size={hidden_size}. "
                f"Known: {list(self.stash_buffers)}"
            ) from None

    def reset_buffers(self) -> None:
        """Restore every freelist and clear the spill flag, allocating nothing.

        The overflow flag is sticky: an overflow is fatal, and only the
        ``blocking`` rerun clears it. The deferred
        check reads a copy taken on an earlier step, and the CPU can run several
        steps ahead of that copy; if each step cleared the flag, a later copy
        could overwrite an unread one and lose the overflow.
        """
        for buffer in self.stash_buffers.values():
            buffer.reset()
        assert self.overflow is not None and self.host_spill is not None
        self.host_spill.zero_()

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
        self._parked_tensors.clear()
        self._records.clear()
        self._num_stashed.clear()
        self._backward_microbatch.clear()
        self._stashes_in_flight.clear()
        self._live_pages = {key: 0 for key in self._live_pages}
        if self.paging_disabled:
            return

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
            self.reset_buffers()

    @property
    def active(self) -> bool:
        """True when the ops should page through real buffers."""
        return (
            self.enabled
            and not self.paging_disabled
            and self.status == "captured"
            and bool(self.stash_buffers)
        )


def _flag_is_set(flag: torch.Tensor | None) -> torch.Tensor:
    manager = PagedStashManager.get_instance()
    if not manager.enabled or flag is None:
        device = manager.device or torch.device("cuda")
        return torch.zeros(1, dtype=torch.bool, device=device)
    return flag.ne(0)


def check_paged_stash_overflow() -> torch.Tensor:
    """Device-side flag: did any stash fail to find a free page this step?"""
    return _flag_is_set(PagedStashManager.get_instance().overflow)


def check_paged_stash_host_spill() -> torch.Tensor:
    """Device-side flag: did any activation spill to pinned host memory?"""
    return _flag_is_set(PagedStashManager.get_instance().host_spill)


# ===========================================================================
# Custom ops that page MoE activations in and out of the stash buffers
# ===========================================================================
#
# These are the FX-visible form of Megatron's ``PagedTensor.offload_to_stash`` /
# ``reload_from_stash`` and of the two autograd functions that drive them around
# the grouped MLP, ``paged_stash_group_start`` and ``paged_stash_group_commit``.
# Megatron runs all of it from ``saved_tensors_hooks`` and those functions'
# forward and backward; GraphTrainer has no autograd tape at graph-execution
# time, so the same points are ops that ``apply_paged_stash_pass`` places. Per
# paged layer, the resulting IR is::
#
#     Forward:  x = paged_stash.group_start(x)    # the dispatch output, before
#                                                  # the expert MLP reads it
#               ... the expert MLP ...
#               handle = aten.empty([max_pages + 1], int64)
#               paged_stash.stash_(handle, act, num_tokens, slot, prefetch)
#
#     Backward: act = paged_stash.reload(handle, num_tokens, slot, size, dtype)
#               act = paged_stash.wait_reload(act)
#               ... the expert MLP's backward ...
#               dx = paged_stash.reload_next(dx, slot, prefetch)
#                                                  # the dispatch output's gradient
#
# ``stash_`` launches the copy of the activation's live rows on the side stream,
# at the end of the expert MLP like Megatron's group commit. ``group_start`` is
# Megatron's group start: it joins every copy still in flight -- the previous
# paged layer's, which may have run in an earlier pipeline action -- and only
# then lets the padded activations go. A backward joins them too, at its first
# ``reload``, in case no group start ran in between.
#
# Reloads run one schedule entry ahead, as Megatron's do. A forward's last stash
# and every ``reload_next`` start the reloads of the next paged layer's backward
# whenever that is what the schedule runs next: the same backward's next layer,
# or the first layer of another microbatch's or stage's backward, possibly in a
# later pipeline action. ``reload`` hands backward the prefetched tensor, or
# pops the pages itself when nothing was prefetched, and ``wait_reload`` joins
# the copy before backward reads it.
#
# ``handle`` is one small int64 tensor: ``[:max_pages]`` records which page backs
# each chunk of the activation, and the last element is the spill flag. It is the
# only thing that crosses into backward, in place of the padded activation.
#
# The stash is a *mutation* of ``handle`` rather than a value-returning op, and
# that is load-bearing rather than cosmetic. GraphPP defines its forward graph
# as the dependency closure of the forward user outputs; a value-returning
# stash reaches no forward output, so the whole chain would be classified as
# backward, the activation would still be saved, and the pass would achieve
# nothing. Expressing the stash as a mutation in forward puts it through
# GraphPP's existing ``_forward_mutations_to_materialize`` path, which runs the
# mutation in the forward callable and saves the mutated base -- the handle --
# for backward. ``group_start`` stays in forward and ``reload_next`` in
# backward the other way: each passes a value through, so forward outputs
# depend on the first and backward outputs on the second.
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
# the reload kernels run on a side stream while their *inputs* are ordinary
# graph values owned by the compute stream. FX emits a ``del`` for a value right
# after its last graph use, so without an extra reference the caching allocator
# would hand ``page_record`` / ``spilled`` / ``num_tokens`` to the next
# compute-stream allocation while the pop kernel is still reading them.
# Megatron never hits this because its ``PagedTensor`` keeps those tensors alive
# for the whole iteration; retaining them until the matching ``wait_reload``
# reproduces that lifetime. A stash's inputs are held the same way, by the
# manager, until a group start or a backward joins the copy.
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
            "Every reload must be paired with exactly one wait_reload before "
            "the next use of that buffer."
        )
    event = torch.Event()
    _wait_registry[key] = (event, device, retain + (tensor,))
    return event


def _wait_for_copy(tensor: torch.Tensor) -> None:
    """Join the compute stream to ``tensor``'s pending copy.

    Once the compute stream is ordered after the copy kernel, the inputs the
    registry retained for it may go back to the allocator. A no-op in measure
    mode, where nothing was copied.
    """
    registered = _wait_registry.pop(tensor.data_ptr(), None)
    if registered is None:
        return
    completion_event, device, _retained = registered
    torch.cuda.current_stream(device).wait_event(completion_event)


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


def _launch_stash(
    slot: StashSlot,
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
) -> None:
    """Copy ``activation``'s live rows into pages on the side stream, recording
    them in ``handle``; the manager holds what the copy reads until it is
    joined."""
    manager = PagedStashManager.get_instance()
    device = activation.device
    buffer = manager.get_buffer(slot.dtype, slot.hidden_size)
    current_stream = torch.cuda.current_stream(device)
    page_record, spilled = _handle_views(handle, handle.numel() - 1)

    # Derive the token count *before* the stream join. For a scale slot
    # ``token_divisor != 1``, so this enqueues a division on the compute
    # stream, and ``wait_stream`` only covers work already enqueued when it
    # runs -- ordering it afterwards lets the copy kernel read the count before
    # the division has written it. Data slots divide by 1 and launch nothing,
    # which is why only the MXFP8 scale slots ever raced.
    tokens = _num_tokens_arg(num_tokens, slot)

    stream = manager.stream
    stream.wait_stream(current_stream)
    completion_event = torch.Event()
    with torch.cuda.stream(stream):
        # Freelist scratch never leaves the side stream, so it must be
        # allocated here: allocating on the compute stream and then freeing it
        # at op exit would let the allocator hand the block to another
        # compute-stream tensor while the copy kernel is still reading it.
        new_free_list_head = torch.empty(2, dtype=torch.int64, device=device)
        handle.zero_()
        source = activation.contiguous().view(-1, slot.hidden_size)
        buffer.launch_copy(
            source.view(buffer.cuda_buffer.dtype),
            tokens,
            page_record,
            spilled,
            new_free_list_head,
            hidden_size=slot.hidden_size,
            num_blocks=min(slot.max_stash_tokens, _MAX_BLOCKS),
        )
        buffer.free_list_head.copy_(new_free_list_head)
        stream.record_event(completion_event)
    manager.add_stash_in_flight(completion_event, (activation, num_tokens, tokens))


def _launch_reload(
    slot: StashSlot,
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pop ``handle``'s pages into a new tensor on the side stream.

    The destination is allocated on the compute stream (so the caching
    allocator owns it there) and filled on the side stream, matching both
    Megatron's ``reload_paged_tensors`` and upstream ``ao::reload``;
    ``wait_reload`` joins the copy.
    """
    manager = PagedStashManager.get_instance()
    device = handle.device
    buffer = manager.get_buffer(slot.dtype, slot.hidden_size)
    current_stream = torch.cuda.current_stream(device)
    page_record, spilled = _handle_views(handle, handle.numel() - 1)

    result = torch.empty(size, dtype=dtype, device=device)

    # Before the join, for the reason given in ``_launch_stash``.
    tokens = _num_tokens_arg(num_tokens, slot)

    stream = manager.stream
    stream.wait_stream(current_stream)

    # ``handle`` and the token count are dead in the graph the moment the
    # reload returns, but the pop kernel may not have run yet.
    completion_event = _register_wait(
        result, device, retain=(handle, page_record, spilled, num_tokens, tokens)
    )

    with torch.cuda.stream(stream):
        # Side-stream-local scratch; see the note in ``_launch_stash``.
        new_free_list_tail = torch.empty(2, dtype=torch.int64, device=device)
        buffer.launch_pop(
            result.view(-1, slot.hidden_size).view(buffer.cuda_buffer.dtype),
            tokens,
            page_record,
            spilled,
            new_free_list_tail,
            hidden_size=slot.hidden_size,
            num_blocks=min(slot.max_stash_tokens, _MAX_BLOCKS),
        )
        buffer.free_list_tail.copy_(new_free_list_tail)
        stream.record_event(completion_event)

    return result


@custom_op("paged_stash::stash_", mutates_args=("handle",))
def stash_(
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    prefetch: int = 0,
) -> None:
    """Copy the live rows of ``activation`` into the paged stash, into ``handle``.

    ``handle`` must be an ``int64`` tensor of ``max_pages + 1`` elements. Once
    the copy completes it holds the page id backing each ``page_size``-token
    chunk, plus a trailing flag recording whether the stash landed in the
    pinned host buffer.

    The caller allocates ``handle`` and the stash fills it in place rather than
    returning it, because GraphPP keeps an op in the forward callable only if a
    forward output depends on it or it mutates a forward value. A returned
    handle would feed nothing but backward's ``reload``, so the stash would run
    in backward and the padded activation would be saved for it; see the
    section comment above.

    The copy runs on the side stream, and the manager keeps ``activation``
    alive until the next ``group_start``, or a backward's first ``reload``,
    joins it. When the schedule runs this layer's backward next, nothing is
    copied and the activation stays resident instead (see
    ``PagedStashManager.should_skip_stash``).

    ``prefetch`` is nonzero on a layer's last stash, where Megatron's group
    commit runs: once the layer's copies are launched, the reloads of every
    backward among the next ``prefetch`` schedule entries start.
    """
    manager = PagedStashManager.get_instance()
    slot = _slot(slot_id)

    if not manager.active:
        # Not paging: keep the activation alive and leave the handle zeroed so
        # the graph shape never changes between phases. The measure step also
        # reads the real token count off the device and records it for sizing,
        # as Megatron's capture iteration does; a ``blocking`` rerun, which runs
        # with paging disabled, sizes nothing.
        manager.device = activation.device
        count = None
        if manager.status == "capture":
            count = int(num_tokens.reshape(-1)[0].item()) // slot.token_divisor
            manager.record_stash(slot, count)
        manager.park_tensor(slot_id, activation, count)
        handle.zero_()
        return

    _check_slot_dims(activation, slot)
    entry = manager.stash_entry(slot)
    record = _StashRecord(
        entry.microbatch, handle, num_tokens, list(activation.shape), activation.dtype
    )
    if manager.should_skip_stash(entry):
        # Megatron's `remove_paged_tensor_from_stash`: no copy, no pages, the
        # activation just stays where it is. Zero the handle so the graph shape
        # is unchanged and nothing reads a stale page record.
        handle.zero_()
        record.resident = activation
    else:
        _launch_stash(slot, handle, activation, num_tokens)
    manager.add_stash_record(slot_id, record)
    manager.prefetch_after(entry, prefetch)


@stash_.register_fake
def _stash_fake(
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    prefetch: int = 0,
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
    """Restore a stashed activation from its pages.

    Hands back the tensor a prefetch already started filling, or pops the pages
    now if nothing was prefetched (see ``_launch_reload``); a stash that stayed
    resident hands back the activation itself.
    """
    manager = PagedStashManager.get_instance()
    slot = _slot(slot_id)

    if not manager.active:
        # Not paging: hand back the activation the stash parked, so the step is
        # numerically identical to a run without paged stash. The measure step
        # also releases its pages from the live counters; without the release
        # the "peak" would be the sum of every stash in the step rather than
        # the largest set alive at one time -- which with pipeline parallelism
        # over-sizes the buffers by the microbatch count.
        activation, count = manager.pop_parked_tensor(slot_id)
        if count is not None:
            manager.record_reload(slot, count)
        return activation

    # Megatron's backward start: copies still in flight -- the last forward's,
    # when no group start has run since -- complete first.
    manager.join_stashes()
    record = manager.take_stash_record(slot, handle)
    if record.resident is not None:
        return record.resident
    if record.reloaded is not None:
        return record.reloaded
    return _launch_reload(slot, handle, num_tokens, size, dtype)


@reload.register_fake
def _reload_fake(
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(size, dtype=dtype, device=handle.device)


# The remaining ops pass their input through: the output aliases the input,
# which ``custom_op`` forbids, so they are defined through torch.library
# directly -- the same reason ``ao::wait_tensor`` is. Passing a value through is
# what places each one: the consumers of that value run after it, and in GraphPP
# it lands in the callable whose outputs depend on that value.
_lib = torch.library.Library("paged_stash", "FRAGMENT")
_lib.define("group_start(Tensor(a) tensor) -> Tensor(a)")
_lib.define("wait_reload(Tensor(a) tensor) -> Tensor(a)")
_lib.define("reload_next(Tensor(a) tensor, int slot_id, int count) -> Tensor(a)")


@torch.library.impl("paged_stash::group_start", "CompositeExplicitAutograd")
def _paged_stash_group_start(tensor: torch.Tensor) -> torch.Tensor:
    """Megatron's ``paged_stash_group_start``: join the copies still in flight
    before the expert MLP reads ``tensor``, which frees the padded activations
    they read."""
    PagedStashManager.get_instance().join_stashes()
    return tensor


@torch.library.impl("paged_stash::wait_reload", "CompositeExplicitAutograd")
def _paged_stash_wait_reload(tensor: torch.Tensor) -> torch.Tensor:
    """Join the side stream into compute before backward reads the reload."""
    _wait_for_copy(tensor)
    return tensor


@torch.library.impl("paged_stash::reload_next", "CompositeExplicitAutograd")
def _paged_stash_reload_next(
    tensor: torch.Tensor, slot_id: int, count: int
) -> torch.Tensor:
    """Megatron's ``PipelinePreScheduleFunction.backward``: once the expert
    MLP's backward has produced ``tensor``, the gradient of its input, start the
    reloads of every backward among the next ``count`` schedule entries.
    ``slot_id`` is any of the layer's slots; it names the layer."""
    manager = PagedStashManager.get_instance()
    entry = manager.backward_entry(_slot(slot_id)) if manager.active else None
    if entry is not None:
        manager.prefetch_after(entry, count)
    return tensor


@torch.library.register_fake("paged_stash::group_start")
def _paged_stash_group_start_fake(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.library.register_fake("paged_stash::wait_reload")
def _paged_stash_wait_reload_fake(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.library.register_fake("paged_stash::reload_next")
def _paged_stash_reload_next_fake(
    tensor: torch.Tensor, slot_id: int, count: int
) -> torch.Tensor:
    return tensor


# The backward ops are side effects to FX, which keeps dead-code elimination and
# reordering off them. ``group_start`` must not be: GraphPP's backward
# extraction copies every side effect whose inputs it can compute, and backward
# can recompute the dispatch output from what forward saves, so it would run the
# dispatch's all-to-all a second time just to feed a join. The expert MLP
# reading its output is what keeps it in place.
has_side_effect(torch.ops.paged_stash.wait_reload.default)
has_side_effect(torch.ops.paged_stash.reload_next.default)


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
    for buffer in manager.stash_buffers.values():
        scratch = torch.zeros(
            (buffer.page_size, buffer.hidden_size), dtype=buffer.dtype, device=device
        )
        page_record = torch.zeros(1, dtype=torch.int64, device=device)
        spilled = torch.zeros(1, dtype=torch.int64, device=device)
        new_free_list_ends = torch.zeros(2, dtype=torch.int64, device=device)
        for launch in (buffer.launch_copy, buffer.launch_pop):
            launch(
                scratch,
                zero_tokens,
                page_record,
                spilled,
                new_free_list_ends,
                hidden_size=buffer.hidden_size,
                num_blocks=1,
            )

    # The warmup launches are no-ops, but reset anyway so the step starts from
    # a pristine freelist regardless of kernel-side bookkeeping.
    manager.reset_buffers()
    torch.cuda.synchronize(device)
    manager.needs_kernel_warmup = False
    logger.info(
        "Paged stash: compiled Triton kernels for %d buffer configuration(s).",
        len(manager.stash_buffers),
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
#
# The region is annotated on the model before tracing, the trace-time
# counterpart of Megatron's context: ``annotate_paged_stash_region`` wraps the
# region's modules with ``annotate_fn``, as ``annotate_module_fqns`` does for
# FQNs, so every node traced inside carries ``PAGED_STASH_REGION`` in
# ``node.meta["custom"]``. Opting into paged stashing hands it that region:
# ``tag_paged_stash_activations`` stashes every activation there that backward
# reads and the stash can hold, whatever SAC or CPU offload decided before it.


# node.meta key set by this pass and consumed by apply_paged_stash_pass.
PAGED_STASH_TAG = "paged_stash"

# node.meta["custom"] key marking the paged-stash region.
PAGED_STASH_REGION = "paged_stash_region"

# Default module-FQN pattern of the paged-stash region: the routed experts, so
# the region holds the dispatcher's output and everything inside
# ``inner_experts``.
DEFAULT_MODULE_FQN = "layers.*.moe.routed_experts"

# Paging a tensor costs one full round trip through the stash, so tiny tensors
# are never worth it. Same rationale as cpu_offload's ``_MIN_OFFLOAD_BYTES``.
_MIN_STASH_BYTES = 1 << 20  # 1 MiB


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Where a layer's padded and live token counts come from.

    ``padded`` is the static leading dimension of every permuted activation in
    the layer -- the capacity-factor-sized buffer the dispatcher writes into.
    ``per_expert_node`` produces the device-side per-local-expert token counts;
    summing it gives the live row count, which is what the stash actually
    copies. This pair is the GraphTrainer analogue of Megatron's
    ``(max_num_tokens, num_tokens_tensor)``. ``hidden_node`` is the dispatch
    output, the expert MLP's input.
    """

    layer_id: int
    padded: int
    per_expert_node: Node
    hidden_node: Node


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
            hidden_node=hidden_node,
        )
    return counts


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

    Scale inverses are recognized by dtype. Megatron learns that a saved tensor
    is one from the ``grouped_tensor_scale_inv`` flag Transformer Engine stamps
    on it; nothing marks ours, but e8m0 is only ever used for MXFP8 block
    scales, so the dtype carries the same information.
    """
    if val.dtype is torch.float8_e8m0fnu:
        # 32 tokens share one scale row, so the row count is padded // 32.
        if padded % SCALE_INV_BLOCK_SIZE:
            return None
        divisor, rows = SCALE_INV_BLOCK_SIZE, padded // SCALE_INV_BLOCK_SIZE
        if val.dim() > 1 and int(val.shape[0]) != rows:
            # A shaped buffer whose rows are not 32-token blocks may be stored
            # column by column, each column's live tokens strided through it
            # rather than sitting in a prefix, and paging a prefix of that
            # would restore the wrong values. A flat buffer has no rows to say
            # so and gets the token reading, as Transformer Engine's flat
            # buffers do in Megatron. ``_slot_dims`` also tries its reshapes.
            return None
    else:
        if int(val.shape[0]) != padded:
            return None
        divisor, rows = 1, padded
    if rows <= 0 or val.numel() % rows:
        return None
    return divisor, val.numel() // rows


def _whole_storage_views(node: Node) -> Iterator[Node]:
    """Views of ``node``, through any chain of views, that reshape all of it.

    Same element count, no offset and contiguous, so each describes the same
    bytes in the same order. Backward views count too: remat moves a view whose
    only readers are in backward into backward.
    """
    numel = node.meta["val"].numel()
    pending = [user for user in node.users if _is_view(user)]
    while pending:
        view = pending.pop()
        val = view.meta.get("val")
        if (
            isinstance(val, torch.Tensor)
            and val.numel() == numel
            and val.storage_offset() == 0
            and _tensor_is_contiguous(val)
        ):
            yield view
        pending.extend(user for user in view.users if _is_view(user))


def _slot_dims(node: Node, padded: int) -> tuple[int, int] | None:
    """``(token_divisor, hidden_size)`` read off ``node``'s shape, or off a view
    that reshapes all of it.

    A traced graph often holds a buffer under a shape made for some kernel's
    API rather than for its tokens. A fused MXFP8 expert MLP, for instance,
    gathers its FC1-input block scales as ``(1536, 512)`` tiles, views them as
    ``(256, 3072)`` and saves them flattened for backward: three shapes of one
    token-ordered buffer, of which only the flat one reads as token rows. Each
    view describes the same bytes, so one that reads as token rows is as good a
    witness as the buffer's own shape. That keeps the region annotation the
    only thing paging needs; nothing in the region declares its layout.
    """
    dims = infer_slot_dims(node.meta["val"], padded)
    if dims is not None:
        return dims
    for view in _whole_storage_views(node):
        dims = infer_slot_dims(view.meta["val"], padded)
        if dims is not None:
            return dims
    return None


# Views that hand their outputs out through ``getitem``. CPU offload's view
# chain stops at the ``getitem``, which would hide a backward read through one:
# a fused gate/up projection, for instance, is read in backward through the
# ``unbind`` that separates its two halves.
_MULTI_OUTPUT_VIEW_OPS = frozenset(
    {
        torch.ops.aten.unbind,
        torch.ops.aten.split,
        torch.ops.aten.split_with_sizes,
        torch.ops.aten.chunk,
        torch.ops.aten.tensor_split,
    }
)


def _is_stash_view(node: Node) -> bool:
    """``_is_view``, extended through multi-output views and their ``getitem``."""
    if _is_view(node) or _get_aten_target(node) in _MULTI_OUTPUT_VIEW_OPS:
        return True
    return (
        node.target is operator.getitem
        and isinstance(node.args[0], Node)
        and _get_aten_target(node.args[0]) in _MULTI_OUTPUT_VIEW_OPS
    )


def _is_pageable(node: Node, padded: int) -> bool:
    """Whether ``node``'s output is a permuted activation worth paging.

    Starts from what CPU offload requires of any activation it moves out of the
    way until backward -- a materialized, contiguous, non-view tensor -- and
    adds the row layout the stash addresses. A view is paged through its base.
    """
    if _is_backward_node(node) or _is_stash_view(node) or not _can_offload_node(node):
        return False
    val = node.meta["val"]
    if val.device.type != "cuda":
        return False
    if val.dim() == 0 or any(isinstance(s, torch.SymInt) for s in val.shape):
        return False
    dims = _slot_dims(node, padded)
    if dims is None:
        return False
    divisor, _ = dims
    # A block-scale buffer is 1/32 of the tensor it describes and is only ever
    # paged alongside it -- Transformer Engine marks the pair together and
    # Megatron stashes a layer's tensors as one list. Judging it on its own
    # bytes would reject it for being small when the round trip it shares is
    # already paid for, so weigh it by the data it scales.
    return _tensor_bytes(val) * divisor >= _MIN_STASH_BYTES


def annotate_paged_stash_region(
    model: nn.Module, module_fqn_pattern: str = DEFAULT_MODULE_FQN
) -> None:
    """Annotate the modules matching ``module_fqn_pattern`` as the paged-stash
    region.

    Wraps each matching module's forward with ``annotate_fn``, as
    ``annotate_module_fqns`` does, so every node traced inside it, including
    inside its submodules, carries ``PAGED_STASH_REGION`` in
    ``node.meta["custom"]``. Call before the model is traced.
    """
    for fqn, module in model.named_modules():
        if matches_module_fqn_pattern(module_fqn_pattern, fqn):
            module.forward = annotate_fn({PAGED_STASH_REGION: True})(module.forward)


def tag_paged_stash_activations(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Stash every activation in the paged-stash region that backward reads.

    Paged stashing owns the region ``annotate_paged_stash_region`` marks: each
    activation there that backward reads and the stash can hold is saved and
    paged, whatever SAC or CPU offload decided before, and nothing in it is
    recomputed. For the stock BF16 SwiGLU experts that is the dispatch output,
    the fused gate/up projection -- which backward reads through the ``unbind``
    that splits it -- the SiLU, the gate product and the down projection's
    output. Activations the stash cannot hold keep SAC's decision: the dispatch
    and combine collectives themselves, the per-row router scores, and the
    expert weights.
    """
    token_counts = find_layer_token_counts(gm)
    if not token_counts:
        logger.info(
            "Paged stash: no static-capacity token dispatch found; nothing tagged."
        )
        return gm

    tagged = 0
    tagged_bytes = 0
    per_layer: dict[int, int] = defaultdict(int)
    for node in gm.graph.nodes:
        if PAGED_STASH_REGION not in node.meta.get("custom", {}):
            continue
        counts = token_counts.get(_get_layer_id(node))
        if counts is None or not _is_pageable(node, counts.padded):
            continue
        _, has_backward = _get_storage_chain(node, is_view=_is_stash_view)
        if not has_backward:
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        node.meta[PAGED_STASH_TAG] = True
        tagged += 1
        tagged_bytes += _tensor_bytes(node.meta["val"])
        per_layer[counts.layer_id] += 1

    if tagged:
        logger.info(
            "Paged stash: tagged %d activation(s) across %d layer(s) "
            "(%.2f MiB padded)",
            tagged,
            len(per_layer),
            tagged_bytes / (1024 * 1024),
        )
        for layer_id in sorted(per_layer):
            logger.debug(
                "  layer %d: %d tagged, padded tokens %d",
                layer_id,
                per_layer[layer_id],
                token_counts[layer_id].padded,
            )
    else:
        logger.info("Paged stash: no activations in the paged-stash region")
    return gm


# ===========================================================================
# Insert paged stash / reload ops around tagged activations
# ===========================================================================
#
# This is the structural counterpart of ``cpu_offload.apply_cpu_offload_pass``,
# and of Megatron's ``PipelinePreScheduleFunction`` /
# ``PipelinePostScheduleFunction`` pair. Megatron needs those autograd functions
# because it discovers stash and reload points while the tape runs; here the
# joint graph already contains both halves, so a single pass places every op
# and rewires backward directly.
#
# Rewrite performed for each paged layer::
#
#     forward:
#         x = hybridep.dispatch(...)[0]
#         x' = paged_stash.group_start(x)
#         ... the expert MLP, reading x' ...
#         # for every activation tagged by tag_paged_stash_activations, in order:
#         handle = aten.empty([max_pages + 1], int64)
#         paged_stash.stash_(handle, act, num_tokens, slot, prefetch)
#
#     backward:
#         # for every stashed activation:
#         act' = paged_stash.reload(handle, num_tokens, slot, size, dtype)
#         act' = paged_stash.wait_reload(act')
#         ... backward consumers rewired from act to act' ...
#         dx' = paged_stash.reload_next(dx, slot, prefetch)
#         hybridep.combine_bwd(dx', ...)  # the dispatch's backward
#
# After the rewrite nothing in backward reads a padded activation, so nothing
# keeps it once the next group start joins its copy; ``handle`` and
# ``num_tokens`` -- a few kilobytes -- are what cross into backward. Under
# GraphPP that means the stage's ``saved_values_for_backward`` carries the
# handle instead of the padded tensor, which is where the memory is actually
# recovered.
#
# The points are Megatron's. A layer's stashes launch together once the expert
# MLP has produced everything it saves (``paged_stash_group_commit``), so no
# copy competes with the expert GEMMs, and the next layer's ``group_start``
# joins them (``paged_stash_group_start``), so the copies overlap the combine
# and the next layer's attention and dispatch. Each backward's reloads are
# issued at the schedule entry before it -- a forward's last stash, or the
# ``reload_next`` that ends a backward's expert MLP (the backward of Megatron's
# group start) -- and wait in that microbatch's stash records, so the entry
# before may be another pipeline action's graph.
#
# The backward half is CPU offload's: its helpers find the backward consumers
# and rewire them to the reload, replaying any view chain.


aten = torch.ops.aten
paged_stash_ops = torch.ops.paged_stash


@dataclass(frozen=True, slots=True)
class _StashPlan:
    offload: _OffloadInfo
    slot: StashSlot
    num_tokens_node: Node


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
    # This is a forward value even though backward reads it.
    node.meta.pop("autograd_backward", None)
    return node


def _plan_stashes(
    gm: torch.fx.GraphModule,
    tagged: list[Node],
    token_counts: dict[int, TokenCounts],
) -> tuple[list[_StashPlan], dict[Node, int]]:
    """Register a stash slot for every tagged node that backward reads.

    First inserts the per-layer live-token-count nodes the plans refer to, then
    returns the plans with the node index they were made against.
    """
    manager = PagedStashManager.get_instance()

    # One live-token-count node per MoE layer, shared by every stash in it.
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
    node_to_index = {n: i for i, n in enumerate(gm.graph.nodes)}

    plans: list[_StashPlan] = []
    for node in tagged:
        val = node.meta["val"]
        layer_id = _get_layer_id(node)
        padded = token_counts[layer_id].padded
        dims = _slot_dims(node, padded)
        if dims is None:
            # tag_paged_stash_activations only tags nodes with slot dims, so
            # this means the two disagree about the token count.
            raise RuntimeError(
                f"paged stash: tagged node {node.name} has no slot dims "
                f"for layer {layer_id}; tagging and rewriting disagree."
            )
        offload = _collect_offload_info(node, node_to_index, is_view=_is_stash_view)
        if offload is None:
            continue

        token_divisor, hidden_size = dims
        slot = StashSlot(
            slot_id=manager.next_slot_id(),
            dtype=val.dtype,
            hidden_size=int(hidden_size),
            max_num_tokens=padded,
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
        plans.append(
            _StashPlan(
                offload=offload,
                slot=slot,
                num_tokens_node=num_tokens_nodes[layer_id],
            )
        )
    return plans, node_to_index


def _stash_group_commits(
    plans: list[_StashPlan], node_to_index: dict[Node, int]
) -> dict[int, Node]:
    """Per layer, the latest of its stashed activations and their token counts.

    That is where the layer's stashes launch, together: once the expert MLP has
    produced everything it saves, as Megatron's ``paged_stash_group_commit``
    launches them after the grouped MLP. Launching each copy as soon as its
    activation exists would put it alongside the expert GEMMs, and a copy
    kernel occupies every SM while it runs.
    """
    commits: dict[int, Node] = {}
    for plan in plans:
        for candidate in (plan.offload.node, plan.num_tokens_node):
            current = commits.get(plan.slot.layer_id)
            if current is None or node_to_index[candidate] > node_to_index[current]:
                commits[plan.slot.layer_id] = candidate
    return commits


def _meta_from(node: Node) -> dict[str, Any]:
    """``node``'s metadata minus what describes its value or its tagging, so
    the ops inserted for it are attributed to the same module in tlparse dumps
    and kernel annotations."""
    return {
        k: v
        for k, v in node.meta.items()
        if k not in ("val", "recompute", PAGED_STASH_TAG)
    }


def _insert_stash_ops(
    gm: torch.fx.GraphModule,
    plan: _StashPlan,
    *,
    page_size: int,
    prefetch: int,
    insert_after: Node,
) -> tuple[Node, Node]:
    """Insert the forward stash and backward reload for ``plan``.

    The stash is inserted after ``insert_after``: the layer's commit (see
    ``_stash_group_commits``) or the layer's previous stash, so a layer's
    stashes launch together and in order. ``prefetch`` is the stash's
    ``prefetch`` argument, set on the layer's last one. Returns the stash node
    and the ``wait_reload`` node, placed in front of the first backward
    consumer; rewiring the consumers to it is left to the caller.
    """
    node = plan.offload.node
    val = node.meta["val"]
    max_pages = _num_pages(plan.slot.max_stash_tokens, page_size)
    src_meta = _meta_from(node)

    with gm.graph.inserting_after(insert_after):
        # One int64 tensor per stashed activation: page ids followed by the
        # spill flag. Keeping it as a single tensor gives the forward
        # mutation a single base for GraphPP to save.
        handle = gm.graph.call_function(
            aten.empty.memory_format,
            args=([max_pages + 1],),
            kwargs={
                "dtype": torch.int64,
                "device": val.device,
                "pin_memory": False,
            },
        )
        handle.meta.update(src_meta)
        handle.meta["val"] = val.new_empty((max_pages + 1,), dtype=torch.int64)
    with gm.graph.inserting_after(handle):
        stash_node = gm.graph.call_function(
            paged_stash_ops.stash_.default,
            args=(handle, node, plan.num_tokens_node, plan.slot.slot_id, prefetch),
        )
        stash_node.meta.update(src_meta)
        stash_node.meta["val"] = None

    # Backward: page the activation back in just before its first consumer.
    with gm.graph.inserting_before(plan.offload.first_bwd_consumer):
        reload_node = gm.graph.call_function(
            paged_stash_ops.reload.default,
            args=(
                handle,
                plan.num_tokens_node,
                plan.slot.slot_id,
                list(val.size()),
                val.dtype,
            ),
        )
        reload_node.meta.update(src_meta)
        reload_node.meta["val"] = val
        reload_node.meta["autograd_backward"] = True

        reload_wait = gm.graph.call_function(
            paged_stash_ops.wait_reload.default,
            args=(reload_node,),
        )
        reload_wait.meta.update(src_meta)
        reload_wait.meta["val"] = val
        reload_wait.meta["autograd_backward"] = True
    return stash_node, reload_wait


def _insert_group_start(gm: torch.fx.GraphModule, hidden: Node) -> Node | None:
    """Route the expert MLP's reads of the dispatch output ``hidden`` through a
    ``group_start`` placed just before the first of them.

    Returns the ``group_start`` node, or None if nothing in forward reads
    ``hidden``.
    """
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    readers = [
        user
        for user in hidden.users
        if not _is_backward_node(user)
        and user.target is not paged_stash_ops.stash_.default
    ]
    if not readers:
        return None
    with gm.graph.inserting_before(min(readers, key=order.__getitem__)):
        start = gm.graph.call_function(paged_stash_ops.group_start.default, (hidden,))
    start.meta.update(_meta_from(hidden))
    start.meta["val"] = hidden.meta["val"]
    for reader in readers:
        reader.replace_input_with(hidden, start)
    return start


def _dispatch_backward(dispatch: Node) -> Node | None:
    """HybridEP's backward of ``dispatch``: the ``combine_bwd`` that reads the
    gradient of the dispatch output, which is the expert MLP's input.

    Autograd gives the ops of a forward op's backward that op's ``seq_nr``.
    """
    seq_nr = dispatch.meta.get("seq_nr")
    if seq_nr is None:
        return None
    for node in dispatch.graph.nodes:
        if (
            node.target is torch.ops.hybridep.combine_bwd.default
            and _is_backward_node(node)
            and node.meta.get("seq_nr") == seq_nr
        ):
            return node
    return None


def _insert_reload_next(
    gm: torch.fx.GraphModule, hidden: Node, slot_id: int, prefetch: int
) -> Node | None:
    """Pass the gradient of the dispatch output ``hidden`` through a
    ``reload_next`` as soon as the expert MLP's backward has produced it, on its
    way to the dispatch's backward.

    Returns the ``reload_next`` node, or None if the dispatch's backward is not
    in the graph.
    """
    dispatch_backward = _dispatch_backward(hidden.args[0])
    if dispatch_backward is None or not isinstance(dispatch_backward.args[0], Node):
        return None
    grad = dispatch_backward.args[0]
    anchor = (
        gm.graph.inserting_after(grad)
        if _is_backward_node(grad)
        else gm.graph.inserting_before(dispatch_backward)
    )
    with anchor:
        node = gm.graph.call_function(
            paged_stash_ops.reload_next.default, (grad, slot_id, prefetch)
        )
    # The dispatch backward's metadata, not the gradient's: an accumulated
    # gradient is a bare ``add`` that carries no module annotation.
    node.meta.update(_meta_from(dispatch_backward))
    node.meta["val"] = grad.meta.get("val")
    dispatch_backward.replace_input_with(grad, node)
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
        prefetch_lookahead: How many paged-layer schedule entries ahead of a
            backward its reloads are issued, across pipeline actions. 1 is
            Megatron's; 0 pops each activation's pages when backward first
            reads it.

    Returns:
        The transformed GraphModule.
    """
    tagged = [n for n in gm.graph.nodes if n.meta.get(PAGED_STASH_TAG)]
    if not tagged:
        return gm

    PagedStashManager.get_instance().page_size = page_size
    token_counts = find_layer_token_counts(gm)
    plans, node_to_index = _plan_stashes(gm, tagged, token_counts)
    commits = _stash_group_commits(plans, node_to_index)
    # Each layer's last plan, whose stash ends the group.
    last_plans = {plan.slot.layer_id: plan for plan in plans}
    last_stash: dict[int, Node] = {}

    stashed_bytes = 0
    replay_count = 0
    for plan in plans:
        layer_id = plan.slot.layer_id
        commit = commits[layer_id]
        assert node_to_index[commit] < node_to_index[plan.offload.first_bwd_consumer], (
            f"paged stash: layer {layer_id} launches its stashes at "
            f"{commit.name}, after {plan.offload.node.name}'s first backward "
            f"reader {plan.offload.first_bwd_consumer.name}; the joint graph "
            "must trace forward before backward."
        )
        last_stash[layer_id], reloaded = _insert_stash_ops(
            gm,
            plan,
            page_size=page_size,
            prefetch=prefetch_lookahead if plan is last_plans[layer_id] else 0,
            insert_after=last_stash.get(layer_id, commit),
        )
        if _redirect_backward_consumers(gm, plan.offload, reloaded, node_to_index):
            replay_count += 1

        val = plan.offload.node.meta["val"]
        stashed_bytes += _tensor_bytes(val)
        logger.debug(
            "Paged stash: slot %d <- %s (%s %s, %.2f MiB padded)",
            plan.slot.slot_id,
            plan.offload.node.name,
            val.dtype,
            tuple(val.shape),
            _tensor_bytes(val) / (1024 * 1024),
        )

    unanchored = []
    for layer_id, plan in last_plans.items():
        hidden = token_counts[layer_id].hidden_node
        _insert_group_start(gm, hidden)
        if prefetch_lookahead > 0 and (
            _insert_reload_next(gm, hidden, plan.slot.slot_id, prefetch_lookahead)
            is None
        ):
            unanchored.append(layer_id)
    if unanchored:
        logger.warning(
            "Paged stash: found no dispatch backward in layer(s) %s, so the "
            "reloads that would follow their backward are issued when backward "
            "first reads them instead.",
            unanchored,
        )

    _warn_unpaged_padded_activations(gm)
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Paged stash: paged %d activation(s) (%.2f MiB padded), %d with view "
        "replay; reloads issued %d schedule entr%s ahead",
        len(plans),
        stashed_bytes / (1024 * 1024),
        replay_count,
        prefetch_lookahead,
        "y" if prefetch_lookahead == 1 else "ies",
    )
    return gm


def _warn_unpaged_padded_activations(gm: torch.fx.GraphModule) -> None:
    """Warn about capacity-padded activations in the paged-stash region that
    backward still reads once the stashes are in place.

    Tagging sees the graph before remat, and remat can add backward readers
    afterwards: a region tensor the stash could not hold is left to SAC, and if
    SAC recomputes it, backward rebuilds it from inputs that then cross the
    boundary at their full padded size. Nothing else reports that, and the
    input can be far larger than the tensor that pulled it in.
    """
    token_counts = find_layer_token_counts(gm)
    unpaged = []
    for node in gm.graph.nodes:
        if PAGED_STASH_REGION not in node.meta.get("custom", {}):
            continue
        counts = token_counts.get(_get_layer_id(node))
        if counts is None or not _is_pageable(node, counts.padded):
            continue
        _, has_backward = _get_storage_chain(node, is_view=_is_stash_view)
        if has_backward:
            unpaged.append(node)
    if unpaged:
        logger.warning(
            "Paged stash: %d capacity-padded activation(s) in the paged-stash "
            "region cross into backward unpaged (%.2f MiB padded): %s. Backward "
            "rebuilds something from them that the stash could not hold, "
            "usually a tensor whose shape does not show its token rows; "
            "storing that tensor token-major lets the stash take it instead.",
            len(unpaged),
            sum(_tensor_bytes(node.meta["val"]) for node in unpaged) / (1024 * 1024),
            ", ".join(node.name for node in unpaged),
        )


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


def paged_layer_schedule(
    actions: list[_Action], layers: dict[int, Sequence[int]]
) -> list[ScheduleEntry]:
    """Megatron's ``_pp_schedule`` for one rank: every paged layer's forward
    and backward, in the order the rank runs them.

    Megatron records it while its capture iteration runs, one entry per MoE
    layer of every microbatch's forward and backward, and decides from the
    entry after the current one whether to skip a stash (its own backward) and
    which reloads to start (a backward). A GraphPP action list already is the
    execution order, so each compute action of a stage with paged layers
    expands into those layers -- ascending for a forward, descending for a
    backward. Stages without paged layers add nothing, as dense layers add
    nothing to Megatron's; sends, receives and unshards stash nothing, and
    neither does a ``BACKWARD_WEIGHT``, which consumes a different set of saved
    values -- the reason :func:`replay_schedule_page_peak` ignores it too.

    Args:
        actions: The rank's ordered pipeline actions.
        layers: Each stage's paged layers, in forward order.

    Returns:
        The rank's schedule entries, in execution order.

    Raises:
        ValueError: If a paged stage's forward has no backward that consumes
            its saved values, e.g. because ``actions`` name their backwards
            otherwise. Sizing would then keep those stashes live to the end
            of the step, and nothing would be skipped or prefetched.
    """
    schedule: list[ScheduleEntry] = []
    for action in actions:
        for sub in _iter_compute_actions(action):
            stage_layers = layers.get(sub.stage_index)
            if not stage_layers:
                continue
            if sub.computation_type is FORWARD:
                forward, order = True, stage_layers
            elif sub.computation_type in (FULL_BACKWARD, BACKWARD_INPUT):
                forward, order = False, list(reversed(stage_layers))
            else:
                continue
            schedule.extend(
                ScheduleEntry(forward, sub.stage_index, layer, sub.microbatch_index)
                for layer in order
            )
    forwards = {(e.stage_index, e.microbatch) for e in schedule if e.forward}
    backwards = {(e.stage_index, e.microbatch) for e in schedule if not e.forward}
    if forwards != backwards:
        raise ValueError(
            "Paged stash: every forward of a paged stage needs a FULL_BACKWARD "
            "or BACKWARD_INPUT of the same microbatch in the schedule; "
            f"unmatched (stage, microbatch): {sorted(forwards ^ backwards)}"
        )
    return schedule


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


def apply_graph_pp_schedule(schedule: _PipelineScheduleRuntime) -> None:
    """Install this rank's GraphPP actions on the manager, which reads the
    paged-layer schedule off them (see :func:`paged_layer_schedule`).

    Sizing takes the max over ranks so buffers stay uniform; the schedule is the
    opposite -- it must be *this* rank's action list, because what runs after a
    forward or backward is a property of the local schedule. A rank whose
    action list cannot be read gets an empty schedule, which skips nothing and
    prefetches nothing: correct, just not optimized.
    """
    manager = PagedStashManager.get_instance()
    if not manager.enabled:
        return
    # ``pipeline_order`` is keyed by pipeline-group rank, which differs from the
    # global rank whenever another parallelism (e.g. data parallel) sits
    # alongside PP.
    pipeline_order = getattr(schedule, "pipeline_order_with_comms", None) or {}
    manager.set_pipeline_actions(list(pipeline_order.get(schedule.rank, ())))


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
    if estimate and estimate != manager.schedule_estimate:
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
    manager.schedule_estimate = estimate
    return estimate


# ===========================================================================
# Overflow detection and recovery around one forward/backward step
# ===========================================================================
#
# Port of Megatron-Core's ``PagedStashRunner``. The detection half is identical:
# the Triton kernels raise a device-side ``overflow`` flag when neither the CUDA
# nor the pinned-host freelist can satisfy a stash, and both kernels then become
# no-ops so nothing is corrupted further.
#
# **Reading the verdict.** Megatron combines the flags across ranks with one
# all-reduce and calls ``.item()`` on them at every step boundary. That is a
# device sync on a value that is almost always zero: it forfeits CPU run-ahead
# and turns each step's CPU work into a GPU idle bubble. ``deferred``, the
# default, instead copies each rank's own flags into pinned host memory
# asynchronously and reads the copy on a later step once ``Event.query()`` says
# it landed, so no step syncs. ``blocking`` keeps Megatron's all-reduce and
# synchronous read, since every rank must agree to rerun the step.
#
# **Recovering.** ``blocking`` follows Megatron's ``PagedStashRunner``: discard
# the step's gradients, drop the CUDA graph, release the pages, and rerun the
# step once with paging disabled, which cannot overflow; the next step
# reallocates buffers of the same size and pages again. Megatron disables paging
# by switching off ``moe_paged_stash``, a flag its eager forward reads. Here the
# stash ops are *nodes in the compiled graph*, so they disable it themselves:
# they park each activation instead of paging it, as in the measure step, but
# measure nothing. Megatron also reruns a step whose dispatch dropped tokens,
# with the capacity factor cleared so the dispatch sizes itself; here the
# capacity is compiled into the graph, so a drop is reported instead.
#
# Overflow means backward read activations that were never written back, so the
# step's gradients are invalid. Neither mode ever applies them:
#
# ``deferred`` (default)
#     No per-step sync. Overflow is fatal and rank-local: a rank raises when it
#     reads its own verdict, typically two steps after the overflow, and the
#     other ranks exit when the job is torn down. The overflow flag is sticky --
#     ``reset_buffers`` leaves it set -- so however far the CPU runs ahead of the
#     copies, a later one still carries it, and a blocking read at shutdown
#     catches an overflow on the final steps, which no later step reads.
#
#     The steps until the raise are already enqueued, so they are skipped on
#     device instead, the way AMP skips a step whose gradients hold an inf.
#     Every optimizer step all-reduces the flag and hands it to the optimizer as
#     ``found_inf``, which makes fused Adam and AdamW leave the parameters and
#     their state untouched; the expert biases the step's MoE load-balancing
#     hook moved are put back. All ranks skip, not just the one that
#     overflowed, because FSDP's reduce-scatter and EP's all-to-all carry its
#     gradients to every rank. A checkpoint saved meanwhile is therefore the
#     last state before the overflow, with the step count, learning-rate
#     schedule and data position advanced, as AMP's skipped steps leave them.
#     Optimizers without ``found_inf`` support cannot skip a step on device, so
#     they are only accepted in ``blocking`` mode.
#
#     TODO: a pop after an overflow copies nothing, so backward reads
#     uninitialized rows. When those make the gradient norm non-finite, the
#     trainer's non-finite guard ends the job with a device-side assert before
#     this check raises, and the user sees that error instead of the paged
#     stash one. Zeroing the skipped rows would keep the step finite until the
#     overflow error is raised.
# ``blocking``
#     The verdict is known before the step returns, so an overflowing step is
#     rerun with paging disabled. Recovers in process and works with any
#     optimizer; costs one device sync per step.


OverflowCheckMode = Literal["deferred", "blocking"]


@dataclass(frozen=True, slots=True)
class OverbudgetProbe:
    """A token dispatcher's device-side "dropped tokens this step" flag.

    Reported with the verdict as the ``overbudget`` slot of Megatron's
    ``check_moe_overflow``, which Megatron fills from
    ``token_dispatcher.check_over_budget()``. ``reset`` clears the flag at the
    start of every attempt, so each verdict covers exactly one step.
    """

    read: Callable[[], torch.Tensor | None]
    reset: Callable[[], None]


def skips_step_on_found_inf(optimizer: torch.optim.Optimizer) -> bool:
    """Whether ``optimizer.step()`` leaves the parameters and their state
    untouched when AMP's ``found_inf`` attribute is 1, as ``GradScaler``
    relies on: true for fused Adam and AdamW.

    Checked per param group because torchtitan sets ``fused`` there, which
    leaves ``_step_supports_amp_scaling`` -- set only for a constructor-level
    ``fused=True`` -- unset.
    """
    return isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)) and all(
        group["fused"] for group in optimizer.param_groups
    )


def moe_expert_biases(model_parts: Sequence[nn.Module]) -> list[torch.Tensor]:
    """The MoE expert biases, as local tensors that write through to them.

    Besides the parameters, the optimizer step moves these:
    ``register_moe_load_balancing_hook`` adds to them in a step pre-hook. They
    are found the way that hook finds them, MTP decoder blocks included.
    """
    biases = []
    for model_part in model_parts:
        layer_containers = [model_part.get_submodule("layers")]
        mtp_layers = getattr(model_part, "mtp_layers", None)
        if mtp_layers is not None:
            layer_containers.append(mtp_layers)
        for layers in layer_containers:
            for block in layers.children():
                if not getattr(block, "moe_enabled", False):
                    continue
                bias = block.moe.expert_bias_E
                if bias is not None:
                    # A buffer, so no gradient flows through ``to_local``.
                    biases.append(
                        bias.to_local() if isinstance(bias, DTensor) else bias
                    )
    return biases


class PagedStashRunner:
    """Run one forward/backward step with paged-stash overflow recovery.

    Sits exactly where Megatron's ``PagedStashRunner`` sits: wrapped around the
    call that runs a whole schedule of microbatches, so a single reset and a
    single overflow verdict cover the microbatches that share the stash buffers.

    Args:
        optimizers: The trainer's optimizers. In ``deferred`` mode the runner
            hooks their step to skip it while a stash overflow is flagged, so
            every one of them must be able to skip a step on device (see
            ``skips_step_on_found_inf``). ``blocking`` mode reruns an
            overflowing step instead and leaves them alone.
        expert_biases: Tensors the optimizer step updates besides the
            parameters (see ``moe_expert_biases``). A skipped step puts them
            back.
        overbudget_probes: Token-drop flags of the model's dispatchers, reported
            with the verdict.
        max_attempts: Attempts per step in ``blocking`` mode, including the
            first. Megatron asserts on a third attempt; the same bound applies
            here.
        process_group: Group for the flag all-reduce. Defaults to WORLD, which
            is what Megatron uses.
        overflow_check: ``deferred`` for the non-blocking pinned-mirror read,
            which skips the optimizer steps after an overflow and fails the job;
            ``blocking`` for Megatron's synchronous per-step read, which reruns
            an overflowing step.
    """

    def __init__(
        self,
        optimizers: OptimizersContainer,
        *,
        expert_biases: Sequence[torch.Tensor] = (),
        overbudget_probes: Sequence[OverbudgetProbe] = (),
        max_attempts: int = 2,
        process_group: torch.distributed.ProcessGroup | None = None,
        overflow_check: OverflowCheckMode = "deferred",
    ) -> None:
        if overflow_check not in ("deferred", "blocking"):
            raise ValueError(
                "paged stash: overflow_check must be 'deferred' or 'blocking', "
                f"got {overflow_check!r}."
            )
        self.manager = PagedStashManager.get_instance()
        self.overbudget_probes = tuple(overbudget_probes)
        self.max_attempts = max_attempts
        self.process_group = process_group
        self.overflow_check = overflow_check
        self._optimizers = optimizers
        if overflow_check == "deferred":
            self._skip_overflowed_steps(optimizers, expert_biases)

        # Deferred-read state. Allocated up front because ``cudaHostAlloc``
        # synchronizes: doing it lazily on the first captured step would put one
        # surprise sync in the training path, which is exactly what this mode
        # exists to avoid. ``_pending_step`` is the step the in-flight copy
        # covers, which is what the overflow error names.
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
        for probe in self.overbudget_probes:
            flag = probe.read()
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

    def _enqueue_probe(self) -> None:
        """Start an asynchronous read of this rank's flags. Never blocks.

        The flags are not reduced across ranks, so the raise they lead to is
        rank-local: only a rank that overflowed raises.

        The spill flag is cleared by the next ``begin_step``, which is enqueued
        after this copy on the same stream, so the mirror sees this step's
        spills. The overflow flag is sticky (see ``reset_buffers``), so
        overwriting a copy that was never read loses nothing: this copy still
        carries it.
        """
        self._flags_host.copy_(self._fill_local_flags(), non_blocking=True)
        self._flags_event.record()
        self._pending_step = self.manager.step

    def _take_verdict(
        self, *, blocking: bool
    ) -> tuple[int, tuple[int, int, int]] | None:
        """Consume the in-flight verdict if it has landed.

        Returns this rank's ``(step, (overflow, overbudget, host_spill))`` or
        ``None`` when nothing is pending, or when the copy is still in flight
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
    # Skipping overflowed optimizer steps (``deferred`` mode)
    # ------------------------------------------------------------------
    def _skip_overflowed_steps(
        self,
        optimizers: OptimizersContainer,
        expert_biases: Sequence[torch.Tensor],
    ) -> None:
        """Hook ``optimizers`` so that, while any rank's overflow flag is set,
        every rank skips its step on device and puts back ``expert_biases``."""
        unsupported = set()
        for optimizer in optimizers:
            if not skips_step_on_found_inf(optimizer):
                name = type(optimizer).__name__
                adam = isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
                unsupported.add(f"non-fused {name}" if adam else name)
        if unsupported:
            raise ValueError(
                "Paged stash's default overflow check "
                "(compile.paged_stash.overflow_check = 'deferred') learns of an "
                "overflow a step or two after it happens and skips the "
                "optimizer steps in between on the GPU, which only fused Adam "
                "and AdamW can do (fused=True, their default). This job uses: "
                f"{', '.join(sorted(unsupported))}. For now, other optimizers "
                "are only supported with Megatron-style blocking overflow "
                "checks, which rerun an overflowing step instead: "
                "compile.paged_stash.overflow_check = 'blocking'"
            )
        self._expert_biases = list(expert_biases)
        self._expert_bias_snapshots = [bias.clone() for bias in self._expert_biases]
        # Exactly 0.0 or 1.0: the fused kernels skip only on ``found_inf == 1``
        # and subtract it from the step count to undo the step's increment.
        self._skip_step = torch.zeros(
            (), dtype=torch.float32, device=self.manager.device
        )
        optimizers.register_step_pre_hook(self._flag_overflowed_step)
        optimizers.register_step_post_hook(self._undo_overflowed_step)

    def _snapshot_expert_biases(self) -> None:
        """Record the expert biases the next optimizer step starts from. They
        only change in that step, so after any forward/backward will do."""
        if self._expert_biases:
            torch._foreach_copy_(self._expert_bias_snapshots, self._expert_biases)

    def _flag_overflowed_step(
        self, optimizers: OptimizersContainer, args: Any, kwargs: Any
    ) -> None:
        """Step pre-hook: hand every optimizer whether any rank has overflowed,
        as AMP's ``found_inf``. All-reduced on device, so nothing syncs."""
        overflow = self.manager.overflow
        if overflow is None:  # nothing has been paged yet
            self._skip_step.zero_()
        else:
            self._skip_step.copy_(overflow.view(()))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # MAX rather than the verdict's SUM, to keep the flag 0 or 1.
            torch.distributed.all_reduce(
                self._skip_step,
                op=torch.distributed.ReduceOp.MAX,
                group=self.process_group,
            )
        for optimizer in optimizers:
            optimizer.found_inf = self._skip_step

    def _undo_overflowed_step(
        self, optimizers: OptimizersContainer, args: Any, kwargs: Any
    ) -> None:
        """Step post-hook: drop ``found_inf``, as ``GradScaler`` does, and on a
        skipped step put back the expert biases the MoE load-balancing
        pre-hook moved."""
        for optimizer in optimizers:
            del optimizer.found_inf
        if self._expert_biases:
            skipped = self._skip_step.bool()
            for bias, snapshot in zip(
                self._expert_biases, self._expert_bias_snapshots, strict=True
            ):
                torch.where(skipped, snapshot, bias, out=bias)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def _prepare_for_rerun(self) -> None:
        """Megatron's ``prepare_for_rerun``: the next attempt runs with paging
        disabled, which cannot overflow, and the step after it reallocates
        buffers of the same size and pages again."""
        logger.info(
            "Paged stash: rerunning forward-backward with paged stash disabled."
        )
        self.manager.overflow.zero_()
        self.manager.host_spill.zero_()
        # Discard the overflowed attempt's gradients. The runner wraps all of
        # the step's forward-backward, and the engine sets every gradient to
        # None just before it, so None is exactly the state the attempt started
        # from. Holding the attempt's gradients back instead would not be
        # enough: under CUDA graphs a parameter's ``grad`` can be the graph's
        # own output buffer, which a replay overwrites in place.
        self._optimizers.zero_grad(set_to_none=True)
        # Drop the CUDA graphs before releasing the pages they captured
        # pointers to, as Megatron's ``reset_cuda_graph`` precedes
        # ``release_stash_buffers``. The rerun runs eagerly, and the next step
        # captures again over the reallocated buffers.
        reset_cuda_graphs()
        self.manager.release_stash_buffers()
        self.manager.paging_disabled = True

    # ------------------------------------------------------------------
    # Verdict handling
    # ------------------------------------------------------------------
    @staticmethod
    def _log_diagnostics(overbudget: int, host_spill: int, *, local: bool) -> None:
        """Report the two verdict flags that do not invalidate the step: this
        rank's own when ``local``, otherwise counts of ranks."""

        def where(count: int) -> str:
            return "this rank" if local else f"{count} rank(s)"

        if overbudget > 0:
            logger.info(
                "Paged stash: token drop during MoE token dispatch (over budget) on "
                "%s. Consider increasing the token dispatcher capacity factor.",
                where(overbudget),
            )
        if host_spill > 0:
            logger.info(
                "Paged stash: spilled activations to pinned host on %s (CUDA "
                "stash full). Consider increasing "
                "compile.paged_stash.buffer_size_factor_cuda for potentially "
                "better performance.",
                where(host_spill),
            )

    def _handle_deferred_verdict(
        self, verdict: tuple[int, tuple[int, int, int]] | None
    ) -> None:
        """Act on this rank's verdict for an already-completed step: report the
        diagnostics, and raise on an overflow.

        The raise is rank-local: only a rank whose stash overflowed raises. The
        others block in their next collective until the process group times out
        or torchrun tears the job down.
        """
        if verdict is None:
            return
        step, (overflow, overbudget, host_spill) = verdict
        self._log_diagnostics(overbudget, host_spill, local=True)
        if overflow:
            raise RuntimeError(
                f"Paged stash buffer overflow detected on this rank by step "
                f"{step}. The CUDA paged stash buffer is too small for the "
                "current routing pattern.\n\n"
                "Every rank skipped its optimizer steps from the overflow on, so "
                "the parameters, optimizer state and MoE expert biases, including "
                "those of any checkpoint saved since, are from before it. As "
                "with AMP's skipped steps, the step count, learning rate "
                "schedule and data position still advanced.\n\n"
                "To fix, try one of:\n"
                "  1. Enable host spillover: "
                "compile.paged_stash.buffer_size_factor_cpu = 1.0\n"
                "  2. Increase buffer size: "
                "compile.paged_stash.buffer_size_factor_cuda = 2.0\n"
                "  3. Rerun overflowing steps with paging disabled, as Megatron "
                "does: compile.paged_stash.overflow_check = 'blocking'\n"
                "  4. Disable paged stash: compile.memory_policy = 'default'"
            )

    def final_overflow_check(self) -> None:
        """Drain the last in-flight verdict at shutdown, blocking.

        The deferred check needs a later step to observe an overflow, so an
        overflow on the final steps would otherwise escape and the job would
        exit cleanly with those steps silently skipped. One synchronizing read
        at shutdown closes that window at zero per-step cost.
        """
        if not self.manager.enabled or self.overflow_check == "blocking":
            return
        self._handle_deferred_verdict(self._take_verdict(blocking=True))

    # ------------------------------------------------------------------
    # Step entry points
    # ------------------------------------------------------------------
    def __call__(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run ``fwd_bwd_fn`` under the configured overflow policy."""
        if not self.manager.enabled:
            return fwd_bwd_fn()
        if self.overflow_check == "blocking":
            return self._call_blocking(fwd_bwd_fn)
        return self._call_deferred(fwd_bwd_fn)

    def _run_attempt(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Reset the stash and the over-budget probes, then run ``fwd_bwd_fn``."""
        self.manager.begin_step()
        for probe in self.overbudget_probes:
            probe.reset()
        # Triton compiles on first launch, which a CUDA graph capture cannot
        # tolerate; prime the cache while we are still eager.
        warmup_paged_stash_kernels()
        return fwd_bwd_fn()

    def _call_deferred(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        # Read an earlier step's verdict if its copy has landed; never blocks.
        self._handle_deferred_verdict(self._take_verdict(blocking=False))
        result = self._run_attempt(fwd_bwd_fn)
        if self.manager.active:
            self._enqueue_probe()
        self._snapshot_expert_biases()
        return result

    def _call_blocking(self, fwd_bwd_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Megatron's ``PagedStashRunner.__call__``: read the verdict at the end
        of every attempt, and rerun an overflowing step once with paging
        disabled."""
        attempts = 0
        while True:
            attempts += 1
            assert attempts <= self.max_attempts, (
                f"PagedStashRunner: {attempts} attempts exceeded the maximum "
                f"of {self.max_attempts}."
            )
            result = self._run_attempt(fwd_bwd_fn)
            # Nothing was paged in the measure step or a rerun, so there is no
            # verdict to reach.
            overflow_ranks = 0
            if self.manager.active:
                (
                    overflow_ranks,
                    overbudget_ranks,
                    host_spill_ranks,
                ) = self.check_moe_overflow()
                # Megatron also reruns a step whose dispatch dropped tokens,
                # with the capacity factor cleared. Here the capacity is
                # compiled into the graph, so a rerun would drop the same
                # tokens again; the drop is reported instead.
                self._log_diagnostics(overbudget_ranks, host_spill_ranks, local=False)
            if overflow_ranks == 0:
                # Page again from the next step, as Megatron restores
                # moe_paged_stash after a successful rerun.
                self.manager.paging_disabled = False
                return result

            logger.info(
                "Paged stash: stashing buffer overflow on %d rank(s). "
                "Consider increasing compile.paged_stash.buffer_size_factor_cuda "
                "or compile.paged_stash.buffer_size_factor_cpu.",
                overflow_ranks,
            )
            self._prepare_for_rerun()

    def apply_pp_schedule(self, schedule: _PipelineScheduleRuntime) -> None:
        """Feed the GraphPP schedule into buffer sizing and into the
        paged-layer schedule that skips stashes and prefetches reloads.

        This is the GraphTrainer replacement for Megatron's virtual-pipeline
        bookkeeping: instead of recording the schedule while it runs, read it
        off the runtime PP schedule. Called before every step, so the first
        allocation after the measure step already accounts for it.

        ``schedule`` is ``GraphRuntime.pipeline_liveness_schedule``: the
        stash lifetimes are read in PyTorch's action types, and the schedule
        GraphRuntime executes renames its backward actions.
        """
        apply_graph_pp_sizing(schedule)
        apply_graph_pp_schedule(schedule)


# ===========================================================================
# Memory policy registration and trainer setup
# ===========================================================================


@register_memory_policy("sac_and_paged_stash")
def _sac_and_paged_stash_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: GraphTrainer.Config,
) -> torch.fx.GraphModule:
    """SAC + MoE paged stash: default SAC everywhere, except that the paged
    stash decides the activations it can hold in its annotated region."""
    _default_memory_policy_pass(gm, config=config)
    tag_paged_stash_activations(gm)
    return gm


def build_paged_stash_runner(
    paged_stash_config: PagedStashConfig,
    *,
    optimizers: OptimizersContainer,
    model_parts: Sequence[nn.Module],
    device: torch.device,
    pp_enabled: bool,
) -> PagedStashRunner:
    """Enable MoE paged stashing and return its step runner.

    Called only under ``compile.memory_policy = "sac_and_paged_stash"``. Buffers
    are not created here: their size comes from the first measured step. What
    is created up front is everything that must outlive a step -- the manager
    and its sizing configuration -- matching Megatron, which also defers
    ``allocate_stash_buffers`` to the end of its capture iteration.
    """
    # Annotate the region only when the feature is on, so a run without paged
    # stashing traces exactly the graph it did before. It must happen before
    # the first trace, which happens on the first step.
    for model in model_parts:
        annotate_paged_stash_region(model, paged_stash_config.module_fqn)

    # Token drops in the HybridEP dispatcher fold into the same verdict as
    # stash overflow -- the ``overbudget`` slot of Megatron's check_moe_overflow.
    from torchtitan.distributed.deepep.hybridep import (
        check_hybridep_over_budget,
        reset_hybridep_over_budget,
    )

    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = device
    manager.page_size = paged_stash_config.page_size
    manager.buffer_size_factor_cuda = paged_stash_config.buffer_size_factor_cuda
    manager.buffer_size_factor_cpu = paged_stash_config.buffer_size_factor_cpu
    manager.skip_immediate_backward = paged_stash_config.skip_immediate_backward
    # Without pipeline parallelism the schedule is the joint graph's own --
    # every forward, then every backward -- until the GraphPP one is installed.
    manager.set_pipeline_actions([] if pp_enabled else None)
    logger.info(
        "Paged stash enabled: page_size=%d, factor_cuda=%.2f, factor_cpu=%.2f, "
        "overflow_check=%s",
        manager.page_size,
        manager.buffer_size_factor_cuda,
        manager.buffer_size_factor_cpu,
        paged_stash_config.overflow_check,
    )
    return PagedStashRunner(
        optimizers,
        expert_biases=moe_expert_biases(model_parts),
        overbudget_probes=[
            OverbudgetProbe(check_hybridep_over_budget, reset_hybridep_over_budget)
        ],
        overflow_check=paged_stash_config.overflow_check,
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
    "reload",
    "ScheduleEntry",
    "stash_",
    "warmup_paged_stash_kernels",
    "DEFAULT_MODULE_FQN",
    "PAGED_STASH_REGION",
    "PAGED_STASH_TAG",
    "TokenCounts",
    "annotate_paged_stash_region",
    "find_layer_token_counts",
    "tag_paged_stash_activations",
    "apply_paged_stash_pass",
    "apply_graph_pp_schedule",
    "apply_graph_pp_sizing",
    "paged_layer_schedule",
    "max_live_stash_generations",
    "replay_schedule_page_peak",
    "schedule_page_peak",
    "OverflowCheckMode",
    "PagedStashRunner",
    "OverbudgetProbe",
    "moe_expert_biases",
    "skips_step_on_found_inf",
    "build_paged_stash_runner",
]
