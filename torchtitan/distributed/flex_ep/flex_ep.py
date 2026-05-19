# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Router helpers for PyTorch ``flex_ep`` MoE execution."""

from __future__ import annotations

import importlib

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch.distributed.tensor import DeviceMesh

DEFAULT_NUM_CTAS = 1024
MAX_TLX_NUM_CTAS = 152
TLX_NUM_STAGES = 8
EP_TIMEOUT_SECONDS = 30.0
TOKEN_ALIGNMENT = 128
logger = logging.getLogger(__name__)

_REQUIRED_EP_BACKEND_OPS = (
    "flex_ep_allgather",
    "flex_ep_router_compute_all_expert_offsets",
    "flex_ep_router_compute_dest_offsets",
    "flex_ep_router_dispatch",
    "flex_ep_router_combine",
    "flex_ep_barrier_arrive",
    "flex_ep_barrier_wait",
    "flex_ep_swiglu_forward",
    "flex_ep_swiglu_backward",
    "flex_ep_swiglu_forward_with_offsets",
    "flex_ep_swiglu_backward_with_offsets",
    "flex_ep_clone_valid_prefix",
    "flex_ep_weighted_sum_forward",
    "flex_ep_weighted_sum_backward",
    "flex_ep_zfill_ranges_inplace",
)


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _align_up_tensor(x: torch.Tensor, alignment: int) -> torch.Tensor:
    return ((x + alignment - 1) // alignment) * alignment


def _compute_all_expert_offsets_reference(
    all_expert_counts: torch.Tensor,
    ep_rank: int,
    local_experts: int,
    token_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ep_size = all_expert_counts.shape[0]
    offsets = all_expert_counts.new_empty(
        (ep_size, local_experts, ep_size + 1),
        dtype=torch.int32,
    )
    expert_start = all_expert_counts.new_empty(
        (ep_size, local_experts + 1),
        dtype=torch.int32,
    )
    grand_total = all_expert_counts.sum(0).new_empty((ep_size,), dtype=torch.int64)
    for dest in range(ep_size):
        counts = all_expert_counts[
            :,
            dest * local_experts : (dest + 1) * local_experts,
        ]
        total_per_expert = counts.sum(0)
        grand_total[dest] = total_per_expert.sum()
        aligned = _align_up_tensor(total_per_expert, token_alignment).to(torch.int32)
        starts = torch.cat(
            (
                torch.zeros(1, device=all_expert_counts.device, dtype=torch.int32),
                aligned.cumsum(0),
            )
        )
        expert_start[dest].copy_(starts)
        offsets[dest, :, 0].copy_(starts[:-1])
        offsets[dest, :, 1:].copy_(
            starts[:-1].unsqueeze(1) + counts.cumsum(0, dtype=torch.int32).T
        )
    return offsets, grand_total[ep_rank].clone(), expert_start[ep_rank].clone()


def _compute_dest_offsets_reference(
    topk_idx: torch.Tensor,
    recv_ofs: torch.Tensor,
    ep_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, top_k = topk_idx.shape
    num_experts = recv_ofs.shape[0]
    local_experts = num_experts // ep_size
    flat_idx = topk_idx.reshape(-1)
    dest_ranks = torch.div(flat_idx, local_experts, rounding_mode="floor").to(
        torch.int32
    )
    dest_offsets = torch.empty(
        flat_idx.shape,
        device=topk_idx.device,
        dtype=torch.int64,
    )
    for expert in range(num_experts):
        is_mine = flat_idx == expert
        rank_in_expert = is_mine.to(torch.int64).cumsum(0) - 1
        dest_offsets = torch.where(
            is_mine,
            recv_ofs[expert].to(torch.int64) + rank_in_expert,
            dest_offsets,
        )
    return dest_ranks.view(batch, top_k), dest_offsets.view(batch, top_k)


def _ensure_flex_ep_imported() -> None:
    try:
        import torch._higher_order_ops.flex_ep  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "FlexGroupedExperts requires a PyTorch build with "
            "torch._higher_order_ops.flex_ep."
        ) from e


def _missing_ep_backend_ops() -> list[str]:
    return [
        op_name
        for op_name in _REQUIRED_EP_BACKEND_OPS
        if not hasattr(torch.ops.inductor, op_name)
    ]


def _register_ep_backend_ops() -> None:
    if not _missing_ep_backend_ops():
        return

    try:
        import torch.distributed._symmetric_memory._shmem_triton as shmem_triton
        import triton
        import triton.language as tl

        from torch.distributed._symmetric_memory._shmem_triton import requires_shmem
    except ImportError as e:
        raise ImportError(
            "FlexGroupedExperts EP>1 requires Triton symmetric-memory kernels."
        ) from e

    try:
        tlx = cast(Any, importlib.import_module("triton.language.extra.tlx"))
    except ImportError:
        tlx = None

    shmem_backend = shmem_triton.get_shmem_backend_module()

    @triton.jit
    def _barrier_arrive_kernel(flag_ptr, out_ptr):
        old = tl.atomic_add(
            flag_ptr,
            tl.full([], 1, dtype=tl.int32),
            sem="release",
            scope="sys",
        )
        tl.store(out_ptr, old + 1)

    @triton.jit
    def _barrier_wait_kernel(
        cuda_ptrs_ptr,
        expected_ptr,
        n_flags,
        offs_flag,
        timeout_ns,
        BLOCK: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_flags
        expected = tl.load(expected_ptr)
        base_ptrs = tl.load(cuda_ptrs_ptr + offs, mask=mask, other=0)
        flag_ptrs = (base_ptrs + offs_flag).to(tl.pointer_type(tl.int32))
        zero = tl.zeros([BLOCK], dtype=tl.int32)
        start = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            "=l",
            [],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
        done = tl.zeros([], dtype=tl.int32)
        while done == 0:
            vals = tl.atomic_add(
                flag_ptrs,
                zero,
                mask=mask,
                sem="acquire",
                scope="sys",
            )
            waiting = tl.sum(tl.where(mask & (vals < expected), 1, 0))
            if waiting == 0:
                done = tl.full([], 1, dtype=tl.int32)
            else:
                now = tl.inline_asm_elementwise(
                    "mov.u64 $0, %globaltimer;",
                    "=l",
                    [],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
                if (now - start) > timeout_ns:  # pyrefly: ignore[unsupported-operation]
                    tl.device_print("flex_ep barrier_wait timed out")
                    tl.inline_asm_elementwise(
                        "trap;",
                        "=r",
                        [],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    done = tl.full([], 1, dtype=tl.int32)

    @requires_shmem
    @triton.jit
    def _ep_allgather_kernel(
        input_ptr,
        buffers_cuda_ptrs_ptr,
        offs_output,
        ep_rank,
        input_num_bytes: tl.constexpr,
    ):
        peer = tl.program_id(0)
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        dst_ptr = (base_ptr + offs_output + ep_rank * input_num_bytes).to(
            tl.pointer_type(tl.uint8)
        )
        shmem_backend.put(dst_ptr, input_ptr, input_num_bytes, peer)
        shmem_backend.quiet()

    @triton.jit
    def _compute_expert_offsets_kernel(
        counts_ptr,
        offsets_ptr,
        expert_start_ptr,
        grand_total_ptr,
        LOCAL_EXPERTS: tl.constexpr,
        EP_SIZE: tl.constexpr,
        TOKEN_ALIGNMENT: tl.constexpr,
    ):
        dest = tl.program_id(0)
        num_experts = EP_SIZE * LOCAL_EXPERTS

        src_idx = tl.arange(0, EP_SIZE)
        expert_idx = tl.arange(0, LOCAL_EXPERTS)
        counts = tl.load(
            counts_ptr
            + src_idx[:, None] * num_experts
            + dest * LOCAL_EXPERTS
            + expert_idx[None, :]
        )

        total_per_expert = tl.sum(counts, axis=0)
        tl.store(grand_total_ptr + dest, tl.sum(total_per_expert))

        aligned = (
            (total_per_expert + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT
        ) * TOKEN_ALIGNMENT
        aligned_inclusive = tl.cumsum(aligned, axis=0)
        aligned_exclusive = aligned_inclusive - aligned

        tl.store(
            expert_start_ptr + dest * (LOCAL_EXPERTS + 1) + expert_idx,
            aligned_exclusive.to(tl.int32),
        )
        tl.store(
            expert_start_ptr + dest * (LOCAL_EXPERTS + 1) + LOCAL_EXPERTS,
            tl.sum(aligned).to(tl.int32),
        )

        counts_inclusive = tl.cumsum(counts, axis=0)
        tl.store(
            offsets_ptr
            + dest * (LOCAL_EXPERTS * (EP_SIZE + 1))
            + expert_idx * (EP_SIZE + 1),
            aligned_exclusive.to(tl.int32),
        )
        tl.store(
            offsets_ptr
            + dest * (LOCAL_EXPERTS * (EP_SIZE + 1))
            + expert_idx[None, :] * (EP_SIZE + 1)
            + (src_idx[:, None] + 1),
            (aligned_exclusive[None, :] + counts_inclusive).to(tl.int32),
        )

    @triton.jit
    def _compute_dest_offsets_kernel(
        topk_idx_ptr,
        recv_ofs_ptr,
        recv_ofs_stride,
        dest_offsets_ptr,
        batch,
        TOPK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        expert = tl.program_id(0)
        flat_offsets = tl.arange(0, BLOCK)
        total = batch * TOPK
        base = tl.load(recv_ofs_ptr + expert * recv_ofs_stride).to(tl.int64)
        count = tl.zeros([], dtype=tl.int64)

        for flat_begin in range(0, total, BLOCK):
            flat_idx = flat_begin + flat_offsets
            valid = flat_idx < total
            routed_expert = tl.load(topk_idx_ptr + flat_idx, mask=valid, other=-1)
            is_mine = (routed_expert == expert) & valid
            match = is_mine.to(tl.int64)  # pyrefly: ignore[missing-attribute]
            inclusive = tl.cumsum(match, axis=0)
            exclusive = inclusive - match

            tl.store(
                dest_offsets_ptr + flat_idx,
                base + count + exclusive,
                mask=is_mine,
            )
            count += tl.sum(match)

    @triton.jit
    def _swiglu_forward_kernel(
        y1_ptr,
        y2_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        col = offs % hidden_dim
        row = offs // hidden_dim
        gate = tl.load(y1_ptr + row * hidden_dim * 2 + col, mask=mask).to(tl.float32)
        up = tl.load(
            y1_ptr + row * hidden_dim * 2 + hidden_dim + col,
            mask=mask,
        ).to(tl.float32)
        y2 = gate * tl.sigmoid(gate) * up
        tl.store(y2_ptr + offs, y2, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        dy2_ptr,
        y1_ptr,
        dy1_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        col = offs % hidden_dim
        row = offs // hidden_dim
        y1_base = row * hidden_dim * 2
        gate = tl.load(y1_ptr + y1_base + col, mask=mask).to(tl.float32)
        up = tl.load(y1_ptr + y1_base + hidden_dim + col, mask=mask).to(tl.float32)
        dy2 = tl.load(dy2_ptr + offs, mask=mask).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        dgate = dy2 * up * sig * (1.0 + gate * (1.0 - sig))
        dup = dy2 * silu_gate
        tl.store(dy1_ptr + y1_base + col, dgate, mask=mask)
        tl.store(dy1_ptr + y1_base + hidden_dim + col, dup, mask=mask)

    @triton.jit
    def _swiglu_forward_with_offsets_kernel(
        y1_ptr,
        token_end_ptr,
        y2_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token_end = tl.load(token_end_ptr)
        mask = (offs < total) & (offs < token_end * hidden_dim)
        col = offs % hidden_dim
        row = offs // hidden_dim
        gate = tl.load(y1_ptr + row * hidden_dim * 2 + col, mask=mask).to(tl.float32)
        up = tl.load(
            y1_ptr + row * hidden_dim * 2 + hidden_dim + col,
            mask=mask,
        ).to(tl.float32)
        y2 = gate * tl.sigmoid(gate) * up
        tl.store(y2_ptr + offs, y2, mask=mask)

    @triton.jit
    def _swiglu_backward_with_offsets_kernel(
        dy2_ptr,
        y1_ptr,
        token_end_ptr,
        dy1_ptr,
        total,
        hidden_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token_end = tl.load(token_end_ptr)
        mask = (offs < total) & (offs < token_end * hidden_dim)
        col = offs % hidden_dim
        row = offs // hidden_dim
        y1_base = row * hidden_dim * 2
        gate = tl.load(y1_ptr + y1_base + col, mask=mask).to(tl.float32)
        up = tl.load(y1_ptr + y1_base + hidden_dim + col, mask=mask).to(tl.float32)
        dy2 = tl.load(dy2_ptr + offs, mask=mask).to(tl.float32)
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        dgate = dy2 * up * sig * (1.0 + gate * (1.0 - sig))
        dup = dy2 * silu_gate
        tl.store(dy1_ptr + y1_base + col, dgate, mask=mask)
        tl.store(dy1_ptr + y1_base + hidden_dim + col, dup, mask=mask)

    @triton.jit
    def _clone_valid_prefix_kernel(
        input_ptr,
        token_end_ptr,
        out_ptr,
        total,
        row_width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        token_end = tl.load(token_end_ptr)
        mask = (offs < total) & (offs < token_end * row_width)
        values = tl.load(input_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, values, mask=mask)

    @triton.jit
    def _weighted_sum_forward_kernel(
        y_partial_ptr,
        scores_ptr,
        out_ptr,
        batch,
        dim: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        dim_block = tl.program_id(1)
        offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (token < batch) & (offs_d < dim)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for expert_slot in tl.static_range(0, TOPK):
            y = tl.load(
                y_partial_ptr + (token * TOPK + expert_slot) * dim + offs_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.load(scores_ptr + token * TOPK + expert_slot).to(tl.float32)
            acc += y * score
        tl.store(out_ptr + token * dim + offs_d, acc, mask=mask)

    @triton.jit
    def _weighted_sum_backward_kernel(
        grad_out_ptr,
        y_partial_ptr,
        scores_ptr,
        grad_y_partial_ptr,
        grad_scores_ptr,
        batch,
        dim: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        expert_slot = tl.program_id(1)
        dim_block = tl.program_id(2)
        offs_d = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (token < batch) & (offs_d < dim)
        grad_out = tl.load(
            grad_out_ptr + token * dim + offs_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        y = tl.load(
            y_partial_ptr + (token * TOPK + expert_slot) * dim + offs_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score = tl.load(scores_ptr + token * TOPK + expert_slot).to(tl.float32)
        tl.store(
            grad_y_partial_ptr + (token * TOPK + expert_slot) * dim + offs_d,
            grad_out * score,
            mask=mask,
        )
        partial_score_grad = tl.sum(grad_out * y, axis=0)
        tl.atomic_add(
            grad_scores_ptr + token * TOPK + expert_slot,
            partial_score_grad,
            sem="relaxed",
        )

    @requires_shmem
    @triton.jit
    def _router_dispatch_kernel(
        my_tokens_ptr,
        dest_ranks_ptr,
        dest_offsets_ptr,
        buffers_cuda_ptrs_ptr,
        dispatch_recv_origin_global_token_id_ptr,
        dispatch_recv_weights_ptr,
        offs_recv_tokens,
        offs_recv_origin_global_token_id,
        ep_rank,
        max_B,
        total_copies,
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
        WRITE_MAPPING: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        num_programs = tl.num_programs(0)
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)

        while copy_id < total_copies:
            token_id = copy_id // TOPK
            expert_slot = copy_id - token_id * TOPK
            dest_rank = tl.load(dest_ranks_ptr + copy_id)
            dest_offset = tl.load(dest_offsets_ptr + copy_id)
            valid = dest_rank >= 0

            src_ptr = my_tokens_ptr + copy_id * D_BYTES
            dst_ptr = (base_ptr + offs_recv_tokens + dest_offset * D_BYTES).to(
                tl.pointer_type(tl.uint8)
            )

            if valid:
                shmem_backend.put(dst_ptr, src_ptr, D_BYTES, dest_rank)

            if valid and WRITE_MAPPING:
                id_ptr = (base_ptr + offs_recv_origin_global_token_id).to(
                    tl.pointer_type(tl.int64)
                )
                global_token_id = ep_rank * max_B * TOPK + token_id * TOPK + expert_slot
                scratch_id_ptr = dispatch_recv_weights_ptr.to(tl.pointer_type(tl.int64))
                scratch_id_ptr += copy_id
                tl.store(scratch_id_ptr, global_token_id)
                shmem_backend.put(id_ptr + dest_offset, scratch_id_ptr, 1, dest_rank)

            copy_id += num_programs
        shmem_backend.quiet()

    @requires_shmem
    @triton.jit
    def _router_combine_kernel(
        send_tokens_ptr,
        token_send_end_ptr,
        send_origin_global_token_id_ptr,
        buffers_cuda_ptrs_ptr,
        offs_combine_recv_tokens,
        ep_rank,
        ep_size,
        max_B,
        total_copies,
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        num_programs = tl.num_programs(0)
        token_send_end = tl.load(token_send_end_ptr)
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        max_B_topk = max_B * TOPK

        while copy_id < total_copies:
            valid = copy_id < token_send_end
            origin_id = tl.load(
                send_origin_global_token_id_ptr + copy_id,
                mask=valid,
                other=-1,
            )
            valid = valid & (origin_id >= 0)

            from_ep_rank = origin_id // max_B_topk
            dest_idx = origin_id - from_ep_rank * max_B_topk
            valid = valid & (from_ep_rank >= 0) & (from_ep_rank < ep_size)

            src_ptr = send_tokens_ptr + copy_id * D_BYTES
            dst_ptr = (base_ptr + offs_combine_recv_tokens + dest_idx * D_BYTES).to(
                tl.pointer_type(tl.uint8)
            )
            if valid:
                shmem_backend.put(dst_ptr, src_ptr, D_BYTES, from_ep_rank.to(tl.int32))

            copy_id += num_programs
        shmem_backend.quiet()

    _router_dispatch_tlx_kernel_untyped: Any = None
    _router_combine_tlx_kernel_untyped: Any = None

    if tlx is not None:
        tlx_mod = cast(Any, tlx)

        @triton.jit
        def _threadfence_system():
            tl.inline_asm_elementwise(
                "fence.sc.sys; // $0",
                "=r",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )

        @triton.jit
        def _router_dispatch_tlx_kernel(
            my_tokens_ptr,
            dest_ranks_ptr,
            dest_offsets_ptr,
            buffers_cuda_ptrs_ptr,
            dispatch_recv_origin_global_token_id_ptr,
            dispatch_recv_weights_ptr,
            offs_recv_tokens,
            offs_recv_origin_global_token_id,
            ep_rank,
            max_B,
            total_copies,
            D_BYTES: tl.constexpr,
            SMEM_SIZE: tl.constexpr,
            NUM_STAGES: tl.constexpr,
            TOPK: tl.constexpr,
            EP_SIZE: tl.constexpr,
            WRITE_MAPPING: tl.constexpr,
        ):
            pid = tl.program_id(0)
            smem = tlx_mod.local_alloc((SMEM_SIZE,), tl.uint8, num=NUM_STAGES)
            bars_full = tlx_mod.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
            bars_empty = tlx_mod.alloc_barriers(
                num_barriers=NUM_STAGES,
                arrive_count=1,
            )

            num_programs = tl.num_programs(0)
            copy_begin = pid * total_copies // num_programs
            copy_end = (pid + 1) * total_copies // num_programs
            num_copies = copy_end - copy_begin

            with tlx_mod.async_tasks():
                with tlx_mod.async_task("default"):
                    p = 1
                    for i in range(num_copies):
                        copy_id = copy_begin + i
                        buf = i % NUM_STAGES  # pyrefly: ignore[unsupported-operation]

                        empty_bar = tlx_mod.local_view(bars_empty, buf)
                        tlx_mod.barrier_wait(empty_bar, p)

                        src_base = my_tokens_ptr + copy_id * D_BYTES
                        full_bar = tlx_mod.local_view(bars_full, buf)
                        smem_buf = tlx_mod.local_view(smem, buf)
                        tlx_mod.barrier_expect_bytes(full_bar, D_BYTES)
                        tlx_mod.async_load(
                            src_base,
                            smem_buf,
                            bulk=True,
                            bulk_size=D_BYTES,
                            barrier=full_bar,
                        )
                        p = p ^ (buf == (NUM_STAGES - 1))

                with tlx_mod.async_task(num_warps=1, replicate=NUM_STAGES):
                    replica_id = tlx_mod.async_task_replica_id()
                    p = 0
                    num_consumer_iters = (
                        num_copies - replica_id + NUM_STAGES - 1
                    ) // NUM_STAGES

                    for j in range(num_consumer_iters):
                        copy_id = copy_begin + j * NUM_STAGES + replica_id
                        token_id = copy_id // TOPK
                        expert_slot = copy_id - token_id * TOPK
                        flat_idx = token_id * TOPK + expert_slot
                        dest_rank = tl.load(dest_ranks_ptr + flat_idx)
                        dest_offset = tl.load(dest_offsets_ptr + flat_idx)
                        valid = (dest_rank >= 0) & (dest_rank < EP_SIZE)

                        full_bar = tlx_mod.local_view(bars_full, replica_id)
                        tlx_mod.barrier_wait(full_bar, p)

                        if valid:
                            dest_buf_addr = tl.load(
                                buffers_cuda_ptrs_ptr + dest_rank.to(tl.int64)
                            )
                            dst_base = (
                                dest_buf_addr + offs_recv_tokens + dest_offset * D_BYTES
                            ).to(tl.pointer_type(tl.uint8))
                            smem_buf = tlx_mod.local_view(smem, replica_id)
                            tlx_mod.async_store(dst_base, smem_buf, D_BYTES)
                            tlx_mod.async_descriptor_store_wait(0)

                            if WRITE_MAPPING:
                                id_ptr = (
                                    dest_buf_addr + offs_recv_origin_global_token_id
                                ).to(tl.pointer_type(tl.int64))
                                ep_rank_i64 = ep_rank + tl.zeros([], dtype=tl.int64)
                                max_B_i64 = max_B + tl.zeros([], dtype=tl.int64)
                                token_id_i64 = token_id + tl.zeros([], dtype=tl.int64)
                                expert_slot_i64 = expert_slot + tl.zeros(
                                    [],
                                    dtype=tl.int64,
                                )
                                topk_i64 = tl.constexpr(TOPK) + tl.zeros(
                                    [],
                                    dtype=tl.int64,
                                )
                                global_token_id = (
                                    ep_rank_i64 * max_B_i64 * topk_i64
                                    + token_id_i64 * topk_i64
                                    + expert_slot_i64
                                )
                                tl.store(id_ptr + dest_offset, global_token_id)

                        empty_bar = tlx_mod.local_view(bars_empty, replica_id)
                        tlx_mod.barrier_arrive(empty_bar)
                        p = p ^ 1

                    _threadfence_system()

        @triton.jit
        def _router_combine_tlx_kernel(
            send_tokens_ptr,
            token_send_end_ptr,
            send_origin_global_token_id_ptr,
            buffers_cuda_ptrs_ptr,
            offs_combine_recv_tokens,
            max_B,
            D_BYTES: tl.constexpr,
            SMEM_SIZE: tl.constexpr,
            NUM_STAGES: tl.constexpr,
            TOPK: tl.constexpr,
            EP_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            smem = tlx_mod.local_alloc((SMEM_SIZE,), tl.uint8, num=NUM_STAGES)
            bars_full = tlx_mod.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
            bars_empty = tlx_mod.alloc_barriers(
                num_barriers=NUM_STAGES,
                arrive_count=1,
            )

            token_send_end = tl.load(token_send_end_ptr)
            num_programs = tl.num_programs(0)
            copy_begin = pid * token_send_end // num_programs
            copy_end = (pid + 1) * token_send_end // num_programs
            num_copies = (copy_end - copy_begin).to(tl.int32)

            with tlx_mod.async_tasks():
                with tlx_mod.async_task("default"):
                    p = 1
                    for i in range(num_copies):
                        copy_id = (copy_begin + i).to(tl.int64)
                        buf = i % NUM_STAGES  # pyrefly: ignore[unsupported-operation]

                        empty_bar = tlx_mod.local_view(bars_empty, buf)
                        tlx_mod.barrier_wait(empty_bar, p)

                        src_base = send_tokens_ptr + copy_id * D_BYTES
                        full_bar = tlx_mod.local_view(bars_full, buf)
                        smem_buf = tlx_mod.local_view(smem, buf)
                        tlx_mod.barrier_expect_bytes(full_bar, D_BYTES)
                        tlx_mod.async_load(
                            src_base,
                            smem_buf,
                            bulk=True,
                            bulk_size=D_BYTES,
                            barrier=full_bar,
                        )
                        p = p ^ (buf == (NUM_STAGES - 1))

                with tlx_mod.async_task(num_warps=1, replicate=NUM_STAGES):
                    replica_id = tlx_mod.async_task_replica_id()
                    p = 0
                    num_consumer_iters = (
                        num_copies - replica_id + NUM_STAGES - 1
                    ) // NUM_STAGES
                    max_B_i64 = max_B + tl.zeros([], dtype=tl.int64)
                    topk_i64 = tl.constexpr(TOPK) + tl.zeros([], dtype=tl.int64)
                    max_B_topk = max_B_i64 * topk_i64

                    for j in range(num_consumer_iters):
                        copy_id = (copy_begin + j * NUM_STAGES + replica_id).to(
                            tl.int64
                        )
                        origin_id = tl.load(send_origin_global_token_id_ptr + copy_id)
                        from_ep_rank = origin_id // max_B_topk
                        dest_idx = origin_id - from_ep_rank * max_B_topk
                        valid = (
                            (origin_id >= 0)
                            & (from_ep_rank >= 0)
                            & (from_ep_rank < EP_SIZE)
                        )

                        full_bar = tlx_mod.local_view(bars_full, replica_id)
                        tlx_mod.barrier_wait(full_bar, p)

                        if valid:
                            dest_buf_addr = tl.load(
                                buffers_cuda_ptrs_ptr + from_ep_rank
                            )
                            dst_base = (
                                dest_buf_addr
                                + offs_combine_recv_tokens
                                + dest_idx * D_BYTES
                            ).to(tl.pointer_type(tl.uint8))
                            smem_buf = tlx_mod.local_view(smem, replica_id)
                            tlx_mod.async_store(dst_base, smem_buf, D_BYTES)
                            tlx_mod.async_descriptor_store_wait(0)

                        empty_bar = tlx_mod.local_view(bars_empty, replica_id)
                        tlx_mod.barrier_arrive(empty_bar)
                        p = p ^ 1

                    _threadfence_system()

        _router_dispatch_tlx_kernel_untyped = cast(
            Any,
            _router_dispatch_tlx_kernel,
        )
        _router_combine_tlx_kernel_untyped = cast(
            Any,
            _router_combine_tlx_kernel,
        )

    @triton.jit
    def _zfill_ranges_kernel(
        input_ptr,
        begin_ofs_ptr,
        end_ofs_ptr,
        row_num_bytes: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        range_idx = tl.program_id(0)
        row_in_range = tl.program_id(1)
        begin = tl.load(begin_ofs_ptr + range_idx)
        end = tl.load(end_ofs_ptr + range_idx)
        row = begin + row_in_range
        byte_offsets = tl.arange(0, BLOCK_BYTES)
        mask = (row < end) & (byte_offsets < row_num_bytes)
        tl.store(
            input_ptr + row * row_num_bytes + byte_offsets,
            tl.zeros([BLOCK_BYTES], dtype=tl.uint8),
            mask=mask,
        )

    _barrier_arrive_kernel_untyped = cast(Any, _barrier_arrive_kernel)
    _barrier_wait_kernel_untyped = cast(Any, _barrier_wait_kernel)
    _ep_allgather_kernel_untyped = cast(Any, _ep_allgather_kernel)
    _compute_expert_offsets_kernel_untyped = cast(Any, _compute_expert_offsets_kernel)
    _compute_dest_offsets_kernel_untyped = cast(Any, _compute_dest_offsets_kernel)
    _swiglu_forward_kernel_untyped = cast(Any, _swiglu_forward_kernel)
    _swiglu_backward_kernel_untyped = cast(Any, _swiglu_backward_kernel)
    _swiglu_forward_with_offsets_kernel_untyped = cast(
        Any,
        _swiglu_forward_with_offsets_kernel,
    )
    _swiglu_backward_with_offsets_kernel_untyped = cast(
        Any,
        _swiglu_backward_with_offsets_kernel,
    )
    _clone_valid_prefix_kernel_untyped = cast(Any, _clone_valid_prefix_kernel)
    _weighted_sum_forward_kernel_untyped = cast(Any, _weighted_sum_forward_kernel)
    _weighted_sum_backward_kernel_untyped = cast(Any, _weighted_sum_backward_kernel)
    _router_dispatch_kernel_untyped = cast(Any, _router_dispatch_kernel)
    _router_combine_kernel_untyped = cast(Any, _router_combine_kernel)
    _zfill_ranges_kernel_untyped = cast(Any, _zfill_ranges_kernel)

    if not hasattr(torch.ops.inductor, "flex_ep_barrier_arrive"):

        @torch.library.custom_op(
            "inductor::flex_ep_barrier_arrive",
            mutates_args=("flag",),
        )
        def _flex_ep_barrier_arrive(flag: torch.Tensor) -> torch.Tensor:
            out = torch.empty(1, dtype=torch.int32, device=flag.device)
            _barrier_arrive_kernel_untyped[(1,)](flag, out, num_warps=1)
            return out

    if not hasattr(torch.ops.inductor, "flex_ep_barrier_wait"):

        @torch.library.custom_op("inductor::flex_ep_barrier_wait", mutates_args=())
        def _flex_ep_barrier_wait(
            cuda_ptrs: torch.Tensor,
            offs_flag: int,
            expected: torch.Tensor,
            timeout_s: float = 5.0,
        ) -> None:
            block = triton.next_power_of_2(cuda_ptrs.numel())
            timeout_ns = int(timeout_s * 1e9)
            _barrier_wait_kernel_untyped[(1,)](
                cuda_ptrs,
                expected,
                cuda_ptrs.numel(),
                offs_flag,
                timeout_ns,
                BLOCK=block,
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_allgather"):

        @torch.library.custom_op(
            "inductor::flex_ep_allgather",
            mutates_args=("output",),
        )
        def _flex_ep_allgather(
            output: torch.Tensor,
            input: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            offs_output: int,
            ep_rank: int,
        ) -> None:
            input_u8 = input.contiguous().view(torch.uint8)
            _ep_allgather_kernel_untyped[(buffers_cuda_ptrs.numel(),)](
                input_u8,
                buffers_cuda_ptrs,
                offs_output,
                ep_rank,
                input_num_bytes=input_u8.numel(),
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_router_compute_all_expert_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_compute_all_expert_offsets",
            mutates_args=(),
        )
        def _flex_ep_router_compute_all_expert_offsets(
            all_expert_counts: torch.Tensor,
            ep_rank: int,
            local_experts: int,
            token_alignment: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if all_expert_counts.dim() != 2:
                raise ValueError(
                    "flex_ep_router_compute_all_expert_offsets expects "
                    f"[ep_size, num_experts], got {all_expert_counts.shape}."
                )
            ep_size, num_experts = all_expert_counts.shape
            if local_experts <= 0 or num_experts != ep_size * local_experts:
                raise ValueError(
                    "flex_ep_router_compute_all_expert_offsets expects "
                    f"num_experts == ep_size * local_experts, got "
                    f"{num_experts=} {ep_size=} {local_experts=}."
                )
            if not 0 <= ep_rank < ep_size:
                raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}.")
            if (
                all_expert_counts.device.type != "cuda"
                or all_expert_counts.dtype != torch.int64
                or not all_expert_counts.is_contiguous()
                or not _is_power_of_2(ep_size)
                or not _is_power_of_2(local_experts)
            ):
                return _compute_all_expert_offsets_reference(
                    all_expert_counts,
                    ep_rank,
                    local_experts,
                    token_alignment,
                )

            offsets = torch.empty(
                (ep_size, local_experts, ep_size + 1),
                dtype=torch.int32,
                device=all_expert_counts.device,
            )
            expert_start = torch.empty(
                (ep_size, local_experts + 1),
                dtype=torch.int32,
                device=all_expert_counts.device,
            )
            grand_total = torch.empty(
                (ep_size,),
                dtype=torch.int64,
                device=all_expert_counts.device,
            )
            _compute_expert_offsets_kernel_untyped[(ep_size,)](
                all_expert_counts,
                offsets,
                expert_start,
                grand_total,
                LOCAL_EXPERTS=local_experts,
                EP_SIZE=ep_size,
                TOKEN_ALIGNMENT=token_alignment,
                num_warps=4,
            )
            return offsets, grand_total[ep_rank].clone(), expert_start[ep_rank].clone()

        @_flex_ep_router_compute_all_expert_offsets.register_fake
        def _flex_ep_router_compute_all_expert_offsets_fake(
            all_expert_counts: torch.Tensor,
            ep_rank: int,
            local_experts: int,
            token_alignment: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del ep_rank, token_alignment
            ep_size = all_expert_counts.shape[0]
            return (
                torch.empty(
                    (ep_size, local_experts, ep_size + 1),
                    dtype=torch.int32,
                    device=all_expert_counts.device,
                ),
                torch.empty((), dtype=torch.int64, device=all_expert_counts.device),
                torch.empty(
                    (local_experts + 1,),
                    dtype=torch.int32,
                    device=all_expert_counts.device,
                ),
            )

    if not hasattr(torch.ops.inductor, "flex_ep_router_compute_dest_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_compute_dest_offsets",
            mutates_args=(),
        )
        def _flex_ep_router_compute_dest_offsets(
            topk_idx: torch.Tensor,
            recv_ofs: torch.Tensor,
            ep_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if topk_idx.dim() != 2 or recv_ofs.dim() != 1:
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects topk_idx [batch, top_k] "
                    f"and recv_ofs [num_experts], got {topk_idx.shape} and {recv_ofs.shape}."
                )
            if ep_size <= 0 or recv_ofs.numel() % ep_size != 0:
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects num_experts "
                    f"to be divisible by ep_size, got num_experts={recv_ofs.numel()} "
                    f"and {ep_size=}."
                )
            if topk_idx.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects int32 or int64 topk_idx."
                )
            if recv_ofs.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "flex_ep_router_compute_dest_offsets expects int32 or int64 recv_ofs."
                )

            batch, top_k = topk_idx.shape
            num_experts = recv_ofs.numel()
            local_experts = num_experts // ep_size
            if topk_idx.numel() == 0:
                return (
                    torch.empty_like(topk_idx, dtype=torch.int32),
                    torch.empty_like(topk_idx, dtype=torch.int64),
                )
            if (
                topk_idx.device.type != "cuda"
                or not topk_idx.is_contiguous()
                or not _is_power_of_2(num_experts)
            ):
                return _compute_dest_offsets_reference(topk_idx, recv_ofs, ep_size)

            dest_ranks = torch.div(
                topk_idx,
                local_experts,
                rounding_mode="floor",
            ).to(torch.int32)
            dest_offsets = torch.empty(
                topk_idx.shape,
                dtype=torch.int64,
                device=topk_idx.device,
            )
            block = 512
            _compute_dest_offsets_kernel_untyped[(num_experts,)](
                topk_idx,
                recv_ofs,
                recv_ofs.stride(0),
                dest_offsets,
                batch,
                TOPK=top_k,
                BLOCK=block,
                num_warps=4,
            )
            return dest_ranks, dest_offsets

        @_flex_ep_router_compute_dest_offsets.register_fake
        def _flex_ep_router_compute_dest_offsets_fake(
            topk_idx: torch.Tensor,
            recv_ofs: torch.Tensor,
            ep_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del recv_ofs, ep_size
            return (
                torch.empty_like(topk_idx, dtype=torch.int32),
                torch.empty_like(topk_idx, dtype=torch.int64),
            )

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_forward"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_forward",
            mutates_args=(),
        )
        def _flex_ep_swiglu_forward(y1: torch.Tensor) -> torch.Tensor:
            if y1.dtype != torch.bfloat16:
                raise ValueError("flex_ep_swiglu_forward supports BF16 input only.")
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    f"flex_ep_swiglu_forward expects [tokens, 2 * hidden], got {y1.shape}."
                )
            if not y1.is_contiguous():
                raise ValueError("flex_ep_swiglu_forward requires contiguous input.")
            hidden_dim = y1.shape[-1] // 2
            y2 = torch.empty(
                (y1.shape[0], hidden_dim),
                device=y1.device,
                dtype=y1.dtype,
            )
            total = y2.numel()
            if total == 0:
                return y2
            block = 1024
            _swiglu_forward_kernel_untyped[(triton.cdiv(total, block),)](
                y1,
                y2,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return y2

        @_flex_ep_swiglu_forward.register_fake
        def _flex_ep_swiglu_forward_fake(y1: torch.Tensor) -> torch.Tensor:
            return torch.empty(
                (y1.shape[0], y1.shape[-1] // 2),
                device=y1.device,
                dtype=y1.dtype,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_backward"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_backward",
            mutates_args=(),
        )
        def _flex_ep_swiglu_backward(
            dy2: torch.Tensor,
            y1: torch.Tensor,
        ) -> torch.Tensor:
            if dy2.dtype != torch.bfloat16 or y1.dtype != torch.bfloat16:
                raise ValueError("flex_ep_swiglu_backward supports BF16 inputs only.")
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    f"flex_ep_swiglu_backward expects y1 [tokens, 2 * hidden], got {y1.shape}."
                )
            if dy2.shape != (y1.shape[0], y1.shape[-1] // 2):
                raise ValueError(
                    "flex_ep_swiglu_backward dy2 shape must match "
                    f"[tokens, hidden], got dy2={dy2.shape}, y1={y1.shape}."
                )
            if not dy2.is_contiguous() or not y1.is_contiguous():
                raise ValueError("flex_ep_swiglu_backward requires contiguous inputs.")
            hidden_dim = dy2.shape[-1]
            dy1 = torch.empty_like(y1)
            total = dy2.numel()
            if total == 0:
                return dy1
            block = 1024
            _swiglu_backward_kernel_untyped[(triton.cdiv(total, block),)](
                dy2,
                y1,
                dy1,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return dy1

        @_flex_ep_swiglu_backward.register_fake
        def _flex_ep_swiglu_backward_fake(
            dy2: torch.Tensor,
            y1: torch.Tensor,
        ) -> torch.Tensor:
            del dy2
            return torch.empty_like(y1)

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_forward_with_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_forward_with_offsets",
            mutates_args=(),
        )
        def _flex_ep_swiglu_forward_with_offsets(
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            if y1.dtype != torch.bfloat16:
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets supports BF16 input only."
                )
            if token_end.dtype != torch.int64 or token_end.numel() != 1:
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets expects token_end "
                    "to be a single int64 tensor."
                )
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets expects "
                    f"[tokens, 2 * hidden], got {y1.shape}."
                )
            if not y1.is_contiguous() or not token_end.is_contiguous():
                raise ValueError(
                    "flex_ep_swiglu_forward_with_offsets requires contiguous inputs."
                )
            hidden_dim = y1.shape[-1] // 2
            y2 = torch.empty(
                (y1.shape[0], hidden_dim),
                device=y1.device,
                dtype=y1.dtype,
            )
            total = y2.numel()
            if total == 0:
                return y2
            block = 1024
            _swiglu_forward_with_offsets_kernel_untyped[(triton.cdiv(total, block),)](
                y1,
                token_end,
                y2,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return y2

        @_flex_ep_swiglu_forward_with_offsets.register_fake
        def _flex_ep_swiglu_forward_with_offsets_fake(
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            del token_end
            return torch.empty(
                (y1.shape[0], y1.shape[-1] // 2),
                device=y1.device,
                dtype=y1.dtype,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_swiglu_backward_with_offsets"):

        @torch.library.custom_op(
            "inductor::flex_ep_swiglu_backward_with_offsets",
            mutates_args=(),
        )
        def _flex_ep_swiglu_backward_with_offsets(
            dy2: torch.Tensor,
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            if dy2.dtype != torch.bfloat16 or y1.dtype != torch.bfloat16:
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets supports BF16 inputs only."
                )
            if token_end.dtype != torch.int64 or token_end.numel() != 1:
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets expects token_end "
                    "to be a single int64 tensor."
                )
            if y1.dim() != 2 or y1.shape[-1] % 2 != 0:
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets expects y1 "
                    f"[tokens, 2 * hidden], got {y1.shape}."
                )
            if dy2.shape != (y1.shape[0], y1.shape[-1] // 2):
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets dy2 shape must match "
                    f"[tokens, hidden], got dy2={dy2.shape}, y1={y1.shape}."
                )
            if (
                not dy2.is_contiguous()
                or not y1.is_contiguous()
                or not token_end.is_contiguous()
            ):
                raise ValueError(
                    "flex_ep_swiglu_backward_with_offsets requires contiguous inputs."
                )
            hidden_dim = dy2.shape[-1]
            dy1 = torch.empty_like(y1)
            total = dy2.numel()
            if total == 0:
                return dy1
            block = 1024
            _swiglu_backward_with_offsets_kernel_untyped[(triton.cdiv(total, block),)](
                dy2,
                y1,
                token_end,
                dy1,
                total,
                hidden_dim,
                BLOCK=block,
                num_warps=4,
            )
            return dy1

        @_flex_ep_swiglu_backward_with_offsets.register_fake
        def _flex_ep_swiglu_backward_with_offsets_fake(
            dy2: torch.Tensor,
            y1: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            del dy2, token_end
            return torch.empty_like(y1)

    if not hasattr(torch.ops.inductor, "flex_ep_clone_valid_prefix"):

        @torch.library.custom_op(
            "inductor::flex_ep_clone_valid_prefix",
            mutates_args=(),
        )
        def _flex_ep_clone_valid_prefix(
            input: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            if input.dim() < 1:
                raise ValueError(
                    "flex_ep_clone_valid_prefix expects at least 1D input."
                )
            if token_end.dtype != torch.int64 or token_end.numel() != 1:
                raise ValueError(
                    "flex_ep_clone_valid_prefix expects token_end to be a "
                    "single int64 tensor."
                )
            if not input.is_contiguous() or not token_end.is_contiguous():
                raise ValueError(
                    "flex_ep_clone_valid_prefix requires contiguous inputs."
                )
            out = torch.empty_like(input)
            total = input.numel()
            if total == 0:
                return out
            row_width = total // input.shape[0]
            block = 1024
            _clone_valid_prefix_kernel_untyped[(triton.cdiv(total, block),)](
                input,
                token_end,
                out,
                total,
                row_width,
                BLOCK=block,
                num_warps=4,
            )
            return out

        @_flex_ep_clone_valid_prefix.register_fake
        def _flex_ep_clone_valid_prefix_fake(
            input: torch.Tensor,
            token_end: torch.Tensor,
        ) -> torch.Tensor:
            del token_end
            return torch.empty_like(input)

    if not hasattr(torch.ops.inductor, "flex_ep_weighted_sum_forward"):

        @torch.library.custom_op(
            "inductor::flex_ep_weighted_sum_forward",
            mutates_args=(),
        )
        def _flex_ep_weighted_sum_forward(
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> torch.Tensor:
            if y_partial.dtype != torch.bfloat16 or top_scores.dtype != torch.float32:
                raise ValueError(
                    "flex_ep_weighted_sum_forward expects BF16 y_partial and "
                    "FP32 top_scores."
                )
            if y_partial.dim() != 3 or top_scores.dim() != 2:
                raise ValueError(
                    "flex_ep_weighted_sum_forward expects y_partial [tokens, topk, dim] "
                    f"and top_scores [tokens, topk], got {y_partial.shape} and {top_scores.shape}."
                )
            if y_partial.shape[:2] != top_scores.shape:
                raise ValueError(
                    "flex_ep_weighted_sum_forward top_scores shape must match "
                    f"y_partial leading dims, got {top_scores.shape} and {y_partial.shape[:2]}."
                )
            if not y_partial.is_contiguous() or not top_scores.is_contiguous():
                raise ValueError(
                    "flex_ep_weighted_sum_forward requires contiguous inputs."
                )
            out = torch.empty(
                (y_partial.shape[0], y_partial.shape[2]),
                device=y_partial.device,
                dtype=y_partial.dtype,
            )
            if out.numel() == 0:
                return out
            block_d = 256
            _weighted_sum_forward_kernel_untyped[
                (y_partial.shape[0], triton.cdiv(y_partial.shape[2], block_d))
            ](
                y_partial,
                top_scores,
                out,
                y_partial.shape[0],
                dim=y_partial.shape[2],
                TOPK=y_partial.shape[1],
                BLOCK_D=block_d,
                num_warps=4,
            )
            return out

        @_flex_ep_weighted_sum_forward.register_fake
        def _flex_ep_weighted_sum_forward_fake(
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> torch.Tensor:
            del top_scores
            return torch.empty(
                (y_partial.shape[0], y_partial.shape[2]),
                device=y_partial.device,
                dtype=y_partial.dtype,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_weighted_sum_backward"):

        @torch.library.custom_op(
            "inductor::flex_ep_weighted_sum_backward",
            mutates_args=(),
        )
        def _flex_ep_weighted_sum_backward(
            grad_out: torch.Tensor,
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if (
                grad_out.dtype != torch.bfloat16
                or y_partial.dtype != torch.bfloat16
                or top_scores.dtype != torch.float32
            ):
                raise ValueError(
                    "flex_ep_weighted_sum_backward expects BF16 grad/y_partial "
                    "and FP32 top_scores."
                )
            if y_partial.dim() != 3 or top_scores.dim() != 2 or grad_out.dim() != 2:
                raise ValueError(
                    "flex_ep_weighted_sum_backward expects grad_out [tokens, dim], "
                    "y_partial [tokens, topk, dim], and top_scores [tokens, topk]."
                )
            if grad_out.shape != (y_partial.shape[0], y_partial.shape[2]):
                raise ValueError(
                    "flex_ep_weighted_sum_backward grad_out shape must match "
                    f"[tokens, dim], got {grad_out.shape} and {y_partial.shape}."
                )
            if y_partial.shape[:2] != top_scores.shape:
                raise ValueError(
                    "flex_ep_weighted_sum_backward top_scores shape must match "
                    f"y_partial leading dims, got {top_scores.shape} and {y_partial.shape[:2]}."
                )
            if (
                not grad_out.is_contiguous()
                or not y_partial.is_contiguous()
                or not top_scores.is_contiguous()
            ):
                raise ValueError(
                    "flex_ep_weighted_sum_backward requires contiguous inputs."
                )
            grad_y_partial = torch.empty_like(y_partial)
            grad_scores = torch.zeros_like(top_scores)
            if grad_y_partial.numel() == 0:
                return grad_y_partial, grad_scores
            block_d = 1024
            _weighted_sum_backward_kernel_untyped[
                (
                    y_partial.shape[0],
                    y_partial.shape[1],
                    triton.cdiv(y_partial.shape[2], block_d),
                )
            ](
                grad_out,
                y_partial,
                top_scores,
                grad_y_partial,
                grad_scores,
                y_partial.shape[0],
                dim=y_partial.shape[2],
                TOPK=y_partial.shape[1],
                BLOCK_D=block_d,
                num_warps=8,
            )
            return grad_y_partial, grad_scores

        @_flex_ep_weighted_sum_backward.register_fake
        def _flex_ep_weighted_sum_backward_fake(
            grad_out: torch.Tensor,
            y_partial: torch.Tensor,
            top_scores: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del grad_out
            return torch.empty_like(y_partial), torch.empty_like(top_scores)

    if not hasattr(torch.ops.inductor, "flex_ep_router_dispatch"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_dispatch",
            mutates_args=(
                "dispatch_recv_buffer",
                "dispatch_recv_buffer_scaling_factors",
                "dispatch_recv_origin_global_token_id",
                "dispatch_recv_weights",
            ),
            schema="(Tensor my_tokens, Tensor? my_scaling_factors, "
            "Tensor? my_topk_weights, Tensor dest_ranks, Tensor dest_offsets, "
            "Tensor buffers_cuda_ptrs, Tensor(a!) dispatch_recv_buffer, "
            "Tensor(b!) dispatch_recv_buffer_scaling_factors, "
            "Tensor(c!) dispatch_recv_origin_global_token_id, "
            "Tensor(d!) dispatch_recv_weights, int offs_recv_tokens, "
            "int offs_recv_scaling_factors, int offs_recv_weights, "
            "int offs_recv_origin_global_token_id, int ep_rank, int num_ctas, "
            "int max_B) -> ()",
        )
        def _flex_ep_router_dispatch(
            my_tokens: torch.Tensor,
            my_scaling_factors: torch.Tensor | None,
            my_topk_weights: torch.Tensor | None,
            dest_ranks: torch.Tensor,
            dest_offsets: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            dispatch_recv_buffer: torch.Tensor,
            dispatch_recv_buffer_scaling_factors: torch.Tensor,
            dispatch_recv_origin_global_token_id: torch.Tensor,
            dispatch_recv_weights: torch.Tensor,
            offs_recv_tokens: int,
            offs_recv_scaling_factors: int,
            offs_recv_weights: int,
            offs_recv_origin_global_token_id: int,
            ep_rank: int,
            num_ctas: int,
            max_B: int,
        ) -> None:
            del my_scaling_factors, my_topk_weights
            del dispatch_recv_buffer, dispatch_recv_buffer_scaling_factors
            del offs_recv_scaling_factors, offs_recv_weights
            if my_tokens.dtype != torch.bfloat16:
                raise ValueError("flex_ep_router_dispatch supports BF16 tokens only.")
            if not my_tokens.is_contiguous():
                raise ValueError("flex_ep_router_dispatch requires contiguous tokens.")
            if num_ctas < 1:
                raise ValueError(f"num_ctas must be positive, got {num_ctas}.")
            batch, top_k, dim = my_tokens.shape
            my_tokens_u8 = my_tokens.view(torch.uint8)
            dim_bytes = dim * my_tokens.element_size()
            total_copies = batch * top_k
            if total_copies == 0:
                return
            if _router_dispatch_tlx_kernel_untyped is not None:
                if dim_bytes % 16 != 0:
                    raise ValueError(
                        "Token row size must be a multiple of 16 bytes, "
                        f"got {dim_bytes}."
                    )
                launch_ctas = min(num_ctas, MAX_TLX_NUM_CTAS, total_copies)
                _router_dispatch_tlx_kernel_untyped[(launch_ctas,)](
                    my_tokens_u8,
                    dest_ranks,
                    dest_offsets,
                    buffers_cuda_ptrs,
                    dispatch_recv_origin_global_token_id,
                    dispatch_recv_weights,
                    offs_recv_tokens,
                    offs_recv_origin_global_token_id,
                    ep_rank,
                    max_B,
                    total_copies,
                    D_BYTES=dim_bytes,
                    SMEM_SIZE=triton.next_power_of_2(dim_bytes),
                    NUM_STAGES=TLX_NUM_STAGES,
                    TOPK=top_k,
                    EP_SIZE=buffers_cuda_ptrs.numel(),
                    WRITE_MAPPING=offs_recv_origin_global_token_id >= 0,
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(4, 1, 1),
                )
                return
            _router_dispatch_kernel_untyped[(min(num_ctas, total_copies),)](
                my_tokens_u8,
                dest_ranks,
                dest_offsets,
                buffers_cuda_ptrs,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
                offs_recv_tokens,
                offs_recv_origin_global_token_id,
                ep_rank,
                max_B,
                total_copies,
                D_BYTES=dim_bytes,
                TOPK=top_k,
                WRITE_MAPPING=offs_recv_origin_global_token_id >= 0,
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_router_combine"):

        @torch.library.custom_op(
            "inductor::flex_ep_router_combine",
            mutates_args=(
                "combine_recv_buffer",
                "combine_recv_scale_factors",
                "combine_recv_weights",
            ),
            schema="(Tensor send_tokens, Tensor? send_scale_factors, "
            "Tensor? send_weights, Tensor expert_begin_offset_per_ep, "
            "Tensor token_send_end, Tensor send_origin_global_token_id, "
            "Tensor buffers_cuda_ptrs, Tensor(a!) combine_recv_buffer, "
            "Tensor(b!) combine_recv_scale_factors, Tensor(c!) combine_recv_weights, "
            "int offs_combine_recv_tokens, int offs_combine_recv_scale_factors, "
            "int offs_combine_recv_weights, int ep_rank, int B, int TOPK, "
            "int num_ctas, int max_B) -> ()",
        )
        def _flex_ep_router_combine(
            send_tokens: torch.Tensor,
            send_scale_factors: torch.Tensor | None,
            send_weights: torch.Tensor | None,
            expert_begin_offset_per_ep: torch.Tensor,
            token_send_end: torch.Tensor,
            send_origin_global_token_id: torch.Tensor,
            buffers_cuda_ptrs: torch.Tensor,
            combine_recv_buffer: torch.Tensor,
            combine_recv_scale_factors: torch.Tensor,
            combine_recv_weights: torch.Tensor,
            offs_combine_recv_tokens: int,
            offs_combine_recv_scale_factors: int,
            offs_combine_recv_weights: int,
            ep_rank: int,
            batch: int,
            top_k: int,
            num_ctas: int,
            max_B: int,
        ) -> None:
            del send_scale_factors, send_weights, expert_begin_offset_per_ep
            del combine_recv_buffer, combine_recv_scale_factors, combine_recv_weights
            del offs_combine_recv_scale_factors, offs_combine_recv_weights
            del batch
            if send_tokens.dtype != torch.bfloat16:
                raise ValueError("flex_ep_router_combine supports BF16 tokens only.")
            if not send_tokens.is_contiguous():
                raise ValueError("flex_ep_router_combine requires contiguous tokens.")
            if num_ctas < 1:
                raise ValueError(f"num_ctas must be positive, got {num_ctas}.")
            dim = send_tokens.shape[-1]
            send_tokens_u8 = send_tokens.view(torch.uint8)
            dim_bytes = dim * send_tokens.element_size()
            total_copies = send_tokens.shape[0]
            if total_copies == 0:
                return
            if _router_combine_tlx_kernel_untyped is not None:
                if dim_bytes % 16 != 0:
                    raise ValueError(
                        "Token row size must be a multiple of 16 bytes, "
                        f"got {dim_bytes}."
                    )
                launch_ctas = min(num_ctas, MAX_TLX_NUM_CTAS, total_copies)
                _router_combine_tlx_kernel_untyped[(launch_ctas,)](
                    send_tokens_u8,
                    token_send_end,
                    send_origin_global_token_id,
                    buffers_cuda_ptrs,
                    offs_combine_recv_tokens,
                    max_B,
                    D_BYTES=dim_bytes,
                    SMEM_SIZE=triton.next_power_of_2(dim_bytes),
                    NUM_STAGES=TLX_NUM_STAGES,
                    TOPK=top_k,
                    EP_SIZE=buffers_cuda_ptrs.numel(),
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(4, 1, 1),
                )
                return
            _router_combine_kernel_untyped[(min(num_ctas, total_copies),)](
                send_tokens_u8,
                token_send_end,
                send_origin_global_token_id,
                buffers_cuda_ptrs,
                offs_combine_recv_tokens,
                ep_rank,
                buffers_cuda_ptrs.numel(),
                max_B,
                total_copies,
                D_BYTES=dim_bytes,
                TOPK=top_k,
                num_warps=1,
            )

    if not hasattr(torch.ops.inductor, "flex_ep_zfill_ranges_inplace"):

        @torch.library.custom_op(
            "inductor::flex_ep_zfill_ranges_inplace",
            mutates_args=("input",),
        )
        def _flex_ep_zfill_ranges_inplace(
            input: torch.Tensor,
            begin_ofs: torch.Tensor,
            end_ofs: torch.Tensor,
            max_values_per_batch: int,
        ) -> None:
            if not input.is_contiguous() or input.ndim != 2:
                raise ValueError(
                    "flex_ep_zfill_ranges_inplace requires a contiguous 2D tensor."
                )
            block_bytes = triton.next_power_of_2(input.shape[1])
            _zfill_ranges_kernel_untyped[(begin_ofs.numel(), max_values_per_batch)](
                input,
                begin_ofs,
                end_ofs,
                row_num_bytes=input.shape[1],
                BLOCK_BYTES=block_bytes,
                num_warps=1,
            )


def _require_ep_backend_ops() -> None:
    _register_ep_backend_ops()
    missing_ops = _missing_ep_backend_ops()
    if missing_ops:
        formatted_ops = ", ".join(f"torch.ops.inductor.{op}" for op in missing_ops)
        raise RuntimeError(
            "FlexGroupedExperts EP>1 requires PyTorch inductor flex_ep backend "
            f"ops to be registered. Missing: {formatted_ops}. This path uses "
            "PyTorch flex_ep kernels and does not use gb200_moe_sol_cpp."
        )


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _validate_flex_ep_capacity_factor(capacity_factor: float) -> float:
    if not 0 < capacity_factor <= 1.0:
        raise ValueError(
            "FlexEP capacity factor must satisfy 0 < capacity_factor <= 1.0."
        )
    return float(capacity_factor)


def _compute_max_tokens_recv(
    max_tokens: int,
    ep_size: int,
    num_experts: int,
    top_k: int,
    *,
    capacity_factor: float = 1.0,
) -> int:
    capacity_factor = _validate_flex_ep_capacity_factor(capacity_factor)
    local_experts = num_experts // ep_size
    max_unaligned_tokens = math.ceil(
        max_tokens * ep_size * min(local_experts, top_k) * capacity_factor
    )
    return (
        _align_up(max_unaligned_tokens, TOKEN_ALIGNMENT)
        + TOKEN_ALIGNMENT * local_experts
    )


def _validate_flex_ep_capacity(
    local_experts_start: torch.Tensor,
    max_recv_tokens: int,
    *,
    capacity_factor: float,
) -> None:
    error_msg = (
        "FlexGroupedExperts flex_ep receive workspace capacity was exceeded. "
        f"Receive workspace capacity is {max_recv_tokens}; increase the FlexEP "
        f"capacity factor (currently {capacity_factor}) up to 1.0."
    )
    if local_experts_start.is_cuda:
        # Avoid a host sync in the hot path. The CPU branch below keeps the
        # precise ValueError message for unit tests and non-CUDA callers.
        torch._assert_async(  # type: ignore[attr-defined]
            local_experts_start[-1] <= max_recv_tokens,
            error_msg,
        )
        return

    num_recv_tokens = int(local_experts_start[-1].item())
    if num_recv_tokens > max_recv_tokens:
        raise ValueError(f"{error_msg} Received {num_recv_tokens} local expert tokens.")


def _compute_dispatch_recv_weights_numel(
    max_tokens: int,
    max_tokens_received: int,
    top_k: int,
) -> int:
    dispatch_id_scratch_bytes = max_tokens * top_k * torch.int64.itemsize
    dispatch_weight_bytes = max_tokens_received * torch.float32.itemsize
    return (
        max(dispatch_id_scratch_bytes, dispatch_weight_bytes)
        + torch.float32.itemsize
        - 1
    ) // torch.float32.itemsize


def _scale_dim(dim: int) -> int:
    return max(1, dim // 16)


@dataclass(frozen=True)
class FlexEPDispatchPlan:
    recv_origin_global_token_id: torch.Tensor
    expert_begin_offset_per_ep: torch.Tensor
    dest_ranks: torch.Tensor
    dest_offsets: torch.Tensor
    local_experts_start: torch.Tensor
    max_recv_tokens: torch.Tensor
    recv_total_tokens: torch.Tensor
    overflow: torch.Tensor

    @staticmethod
    def field_names() -> tuple[str, ...]:
        return (
            "recv_origin_global_token_id",
            "expert_begin_offset_per_ep",
            "dest_ranks",
            "dest_offsets",
            "local_experts_start",
            "max_recv_tokens",
            "recv_total_tokens",
            "overflow",
        )

    def flatten(self) -> tuple[torch.Tensor, ...]:
        return (
            self.recv_origin_global_token_id,
            self.expert_begin_offset_per_ep,
            self.dest_ranks,
            self.dest_offsets,
            self.local_experts_start,
            self.max_recv_tokens,
            self.recv_total_tokens,
            self.overflow,
        )

    @classmethod
    def from_flat(cls, plan_flat: tuple[Any, ...]) -> "FlexEPDispatchPlan":
        if len(plan_flat) != len(cls.field_names()):
            raise ValueError(
                f"FlexEPDispatchPlan expects {len(cls.field_names())} values, "
                f"got {len(plan_flat)}."
            )
        for name, value in zip(cls.field_names(), plan_flat):
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"FlexEPDispatchPlan field {name} must be a torch.Tensor, "
                    f"got {type(value).__name__}."
                )
        return cls(*cast(tuple[torch.Tensor, ...], plan_flat))


torch.utils._pytree.register_dataclass(FlexEPDispatchPlan)


@dataclass(frozen=True)
class FlexEPRouterOperands:
    raw: torch.Tensor
    buffers_cuda_ptrs: torch.Tensor
    offs_barrier_counter: int
    offs_dispatch_recv_buffer: int
    offs_dispatch_recv_buffer_scaling_factors: int
    offs_dispatch_recv_weights: int
    offs_dispatch_recv_origin_global_token_id: int
    offs_combine_recv_buffer: int
    offs_combine_recv_scale_factors: int
    offs_combine_recv_weights: int
    offs_allgather_expert_counts: int
    ep_rank: int


torch.utils._pytree.register_dataclass(FlexEPRouterOperands)


@dataclass
class NvlSharedBuffer:
    raw: torch.Tensor
    barrier_counter: torch.Tensor
    dispatch_recv_buffer: torch.Tensor
    dispatch_recv_buffer_scaling_factors: torch.Tensor
    dispatch_recv_weights: torch.Tensor
    dispatch_recv_origin_global_token_id: torch.Tensor
    combine_recv_buffer: torch.Tensor
    combine_recv_scale_factors: torch.Tensor
    combine_recv_weights: torch.Tensor
    allgather_expert_counts: torch.Tensor

    @staticmethod
    def tensors_shapes_and_dtypes(
        *,
        max_tokens: int,
        dim: int,
        ep_size: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.0,
    ) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size "
                f"({ep_size})."
            )
        max_tokens_received = _compute_max_tokens_recv(
            max_tokens,
            ep_size,
            num_experts,
            top_k,
            capacity_factor=capacity_factor,
        )
        dispatch_recv_weights_numel = _compute_dispatch_recv_weights_numel(
            max_tokens,
            max_tokens_received,
            top_k,
        )
        scale_dim = _scale_dim(dim)
        return (
            ("barrier_counter", (16,), torch.int32),
            ("dispatch_recv_buffer", (max_tokens_received, dim), torch.uint16),
            (
                "dispatch_recv_buffer_scaling_factors",
                (max_tokens_received, scale_dim),
                torch.uint8,
            ),
            ("dispatch_recv_weights", (dispatch_recv_weights_numel,), torch.float32),
            (
                "dispatch_recv_origin_global_token_id",
                (max_tokens_received,),
                torch.int64,
            ),
            ("combine_recv_buffer", (max_tokens, top_k, dim), torch.uint16),
            ("combine_recv_scale_factors", (max_tokens, top_k, scale_dim), torch.uint8),
            ("combine_recv_weights", (max_tokens, top_k), torch.float32),
            ("allgather_expert_counts", (ep_size, num_experts), torch.int64),
        )

    @classmethod
    def get_buffer_size_bytes(cls, **kwargs: Any) -> int:
        total_size = 0
        for _name, shape, dtype in cls.tensors_shapes_and_dtypes(**kwargs):
            total_size = _align_up(total_size, 16)
            numel = 1
            for shape_dim in shape:
                numel *= shape_dim
            total_size += numel * dtype.itemsize
        return total_size

    @classmethod
    def view_from_buffer(
        cls,
        buffer: torch.Tensor,
        **kwargs: Any,
    ) -> "NvlSharedBuffer":
        if (
            buffer.ndim != 1
            or buffer.dtype != torch.uint8
            or not buffer.is_contiguous()
        ):
            raise ValueError("NvlSharedBuffer requires a contiguous 1D uint8 buffer.")

        offset = 0

        def view_buffer_chunk(
            shape: tuple[int, ...],
            dtype: torch.dtype,
        ) -> torch.Tensor:
            nonlocal offset
            offset = _align_up(offset, 16)
            num_bytes = dtype.itemsize
            for shape_dim in shape:
                num_bytes *= shape_dim
            if offset + num_bytes > buffer.numel():
                raise ValueError("NvlSharedBuffer raw buffer is too small.")
            out = buffer[offset : offset + num_bytes].view(dtype).view(shape)
            offset += num_bytes
            return out

        tensors = {
            name: view_buffer_chunk(shape, dtype)
            for name, shape, dtype in cls.tensors_shapes_and_dtypes(**kwargs)
        }
        return cls(raw=buffer, **tensors)

    def offset_of(self, name: str) -> int:
        return getattr(self, name).data_ptr() - self.raw.data_ptr()


_WorkspaceCacheKey = tuple[str, int | None, int | None, int, int, float]
_FLEX_EP_WORKSPACE_CACHE: dict[_WorkspaceCacheKey, "FlexEPWorkspace"] = {}


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return device.type, device_index


def _require_expert_parallel_mesh(
    ep_mesh: DeviceMesh | None,
) -> tuple[int, int, Any]:
    if ep_mesh is None:
        raise ValueError(
            "FlexGroupedExperts requires expert parallelism "
            "(expert_parallel_degree > 1)."
        )
    ep_size = ep_mesh.size()
    if ep_size <= 1:
        raise ValueError(
            "FlexGroupedExperts requires expert parallelism "
            "(expert_parallel_degree > 1)."
        )
    return ep_size, ep_mesh.get_local_rank(), ep_mesh.get_group()


def _clear_flex_ep_workspace_cache() -> None:
    _FLEX_EP_WORKSPACE_CACHE.clear()


@dataclass
class FlexEPWorkspace:
    device: torch.device
    ep_rank: int
    ep_size: int
    raw: torch.Tensor
    peer_buffers: tuple[torch.Tensor, ...]
    buffers_cuda_ptrs: torch.Tensor
    _views: dict[tuple[int, int, int, int, int, float], NvlSharedBuffer] = field(
        default_factory=dict
    )

    @classmethod
    def get_or_create(
        cls,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        ep_size: int,
        ep_rank: int,
        ep_group: Any,
        capacity_factor: float,
    ) -> "FlexEPWorkspace":
        if ep_size <= 1:
            raise ValueError(
                "FlexGroupedExperts requires expert parallelism "
                "(expert_parallel_degree > 1)."
            )
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size "
                f"({ep_size})."
            )
        if not 0 <= ep_rank < ep_size:
            raise ValueError(f"ep_rank ({ep_rank}) must be in [0, {ep_size}).")
        ep_group_key = id(ep_group)
        device = torch.device(device)
        device_type, device_index = _device_cache_key(device)
        cache_key = (
            device_type,
            device_index,
            ep_group_key,
            ep_rank,
            ep_size,
            capacity_factor,
        )
        nvl_buffer_size = NvlSharedBuffer.get_buffer_size_bytes(
            max_tokens=max_tokens,
            dim=dim,
            ep_size=ep_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )
        workspace = _FLEX_EP_WORKSPACE_CACHE.get(cache_key)
        if workspace is None:
            workspace = cls._allocate(
                nvl_buffer_size,
                max_tokens=max_tokens,
                dim=dim,
                ep_size=ep_size,
                ep_rank=ep_rank,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                device=device,
                ep_group=ep_group,
            )
            _FLEX_EP_WORKSPACE_CACHE[cache_key] = workspace
        elif workspace.raw.numel() < nvl_buffer_size:
            workspace._resize(
                nvl_buffer_size,
                max_tokens=max_tokens,
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                ep_group=ep_group,
            )
        return workspace

    @classmethod
    def _allocate(
        cls,
        nvl_buffer_size: int,
        *,
        max_tokens: int,
        dim: int,
        ep_size: int,
        ep_rank: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
        device: torch.device,
        ep_group: Any,
    ) -> "FlexEPWorkspace":
        raw, peer_buffers, buffers_cuda_ptrs = _allocate_workspace_storage(
            nvl_buffer_size=nvl_buffer_size,
            device=device,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
        )
        workspace = cls(
            device=device,
            ep_rank=ep_rank,
            ep_size=ep_size,
            raw=raw,
            peer_buffers=peer_buffers,
            buffers_cuda_ptrs=buffers_cuda_ptrs,
        )
        workspace._log_allocation(
            nvl_buffer_size,
            max_tokens=max_tokens,
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )
        return workspace

    def _resize(
        self,
        nvl_buffer_size: int,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
        ep_group: Any,
    ) -> None:
        raw, peer_buffers, buffers_cuda_ptrs = _allocate_workspace_storage(
            nvl_buffer_size=nvl_buffer_size,
            device=self.device,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            ep_group=ep_group,
        )
        self.raw = raw
        self.peer_buffers = peer_buffers
        self.buffers_cuda_ptrs = buffers_cuda_ptrs
        self._views.clear()
        self._log_allocation(
            nvl_buffer_size,
            max_tokens=max_tokens,
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )

    def _log_allocation(
        self,
        nvl_buffer_size: int,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
    ) -> None:
        local_experts = num_experts // self.ep_size
        max_recv_tokens = _compute_max_tokens_recv(
            max_tokens,
            self.ep_size,
            num_experts,
            top_k,
            capacity_factor=capacity_factor,
        )
        logger.info(
            "Allocated FlexEP workspace: size_bytes=%s device=%s ep_rank=%s "
            "ep_size=%s max_tokens=%s dim=%s num_experts=%s top_k=%s "
            "local_experts=%s capacity_factor=%s max_recv_tokens=%s",
            nvl_buffer_size,
            self.device,
            self.ep_rank,
            self.ep_size,
            max_tokens,
            dim,
            num_experts,
            top_k,
            local_experts,
            capacity_factor,
            max_recv_tokens,
        )

    def view(
        self,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
    ) -> NvlSharedBuffer:
        view_key = (
            max_tokens,
            dim,
            self.ep_size,
            num_experts,
            top_k,
            capacity_factor,
        )
        view = self._views.get(view_key)
        if view is None:
            view = NvlSharedBuffer.view_from_buffer(
                self.peer_buffers[self.ep_rank],
                max_tokens=max_tokens,
                dim=dim,
                ep_size=self.ep_size,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
            )
            self._views[view_key] = view
        return view


def _allocate_workspace_storage(
    *,
    nvl_buffer_size: int,
    device: torch.device,
    ep_size: int,
    ep_rank: int,
    ep_group: Any,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    if ep_size <= 1:
        raise ValueError(
            "FlexGroupedExperts requires expert parallelism "
            "(expert_parallel_degree > 1)."
        )
    if device.type != "cuda":
        raise ValueError(f"FlexGroupedExperts EP>1 requires CUDA tensors, got {device}.")
    _require_ep_backend_ops()
    try:
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError as e:
        raise ImportError(
            "FlexGroupedExperts EP>1 requires PyTorch symmetric memory."
        ) from e
    if not symm_mem.is_nvshmem_available():
        raise RuntimeError("FlexGroupedExperts EP>1 requires NVSHMEM symmetric memory.")
    symm_mem.set_backend("NVSHMEM")
    allocation_dtype = torch.int64
    raw_storage = symm_mem.empty(
        _align_up(nvl_buffer_size, allocation_dtype.itemsize)
        // allocation_dtype.itemsize,
        device=device,
        dtype=allocation_dtype,
    )
    raw = raw_storage.view(torch.uint8)[:nvl_buffer_size]
    raw.zero_()
    handle = symm_mem.rendezvous(raw_storage, ep_group)
    peer_storages = tuple(
        (
            raw_storage
            if peer == ep_rank
            else handle.get_buffer(peer, raw_storage.shape, raw_storage.dtype)
        )
        for peer in range(ep_size)
    )
    peer_buffers = tuple(
        peer_storage.view(torch.uint8)[:nvl_buffer_size]
        for peer_storage in peer_storages
    )
    dist.barrier(group=ep_group)

    buffer = NvlSharedBuffer.view_from_buffer(
        peer_buffers[ep_rank],
        max_tokens=1,
        dim=1,
        ep_size=ep_size,
        num_experts=ep_size,
        top_k=1,
    )
    buffer.barrier_counter.zero_()
    buffers_cuda_ptrs = torch.tensor(
        [peer_buffer.data_ptr() for peer_buffer in peer_buffers],
        dtype=torch.int64,
        device=device,
    )
    if ep_size > 1:
        _router_barrier(
            buffer.barrier_counter,
            buffer.barrier_counter,
            buffers_cuda_ptrs,
            buffer.offset_of("barrier_counter"),
        )
    return raw, peer_buffers, buffers_cuda_ptrs


def _view_beginning_as(
    x: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    num_bytes = dtype.itemsize
    for shape_dim in shape:
        num_bytes *= shape_dim
    return x.view(-1).view(torch.uint8)[:num_bytes].view(dtype).view(shape)


def _router_barrier(
    dependency: torch.Tensor,
    barrier_counter: torch.Tensor,
    buffers_cuda_ptrs: torch.Tensor,
    offs_barrier_counter: int,
    *,
    nonce: int = 0,
    clone_result: bool = False,
) -> torch.Tensor:
    value = torch.ops._flex_ep.barrier_arrive(
        barrier_counter[:1],
        dependency,
        nonce,
    )
    if clone_result:
        return torch.ops._flex_ep.barrier_wait(
            dependency,
            buffers_cuda_ptrs,
            offs_barrier_counter,
            value,
            EP_TIMEOUT_SECONDS,
        )
    torch.ops._flex_ep.barrier_wait_no_clone(
        dependency,
        buffers_cuda_ptrs,
        offs_barrier_counter,
        value,
        EP_TIMEOUT_SECONDS,
    )
    return dependency


def _weighted_sum_reference(
    y_partial: torch.Tensor,
    top_scores: torch.Tensor,
) -> torch.Tensor:
    return (
        (y_partial.to(torch.float32) * top_scores.to(torch.float32).unsqueeze(-1))
        .sum(dim=1)
        .to(y_partial.dtype)
    )


class _FlexEPWeightedSum(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any,
        y_partial: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        if (
            y_partial.is_cuda
            and top_scores.is_cuda
            and y_partial.dtype == torch.bfloat16
            and top_scores.dtype == torch.float32
            and y_partial.is_contiguous()
            and top_scores.is_contiguous()
        ):
            _require_ep_backend_ops()
            out = torch.ops.inductor.flex_ep_weighted_sum_forward(
                y_partial,
                top_scores,
            )
        else:
            out = _weighted_sum_reference(y_partial, top_scores)
        ctx.save_for_backward(y_partial.clone(), top_scores)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_partial, top_scores = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if (
            grad_out.is_cuda
            and y_partial.is_cuda
            and top_scores.is_cuda
            and grad_out.dtype == torch.bfloat16
            and y_partial.dtype == torch.bfloat16
            and top_scores.dtype == torch.float32
            and y_partial.is_contiguous()
            and top_scores.is_contiguous()
        ):
            _require_ep_backend_ops()
            return torch.ops.inductor.flex_ep_weighted_sum_backward(
                grad_out,
                y_partial,
                top_scores,
            )
        grad_out_fp32 = grad_out.to(torch.float32)
        top_scores_fp32 = top_scores.to(torch.float32)
        y_partial_fp32 = y_partial.to(torch.float32)
        grad_y_partial = (
            grad_out_fp32.unsqueeze(1) * top_scores_fp32.unsqueeze(-1)
        ).to(y_partial.dtype)
        grad_top_scores = (grad_out_fp32.unsqueeze(1) * y_partial_fp32).sum(dim=-1)
        return grad_y_partial, grad_top_scores


def flex_ep_weighted_sum(
    y_partial: torch.Tensor,
    top_scores: torch.Tensor,
) -> torch.Tensor:
    return _FlexEPWeightedSum.apply(y_partial, top_scores)


RouterFns = tuple[
    Callable[..., FlexEPDispatchPlan],
    Callable[..., Any],
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
]


class FlexEPRouter:
    def __init__(
        self,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
        workspace: FlexEPWorkspace,
        num_ctas: int = DEFAULT_NUM_CTAS,
        capacity_factor: float = 1.0,
    ) -> None:
        self.max_tokens = max_tokens
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.num_ctas = num_ctas
        self.workspace = workspace
        self.capacity_factor = capacity_factor
        self.router_fns = self._make_router_fns()

    @property
    def raw(self) -> torch.Tensor:
        return self.workspace.raw

    @property
    def buffers_cuda_ptrs(self) -> torch.Tensor:
        return self.workspace.buffers_cuda_ptrs

    @property
    def peer_buffers(self) -> tuple[torch.Tensor, ...]:
        return self.workspace.peer_buffers

    @classmethod
    def create(
        cls,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        ep_mesh: DeviceMesh,
        num_ctas: int = DEFAULT_NUM_CTAS,
        capacity_factor: float = 1.0,
    ) -> "FlexEPRouter":
        capacity_factor = _validate_flex_ep_capacity_factor(capacity_factor)
        ep_size, ep_rank, ep_group = _require_expert_parallel_mesh(ep_mesh)
        _ensure_flex_ep_imported()
        workspace = FlexEPWorkspace.get_or_create(
            max_tokens=max_tokens,
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            capacity_factor=capacity_factor,
        )
        return cls(
            max_tokens=max_tokens,
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            ep_rank=ep_rank,
            ep_size=ep_size,
            workspace=workspace,
            num_ctas=num_ctas,
            capacity_factor=capacity_factor,
        )

    @property
    def router_operands(self) -> FlexEPRouterOperands:
        buffer = self.workspace.view(
            max_tokens=self.max_tokens,
            dim=self.dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
        )
        return FlexEPRouterOperands(
            self.workspace.raw,
            self.workspace.buffers_cuda_ptrs,
            buffer.offset_of("barrier_counter"),
            buffer.offset_of("dispatch_recv_buffer"),
            buffer.offset_of("dispatch_recv_buffer_scaling_factors"),
            buffer.offset_of("dispatch_recv_weights"),
            buffer.offset_of("dispatch_recv_origin_global_token_id"),
            buffer.offset_of("combine_recv_buffer"),
            buffer.offset_of("combine_recv_scale_factors"),
            buffer.offset_of("combine_recv_weights"),
            buffer.offset_of("allgather_expert_counts"),
            self.ep_rank,
        )

    def _view_buffer(self, raw: torch.Tensor) -> NvlSharedBuffer:
        return NvlSharedBuffer.view_from_buffer(
            raw,
            max_tokens=self.max_tokens,
            dim=self.dim,
            ep_size=self.ep_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
        )

    def _make_router_fns(self) -> RouterFns:
        local_experts = self.num_experts // self.ep_size
        max_recv_tokens = _compute_max_tokens_recv(
            self.max_tokens,
            self.ep_size,
            self.num_experts,
            self.top_k,
            capacity_factor=self.capacity_factor,
        )

        def build_dispatch_plan_fn(
            topk_idx,
            operands,
        ):
            buffer = self._view_buffer(operands.raw)
            barrier_counter = buffer.barrier_counter
            dispatch_recv_origin_global_token_id = (
                buffer.dispatch_recv_origin_global_token_id
            )
            allgather_expert_counts = buffer.allgather_expert_counts

            dispatch_recv_origin_global_token_id = _router_barrier(
                dispatch_recv_origin_global_token_id,
                barrier_counter,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
            )

            dispatch_recv_origin_global_token_id = torch.ops._flex_ep.fill_i64_inplace(
                dispatch_recv_origin_global_token_id,
                -1,
            )
            expert_count_buffer = torch.zeros(
                self.num_experts,
                dtype=torch.int64,
                device=topk_idx.device,
            )
            expert_count_buffer.scatter_(
                0,
                topk_idx.flatten().to(torch.int64),
                1,
                reduce="add",
            )
            allgather_expert_counts = torch.ops._flex_ep.ep_allgather(
                allgather_expert_counts,
                expert_count_buffer,
                operands.buffers_cuda_ptrs,
                operands.offs_allgather_expert_counts,
                operands.ep_rank,
            )
            allgather_expert_counts = _router_barrier(
                allgather_expert_counts,
                barrier_counter,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
                nonce=1,
                clone_result=True,
            )

            (
                all_offsets,
                recv_total_tokens,
                local_experts_start,
            ) = torch.ops._flex_ep.router_compute_all_expert_offsets(
                allgather_expert_counts,
                operands.ep_rank,
                local_experts,
                TOKEN_ALIGNMENT,
            )
            _validate_flex_ep_capacity(
                local_experts_start,
                max_recv_tokens,
                capacity_factor=self.capacity_factor,
            )
            expert_begin_offset = all_offsets[operands.ep_rank]
            recv_ofs = all_offsets[:, :, operands.ep_rank].reshape(-1)
            dest_ranks, dest_offsets = torch.ops._flex_ep.router_compute_dest_offsets(
                topk_idx,
                recv_ofs,
                self.ep_size,
            )
            max_recv_tokens_tensor = torch.full(
                (),
                max_recv_tokens,
                device=topk_idx.device,
                dtype=torch.int32,
            )
            overflow = local_experts_start[-1] > max_recv_tokens_tensor
            return FlexEPDispatchPlan(
                dispatch_recv_origin_global_token_id[:max_recv_tokens],
                expert_begin_offset,
                dest_ranks,
                dest_offsets,
                local_experts_start,
                max_recv_tokens_tensor,
                recv_total_tokens,
                overflow,
            )

        def dispatch_fn(
            x_expanded,
            plan,
            operands,
        ):
            recv_origin_global_token_id = plan.recv_origin_global_token_id
            expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
            dest_ranks = plan.dest_ranks
            dest_offsets = plan.dest_offsets
            local_experts_start = plan.local_experts_start

            buffer = self._view_buffer(operands.raw)
            barrier_counter = buffer.barrier_counter
            dispatch_recv_buffer = buffer.dispatch_recv_buffer
            dispatch_recv_buffer_scaling_factors = (
                buffer.dispatch_recv_buffer_scaling_factors
            )
            dispatch_recv_weights = buffer.dispatch_recv_weights

            (
                dispatch_recv_buffer,
                dispatch_recv_buffer_scaling_factors,
                recv_origin_global_token_id,
                dispatch_recv_weights,
            ) = torch.ops._flex_ep.router_dispatch(
                x_expanded,
                None,
                None,
                dest_ranks,
                dest_offsets,
                operands.buffers_cuda_ptrs,
                dispatch_recv_buffer,
                dispatch_recv_buffer_scaling_factors,
                recv_origin_global_token_id,
                dispatch_recv_weights,
                operands.offs_dispatch_recv_buffer,
                operands.offs_dispatch_recv_buffer_scaling_factors,
                operands.offs_dispatch_recv_weights,
                operands.offs_dispatch_recv_origin_global_token_id,
                operands.ep_rank,
                self.num_ctas,
                self.max_tokens,
            )
            barrier = torch.ops._flex_ep.barrier_arrive(
                barrier_counter[:1],
                dispatch_recv_buffer,
                2,
            )
            recv_x = _view_beginning_as(
                dispatch_recv_buffer,
                (max_recv_tokens, x_expanded.shape[-1]),
                x_expanded.dtype,
            )
            recv_x_u8 = torch.ops._flex_ep.zfill_ranges_inplace(
                recv_x.view(torch.uint8),
                expert_begin_offset_per_ep[:, -1].contiguous(),
                local_experts_start[1:].contiguous(),
                TOKEN_ALIGNMENT,
            )
            recv_x = recv_x_u8.view(recv_x.dtype).view(recv_x.shape)
            recv_x_u8 = torch.ops._flex_ep.barrier_wait_no_clone(
                recv_x_u8,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
                barrier,
                EP_TIMEOUT_SECONDS,
            )
            recv_x = recv_x_u8.view(recv_x.dtype).view(recv_x.shape)
            return recv_x

        def combine_fn(
            y3,
            plan,
            operands,
        ):
            recv_origin_global_token_id = plan.recv_origin_global_token_id
            expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
            local_experts_start = plan.local_experts_start

            buffer = self._view_buffer(operands.raw)
            barrier_counter = buffer.barrier_counter
            combine_recv_buffer = buffer.combine_recv_buffer
            combine_recv_scale_factors = buffer.combine_recv_scale_factors
            combine_recv_weights = buffer.combine_recv_weights

            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
            )
            (
                combine_recv_buffer,
                combine_recv_scale_factors,
                combine_recv_weights,
            ) = torch.ops._flex_ep.router_combine(
                y3,
                None,
                None,
                expert_begin_offset_per_ep,
                local_experts_start[-1:].to(torch.int64),
                recv_origin_global_token_id,
                operands.buffers_cuda_ptrs,
                combine_recv_buffer,
                combine_recv_scale_factors,
                combine_recv_weights,
                operands.offs_combine_recv_buffer,
                operands.offs_combine_recv_scale_factors,
                operands.offs_combine_recv_weights,
                operands.ep_rank,
                self.max_tokens,
                self.top_k,
                self.num_ctas,
                self.max_tokens,
            )
            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
                nonce=1,
            )
            return _view_beginning_as(
                combine_recv_buffer,
                (self.max_tokens, self.top_k, y3.shape[-1]),
                y3.dtype,
            )

        def combine_bwd_fn(
            dy,
            plan,
            operands,
        ):
            expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
            dest_ranks = plan.dest_ranks
            dest_offsets = plan.dest_offsets
            local_experts_start = plan.local_experts_start

            buffer = self._view_buffer(operands.raw)
            barrier_counter = buffer.barrier_counter
            dispatch_recv_buffer = buffer.dispatch_recv_buffer
            dispatch_recv_buffer_scaling_factors = (
                buffer.dispatch_recv_buffer_scaling_factors
            )
            dispatch_recv_weights = buffer.dispatch_recv_weights
            dispatch_recv_origin_global_token_id = (
                buffer.dispatch_recv_origin_global_token_id
            )

            dispatch_recv_buffer = _router_barrier(
                dispatch_recv_buffer,
                barrier_counter,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
            )
            (
                dispatch_recv_buffer,
                dispatch_recv_buffer_scaling_factors,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
            ) = torch.ops._flex_ep.router_dispatch(
                dy.contiguous(),
                None,
                None,
                dest_ranks,
                dest_offsets,
                operands.buffers_cuda_ptrs,
                dispatch_recv_buffer,
                dispatch_recv_buffer_scaling_factors,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
                operands.offs_dispatch_recv_buffer,
                operands.offs_dispatch_recv_buffer_scaling_factors,
                operands.offs_dispatch_recv_weights,
                -1,
                operands.ep_rank,
                self.num_ctas,
                self.max_tokens,
            )
            barrier = torch.ops._flex_ep.barrier_arrive(
                barrier_counter[:1],
                dispatch_recv_buffer,
                1,
            )
            dy3 = _view_beginning_as(
                dispatch_recv_buffer,
                (max_recv_tokens, dy.shape[-1]),
                dy.dtype,
            )
            dy3_u8 = torch.ops._flex_ep.zfill_ranges_inplace(
                dy3.view(torch.uint8),
                expert_begin_offset_per_ep[:, -1].contiguous(),
                local_experts_start[1:].contiguous(),
                TOKEN_ALIGNMENT,
            )
            dy3 = dy3_u8.view(dy3.dtype).view(dy3.shape)
            dy3_u8 = torch.ops._flex_ep.barrier_wait_no_clone(
                dy3_u8,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
                barrier,
                EP_TIMEOUT_SECONDS,
            )
            dy3 = dy3_u8.view(dy3.dtype).view(dy3.shape)
            return dy3

        def dispatch_bwd_fn(
            dx_recv,
            plan,
            operands,
        ):
            recv_origin_global_token_id = plan.recv_origin_global_token_id
            expert_begin_offset_per_ep = plan.expert_begin_offset_per_ep
            local_experts_start = plan.local_experts_start

            buffer = self._view_buffer(operands.raw)
            barrier_counter = buffer.barrier_counter
            combine_recv_buffer = buffer.combine_recv_buffer
            combine_recv_scale_factors = buffer.combine_recv_scale_factors
            combine_recv_weights = buffer.combine_recv_weights

            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
            )
            (
                combine_recv_buffer,
                combine_recv_scale_factors,
                combine_recv_weights,
            ) = torch.ops._flex_ep.router_combine(
                dx_recv,
                None,
                None,
                expert_begin_offset_per_ep,
                local_experts_start[-1:].to(torch.int64),
                recv_origin_global_token_id,
                operands.buffers_cuda_ptrs,
                combine_recv_buffer,
                combine_recv_scale_factors,
                combine_recv_weights,
                operands.offs_combine_recv_buffer,
                operands.offs_combine_recv_scale_factors,
                operands.offs_combine_recv_weights,
                operands.ep_rank,
                self.max_tokens,
                self.top_k,
                self.num_ctas,
                self.max_tokens,
            )
            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                operands.buffers_cuda_ptrs,
                operands.offs_barrier_counter,
                nonce=1,
            )
            return _view_beginning_as(
                combine_recv_buffer,
                (self.max_tokens, self.top_k, dx_recv.shape[-1]),
                dx_recv.dtype,
            )

        return (
            build_dispatch_plan_fn,
            dispatch_fn,
            combine_fn,
            combine_bwd_fn,
            dispatch_bwd_fn,
        )
