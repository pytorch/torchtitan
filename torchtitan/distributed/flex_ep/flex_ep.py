# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Router helpers for PyTorch ``flex_ep`` MoE execution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributed.tensor import DeviceMesh

DEFAULT_NUM_CTAS = 152
EP_TIMEOUT_SECONDS = 30.0
TOKEN_ALIGNMENT = 128
logger = logging.getLogger(__name__)

_REQUIRED_EP_BACKEND_OPS = (
    "flex_ep_allgather",
    "flex_ep_router_dispatch",
    "flex_ep_router_combine",
    "flex_ep_barrier_arrive",
    "flex_ep_barrier_wait",
    "flex_ep_zfill_ranges_inplace",
)


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
        import triton
        import triton.language as tl
        import torch.distributed._symmetric_memory._shmem_triton as shmem_triton

        from torch.distributed._symmetric_memory._shmem_triton import requires_shmem
    except ImportError as e:
        raise ImportError(
            "FlexGroupedExperts EP>1 requires Triton symmetric-memory kernels."
        ) from e

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
                done = 1
            else:
                now = tl.inline_asm_elementwise(
                    "mov.u64 $0, %globaltimer;",
                    "=l",
                    [],
                    dtype=tl.int64,
                    is_pure=False,
                    pack=1,
                )
                if (now - start) > timeout_ns:
                    tl.device_print("flex_ep barrier_wait timed out")
                    tl.inline_asm_elementwise(
                        "trap;",
                        "=r",
                        [],
                        dtype=tl.int32,
                        is_pure=False,
                        pack=1,
                    )
                    done = 1

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
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
        WRITE_MAPPING: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        token_id = copy_id // TOPK
        expert_slot = copy_id - token_id * TOPK
        dest_rank = tl.load(dest_ranks_ptr + copy_id)
        dest_offset = tl.load(dest_offsets_ptr + copy_id)
        valid = dest_rank >= 0

        src_ptr = my_tokens_ptr + copy_id * D_BYTES
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
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
        D_BYTES: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        copy_id = tl.program_id(0)
        token_send_end = tl.load(token_send_end_ptr)
        valid = copy_id < token_send_end
        origin_id = tl.load(
            send_origin_global_token_id_ptr + copy_id,
            mask=valid,
            other=-1,
        )
        valid = valid & (origin_id >= 0)

        max_B_topk = max_B * TOPK
        from_ep_rank = origin_id // max_B_topk
        dest_idx = origin_id - from_ep_rank * max_B_topk
        valid = valid & (from_ep_rank >= 0) & (from_ep_rank < ep_size)

        src_ptr = send_tokens_ptr + copy_id * D_BYTES
        base_ptr = tl.load(buffers_cuda_ptrs_ptr + ep_rank)
        dst_ptr = (base_ptr + offs_combine_recv_tokens + dest_idx * D_BYTES).to(
            tl.pointer_type(tl.uint8)
        )
        if valid:
            shmem_backend.put(dst_ptr, src_ptr, D_BYTES, from_ep_rank.to(tl.int32))
        shmem_backend.quiet()

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

    if not hasattr(torch.ops.inductor, "flex_ep_barrier_arrive"):

        @torch.library.custom_op(
            "inductor::flex_ep_barrier_arrive",
            mutates_args=("flag",),
        )
        def _flex_ep_barrier_arrive(flag: torch.Tensor) -> torch.Tensor:
            out = torch.empty(1, dtype=torch.int32, device=flag.device)
            _barrier_arrive_kernel[(1,)](flag, out, num_warps=1)
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
            _barrier_wait_kernel[(1,)](
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
            _ep_allgather_kernel[(buffers_cuda_ptrs.numel(),)](
                input_u8,
                buffers_cuda_ptrs,
                offs_output,
                ep_rank,
                input_num_bytes=input_u8.numel(),
                num_warps=1,
            )

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
            del offs_recv_scaling_factors, offs_recv_weights, num_ctas
            if my_tokens.dtype != torch.bfloat16:
                raise ValueError("flex_ep_router_dispatch supports BF16 tokens only.")
            if not my_tokens.is_contiguous():
                raise ValueError("flex_ep_router_dispatch requires contiguous tokens.")
            batch, top_k, dim = my_tokens.shape
            my_tokens_u8 = my_tokens.view(torch.uint8)
            dim_bytes = dim * my_tokens.element_size()
            _router_dispatch_kernel[(batch * top_k,)](
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
            del batch, num_ctas
            if send_tokens.dtype != torch.bfloat16:
                raise ValueError("flex_ep_router_combine supports BF16 tokens only.")
            if not send_tokens.is_contiguous():
                raise ValueError("flex_ep_router_combine requires contiguous tokens.")
            dim = send_tokens.shape[-1]
            send_tokens_u8 = send_tokens.view(torch.uint8)
            dim_bytes = dim * send_tokens.element_size()
            _router_combine_kernel[(send_tokens.shape[0],)](
                send_tokens_u8,
                token_send_end,
                send_origin_global_token_id,
                buffers_cuda_ptrs,
                offs_combine_recv_tokens,
                ep_rank,
                buffers_cuda_ptrs.numel(),
                max_B,
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
            _zfill_ranges_kernel[(begin_ofs.numel(), max_values_per_batch)](
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


def _compute_max_tokens_recv(
    max_tokens: int,
    ep_size: int,
    num_experts: int,
    top_k: int,
) -> int:
    local_experts = num_experts // ep_size
    capacity_per_local_expert = _align_up(
        ep_size * ((max_tokens * top_k + num_experts - 1) // num_experts),
        TOKEN_ALIGNMENT,
    )
    return capacity_per_local_expert * local_experts


def _validate_balanced_routing_capacity(
    local_experts_start: torch.Tensor,
    max_recv_tokens: int,
) -> None:
    num_recv_tokens = int(local_experts_start[-1].item())
    if num_recv_tokens > max_recv_tokens:
        raise ValueError(
            "FlexGroupedExperts flex_ep workspace capacity was exceeded. "
            "This v1 path requires balanced routing; pass "
            "--debug.moe_force_load_balance for DeepSeekV3 FlexEP testing. "
            f"Received {num_recv_tokens} local expert tokens, but the balanced "
            f"workspace capacity is {max_recv_tokens}."
        )


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


_WorkspaceCacheKey = tuple[str, int | None, int | None, int, int]
_FLEX_EP_WORKSPACE_CACHE: dict[_WorkspaceCacheKey, "FlexEPWorkspace"] = {}


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return device.type, device_index


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
    _views: dict[tuple[int, int, int, int, int], NvlSharedBuffer] = field(
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
        ep_mesh: DeviceMesh | None,
    ) -> "FlexEPWorkspace":
        ep_size = 1 if ep_mesh is None else ep_mesh.size()
        ep_rank = 0 if ep_mesh is None else ep_mesh.get_local_rank()
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size "
                f"({ep_size})."
            )

        if ep_size == 1:
            ep_group = None
            ep_group_key = None
        else:
            assert ep_mesh is not None
            ep_group = ep_mesh.get_group()
            ep_group_key = id(ep_group)
        device = torch.device(device)
        device_type, device_index = _device_cache_key(device)
        cache_key = (device_type, device_index, ep_group_key, ep_rank, ep_size)
        nvl_buffer_size = NvlSharedBuffer.get_buffer_size_bytes(
            max_tokens=max_tokens,
            dim=dim,
            ep_size=ep_size,
            num_experts=num_experts,
            top_k=top_k,
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
        )

    def _log_allocation(
        self,
        nvl_buffer_size: int,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        local_experts = num_experts // self.ep_size
        capacity_per_local_expert = (
            _compute_max_tokens_recv(
                max_tokens,
                self.ep_size,
                num_experts,
                top_k,
            )
            // local_experts
        )
        logger.info(
            "Allocated FlexEP workspace: size_bytes=%s device=%s ep_rank=%s "
            "ep_size=%s max_tokens=%s dim=%s num_experts=%s top_k=%s "
            "local_experts=%s balanced_capacity_per_local_expert=%s",
            nvl_buffer_size,
            self.device,
            self.ep_rank,
            self.ep_size,
            max_tokens,
            dim,
            num_experts,
            top_k,
            local_experts,
            capacity_per_local_expert,
        )

    def view(
        self,
        *,
        max_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
    ) -> NvlSharedBuffer:
        view_key = (max_tokens, dim, self.ep_size, num_experts, top_k)
        view = self._views.get(view_key)
        if view is None:
            view = NvlSharedBuffer.view_from_buffer(
                self.peer_buffers[self.ep_rank],
                max_tokens=max_tokens,
                dim=dim,
                ep_size=self.ep_size,
                num_experts=num_experts,
                top_k=top_k,
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
    if ep_size == 1:
        raw = torch.empty(nvl_buffer_size, device=device, dtype=torch.uint8)
        raw.zero_()
        peer_buffers = (raw,)
    else:
        if device.type != "cuda":
            raise ValueError(
                f"FlexGroupedExperts EP>1 requires CUDA tensors, got {device}."
            )
        _require_ep_backend_ops()
        try:
            import torch.distributed as dist
            import torch.distributed._symmetric_memory as symm_mem
        except ImportError as e:
            raise ImportError(
                "FlexGroupedExperts EP>1 requires PyTorch symmetric memory."
            ) from e
        if not symm_mem.is_nvshmem_available():
            raise RuntimeError(
                "FlexGroupedExperts EP>1 requires NVSHMEM symmetric memory."
            )
        symm_mem.set_backend("NVSHMEM")
        raw = symm_mem.empty(nvl_buffer_size, device=device, dtype=torch.uint8)
        raw.zero_()
        handle = symm_mem.rendezvous(raw, ep_group)
        peer_buffers = tuple(
            (raw if peer == ep_rank else handle.get_buffer(peer, raw.shape, raw.dtype))
            for peer in range(ep_size)
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
    waited = torch.ops._flex_ep.barrier_wait(
        dependency,
        buffers_cuda_ptrs,
        offs_barrier_counter,
        value,
        EP_TIMEOUT_SECONDS,
    )
    if clone_result:
        return waited
    return dependency


RouterFns = tuple[
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
    ) -> None:
        self.max_tokens = max_tokens
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.num_ctas = num_ctas
        self.workspace = workspace
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
        ep_mesh: DeviceMesh | None,
        num_ctas: int = DEFAULT_NUM_CTAS,
    ) -> "FlexEPRouter":
        _ensure_flex_ep_imported()
        ep_size = 1 if ep_mesh is None else ep_mesh.size()
        ep_rank = 0 if ep_mesh is None else ep_mesh.get_local_rank()
        workspace = FlexEPWorkspace.get_or_create(
            max_tokens=max_tokens,
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            device=device,
            ep_mesh=ep_mesh,
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
        )

    @property
    def router_operands(self) -> tuple[Any, ...]:
        buffer = self.workspace.view(
            max_tokens=self.max_tokens,
            dim=self.dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        return (
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
        )

    def _make_router_fns(self) -> RouterFns:
        local_experts = self.num_experts // self.ep_size
        max_recv_tokens = _compute_max_tokens_recv(
            self.max_tokens,
            self.ep_size,
            self.num_experts,
            self.top_k,
        )

        def dispatch_fn(
            x_expanded,
            topk_idx,
            raw,
            buffers_cuda_ptrs,
            offs_barrier_counter,
            offs_dispatch_recv_buffer,
            offs_dispatch_recv_buffer_scaling_factors,
            offs_dispatch_recv_weights,
            offs_dispatch_recv_origin_global_token_id,
            offs_combine_recv_buffer,
            offs_combine_recv_scale_factors,
            offs_combine_recv_weights,
            offs_allgather_expert_counts,
            ep_rank,
        ):
            del offs_combine_recv_buffer
            del offs_combine_recv_scale_factors, offs_combine_recv_weights

            buffer = self._view_buffer(raw)
            barrier_counter = buffer.barrier_counter
            dispatch_recv_buffer = buffer.dispatch_recv_buffer
            dispatch_recv_buffer_scaling_factors = (
                buffer.dispatch_recv_buffer_scaling_factors
            )
            dispatch_recv_weights = buffer.dispatch_recv_weights
            dispatch_recv_origin_global_token_id = (
                buffer.dispatch_recv_origin_global_token_id
            )
            allgather_expert_counts = buffer.allgather_expert_counts

            dispatch_recv_origin_global_token_id = _router_barrier(
                dispatch_recv_origin_global_token_id,
                barrier_counter,
                buffers_cuda_ptrs,
                offs_barrier_counter,
            )

            dispatch_recv_origin_global_token_id = torch.ops._flex_ep.fill_i64_inplace(
                dispatch_recv_origin_global_token_id,
                -1,
            )
            expert_count_buffer = torch.zeros(
                self.num_experts,
                dtype=torch.int64,
                device=x_expanded.device,
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
                buffers_cuda_ptrs,
                offs_allgather_expert_counts,
                ep_rank,
            )
            allgather_expert_counts = _router_barrier(
                allgather_expert_counts,
                barrier_counter,
                buffers_cuda_ptrs,
                offs_barrier_counter,
                nonce=1,
                clone_result=True,
            )

            all_offsets, recv_total_tokens, local_experts_start = (
                torch.ops._flex_ep.router_compute_all_expert_offsets(
                    allgather_expert_counts,
                    ep_rank,
                    local_experts,
                    TOKEN_ALIGNMENT,
                )
            )
            _validate_balanced_routing_capacity(
                local_experts_start,
                max_recv_tokens,
            )
            expert_begin_offset = all_offsets[ep_rank]
            recv_ofs = all_offsets[:, :, ep_rank].reshape(-1)
            dest_ranks, dest_offsets = torch.ops._flex_ep.router_compute_dest_offsets(
                topk_idx,
                recv_ofs,
                self.ep_size,
            )

            (
                dispatch_recv_buffer,
                dispatch_recv_buffer_scaling_factors,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
            ) = torch.ops._flex_ep.router_dispatch(
                x_expanded,
                None,
                None,
                dest_ranks,
                dest_offsets,
                buffers_cuda_ptrs,
                dispatch_recv_buffer,
                dispatch_recv_buffer_scaling_factors,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
                offs_dispatch_recv_buffer,
                offs_dispatch_recv_buffer_scaling_factors,
                offs_dispatch_recv_weights,
                offs_dispatch_recv_origin_global_token_id,
                ep_rank,
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
                expert_begin_offset[:, -1],
                local_experts_start[1:],
                TOKEN_ALIGNMENT,
            )
            recv_x = recv_x_u8.view(recv_x.dtype).view(recv_x.shape)
            recv_x_u8 = torch.ops._flex_ep.barrier_wait(
                recv_x_u8,
                buffers_cuda_ptrs,
                offs_barrier_counter,
                barrier,
                EP_TIMEOUT_SECONDS,
            )
            recv_x = recv_x_u8.view(recv_x.dtype).view(recv_x.shape)
            return (
                recv_x,
                dispatch_recv_origin_global_token_id[:max_recv_tokens].clone(),
                expert_begin_offset,
                dest_ranks,
                dest_offsets,
                torch.full(
                    (),
                    max_recv_tokens,
                    device=x_expanded.device,
                    dtype=torch.int32,
                ),
                recv_total_tokens,
                local_experts_start,
                self.max_tokens,
            )

        def combine_fn(
            y3,
            recv_origin_global_token_id,
            expert_begin_offset_per_ep,
            dest_ranks,
            dest_offsets,
            max_recv_tokens_tensor,
            recv_total_tokens,
            local_experts_start,
            batch_size,
            raw,
            buffers_cuda_ptrs,
            offs_barrier_counter,
            offs_dispatch_recv_buffer,
            offs_dispatch_recv_buffer_scaling_factors,
            offs_dispatch_recv_weights,
            offs_dispatch_recv_origin_global_token_id,
            offs_combine_recv_buffer,
            offs_combine_recv_scale_factors,
            offs_combine_recv_weights,
            offs_allgather_expert_counts,
            ep_rank,
        ):
            del dest_ranks, dest_offsets, max_recv_tokens_tensor
            del recv_total_tokens, offs_dispatch_recv_buffer
            del offs_dispatch_recv_buffer_scaling_factors
            del offs_dispatch_recv_weights, offs_dispatch_recv_origin_global_token_id
            del offs_allgather_expert_counts

            buffer = self._view_buffer(raw)
            barrier_counter = buffer.barrier_counter
            combine_recv_buffer = buffer.combine_recv_buffer
            combine_recv_scale_factors = buffer.combine_recv_scale_factors
            combine_recv_weights = buffer.combine_recv_weights

            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                buffers_cuda_ptrs,
                offs_barrier_counter,
            )
            valid_send_tokens = recv_origin_global_token_id[: y3.shape[0]] >= 0
            y3 = y3.masked_fill(~valid_send_tokens.unsqueeze(-1), 0)
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
                buffers_cuda_ptrs,
                combine_recv_buffer,
                combine_recv_scale_factors,
                combine_recv_weights,
                offs_combine_recv_buffer,
                offs_combine_recv_scale_factors,
                offs_combine_recv_weights,
                ep_rank,
                batch_size,
                self.top_k,
                self.num_ctas,
                self.max_tokens,
            )
            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                buffers_cuda_ptrs,
                offs_barrier_counter,
                nonce=1,
                clone_result=True,
            )
            return _view_beginning_as(
                combine_recv_buffer,
                (batch_size, self.top_k, y3.shape[-1]),
                y3.dtype,
            )

        def combine_bwd_fn(
            dy,
            recv_origin_global_token_id,
            expert_begin_offset_per_ep,
            dest_ranks,
            dest_offsets,
            max_recv_tokens_tensor,
            recv_total_tokens,
            local_experts_start,
            batch_size,
            raw,
            buffers_cuda_ptrs,
            offs_barrier_counter,
            offs_dispatch_recv_buffer,
            offs_dispatch_recv_buffer_scaling_factors,
            offs_dispatch_recv_weights,
            offs_dispatch_recv_origin_global_token_id,
            offs_combine_recv_buffer,
            offs_combine_recv_scale_factors,
            offs_combine_recv_weights,
            offs_allgather_expert_counts,
            ep_rank,
        ):
            del recv_origin_global_token_id
            del max_recv_tokens_tensor, recv_total_tokens
            del offs_combine_recv_buffer
            del offs_combine_recv_scale_factors, offs_combine_recv_weights
            del offs_allgather_expert_counts

            buffer = self._view_buffer(raw)
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
                buffers_cuda_ptrs,
                offs_barrier_counter,
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
                buffers_cuda_ptrs,
                dispatch_recv_buffer,
                dispatch_recv_buffer_scaling_factors,
                dispatch_recv_origin_global_token_id,
                dispatch_recv_weights,
                offs_dispatch_recv_buffer,
                offs_dispatch_recv_buffer_scaling_factors,
                offs_dispatch_recv_weights,
                -1,
                ep_rank,
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
                expert_begin_offset_per_ep[:, -1],
                local_experts_start[1:],
                TOKEN_ALIGNMENT,
            )
            dy3 = dy3_u8.view(dy3.dtype).view(dy3.shape)
            dy3_u8 = torch.ops._flex_ep.barrier_wait(
                dy3_u8,
                buffers_cuda_ptrs,
                offs_barrier_counter,
                barrier,
                EP_TIMEOUT_SECONDS,
            )
            dy3 = dy3_u8.view(dy3.dtype).view(dy3.shape)
            return dy3

        def dispatch_bwd_fn(
            dx_recv,
            recv_origin_global_token_id,
            expert_begin_offset_per_ep,
            dest_ranks,
            dest_offsets,
            max_recv_tokens_tensor,
            recv_total_tokens,
            local_experts_start,
            batch_size,
            raw,
            buffers_cuda_ptrs,
            offs_barrier_counter,
            offs_dispatch_recv_buffer,
            offs_dispatch_recv_buffer_scaling_factors,
            offs_dispatch_recv_weights,
            offs_dispatch_recv_origin_global_token_id,
            offs_combine_recv_buffer,
            offs_combine_recv_scale_factors,
            offs_combine_recv_weights,
            offs_allgather_expert_counts,
            ep_rank,
        ):
            del dest_ranks, dest_offsets, max_recv_tokens_tensor
            del recv_total_tokens, local_experts_start
            del offs_dispatch_recv_buffer
            del offs_dispatch_recv_buffer_scaling_factors
            del offs_dispatch_recv_weights, offs_dispatch_recv_origin_global_token_id
            del offs_allgather_expert_counts

            buffer = self._view_buffer(raw)
            barrier_counter = buffer.barrier_counter
            combine_recv_buffer = buffer.combine_recv_buffer
            combine_recv_scale_factors = buffer.combine_recv_scale_factors
            combine_recv_weights = buffer.combine_recv_weights

            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                buffers_cuda_ptrs,
                offs_barrier_counter,
            )
            valid_send_tokens = recv_origin_global_token_id[: dx_recv.shape[0]] >= 0
            dx_recv = dx_recv.masked_fill(~valid_send_tokens.unsqueeze(-1), 0)
            (
                combine_recv_buffer,
                combine_recv_scale_factors,
                combine_recv_weights,
            ) = torch.ops._flex_ep.router_combine(
                dx_recv,
                None,
                None,
                expert_begin_offset_per_ep,
                expert_begin_offset_per_ep[:, -1].max().view(1).to(torch.int64),
                recv_origin_global_token_id,
                buffers_cuda_ptrs,
                combine_recv_buffer,
                combine_recv_scale_factors,
                combine_recv_weights,
                offs_combine_recv_buffer,
                offs_combine_recv_scale_factors,
                offs_combine_recv_weights,
                ep_rank,
                batch_size,
                self.top_k,
                self.num_ctas,
                self.max_tokens,
            )
            combine_recv_buffer = _router_barrier(
                combine_recv_buffer,
                barrier_counter,
                buffers_cuda_ptrs,
                offs_barrier_counter,
                nonce=1,
                clone_result=True,
            )
            return _view_beginning_as(
                combine_recv_buffer,
                (batch_size, self.top_k, dx_recv.shape[-1]),
                dx_recv.dtype,
            )

        return dispatch_fn, combine_fn, combine_bwd_fn, dispatch_bwd_fn
