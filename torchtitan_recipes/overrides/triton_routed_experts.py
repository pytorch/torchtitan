# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Triton grouped GEMMs for the routed experts' no-grad forward (generation).

One kernel, launched for w13 (+ SwiGLU on the fp32 accumulators) and for w2. With any
dispatcher it runs on the expert-sorted rows. It reads the routes directly, so the expert
sort and the scatter go away and one kernel sums each token's routes, in two cases:

    EP = 1                      no dispatch
    EP > 1, DeepEP, no grad     one row per (token, rank), DeepEP sums the ranks' rows

Forwards with grad, batch-invariant mode and quantized expert linears keep the stock path.

Example:
    >>> cfg = triton_routed_experts(moe_cfg.routed_experts)  # or the override import
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    LocalTokenDispatcher,
)
from triton.language.extra import libdevice

from torchtitan_recipes.overrides.fused_swiglu import FusedSwiGLU

__all__ = [
    "TritonRoutedExperts",
    "triton_expert_mlp_op",
    "triton_routed_experts",
    "triton_routed_experts_op",
]


@dataclass(frozen=True, slots=True)
class _Tiles:
    block_m: int
    persistent: bool
    # (BLOCK_N, BLOCK_K, stages, warps) per GEMM
    w13: tuple[int, int, int, int]
    w2: tuple[int, int, int, int]


# Tiles tuned on GB300. Decode reads weights (16-row blocks, one program per tile; below
# 32 rows only a few experts are active, so narrower w13 tiles give more programs);
# prefill runs a persistent, warp-specialized loop.
_SMALL_TILES = _Tiles(16, False, w13=(64, 128, 6, 4), w2=(128, 128, 4, 4))
_SMALL_ROWS = 32
_DECODE_TILES = _Tiles(16, False, w13=(128, 128, 4, 4), w2=(128, 256, 2, 4))
_MID_TILES = _Tiles(64, True, w13=(128, 128, 4, 4), w2=(128, 128, 3, 4))
_PREFILL_TILES = _Tiles(128, True, w13=(256, 64, 4, 8), w2=(128, 128, 3, 4))
# From this many rows per expert on average, the persistent tiles win; 128-row blocks from
# the larger count (at 32 rows per expert they would be three quarters padding).
_PREFILL_ROWS_PER_EXPERT = 32
_LARGE_ROWS_PER_EXPERT = 64
# From this many, the stock grouped GEMMs are faster (16k tokens at top-8, E=256).
_STOCK_ROWS_PER_EXPERT = 512


# Triton 3.9's descriptor `.load()` drops the cache policy its IR op takes, so this calls the
# private semantic API; replace it once `.load()` takes `eviction_policy`. Weights are
# streamed once, so they shouldn't evict the reused activations from L2. Measured
# exceptions keep the default: below 32 rows, and w2 with the 128-row prefill tiles.
@tl.core.builtin
def _load_weights(desc, offsets, evict_first, _semantic=None):
    policy = (
        "evict_first" if tl.core._unwrap_if_constexpr(evict_first) else "evict_normal"
    )
    return _semantic.descriptor_load(
        desc, offsets, tl.core._CachePolicy("none", policy)
    )


@triton.jit
def _local_expert_ids(experts, first_expert, num_experts):
    """Global expert ids -> this rank's ids, -1 for other ranks' experts and padding.

    Example (rank 1 of 4, 64 experts each): [3, 70, 127, 128, -1] -> [-1, 6, 63, -1, -1]
    """
    local = experts - first_expert
    return tl.where((local >= 0) & (local < num_experts), local, -1)


@triton.jit
def _count_routes_kernel(
    ids_ptr, counts_ptr, N, first_expert, num_experts,
    E: tl.constexpr, BLOCK: tl.constexpr, SKIP: tl.constexpr,
):  # fmt: skip
    """Add this block's routes per expert to ``counts``: one atomic per expert per block."""
    routes = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = routes < N
    experts = tl.load(ids_ptr + routes, mask=valid, other=0).to(tl.int32)
    if SKIP:
        experts = _local_expert_ids(experts, first_expert, num_experts)
        valid = valid & (experts >= 0)
    block_counts = tl.histogram(experts, E, mask=valid)
    tl.atomic_add(
        counts_ptr + tl.arange(0, E), block_counts, mask=block_counts > 0, sem="relaxed"
    )


@triton.jit
def _route_scan_kernel(
    ids_ptr,
    rows_ptr,
    offsets_ptr,
    N,
    first_expert,
    num_experts,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    SKIP: tl.constexpr,
):
    """Program ``e`` writes expert ``e``'s routes to its rows, in route order, and its end row."""
    expert = tl.program_id(0)
    e = tl.arange(0, E)
    counts = tl.zeros((E,), dtype=tl.int32)
    for start in range(0, N, BLOCK):
        routes = start + tl.arange(0, BLOCK)
        valid = routes < N
        experts = tl.load(ids_ptr + routes, mask=valid, other=0).to(tl.int32)
        if SKIP:
            experts = _local_expert_ids(experts, first_expert, num_experts)
            valid = valid & (experts >= 0)
        counts += tl.histogram(experts, E, mask=valid)
    row = tl.sum(tl.where(e < expert, counts, 0), axis=0)
    tl.store(
        offsets_ptr + expert, row + tl.sum(tl.where(e == expert, counts, 0), axis=0)
    )
    for start in range(0, N, BLOCK):
        routes = start + tl.arange(0, BLOCK)
        experts = tl.load(ids_ptr + routes, mask=routes < N, other=-1)
        if SKIP:
            experts = _local_expert_ids(experts.to(tl.int32), first_expert, num_experts)
        mine = (experts == expert).to(tl.int32)
        tl.store(
            rows_ptr + row + tl.cumsum(mine, axis=0) - mine, routes, mask=mine == 1
        )
        row += tl.sum(mine, axis=0)


@triton.jit
def _route_sorted_kernel(
    ids_ptr, counts_ptr, fill_ptr, rows_ptr, offsets_ptr, N, first_expert, num_experts,
    E: tl.constexpr, BLOCK: tl.constexpr, SKIP: tl.constexpr,
):  # fmt: skip
    """Give each route a row of its expert; rows of one expert end up in any order.

    Each program sorts its BLOCK routes by expert, ranks them inside the block, and takes
    one ``atomic_add`` per expert for the block's first row (one atomic per route contends
    on the 256 counters). The order doesn't change any result: each GEMM output row depends
    only on its own input row.
    """
    e = tl.arange(0, E)
    counts = tl.load(counts_ptr + e, mask=e < num_experts, other=0).to(tl.int32)
    row_end = tl.cumsum(counts, axis=0)
    if tl.program_id(0) == 0:
        tl.store(offsets_ptr + e, row_end, mask=e < num_experts)
    i = tl.arange(0, BLOCK)
    routes = tl.program_id(0) * BLOCK + i
    # Padding and other ranks' routes get expert E: they sort last, into the unused bins >= E.
    if SKIP:  # skipped routes too
        experts = tl.load(ids_ptr + routes, mask=routes < N, other=-1).to(tl.int32)
        experts = _local_expert_ids(experts, first_expert, num_experts)
        experts = tl.where(experts < 0, E, experts)
    else:
        experts = tl.load(ids_ptr + routes, mask=routes < N, other=E).to(tl.int32)
    sorted_keys = tl.sort((experts << 16) | i)
    experts = sorted_keys >> 16
    routes = tl.program_id(0) * BLOCK + (sorted_keys & 0xFFFF)
    bins = tl.arange(0, 2 * E)
    block_counts = tl.where(bins < E, tl.histogram(experts, 2 * E), 0)
    block_start = tl.cumsum(block_counts, axis=0) - block_counts
    block_row0 = tl.atomic_add(
        fill_ptr + bins, block_counts, mask=block_counts > 0, sem="relaxed"
    )
    row = (
        tl.gather(row_end - counts, tl.minimum(experts, E - 1), axis=0)
        + tl.gather(block_row0, experts, axis=0)
        + i
        - tl.gather(block_start, experts, axis=0)
    )
    tl.store(rows_ptr + row, routes, mask=experts < E)


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    a_desc,
    w_desc,
    out_ptr,
    offsets_ptr,
    routes_ptr,
    num_experts,
    stride_a,
    E: tl.constexpr,
    KDIM: tl.constexpr,
    N_OUT: tl.constexpr,
    GATED: tl.constexpr,
    TMA_A: tl.constexpr,
    ROUTES: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    PERSISTENT: tl.constexpr,
    EVICT_W: tl.constexpr,
):
    """``out[row] = a[row] @ W[e].T`` for the rows of each expert ``e``.

    Each loop step computes one (block of BLOCK_M rows of one expert, N tile); it finds the
    block's expert from ``offsets``, so nothing writes a schedule.
    GATED (w13, weight (E, 2, F, D)): ``out = silu(a @ Wg.T) * (a @ Wu.T)``; a tile's gate
    and up rows load as one TMA box, so one MMA computes both.
    Else (w2, weight (E, D, F)): ``out = a @ W.T``.
    TMA_A: load a block's rows of ``a`` with one TMA box (rows past the expert are masked
    at the store); else with masked pointer loads.
    ROUTES (the kernels read the routes): expert-sorted row ``i`` holds route ``routes[i] = token * TOP_K + k``;
    w13 reads ``a`` (x) at the route's token, and w2 writes ``out`` (y) at the route.
    """
    BLOCK_OUT: tl.constexpr = BLOCK_N // 2 if GATED else BLOCK_N
    n_tiles: tl.constexpr = N_OUT // BLOCK_OUT
    e = tl.arange(0, E)
    row_end = tl.load(offsets_ptr + e, mask=e < num_experts, other=0)
    row_start = tl.load(offsets_ptr + e - 1, mask=(e > 0) & (e < num_experts), other=0)
    blocks = (row_end - row_start + BLOCK_M - 1) // BLOCK_M
    blocks_end = tl.cumsum(blocks, axis=0)
    num_tiles = tl.sum(blocks, axis=0) * n_tiles
    if not GATED:
        # Launched with programmatic dependent launch: everything above overlapped
        # the w13 kernel's tail; h is complete after this wait.
        tl.extra.cuda.gdc_wait()
    for tile in tl.range(
        tl.program_id(0),
        num_tiles,
        tl.num_programs(0),
        flatten=PERSISTENT,
        warp_specialize=PERSISTENT,
    ):
        # Expert of this block and its rows.
        block = tile // n_tiles
        expert = tl.sum((blocks_end <= block).to(tl.int32), axis=0)
        is_expert = e == expert
        block_in_expert = block - tl.sum(
            tl.where(is_expert, blocks_end - blocks, 0), axis=0
        )
        first_row = tl.sum(tl.where(is_expert, row_start, 0), axis=0)
        first_row += block_in_expert * BLOCK_M
        rows = first_row + tl.arange(0, BLOCK_M)
        valid = rows < tl.sum(tl.where(is_expert, row_end, 0), axis=0)
        a_rows = rows
        out_rows = rows
        if ROUTES:
            routes = tl.load(routes_ptr + rows, mask=valid, other=0)
            if GATED:
                a_rows = routes // TOP_K
            else:
                out_rows = routes
        # Main loop over K.
        out0 = (tile % n_tiles) * BLOCK_OUT
        a_ptrs = (
            a_ptr
            + a_rows[:, None].to(tl.int64) * stride_a
            + tl.arange(0, BLOCK_K)[None, :]
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, KDIM // BLOCK_K, num_stages=STAGES):
            if TMA_A:
                a = a_desc.load([first_row, k * BLOCK_K])
            else:
                a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
                a_ptrs += BLOCK_K
            if GATED:
                w = _load_weights(w_desc, [expert, 0, out0, k * BLOCK_K], EVICT_W)
                w = w.reshape(BLOCK_N, BLOCK_K)
            else:
                w = _load_weights(w_desc, [expert * N_OUT + out0, k * BLOCK_K], EVICT_W)
            acc = tl.dot(a, w.T, acc)
        if not PERSISTENT:
            tl.extra.cuda.gdc_launch_dependents()
        # Epilogue.
        if GATED:
            gate, up = acc.reshape(BLOCK_M, 2, BLOCK_OUT).permute(0, 2, 1).split()
            # The stock path's math (bitwise with its compiled SwiGLU region): its grouped
            # GEMM rounds gate/up to bf16, then SwiGLU computes this in fp32.
            gate = gate.to(tl.bfloat16).to(tl.float32)
            up = up.to(tl.bfloat16).to(tl.float32)
            acc = gate / (1.0 + libdevice.exp(-gate)) * up
        cols = out0 + tl.arange(0, BLOCK_OUT)
        out_ptrs = out_ptr + out_rows[:, None].to(tl.int64) * N_OUT + cols[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=valid[:, None])


def _tiles(num_rows: int, num_experts: int) -> _Tiles:
    if num_rows >= _LARGE_ROWS_PER_EXPERT * num_experts:
        return _PREFILL_TILES
    if num_rows >= _PREFILL_ROWS_PER_EXPERT * num_experts:
        return _MID_TILES
    return _SMALL_TILES if num_rows < _SMALL_ROWS else _DECODE_TILES


def _launch_gemm(a, w, out, offsets_E, tiles, gated, routes_N=None, top_k=None):
    from triton.tools.tensor_descriptor import TensorDescriptor

    num_experts, n_out = w.shape[0], out.shape[1]
    block_n, block_k, stages, warps = tiles.w13 if gated else tiles.w2
    if gated:
        w_desc = TensorDescriptor.from_tensor(w, [1, 2, block_n // 2, block_k])
    else:
        w_desc = TensorDescriptor.from_tensor(
            w.view(num_experts * n_out, w.shape[2]), [block_n, block_k]
        )
    # Only w2 reads its input (our h buffer, expert-sorted) by TMA.
    tma_a = tiles.persistent and not gated
    a_desc = (
        TensorDescriptor.from_tensor(a, [tiles.block_m, block_k]) if tma_a else None
    )
    if tiles.persistent:
        grid = torch.cuda.get_device_properties(a.device).multi_processor_count
    else:
        # Every row's block, plus one partial block per expert at most.
        num_rows = out.shape[0]
        max_blocks = num_rows // tiles.block_m + min(num_rows, num_experts)
        grid = max_blocks * (n_out // (block_n // 2 if gated else block_n))
    _grouped_gemm_kernel[(grid,)](
        a, a_desc, w_desc, out, offsets_E, routes_N, num_experts, a.stride(0),
        E=triton.next_power_of_2(num_experts), KDIM=a.shape[1], N_OUT=n_out,
        GATED=gated, TMA_A=tma_a, ROUTES=routes_N is not None, TOP_K=top_k, BLOCK_M=tiles.block_m, BLOCK_N=block_n,
        BLOCK_K=block_k, STAGES=stages, PERSISTENT=tiles.persistent,
        EVICT_W=tiles is not _SMALL_TILES and (gated or tiles is not _PREFILL_TILES),
        num_warps=warps,
        launch_pdl=not gated,
    )  # fmt: skip


@torch.library.custom_op("torchtitan::triton_expert_mlp", mutates_args=())
def triton_expert_mlp_op(
    routed_input_RD: torch.Tensor,
    offsets_E: torch.Tensor,
    w13_E2FD: torch.Tensor,
    w2_EDF: torch.Tensor,
) -> torch.Tensor:
    """``w2(silu(x @ w13_gate.T) * (x @ w13_up.T))`` for each expert's rows of ``x``.

    Args:
        routed_input_RD: bf16 rows grouped by expert.
        offsets_E: int32 end row of each expert's group, as ``torch._grouped_mm``'s ``offs``.

    Example:
        >>> routed_output_RD = triton_expert_mlp_op(routed_input_RD, offsets_E, w13, w2)
    """
    num_rows = routed_input_RD.shape[0]
    num_experts, _, hidden, dim = w13_E2FD.shape
    out_RD = routed_input_RD.new_empty(num_rows, dim)
    if num_rows == 0:  # an expert-parallel rank can receive no rows
        return out_RD
    tiles = _tiles(num_rows, num_experts)
    h_RF = routed_input_RD.new_empty(num_rows, hidden)
    _launch_gemm(routed_input_RD, w13_E2FD, h_RF, offsets_E, tiles, gated=True)
    _launch_gemm(h_RF, w2_EDF, out_RD, offsets_E, tiles, gated=False)
    return out_RD


@triton_expert_mlp_op.register_fake
def _(routed_input_RD, offsets_E, w13_E2FD, w2_EDF):
    return routed_input_RD.new_empty(routed_input_RD.shape[0], w2_EDF.shape[1])


@triton.jit
def _combine_kernel(
    y_ptr, scores_ptr, ids_ptr, out_ptr, first_expert, num_experts,
    D: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr, SKIP: tl.constexpr,
):  # fmt: skip
    """``out[t] = sum_k score[t, k] * y[t * K + k]`` in fp32, one rounding.

    SKIP (EP > 1): a route to another rank's expert or padding adds nothing; its y row was never
    written, and its score is another rank's (or NaN on a padding row), so neither is used.
    """
    tl.extra.cuda.gdc_wait()  # y comes from the w2 kernel (programmatic dependent launch)
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    for k in tl.static_range(K):
        score = tl.load(scores_ptr + token * K + k).to(tl.float32)
        y_ptrs = y_ptr + (token * K + k).to(tl.int64) * D + cols
        if SKIP:
            expert = tl.load(ids_ptr + token * K + k).to(tl.int32)
            is_local = _local_expert_ids(expert, first_expert, num_experts) >= 0
            score = tl.where(is_local, score, 0.0)
            y = tl.load(y_ptrs, mask=is_local, other=0.0).to(tl.float32)
        else:
            y = tl.load(y_ptrs).to(tl.float32)
        # fp32, rounded once: y0*s0 + y1*s1, then one fma per route.
        if k == 0:
            acc = libdevice.mul_rn(y, score)
        elif k == 1:
            acc = libdevice.add_rn(acc, libdevice.mul_rn(y, score))
        else:
            acc = tl.fma(y, score, acc)
    tl.store(out_ptr + token.to(tl.int64) * D + cols, acc.to(out_ptr.dtype.element_ty))


@torch.library.custom_op("torchtitan::triton_routed_experts", mutates_args=())
def triton_routed_experts_op(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    w13_E2FD: torch.Tensor,
    w2_EDF: torch.Tensor,
    ep_rank: int = 0,
    ep_size: int = 1,
) -> torch.Tensor:
    """Routed-experts output ``(T, D)``: each token's ``K`` routes, weighted by their scores.

    With ``ep_size > 1`` this rank holds ``E = w13_E2FD.shape[0]`` experts, global ids
    ``[ep_rank * E, (ep_rank + 1) * E)``; routes to other experts, or -1, add nothing.

    Example:
        >>> out_TD = triton_routed_experts_op(x_TD, scores_TK, ids_TK, w13, w2)
        >>> part_TD = triton_routed_experts_op(  # experts 64-127 of 256
        ...     x_TD, scores_TK, ids_TK, w13, w2, ep_rank=1, ep_size=4
        ... )
    """
    num_tokens, dim = x_TD.shape
    top_k = topk_expert_ids_TK.shape[1]
    num_experts, _, hidden, _ = w13_E2FD.shape
    num_routes = num_tokens * top_k
    ids_N = topk_expert_ids_TK.reshape(-1)
    routes_N = x_TD.new_empty(num_routes, dtype=torch.int32)
    offsets_E = x_TD.new_empty(num_experts, dtype=torch.int32)
    experts_p2 = triton.next_power_of_2(num_experts)
    skip, first_expert = ep_size > 1, ep_rank * num_experts
    # About 1 / ep_size of the routes are local; their exact count is only on the device.
    tiles = _tiles(num_routes // ep_size, num_experts)
    h_NF = x_TD.new_empty(num_routes, hidden)
    y_ND = x_TD.new_empty(num_routes, dim)
    out_TD = torch.empty_like(x_TD)
    block_d = 1024 if num_tokens >= 1024 and dim % 1024 == 0 else 256
    if tiles.persistent:
        # One zeroed buffer: the per-expert row counters, then the counts.
        zeros = torch.zeros(3 * experts_p2, dtype=torch.int32, device=x_TD.device)
        fill, counts_E = zeros[: 2 * experts_p2], zeros[2 * experts_p2 :]
        _count_routes_kernel[(triton.cdiv(num_routes, 512),)](
            ids_N, counts_E, num_routes, first_expert, num_experts,
            E=experts_p2, BLOCK=512, SKIP=skip,
        )  # fmt: skip
        _route_sorted_kernel[(triton.cdiv(num_routes, 512),)](
            ids_N, counts_E, fill, routes_N, offsets_E, num_routes, first_expert,
            num_experts, E=experts_p2, BLOCK=512, SKIP=skip,
        )  # fmt: skip
    else:
        _route_scan_kernel[(num_experts,)](
            ids_N, routes_N, offsets_E, num_routes, first_expert, num_experts,
            E=experts_p2, BLOCK=min(triton.next_power_of_2(num_routes), 2048), SKIP=skip,
        )  # fmt: skip
    _launch_gemm(
        x_TD,
        w13_E2FD,
        h_NF,
        offsets_E,
        tiles,
        gated=True,
        routes_N=routes_N,
        top_k=top_k,
    )
    _launch_gemm(h_NF, w2_EDF, y_ND, offsets_E, tiles, gated=False, routes_N=routes_N)
    _combine_kernel[(num_tokens, dim // block_d)](
        y_ND, topk_scores_TK, ids_N, out_TD, first_expert, num_experts,
        D=dim, K=top_k, BLOCK_D=block_d, SKIP=skip, launch_pdl=True,
    )  # fmt: skip
    return out_TD


@triton_routed_experts_op.register_fake
def _(x_TD, topk_scores_TK, topk_expert_ids_TK, w13_E2FD, w2_EDF, ep_rank=0, ep_size=1):
    return torch.empty_like(x_TD)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


class TritonRoutedExperts(RoutedExperts):
    """``RoutedExperts`` whose no-grad expert MLP runs as Triton grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor | None,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self._runs_ep1_path()
            and x_TD.dtype == torch.bfloat16
            and x_TD.shape[0] > 0
            and x_TD.is_cuda
            and x_TD.stride(-1) == 1
        ):
            # Padding rows are computed like real tokens, as on the stock path.
            return triton_routed_experts_op(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                _local(self.w13.weight),
                _local(self.w2.weight),
            )
        if self._runs_deepep_path() and x_TD.dtype == torch.bfloat16:
            from torchtitan.distributed.deepep.deepep import (
                combine_routes,
                dispatch_routes,
            )

            dispatcher = self.token_dispatcher
            recv_x_RD, recv_scores_RK, recv_ids_RK, handle = dispatch_routes(
                x_TD,
                selected_experts_indices=topk_expert_ids_TK,
                top_scores=topk_scores_TK,
                num_experts=dispatcher.num_experts,
            )
            # DeepEP's ids are local, so this rank's experts start at 0; ep_size > 1 makes
            # the kernels skip -1 slots and size tiles for T x top-k routes, not EP x T x top-k.
            out_RD = triton_routed_experts_op(
                recv_x_RD,
                recv_scores_RK,
                recv_ids_RK,
                _local(self.w13.weight),
                _local(self.w2.weight),
                ep_rank=0,
                ep_size=dispatcher.ep_mesh.size(),
            )
            return combine_routes(out_RD, handle)
        if num_local_tokens_per_expert_E is None:  # count_tokens left it to us
            ids_N = topk_expert_ids_TK.reshape(-1)
            num_local_tokens_per_expert_E = torch.zeros(
                self.w13.weight.shape[0], dtype=torch.int64, device=ids_N.device
            ).index_add_(0, ids_N, torch.ones_like(ids_N))
        # Other dispatchers: their dispatch and combine around the Triton GEMMs.
        return super().forward(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
            padding_mask_T=padding_mask_T,
        )

    def count_tokens(self, routing_map_TE: torch.Tensor) -> torch.Tensor | None:
        if self._runs_ep1_path() or self._runs_deepep_path():
            return None  # the kernels count the expert ids themselves
        return super().count_tokens(routing_map_TE)

    def _runs_ep1_path(self) -> bool:
        return (
            self._all_experts_local()
            and self.output_postprocess is None
            and self._can_run_triton(_local(self.w13.weight), _local(self.w2.weight))
        )

    def _runs_deepep_path(self) -> bool:
        """DeepEP at inference (EP > 1): send each token once per rank, not once per expert."""
        dispatcher = self.token_dispatcher
        return (
            type(dispatcher) is DeepEPTokenDispatcher
            and dispatcher.cuda_graph_compatible
            and dispatcher.ep_mesh is not None
            and dispatcher.ep_mesh.size() > 1
            and self.output_postprocess is None
            and self._can_run_triton(_local(self.w13.weight), _local(self.w2.weight))
        )

    def _all_experts_local(self) -> bool:
        dispatcher = self.token_dispatcher
        if type(dispatcher) is LocalTokenDispatcher:
            return True
        # Without an EP mesh, the all-to-all dispatcher only reorders tokens locally.
        return type(dispatcher) is AllToAllTokenDispatcher and (
            dispatcher.ep_mesh is None or dispatcher.ep_mesh.size() == 1
        )

    def _expert_mlp(
        self, routed_input_RD: torch.Tensor, offsets_E: torch.Tensor
    ) -> torch.Tensor:
        routed_input_RD = routed_input_RD.bfloat16()
        w13_E2FD, w2_EDF = _local(self.w13.weight), _local(self.w2.weight)
        if not (
            self._can_run_triton(w13_E2FD, w2_EDF)
            and routed_input_RD.is_cuda
            and routed_input_RD.stride(-1) == 1
            and routed_input_RD.shape[0] < _STOCK_ROWS_PER_EXPERT * w13_E2FD.shape[0]
        ):
            return super()._expert_mlp(routed_input_RD, offsets_E)
        return triton_expert_mlp_op(routed_input_RD, offsets_E, w13_E2FD, w2_EDF)

    def _can_run_triton(self, w13_E2FD: torch.Tensor, w2_EDF: torch.Tensor) -> bool:
        if torch.is_grad_enabled() or is_in_batch_invariant_mode():
            return False
        # Quantized expert linears subclass GroupedLinear and keep their own GEMMs.
        if type(self.w13) is not GroupedLinear or type(self.w2) is not GroupedLinear:
            return False
        # Both compute silu(gate) * up; the fused one is the fused_swiglu override's.
        if type(self.activation_fn) not in (SwiGLU, FusedSwiGLU):
            return False
        if any(
            w.dtype != torch.bfloat16 or not w.is_contiguous()
            for w in (w13_E2FD, w2_EDF)
        ):
            return False
        # The largest tiles must divide the dimensions.
        dim, hidden = w2_EDF.shape[1], w2_EDF.shape[2]
        return dim % 256 == 0 and hidden % 256 == 0


@override(
    target=RoutedExperts.Config,
    exact=True,
    description="Run the no-grad routed experts as Triton grouped-GEMM kernels.",
)
def triton_routed_experts(cfg: RoutedExperts.Config) -> TritonRoutedExperts.Config:
    return derive(cfg, TritonRoutedExperts.Config)
