# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Triton grouped GEMMs for the routed experts' no-grad expert MLP (generation).

One kernel, launched for w13 (+ SwiGLU on the fp32 accumulators) and for w2, on the
dispatcher's expert-sorted rows; dispatch and combine are unchanged. Forwards with grad,
batch-invariant mode and quantized expert linears keep the stock path.

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
from triton.language.extra import libdevice

__all__ = ["TritonRoutedExperts", "triton_expert_mlp_op", "triton_routed_experts"]


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
def _grouped_gemm_kernel(
    a_ptr,
    a_desc,
    w_desc,
    out_ptr,
    offsets_ptr,
    num_experts,
    stride_a,
    E: tl.constexpr,
    KDIM: tl.constexpr,
    N_OUT: tl.constexpr,
    GATED: tl.constexpr,
    TMA_A: tl.constexpr,
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
        # Main loop over K.
        out0 = (tile % n_tiles) * BLOCK_OUT
        a_ptrs = (
            a_ptr
            + rows[:, None].to(tl.int64) * stride_a
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
            # The trainer's math, so outputs match it bitwise: its grouped GEMM rounds
            # gate/up to bf16, then its compiled SwiGLU computes this in fp32.
            gate = gate.to(tl.bfloat16).to(tl.float32)
            up = up.to(tl.bfloat16).to(tl.float32)
            acc = gate / (1.0 + libdevice.exp(-gate)) * up
        cols = out0 + tl.arange(0, BLOCK_OUT)
        out_ptrs = out_ptr + rows[:, None].to(tl.int64) * N_OUT + cols[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=valid[:, None])


def _tiles(num_rows: int, num_experts: int) -> _Tiles:
    if num_rows >= _LARGE_ROWS_PER_EXPERT * num_experts:
        return _PREFILL_TILES
    if num_rows >= _PREFILL_ROWS_PER_EXPERT * num_experts:
        return _MID_TILES
    return _SMALL_TILES if num_rows < _SMALL_ROWS else _DECODE_TILES


def _launch_gemm(a, w, out, offsets_E, tiles, gated):
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
        a, a_desc, w_desc, out, offsets_E, num_experts, a.stride(0),
        E=triton.next_power_of_2(num_experts), KDIM=a.shape[1], N_OUT=n_out,
        GATED=gated, TMA_A=tma_a, BLOCK_M=tiles.block_m, BLOCK_N=block_n,
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


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


class TritonRoutedExperts(RoutedExperts):
    """``RoutedExperts`` whose no-grad expert MLP runs as Triton grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

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
        if type(self.activation_fn) is not SwiGLU:
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
