# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Opt-in fused DeepSeek-V3 MLA Q/KV assembly.

Activate with::

    --override.imports torchtitan.overrides.fused_mla.fused_mla

Scope and limitations
---------------------
This override is specific to TorchTitan's DeepSeek-V3 ``Attention`` module and
its packed MLA Q/KV projection layout. It is not a generic RoPE fusion and does
not apply to non-MLA models such as Qwen3, Qwen3.5, or GPT-OSS.

The kernels implement TorchTitan ``ComplexRoPE`` and require its complex-valued
cache. They do not support ``CosSinRoPE`` or ``MRoPE``, whose real cos/sin cache
layouts and rotation conventions require a separate kernel path. A future MLA
model using either of those RoPE implementations cannot use this override
without such an adaptation.

Design provenance
-----------------
The core fusion strategy is borrowed from NVIDIA Megatron Core's fused MLA
design in ``megatron/core/fusions/fused_mla_yarn_rope_apply.py`` (Megatron
Core 0.17.0): apply RoPE to Q in place, directly assemble the expanded K while
applying K RoPE, and fuse KV-gradient packing with the shared K-position
gradient reduction. Credit belongs to the NVIDIA Megatron Core authors for
that design. This file is an independent TorchTitan adaptation and does not
import Megatron Core or TransformerEngine.

The implementation differs from Megatron Core in several important ways:

* It consumes TorchTitan's BSHD tensors and explicit per-example position IDs,
  rather than Megatron's SBHD/THD tensors and packed-sequence/CP indexing.
* It implements TorchTitan ``ComplexRoPE``'s adjacent-pair complex convention,
  rather than Megatron MLA's YaRN cos/sin output layout.
* V remains a zero-copy view of TorchTitan's packed KV projection; Megatron's
  fused KV path materializes a separate V output.
* It preserves TorchTitan eager's BF16/FP16 reduction-rounding boundary and
  wraps local results back into DTensors.
* Flattened offsets use 64-bit arithmetic because the traced 671B local tensors
  exceed 2**31 elements.
* Each head's rope tail is addressed as one contiguous tile and de-interleaved
  in registers. Addressing the adjacent complex pairs as two stride-2 accesses
  over the same bytes prevents vectorization: on Triton 3.8 that emits 32
  scalar 2-byte accesses per program where the tile form emits 4 16-byte
  vector accesses. The Q kernel is 1.77x faster for it on GB300 at the 671B
  shape; the K and KV-backward kernels are already bandwidth-bound on their
  contiguous nope/value copies and do not measurably change.
* Every Triton launch is exposed as a stable ``torch.library`` custom operator,
  so GraphTrainer's fake-tensor ``make_fx`` trace keeps the fused boundaries.

The override keeps the stock Attention parameters and state-dict layout.  It
only replaces the Q/KV layout boundary around ComplexRoPE:

* Q RoPE is applied in place to the positional tail of the Q projection.
* K RoPE, head expansion, and final K materialization are one Triton kernel.
* V remains a view of the packed KV projection (no extra forward copy).
* KV backward packs dK-nope and dV while reducing/inverse-rotating dK-pos.

No Megatron-Core or TransformerEngine dependency is required.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor

from torchtitan.config import derive, override
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rope import _maybe_check_max_pos, ComplexRoPE
from torchtitan.models.deepseek_v3.model import Attention

__all__ = [
    "FusedMLAAttention",
    "fused_mla",
    "fused_mla_q",
    "fused_mla_kv",
]

# Autotuned rather than fixed: the best pair tracks head count. At 128 heads
# any tile from 16 up is within a few percent; at 16 heads one warp beats four
# by ~1.2x on the KV backward. Wider grids were no faster at either count and
# tuned 2.4x slower -- an 8-wide tile alone unrolls the KV backward's
# tl.static_range head loop 16 times.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_H": block_h}, num_warps=num_warps)
    for block_h in (16, 32, 64, 128)
    for num_warps in (1, 4)
]

# Tuning key: the geometry that changes the best tile. Token count does not --
# the best config was stable from 4k to 32k tokens -- so runs with a varying
# microbatch reuse one tuning result instead of re-benchmarking.
_AUTOTUNE_KEY = ["N_HEADS", "Q_NOPE_DIM", "ROPE_DIM"]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=_AUTOTUNE_KEY,
    # This kernel rotates q in place, and the autotuner runs each candidate
    # against the same buffer. Without this, every trial after the first would
    # rotate already-rotated data and the chosen config would be benchmarked
    # (and the caller's tensor left) wrong.
    restore_value=["q"],
)
@triton.jit
def _fused_q_rope_kernel(
    q,
    rope_cache,
    positions,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_L: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_M: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    POS_STRIDE_B: tl.constexpr,
    POS_STRIDE_L: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    Q_NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    INVERSE: tl.constexpr,
) -> None:
    # Derived here rather than passed in: BLOCK_H is chosen by the autotuner,
    # so the host cannot know the block count when it builds the launch.
    num_head_blocks: tl.constexpr = (N_HEADS + BLOCK_H - 1) // BLOCK_H
    # Production DeepSeek shapes exceed 2**31 elements, so every flattened
    # tensor index must be promoted before multiplying by a stride.
    program = tl.program_id(0).to(tl.int64)
    head_block = program % num_head_blocks
    token = program // num_head_blocks
    seq = token % SEQ_LEN
    batch = token // SEQ_LEN

    head = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    pair = tl.arange(0, BLOCK_PAIRS)[None, :]
    # Each head's rope tail is ROPE_DIM contiguous elements. Address it as one
    # tile and de-interleave the adjacent pairs in registers: two stride-2
    # accesses over the same bytes defeat vectorization, so each transaction
    # would carry half the useful bytes.
    lane = tl.arange(0, 2 * BLOCK_PAIRS)[None, :]
    head_mask = head < N_HEADS
    pair_mask = pair < ROPE_DIM // 2
    lane_mask = head_mask & (lane < ROPE_DIM)
    position = tl.load(positions + batch * POS_STRIDE_B + seq * POS_STRIDE_L)

    q_base = (
        batch * Q_STRIDE_B
        + seq * Q_STRIDE_L
        + head * Q_STRIDE_H
        + Q_NOPE_DIM * Q_STRIDE_D
    )
    q_pairs = tl.reshape(
        tl.load(q + q_base + lane * Q_STRIDE_D, mask=lane_mask, other=0.0),
        (BLOCK_H, BLOCK_PAIRS, 2),
    )
    q_even, q_odd = tl.split(q_pairs)
    q_even = q_even.to(tl.float32)
    q_odd = q_odd.to(tl.float32)

    cache_base = position * CACHE_STRIDE_M + pair * CACHE_STRIDE_P
    cos = tl.load(
        rope_cache + cache_base,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        rope_cache + cache_base + CACHE_STRIDE_R,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    if INVERSE:
        out_even = q_even * cos + q_odd * sin
        out_odd = q_odd * cos - q_even * sin
    else:
        out_even = q_even * cos - q_odd * sin
        out_odd = q_even * sin + q_odd * cos

    tl.store(
        q + q_base + lane * Q_STRIDE_D,
        tl.reshape(tl.join(out_even, out_odd), (BLOCK_H, 2 * BLOCK_PAIRS)),
        mask=lane_mask,
    )


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def _fused_k_rope_kernel(
    kv,
    k_pe,
    rope_cache,
    positions,
    k,
    KV_STRIDE_B: tl.constexpr,
    KV_STRIDE_L: tl.constexpr,
    KV_STRIDE_H: tl.constexpr,
    KV_STRIDE_D: tl.constexpr,
    KPE_STRIDE_B: tl.constexpr,
    KPE_STRIDE_L: tl.constexpr,
    KPE_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_M: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    POS_STRIDE_B: tl.constexpr,
    POS_STRIDE_L: tl.constexpr,
    K_STRIDE_B: tl.constexpr,
    K_STRIDE_L: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    Q_NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
) -> None:
    num_head_blocks: tl.constexpr = (N_HEADS + BLOCK_H - 1) // BLOCK_H
    program = tl.program_id(0).to(tl.int64)
    head_block = program % num_head_blocks
    token = program // num_head_blocks
    seq = token % SEQ_LEN
    batch = token // SEQ_LEN

    head = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    dim = tl.arange(0, BLOCK_D)[None, :]
    head_mask = head < N_HEADS
    nope_mask = head_mask & (dim < Q_NOPE_DIM)
    kv_base = batch * KV_STRIDE_B + seq * KV_STRIDE_L + head * KV_STRIDE_H
    k_base = batch * K_STRIDE_B + seq * K_STRIDE_L + head * K_STRIDE_H
    k_nope = tl.load(
        kv + kv_base + dim * KV_STRIDE_D,
        mask=nope_mask,
        other=0.0,
    )
    tl.store(k + k_base + dim * K_STRIDE_D, k_nope, mask=nope_mask)

    # k_pe is shared by every attention head. Rotate it once per head tile,
    # then broadcast the result while storing the tile instead of repeating the
    # FP32 complex multiply in a separate program for every head.
    pair = tl.arange(0, BLOCK_PAIRS)
    pair_mask = pair < ROPE_DIM // 2
    # Load and store the rope tail as one contiguous tile; see the comment in
    # _fused_q_rope_kernel.
    lane = tl.arange(0, 2 * BLOCK_PAIRS)
    lane_mask = lane < ROPE_DIM
    kpe_base = batch * KPE_STRIDE_B + seq * KPE_STRIDE_L
    kpe_pairs = tl.reshape(
        tl.load(k_pe + kpe_base + lane * KPE_STRIDE_D, mask=lane_mask, other=0.0),
        (BLOCK_PAIRS, 2),
    )
    kpe_even, kpe_odd = tl.split(kpe_pairs)
    kpe_even = kpe_even.to(tl.float32)
    kpe_odd = kpe_odd.to(tl.float32)

    position = tl.load(positions + batch * POS_STRIDE_B + seq * POS_STRIDE_L)
    cache_base = position * CACHE_STRIDE_M + pair * CACHE_STRIDE_P
    cos = tl.load(
        rope_cache + cache_base,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        rope_cache + cache_base + CACHE_STRIDE_R,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    out_even = kpe_even * cos - kpe_odd * sin
    out_odd = kpe_even * sin + kpe_odd * cos

    rope_base = k_base + Q_NOPE_DIM * K_STRIDE_D
    rope_mask = head_mask & lane_mask[None, :]
    tl.store(
        k + rope_base + lane[None, :] * K_STRIDE_D,
        tl.reshape(tl.join(out_even, out_odd), (2 * BLOCK_PAIRS,))[None, :],
        mask=rope_mask,
    )


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
@triton.jit
def _fused_kv_backward_kernel(
    grad_k,
    grad_v,
    rope_cache,
    positions,
    grad_kv,
    grad_k_pe,
    GK_STRIDE_B: tl.constexpr,
    GK_STRIDE_L: tl.constexpr,
    GK_STRIDE_H: tl.constexpr,
    GK_STRIDE_D: tl.constexpr,
    GV_STRIDE_B: tl.constexpr,
    GV_STRIDE_L: tl.constexpr,
    GV_STRIDE_H: tl.constexpr,
    GV_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_M: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_R: tl.constexpr,
    POS_STRIDE_B: tl.constexpr,
    POS_STRIDE_L: tl.constexpr,
    GKV_STRIDE_B: tl.constexpr,
    GKV_STRIDE_L: tl.constexpr,
    GKV_STRIDE_H: tl.constexpr,
    GKV_STRIDE_D: tl.constexpr,
    GKPE_STRIDE_B: tl.constexpr,
    GKPE_STRIDE_L: tl.constexpr,
    GKPE_STRIDE_D: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    Q_NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    ROUND_BF16_SUM: tl.constexpr,
    ROUND_FP16_SUM: tl.constexpr,
) -> None:
    token = tl.program_id(0).to(tl.int64)
    seq = token % SEQ_LEN
    batch = token // SEQ_LEN

    dim = tl.arange(0, BLOCK_D)[None, :]
    # Load and store the rope tail as one contiguous tile; see the comment in
    # _fused_q_rope_kernel.
    lane = tl.arange(0, 2 * BLOCK_PAIRS)[None, :]
    grad_pos_even = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
    grad_pos_odd = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)

    for head_start in tl.static_range(0, N_HEADS, BLOCK_H):
        head = head_start + tl.arange(0, BLOCK_H)[:, None]
        head_mask = head < N_HEADS

        gk_base = batch * GK_STRIDE_B + seq * GK_STRIDE_L + head * GK_STRIDE_H
        gv_base = batch * GV_STRIDE_B + seq * GV_STRIDE_L + head * GV_STRIDE_H
        gkv_base = batch * GKV_STRIDE_B + seq * GKV_STRIDE_L + head * GKV_STRIDE_H

        nope_mask = head_mask & (dim < Q_NOPE_DIM)
        grad_nope = tl.load(
            grad_k + gk_base + dim * GK_STRIDE_D,
            mask=nope_mask,
            other=0.0,
        )
        tl.store(
            grad_kv + gkv_base + dim * GKV_STRIDE_D,
            grad_nope,
            mask=nope_mask,
        )

        value_mask = head_mask & (dim < V_DIM)
        grad_value = tl.load(
            grad_v + gv_base + dim * GV_STRIDE_D,
            mask=value_mask,
            other=0.0,
        )
        tl.store(
            grad_kv + gkv_base + (Q_NOPE_DIM + dim) * GKV_STRIDE_D,
            grad_value,
            mask=value_mask,
        )

        lane_mask = head_mask & (lane < ROPE_DIM)
        grad_pairs = tl.reshape(
            tl.load(
                grad_k + gk_base + (Q_NOPE_DIM + lane) * GK_STRIDE_D,
                mask=lane_mask,
                other=0.0,
            ),
            (BLOCK_H, BLOCK_PAIRS, 2),
        )
        grad_even, grad_odd = tl.split(grad_pairs)
        grad_pos_even += tl.sum(grad_even.to(tl.float32), axis=0)
        grad_pos_odd += tl.sum(grad_odd.to(tl.float32), axis=0)

    # Stock expand-backward materializes the head reduction in the input dtype
    # before ComplexRoPE backward upcasts it. Preserve that rounding boundary.
    if ROUND_BF16_SUM:
        grad_pos_even = grad_pos_even.to(tl.bfloat16).to(tl.float32)
        grad_pos_odd = grad_pos_odd.to(tl.bfloat16).to(tl.float32)
    if ROUND_FP16_SUM:
        grad_pos_even = grad_pos_even.to(tl.float16).to(tl.float32)
        grad_pos_odd = grad_pos_odd.to(tl.float16).to(tl.float32)

    pair_1d = tl.arange(0, BLOCK_PAIRS)
    pair_mask_1d = pair_1d < ROPE_DIM // 2
    position = tl.load(positions + batch * POS_STRIDE_B + seq * POS_STRIDE_L)
    cache_base = position * CACHE_STRIDE_M + pair_1d * CACHE_STRIDE_P
    cos = tl.load(
        rope_cache + cache_base,
        mask=pair_mask_1d,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        rope_cache + cache_base + CACHE_STRIDE_R,
        mask=pair_mask_1d,
        other=0.0,
    ).to(tl.float32)
    out_even = grad_pos_even * cos + grad_pos_odd * sin
    out_odd = grad_pos_odd * cos - grad_pos_even * sin

    gkpe_base = batch * GKPE_STRIDE_B + seq * GKPE_STRIDE_L
    lane_1d = tl.arange(0, 2 * BLOCK_PAIRS)
    tl.store(
        grad_k_pe + gkpe_base + lane_1d * GKPE_STRIDE_D,
        tl.reshape(tl.join(out_even, out_odd), (2 * BLOCK_PAIRS,)),
        mask=lane_1d < ROPE_DIM,
    )


@torch.library.custom_op(
    "torchtitan::fused_mla_q_rope_",
    mutates_args={"q"},
    device_types="cuda",
    tags=torch.Tag.inplace,
)
def _fused_mla_q_rope_op(
    q: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    inverse: bool,
) -> torch.Tensor:
    batch, seq_len, n_heads, q_head_dim = q.shape
    rope_dim = q_head_dim - q_nope_dim
    block_pairs = triton.next_power_of_2(rope_dim // 2)
    _fused_q_rope_kernel[
        lambda meta: (batch * seq_len * triton.cdiv(n_heads, meta["BLOCK_H"]),)
    ](
        q,
        rope_cache_real,
        positions,
        Q_STRIDE_B=q.stride(0),
        Q_STRIDE_L=q.stride(1),
        Q_STRIDE_H=q.stride(2),
        Q_STRIDE_D=q.stride(3),
        CACHE_STRIDE_M=rope_cache_real.stride(0),
        CACHE_STRIDE_P=rope_cache_real.stride(1),
        CACHE_STRIDE_R=rope_cache_real.stride(2),
        POS_STRIDE_B=positions.stride(0),
        POS_STRIDE_L=positions.stride(1),
        SEQ_LEN=seq_len,
        N_HEADS=n_heads,
        Q_NOPE_DIM=q_nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_PAIRS=block_pairs,
        INVERSE=inverse,
    )
    return q


@torch.library.custom_op(
    "torchtitan::fused_mla_k_rope",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mla_k_rope_op(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
) -> torch.Tensor:
    batch, seq_len, n_heads, _ = kv.shape
    rope_dim = k_pe.shape[-1]
    k = torch.empty(
        (batch, seq_len, n_heads, q_nope_dim + rope_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    _fused_k_rope_kernel[
        lambda meta: (batch * seq_len * triton.cdiv(n_heads, meta["BLOCK_H"]),)
    ](
        kv,
        k_pe,
        rope_cache_real,
        positions,
        k,
        KV_STRIDE_B=kv.stride(0),
        KV_STRIDE_L=kv.stride(1),
        KV_STRIDE_H=kv.stride(2),
        KV_STRIDE_D=kv.stride(3),
        KPE_STRIDE_B=k_pe.stride(0),
        KPE_STRIDE_L=k_pe.stride(1),
        KPE_STRIDE_D=k_pe.stride(2),
        CACHE_STRIDE_M=rope_cache_real.stride(0),
        CACHE_STRIDE_P=rope_cache_real.stride(1),
        CACHE_STRIDE_R=rope_cache_real.stride(2),
        POS_STRIDE_B=positions.stride(0),
        POS_STRIDE_L=positions.stride(1),
        K_STRIDE_B=k.stride(0),
        K_STRIDE_L=k.stride(1),
        K_STRIDE_H=k.stride(2),
        K_STRIDE_D=k.stride(3),
        SEQ_LEN=seq_len,
        N_HEADS=n_heads,
        Q_NOPE_DIM=q_nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_D=triton.next_power_of_2(q_nope_dim),
        BLOCK_PAIRS=triton.next_power_of_2(rope_dim // 2),
    )
    return k


@_fused_mla_k_rope_op.register_fake
def _fused_mla_k_rope_op_fake(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
) -> torch.Tensor:
    return torch.empty(
        (*kv.shape[:3], q_nope_dim + k_pe.shape[-1]),
        dtype=kv.dtype,
        device=kv.device,
    )


@torch.library.custom_op(
    "torchtitan::fused_mla_kv_backward",
    mutates_args=(),
    device_types="cuda",
)
def _fused_mla_kv_backward_op(
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, n_heads, _ = grad_k.shape
    v_dim = grad_v.shape[-1]
    grad_kv = torch.empty(
        (batch, seq_len, n_heads, q_nope_dim + v_dim),
        dtype=grad_k.dtype,
        device=grad_k.device,
    )
    grad_k_pe = torch.empty(
        (batch, seq_len, rope_dim),
        dtype=grad_k.dtype,
        device=grad_k.device,
    )
    _fused_kv_backward_kernel[(batch * seq_len,)](
        grad_k,
        grad_v,
        rope_cache_real,
        positions,
        grad_kv,
        grad_k_pe,
        GK_STRIDE_B=grad_k.stride(0),
        GK_STRIDE_L=grad_k.stride(1),
        GK_STRIDE_H=grad_k.stride(2),
        GK_STRIDE_D=grad_k.stride(3),
        GV_STRIDE_B=grad_v.stride(0),
        GV_STRIDE_L=grad_v.stride(1),
        GV_STRIDE_H=grad_v.stride(2),
        GV_STRIDE_D=grad_v.stride(3),
        CACHE_STRIDE_M=rope_cache_real.stride(0),
        CACHE_STRIDE_P=rope_cache_real.stride(1),
        CACHE_STRIDE_R=rope_cache_real.stride(2),
        POS_STRIDE_B=positions.stride(0),
        POS_STRIDE_L=positions.stride(1),
        GKV_STRIDE_B=grad_kv.stride(0),
        GKV_STRIDE_L=grad_kv.stride(1),
        GKV_STRIDE_H=grad_kv.stride(2),
        GKV_STRIDE_D=grad_kv.stride(3),
        GKPE_STRIDE_B=grad_k_pe.stride(0),
        GKPE_STRIDE_L=grad_k_pe.stride(1),
        GKPE_STRIDE_D=grad_k_pe.stride(2),
        SEQ_LEN=seq_len,
        N_HEADS=n_heads,
        Q_NOPE_DIM=q_nope_dim,
        ROPE_DIM=rope_dim,
        V_DIM=v_dim,
        BLOCK_D=triton.next_power_of_2(max(q_nope_dim, v_dim)),
        BLOCK_PAIRS=triton.next_power_of_2(rope_dim // 2),
        ROUND_BF16_SUM=grad_k.dtype == torch.bfloat16,
        ROUND_FP16_SUM=grad_k.dtype == torch.float16,
    )
    return grad_kv, grad_k_pe


@_fused_mla_kv_backward_op.register_fake
def _fused_mla_kv_backward_op_fake(
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
    q_nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            (*grad_k.shape[:3], q_nope_dim + grad_v.shape[-1]),
            dtype=grad_k.dtype,
            device=grad_k.device,
        ),
        torch.empty(
            (*grad_k.shape[:2], rope_dim),
            dtype=grad_k.dtype,
            device=grad_k.device,
        ),
    )


class _FusedMLAQ(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(
        output: torch.Tensor,
        *,
        q: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        q_type = (spmd.V, spmd.PartitionSpec(None, ("dp", "cp"), "tp", None))
        positions_type = (spmd.V, spmd.PartitionSpec(None, ("dp", "cp")))
        spmd.assert_type(q, *q_type)
        spmd.assert_type(rope_cache_real, spmd.R)
        spmd.assert_type(positions, *positions_type)
        spmd.assert_type(output, *q_type)

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
        q_nope_dim: int,
    ) -> torch.Tensor:
        ctx.q_nope_dim = q_nope_dim
        ctx.save_for_backward(rope_cache_real, positions)
        ctx.mark_dirty(q)
        q_local = _to_local_for_mutation(q)
        _fused_mla_q_rope_op(
            q_local,
            rope_cache_real,
            positions,
            q_nope_dim,
            False,
        )
        return q

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q: torch.Tensor):
        rope_cache_real, positions = ctx.saved_tensors
        # grad_q may be an expanded tensor with zero strides (for example,
        # from fused_mla_q(...).sum()). Match Megatron's fused MLA path by
        # materializing only non-contiguous gradients before rotating in place.
        grad_q = grad_q.contiguous()
        grad_q_local = _to_local_for_mutation(grad_q)
        _fused_mla_q_rope_op(
            grad_q_local,
            rope_cache_real,
            positions,
            ctx.q_nope_dim,
            True,
        )
        return grad_q, None, None, None


class _FusedMLAKV(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(
        outputs: tuple[torch.Tensor, torch.Tensor],
        *,
        kv: torch.Tensor,
        k_pe: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        kv_partition = spmd.PartitionSpec(None, ("dp", "cp"), "tp", None)
        k_pe_partition = spmd.PartitionSpec(None, ("dp", "cp"), None)
        positions_partition = spmd.PartitionSpec(None, ("dp", "cp"))
        spmd.assert_type(kv, spmd.V, kv_partition)
        spmd.assert_type(k_pe, spmd.V, k_pe_partition)
        spmd.assert_type(rope_cache_real, spmd.R)
        spmd.assert_type(positions, spmd.V, positions_partition)
        k, v = outputs
        spmd.assert_type(k, spmd.V, kv_partition)
        spmd.assert_type(v, spmd.V, kv_partition)

    @staticmethod
    def forward(
        ctx,
        kv: torch.Tensor,
        k_pe: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
        q_nope_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.q_nope_dim = q_nope_dim
        ctx.rope_dim = k_pe.shape[-1]
        ctx.save_for_backward(rope_cache_real, positions)
        k = _fused_mla_k_rope_op(kv, k_pe, rope_cache_real, positions, q_nope_dim)
        # Preserve the stock zero-copy V view. The custom backward combines its
        # gradient with dK-nope directly into the packed KV gradient.
        v = kv[..., q_nope_dim:]
        return k, v

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_k: torch.Tensor, grad_v: torch.Tensor):
        rope_cache_real, positions = ctx.saved_tensors
        grad_kv, grad_k_pe = _fused_mla_kv_backward_op(
            grad_k,
            grad_v,
            rope_cache_real,
            positions,
            ctx.q_nope_dim,
            ctx.rope_dim,
        )
        return grad_kv, grad_k_pe, None, None, None


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _to_local_for_mutation(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        # DTensor.to_local() is an autograd view produced by _ToTorchTensor.
        # Mutating that view inside a custom Function is forbidden because it
        # would obscure the Function's custom backward. The underlying local
        # tensor carries the same autograd history without introducing that
        # extra view; DTensor.from_local() below restores the wrapper around the
        # custom Function's output.
        return tensor._local_tensor
    return tensor


def _from_local(local: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
    if isinstance(spec, DTensor):
        return DTensor.from_local(
            local,
            spec.device_mesh,
            spec.placements,
            run_check=False,
        )
    return local


def _resolve_positions(
    positions: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if positions is not None:
        pos = _to_local(positions)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        batch = reference.shape[0]
        if pos.shape[0] == 1 and batch != 1:
            pos = pos.expand(batch, -1)
        return pos.contiguous()
    batch, seq_len = reference.shape[:2]
    pos = torch.arange(seq_len, device=reference.device, dtype=torch.int32)
    return pos.unsqueeze(0).expand(batch, -1).contiguous()


def fused_mla_q(
    q: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
    q_nope_dim: int,
) -> torch.Tensor:
    """Apply ComplexRoPE in place to Q's positional tail."""
    q_local = _to_local(q)
    cache_local = _to_local(rope_cache)
    positions_local = _resolve_positions(positions, q_local)
    cache_real = torch.view_as_real(cache_local).contiguous()
    return _FusedMLAQ.apply(q, cache_real, positions_local, q_nope_dim)


def fused_mla_kv(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
    q_nope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize K and expose V with a fused custom backward."""
    kv_local = _to_local(kv)
    kpe_local = _to_local(k_pe)
    cache_local = _to_local(rope_cache)
    positions_local = _resolve_positions(positions, kv_local)
    cache_real = torch.view_as_real(cache_local).contiguous()
    k_local, v_local = _FusedMLAKV.apply(
        kv_local,
        kpe_local,
        cache_real,
        positions_local,
        q_nope_dim,
    )
    return _from_local(k_local, kv), _from_local(v_local, kv)


class FusedMLAAttention(Attention):
    """Stock DeepSeek-V3 attention with fused MLA tensor assembly."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if not isinstance(self.rope, ComplexRoPE):
            raise TypeError(
                "FusedMLAAttention currently requires ComplexRoPE, got "
                f"{type(self.rope).__name__}."
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            return super().forward(x, attention_masks, positions)

        num_tokens = x.shape[0]
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        with spmd.local():
            q = q.view(num_tokens, -1, self.qk_head_dim)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    q,
                    spmd.V,
                    spmd.PartitionSpec(("dp", "cp"), "tp", None),
                )

        if positions is not None:
            _maybe_check_max_pos(
                positions,
                max_valid_pos=self.rope.cache.shape[0] - 1,
            )
        q = fused_mla_q(
            q.unsqueeze(0),
            self.rope.cache,
            positions,
            self.qk_nope_head_dim,
        ).squeeze(0)

        kv_down = self.wkv_a(x)
        kv_latent, k_pe = torch.split(
            kv_down,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = self.wkv_b(self.kv_norm(kv_latent))
        with spmd.local():
            kv = kv.view(num_tokens, -1, self.qk_nope_head_dim + self.v_head_dim)
            k, v = fused_mla_kv(
                kv.unsqueeze(0),
                k_pe.unsqueeze(0),
                self.rope.cache,
                positions,
                self.qk_nope_head_dim,
            )
            k, v = k.squeeze(0), v.squeeze(0)
            if (
                get_spmd_backend() == "spmd_types"
                and spmd.is_type_checking()
                and not torch.compiler.is_compiling()
            ):
                for tensor in (k, v):
                    spmd.assert_type(
                        tensor,
                        spmd.V,
                        spmd.PartitionSpec(("dp", "cp"), "tp", None),
                    )

        output = self.inner_attention(
            q,
            k,
            v,
            attention_masks=attention_masks,
            scale=self.softmax_scale,
        ).contiguous()
        output = output.view(num_tokens, -1)
        return self.wo(output)


@override(
    target=Attention.Config,
    description="Fuse DeepSeek-V3 MLA Q/KV RoPE assembly with Triton kernels.",
)
def fused_mla(cfg: Attention.Config) -> FusedMLAAttention.Config:
    return derive(cfg, FusedMLAAttention.Config)
