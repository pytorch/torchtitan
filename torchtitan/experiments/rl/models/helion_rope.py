# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Forward-only Helion cos/sin RoPE kernel for the inference ablation.

Adapted from the experimental Helion kernels in pytorch/torchtitan (PR #3342 /
#3651): the original ``apply_rotary_emb_cos_sin`` fused position lookup,
rotate-half, fp32 math, and cast-back into one GPU kernel. The core
PyTorch path instead materializes a broadcast/gathered cache, casts to fp32,
builds rotate_half with torch.cat, then multiplies/adds -- several intermediate
allocations per call across all layers.

Only the forward kernel is needed for inference. ``rope_cache`` is the raw
``(max_seq_len, head_dim * 2)`` cos/sin cache (CosSinRoPE.cache); the kernel
does the per-token position lookup internally.
"""
from __future__ import annotations

import helion
import helion.language as hl
import torch


@helion.kernel(
    config=helion.Config(block_sizes=[512, 512], num_warps=4),
    static_shapes=True,
)
def rope_cos_sin_fwd(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, seqlen, n_heads, head_dim = xq.size()
    _, _, n_kv_heads, _ = xk.size()

    xq_out = torch.empty_like(xq)
    xk_out = torch.empty_like(xk)

    xq_flat = xq.view(-1)
    xk_flat = xk.view(-1)
    xq_out_flat = xq_out.view(-1)
    xk_out_flat = xk_out.view(-1)
    rope_cache_flat = rope_cache.view(-1)
    positions_flat = positions.view(-1)

    half_head_dim = head_dim // 2
    rope_cache_stride = head_dim * 2

    for tile_q in hl.tile(xq.numel()):
        idx_q = tile_q.index
        head_dim_idx_q = idx_q % head_dim
        seq_idx_q = (idx_q // (head_dim * n_heads)) % seqlen
        batch_idx_q = idx_q // (head_dim * n_heads * seqlen)

        position_q = positions_flat[batch_idx_q * seqlen + seq_idx_q]
        rotated_idx_q = torch.where(
            head_dim_idx_q < half_head_dim,
            idx_q + half_head_dim,
            idx_q - half_head_dim,
        )
        rotated_sign_q = torch.where(head_dim_idx_q < half_head_dim, -1.0, 1.0).to(
            torch.float32
        )

        xq_val = xq_flat[idx_q].to(torch.float32)
        xq_rotated = xq_flat[rotated_idx_q].to(torch.float32) * rotated_sign_q
        cos_q = rope_cache_flat[position_q * rope_cache_stride + head_dim_idx_q].to(
            torch.float32
        )
        sin_q = rope_cache_flat[
            position_q * rope_cache_stride + head_dim + head_dim_idx_q
        ].to(torch.float32)
        xq_out_flat[idx_q] = (xq_val * cos_q + xq_rotated * sin_q).to(xq.dtype)

    for tile_k in hl.tile(xk.numel()):
        idx_k = tile_k.index
        head_dim_idx_k = idx_k % head_dim
        seq_idx_k = (idx_k // (head_dim * n_kv_heads)) % seqlen
        batch_idx_k = idx_k // (head_dim * n_kv_heads * seqlen)

        position_k = positions_flat[batch_idx_k * seqlen + seq_idx_k]
        rotated_idx_k = torch.where(
            head_dim_idx_k < half_head_dim,
            idx_k + half_head_dim,
            idx_k - half_head_dim,
        )
        rotated_sign_k = torch.where(head_dim_idx_k < half_head_dim, -1.0, 1.0).to(
            torch.float32
        )

        xk_val = xk_flat[idx_k].to(torch.float32)
        xk_rotated = xk_flat[rotated_idx_k].to(torch.float32) * rotated_sign_k
        cos_k = rope_cache_flat[position_k * rope_cache_stride + head_dim_idx_k].to(
            torch.float32
        )
        sin_k = rope_cache_flat[
            position_k * rope_cache_stride + head_dim + head_dim_idx_k
        ].to(torch.float32)
        xk_out_flat[idx_k] = (xk_val * cos_k + xk_rotated * sin_k).to(xk.dtype)

    return xq_out, xk_out


@torch.library.custom_op("ttablation::rope_helion_fwd", mutates_args=())
def rope_helion_fwd(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque custom-op wrapper around the Helion kernel.

    torch.compile (aot_eager, used by the cudagraph rung) traces the model with
    symbolic shapes; the Helion kernel uses static_shapes and chokes on symbolic
    dims. Wrapping it as a custom op makes dynamo treat the call as opaque (it
    uses register_fake for shape propagation) while the real kernel runs on the
    concrete tensors at execution / cudagraph-capture time.
    """
    return rope_cos_sin_fwd(
        xq.contiguous(),
        xk.contiguous(),
        rope_cache.contiguous(),
        positions.contiguous(),
    )


@rope_helion_fwd.register_fake
def _rope_helion_fwd_fake(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(xq), torch.empty_like(xk)


def register_rope_helion_sharding() -> None:
    """Register a DTensor sharding strategy for rope_helion_fwd so it can be
    called directly on DTensors (q/k sharded on heads) without to_local/from_local
    round-trips. Per-head op: head_dim (last) never sharded; positions/cache are
    replicated. Idempotent-safe to call once.
    """
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.experimental import register_sharding

    @register_sharding(torch.ops.ttablation.rope_helion_fwd.default)
    def _strategy(xq, xk, rope_cache, positions):
        # inputs: xq, xk (b,s,H,D) Shard(2); rope_cache, positions Replicate.
        # outputs: xq_out, xk_out same placement as xq/xk.
        return [
            (
                [Replicate(), Replicate(), Replicate(), Replicate()],
                [Replicate(), Replicate()],
            ),
            ([Shard(2), Shard(2), Replicate(), Replicate()], [Shard(2), Shard(2)]),
        ]


def can_use_helion_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
) -> bool:
    """Fast-path guard. Other cases fall back to the core CosSinRoPE path."""
    return (
        positions is not None
        and rope_cache.ndim == 2
        and positions.ndim == 2
        and positions.shape == (xq.shape[0], xq.shape[1])
        and xq.is_cuda
        and xk.is_cuda
        and rope_cache.is_cuda
        and positions.is_cuda
        and xq.is_contiguous()
        and xk.is_contiguous()
        and rope_cache.is_contiguous()
        and positions.is_contiguous()
        and xq.shape[-1] == xk.shape[-1]
        and xq.shape[-1] % 2 == 0
    )
