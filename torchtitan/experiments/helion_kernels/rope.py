# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(block_sizes=[512, 512], num_warps=4),
    static_shapes=True,
)
def _rope_cos_sin_fwd_positions(
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


def _nearest_power_of_2_bucket(value: int) -> int:
    assert value > 0
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    return lower if value - lower < upper - value else upper


def _config(
    block_sizes: list[int],
    *,
    num_warps: int,
    num_stages: int,
    pid_type: str = "flat",
    indexing: list[str] | None = None,
    load_eviction_policies: list[str] | None = None,
    maxnreg: int | None = None,
    num_sm_multiplier: int | None = None,
) -> helion.Config:
    kwargs: dict[str, Any] = {
        "block_sizes": block_sizes,
        "num_stages": num_stages,
        "num_warps": num_warps,
        "pid_type": pid_type,
    }
    if indexing is not None:
        kwargs["indexing"] = indexing
    if load_eviction_policies is not None:
        kwargs["load_eviction_policies"] = load_eviction_policies
    if maxnreg is not None:
        kwargs["maxnreg"] = maxnreg
    if num_sm_multiplier is not None:
        kwargs["num_sm_multiplier"] = num_sm_multiplier
    return helion.Config(**kwargs)


_ROPE_HELION_CONFIGS_BY_SEQ_BUCKET: dict[int, helion.Config] = {
    1: _config(
        [512, 128],
        num_warps=16,
        num_stages=5,
        pid_type="persistent_blocked",
        maxnreg=128,
        num_sm_multiplier=1,
    ),
    256: _config(
        [1024, 128],
        num_warps=8,
        num_stages=4,
    ),
    1024: _config(
        [512, 256],
        num_warps=8,
        num_stages=2,
    ),
}


_ROPE_HELION_BWD_CONFIGS_BY_SEQ_BUCKET: dict[int, helion.Config] = {
    1: _config([128, 16], num_warps=4, num_stages=3),
    256: _config([1024, 512], num_warps=16, num_stages=6),
    1024: _config([1024, 512], num_warps=8, num_stages=7),
}


def _rope_cos_sin_config(
    xq: torch.Tensor,
    xk: torch.Tensor,
) -> helion.Config | None:
    batch, seq_len, n_heads, head_dim = xq.shape
    _, k_seq_len, n_kv_heads, k_head_dim = xk.shape
    # These configs were tuned only for the Qwen3-32B TP=8 inference shapes:
    # xq=[1, S, 8, 128], xk=[1, S, 1, 128], S in {1, 192, 1088}.
    if (
        batch != 1
        or seq_len != k_seq_len
        or n_heads != 8
        or n_kv_heads != 1
        or head_dim != 128
        or k_head_dim != 128
    ):
        return None

    seq_bucket = _nearest_power_of_2_bucket(seq_len)
    return _ROPE_HELION_CONFIGS_BY_SEQ_BUCKET.get(seq_bucket)


def _rope_cos_sin_bwd_config(
    grad_xq_out: torch.Tensor,
    grad_xk_out: torch.Tensor,
) -> helion.Config | None:
    batch, seq_len, n_heads, head_dim = grad_xq_out.shape
    _, k_seq_len, n_kv_heads, k_head_dim = grad_xk_out.shape
    # These configs were tuned only for the Qwen3-32B TP=8 inference shapes:
    # grad_xq=[1, S, 8, 128], grad_xk=[1, S, 1, 128], S in {1, 192, 1088}.
    if (
        batch != 1
        or seq_len != k_seq_len
        or n_heads != 8
        or n_kv_heads != 1
        or head_dim != 128
        or k_head_dim != 128
    ):
        return None

    seq_bucket = _nearest_power_of_2_bucket(seq_len)
    return _ROPE_HELION_BWD_CONFIGS_BY_SEQ_BUCKET.get(seq_bucket)


def _rope_cos_sin_fwd_positions_with_config(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = _rope_cos_sin_config(xq, xk)
    if config is None:
        return _rope_cos_sin_fwd_positions(xq, xk, rope_cache, positions)

    bound_kernel = _rope_cos_sin_fwd_positions.bind((xq, xk, rope_cache, positions))
    if getattr(bound_kernel, "_config", None) != config:
        bound_kernel.set_config(config)
    return bound_kernel(xq, xk, rope_cache, positions)


def _rope_cos_sin_bwd_positions_with_config(
    grad_xq_out: torch.Tensor,
    grad_xk_out: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = _rope_cos_sin_bwd_config(grad_xq_out, grad_xk_out)
    if config is None:
        return _rope_cos_sin_bwd_positions(
            grad_xq_out, grad_xk_out, rope_cache, positions
        )

    args = (grad_xq_out, grad_xk_out, rope_cache, positions)
    bound_kernel = _rope_cos_sin_bwd_positions.bind(args)
    if getattr(bound_kernel, "_config", None) != config:
        bound_kernel.set_config(config)
    return bound_kernel(*args)


@helion.kernel(
    config=helion.Config(block_sizes=[512, 512], num_warps=4),
    static_shapes=True,
)
def _rope_cos_sin_bwd_positions(
    grad_xq_out: torch.Tensor,
    grad_xk_out: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, seqlen, n_heads, head_dim = grad_xq_out.size()
    _, _, n_kv_heads, _ = grad_xk_out.size()

    grad_xq = torch.empty_like(grad_xq_out)
    grad_xk = torch.empty_like(grad_xk_out)

    grad_xq_out_flat = grad_xq_out.view(-1)
    grad_xk_out_flat = grad_xk_out.view(-1)
    grad_xq_flat = grad_xq.view(-1)
    grad_xk_flat = grad_xk.view(-1)
    rope_cache_flat = rope_cache.view(-1)
    positions_flat = positions.view(-1)

    half_head_dim = head_dim // 2
    rope_cache_stride = head_dim * 2

    for tile_q in hl.tile(grad_xq_out.numel()):
        idx_q = tile_q.index
        head_dim_idx_q = idx_q % head_dim
        seq_idx_q = (idx_q // (head_dim * n_heads)) % seqlen
        batch_idx_q = idx_q // (head_dim * n_heads * seqlen)

        position_q = positions_flat[batch_idx_q * seqlen + seq_idx_q]
        paired_idx_q = torch.where(
            head_dim_idx_q < half_head_dim,
            idx_q + half_head_dim,
            idx_q - half_head_dim,
        )
        paired_head_dim_idx_q = torch.where(
            head_dim_idx_q < half_head_dim,
            head_dim_idx_q + half_head_dim,
            head_dim_idx_q - half_head_dim,
        )
        paired_sign_q = torch.where(head_dim_idx_q < half_head_dim, 1.0, -1.0).to(
            torch.float32
        )

        grad_self_q = grad_xq_out_flat[idx_q].to(torch.float32)
        grad_pair_q = grad_xq_out_flat[paired_idx_q].to(torch.float32)
        cos_q = rope_cache_flat[position_q * rope_cache_stride + head_dim_idx_q].to(
            torch.float32
        )
        sin_pair_q = rope_cache_flat[
            position_q * rope_cache_stride + head_dim + paired_head_dim_idx_q
        ].to(torch.float32)
        grad_self_term_q = (grad_self_q * cos_q).to(grad_xq.dtype)
        grad_pair_term_q = (grad_pair_q * sin_pair_q * paired_sign_q).to(grad_xq.dtype)
        grad_xq_flat[idx_q] = grad_self_term_q + grad_pair_term_q

    for tile_k in hl.tile(grad_xk_out.numel()):
        idx_k = tile_k.index
        head_dim_idx_k = idx_k % head_dim
        seq_idx_k = (idx_k // (head_dim * n_kv_heads)) % seqlen
        batch_idx_k = idx_k // (head_dim * n_kv_heads * seqlen)

        position_k = positions_flat[batch_idx_k * seqlen + seq_idx_k]
        paired_idx_k = torch.where(
            head_dim_idx_k < half_head_dim,
            idx_k + half_head_dim,
            idx_k - half_head_dim,
        )
        paired_head_dim_idx_k = torch.where(
            head_dim_idx_k < half_head_dim,
            head_dim_idx_k + half_head_dim,
            head_dim_idx_k - half_head_dim,
        )
        paired_sign_k = torch.where(head_dim_idx_k < half_head_dim, 1.0, -1.0).to(
            torch.float32
        )

        grad_self_k = grad_xk_out_flat[idx_k].to(torch.float32)
        grad_pair_k = grad_xk_out_flat[paired_idx_k].to(torch.float32)
        cos_k = rope_cache_flat[position_k * rope_cache_stride + head_dim_idx_k].to(
            torch.float32
        )
        sin_pair_k = rope_cache_flat[
            position_k * rope_cache_stride + head_dim + paired_head_dim_idx_k
        ].to(torch.float32)
        grad_self_term_k = (grad_self_k * cos_k).to(grad_xk.dtype)
        grad_pair_term_k = (grad_pair_k * sin_pair_k * paired_sign_k).to(grad_xk.dtype)
        grad_xk_flat[idx_k] = grad_self_term_k + grad_pair_term_k

    return grad_xq, grad_xk


@helion.kernel(
    config=helion.Config(block_sizes=[8, 8], num_warps=1),
    static_shapes=True,
)
def _qk_rmsnorm_rope_cos_sin_fwd_positions(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seqlen, n_heads, head_dim = xq.size()
    _, _, n_kv_heads, _ = xk.size()
    head_dim = hl.specialize(head_dim)

    xq_2d = xq.reshape(-1, head_dim)
    xk_2d = xk.reshape(-1, head_dim)
    xq_out = torch.empty_like(xq)
    xk_out = torch.empty_like(xk)
    xq_out_2d = xq_out.reshape(-1, head_dim)
    xk_out_2d = xk_out.reshape(-1, head_dim)
    positions_flat = positions.view(-1)

    q_rows = xq_2d.size(0)
    k_rows = xk_2d.size(0)

    for tile_q in hl.tile(q_rows):
        row_q = tile_q.index
        seq_idx_q = (row_q // n_heads) % seqlen
        batch_idx_q = row_q // (n_heads * seqlen)
        position_q = positions_flat[batch_idx_q * seqlen + seq_idx_q]

        xq_tile = xq_2d[tile_q, :].to(torch.float32)
        q_variance = torch.mean(xq_tile * xq_tile, dim=-1)
        q_inv_rms = torch.rsqrt(q_variance + eps)
        xq_norm = xq_tile * q_inv_rms[:, None] * q_weight[None, :].to(torch.float32)
        rope_q = rope_cache[position_q, :].to(torch.float32)
        cos_q, sin_q = hl.split(rope_q.reshape([tile_q, 2, head_dim]).permute(0, 2, 1))
        xq_norm_lo, xq_norm_hi = hl.split(
            xq_norm.reshape([tile_q, 2, head_dim // 2]).permute(0, 2, 1)
        )
        cos_q_lo, cos_q_hi = hl.split(
            cos_q.reshape([tile_q, 2, head_dim // 2]).permute(0, 2, 1)
        )
        sin_q_lo, sin_q_hi = hl.split(
            sin_q.reshape([tile_q, 2, head_dim // 2]).permute(0, 2, 1)
        )

        xq_out_lo = xq_norm_lo * cos_q_lo - xq_norm_hi * sin_q_lo
        xq_out_hi = xq_norm_hi * cos_q_hi + xq_norm_lo * sin_q_hi
        xq_out_2d[tile_q, :] = (
            hl.join(xq_out_lo, xq_out_hi)
            .permute(0, 2, 1)
            .reshape([tile_q, head_dim])
            .to(xq.dtype)
        )

    for tile_k in hl.tile(k_rows):
        row_k = tile_k.index
        seq_idx_k = (row_k // n_kv_heads) % seqlen
        batch_idx_k = row_k // (n_kv_heads * seqlen)
        position_k = positions_flat[batch_idx_k * seqlen + seq_idx_k]

        xk_tile = xk_2d[tile_k, :].to(torch.float32)
        k_variance = torch.mean(xk_tile * xk_tile, dim=-1)
        k_inv_rms = torch.rsqrt(k_variance + eps)
        xk_norm = xk_tile * k_inv_rms[:, None] * k_weight[None, :].to(torch.float32)
        rope_k = rope_cache[position_k, :].to(torch.float32)
        cos_k, sin_k = hl.split(rope_k.reshape([tile_k, 2, head_dim]).permute(0, 2, 1))
        xk_norm_lo, xk_norm_hi = hl.split(
            xk_norm.reshape([tile_k, 2, head_dim // 2]).permute(0, 2, 1)
        )
        cos_k_lo, cos_k_hi = hl.split(
            cos_k.reshape([tile_k, 2, head_dim // 2]).permute(0, 2, 1)
        )
        sin_k_lo, sin_k_hi = hl.split(
            sin_k.reshape([tile_k, 2, head_dim // 2]).permute(0, 2, 1)
        )

        xk_out_lo = xk_norm_lo * cos_k_lo - xk_norm_hi * sin_k_lo
        xk_out_hi = xk_norm_hi * cos_k_hi + xk_norm_lo * sin_k_hi
        xk_out_2d[tile_k, :] = (
            hl.join(xk_out_lo, xk_out_hi)
            .permute(0, 2, 1)
            .reshape([tile_k, head_dim])
            .to(xk.dtype)
        )

    return xq_out, xk_out


class _ApplyRotaryEmbCosSinHelion(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.xq_shape = xq.shape
        ctx.xk_shape = xk.shape
        ctx.save_for_backward(rope_cache, positions)
        return _rope_cos_sin_fwd_positions_with_config(
            xq, xk, rope_cache, positions
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_xq_out: torch.Tensor | None,
        grad_xk_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        rope_cache, positions = ctx.saved_tensors
        if grad_xq_out is None:
            assert grad_xk_out is not None
            grad_xq_out = torch.zeros(
                ctx.xq_shape,
                device=grad_xk_out.device,
                dtype=grad_xk_out.dtype,
            )
        if grad_xk_out is None:
            grad_xk_out = torch.zeros(
                ctx.xk_shape,
                device=grad_xq_out.device,
                dtype=grad_xq_out.dtype,
            )

        grad_xq, grad_xk = _rope_cos_sin_bwd_positions_with_config(
            grad_xq_out.contiguous(),
            grad_xk_out.contiguous(),
            rope_cache,
            positions,
        )
        if not ctx.needs_input_grad[0]:
            grad_xq = None
        if not ctx.needs_input_grad[1]:
            grad_xk = None
        return grad_xq, grad_xk, None, None


def _can_use_helion_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
) -> bool:
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


def _can_use_helion_qk_rmsnorm_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
) -> bool:
    return (
        positions is not None
        and xq.ndim == 4
        and xk.ndim == 4
        and q_weight.ndim == 1
        and k_weight.ndim == 1
        and rope_cache.ndim == 2
        and positions.ndim == 2
        and positions.shape == (xq.shape[0], xq.shape[1])
        and xq.shape[0] == xk.shape[0]
        and xq.shape[1] == xk.shape[1]
        and xq.shape[-1] == xk.shape[-1]
        and xq.shape[-1] == q_weight.shape[0]
        and xk.shape[-1] == k_weight.shape[0]
        and rope_cache.shape[-1] == xq.shape[-1] * 2
        and xq.shape[-1] % 2 == 0
        and xq.is_cuda
        and xk.is_cuda
        and q_weight.is_cuda
        and k_weight.is_cuda
        and rope_cache.is_cuda
        and positions.is_cuda
        and xq.is_contiguous()
        and xk.is_contiguous()
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and rope_cache.is_contiguous()
        and positions.is_contiguous()
        and xq.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and xk.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q_weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and k_weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not (
            torch.is_grad_enabled()
            and (
                xq.requires_grad
                or xk.requires_grad
                or q_weight.requires_grad
                or k_weight.requires_grad
            )
        )
    )


def _qk_rmsnorm_rope_cos_sin_reference(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torchtitan.models.common.rope import apply_rotary_emb_cos_sin

    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, eps)
    return apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)


def apply_rotary_emb_cos_sin_helion(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply cos/sin RoPE with an experimental Helion fused kernel.

    The Helion fast path currently targets the common training case:
    contiguous CUDA ``xq``/``xk`` tensors, a 2D cos/sin cache, and batched
    position IDs of shape ``(batch, seq_len)``. Other cases fall back to the
    core PyTorch implementation.
    """
    if not _can_use_helion_rope(xq, xk, rope_cache, positions):
        from torchtitan.models.common.rope import apply_rotary_emb_cos_sin

        return apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)

    assert positions is not None
    return _ApplyRotaryEmbCosSinHelion.apply(xq, xk, rope_cache, positions)


def apply_qk_rmsnorm_rotary_emb_cos_sin_helion(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Q/K RMSNorm and cos/sin RoPE with an experimental Helion kernel.

    The fast path is forward-only and targets inference-style contiguous CUDA
    tensors shaped ``(batch, seq_len, heads, head_dim)`` with a 2D cos/sin RoPE
    cache and batched position IDs. Other cases fall back to the PyTorch
    composition.
    """
    if not _can_use_helion_qk_rmsnorm_rope(
        xq, xk, q_weight, k_weight, rope_cache, positions
    ):
        return _qk_rmsnorm_rope_cos_sin_reference(
            xq, xk, q_weight, k_weight, rope_cache, positions, eps
        )

    assert positions is not None
    return _qk_rmsnorm_rope_cos_sin_fwd_positions(
        xq, xk, q_weight, k_weight, rope_cache, positions, eps
    )
