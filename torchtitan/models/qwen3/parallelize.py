# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 parallelization scaffold.

This file is intentionally strategy-free. Autoresearch is expected to replace
``parallelize_qwen3`` with an implementation specialized to the exact train
command, model flavor, and cluster/system it is optimizing for.
"""

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.attention import SDPBackend

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.components.quantization.mx import (
    mxfp8_shared_input_gate_up,
    mxfp8_shared_input_gate_up_forward,
    mxfp8_shared_input_gate_up_forward_grad_input_concat,
    mxfp8_shared_input_gate_up_grad_input_concat,
    mxfp8_shared_input_qkv,
    mxfp8_shared_input_qkv_backward_concat,
    mxfp8_shared_input_qkv_forward_grad_input_concat,
    mxfp8_shared_input_qkv_grad_input_concat,
    mxfp8_shared_input_qkv_grad_weight_concat,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.tools.logging import logger


_PAIRWISE_ROPE_BLOCK = 256


@triton.jit
def _sequential_rope_forward_kernel(
    x_ptr,
    rope_ptr,
    out_ptr,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    d = offsets % head_dim
    tmp = offsets // head_dim
    s = (tmp // num_heads) % seq_len
    half = head_dim // 2
    first_half = d < half
    other_offsets = tl.where(first_half, offsets + half, offsets - half)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    other = tl.load(x_ptr + other_offsets, mask=mask).to(tl.float32)
    cos = tl.load(rope_ptr + s * (head_dim * 2) + d, mask=mask).to(tl.float32)
    sin = tl.load(rope_ptr + s * (head_dim * 2) + head_dim + d, mask=mask).to(
        tl.float32
    )
    rotated = tl.where(first_half, -other, other)
    tl.store(out_ptr + offsets, x * cos + rotated * sin, mask=mask)


@triton.jit
def _sequential_rope_backward_kernel(
    grad_ptr,
    rope_ptr,
    out_ptr,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    d = offsets % head_dim
    tmp = offsets // head_dim
    s = (tmp // num_heads) % seq_len
    half = head_dim // 2
    first_half = d < half
    other_offsets = tl.where(first_half, offsets + half, offsets - half)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    other_grad = tl.load(grad_ptr + other_offsets, mask=mask).to(tl.float32)
    cos = tl.load(rope_ptr + s * (head_dim * 2) + d, mask=mask).to(tl.float32)
    sin = tl.load(rope_ptr + s * (head_dim * 2) + head_dim + d, mask=mask).to(
        tl.float32
    )
    rotated_grad = tl.where(first_half, other_grad, -other_grad)
    tl.store(out_ptr + offsets, grad * cos + rotated_grad * sin, mask=mask)


@triton.jit
def _pairwise_rope_forward_kernel(
    x_ptr,
    rope_ptr,
    out_ptr,
    total_pairs: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair_offsets < total_pairs
    half = head_dim // 2
    d = pair_offsets % half
    row = pair_offsets // half
    s = (row // num_heads) % seq_len
    first_offsets = row * head_dim + d
    second_offsets = first_offsets + half
    x0 = tl.load(x_ptr + first_offsets, mask=mask).to(tl.float32)
    x1 = tl.load(x_ptr + second_offsets, mask=mask).to(tl.float32)
    rope_base = s * (head_dim * 2)
    cos0 = tl.load(rope_ptr + rope_base + d, mask=mask).to(tl.float32)
    sin0 = tl.load(rope_ptr + rope_base + head_dim + d, mask=mask).to(tl.float32)
    cos1 = tl.load(rope_ptr + rope_base + d + half, mask=mask).to(tl.float32)
    sin1 = tl.load(rope_ptr + rope_base + head_dim + d + half, mask=mask).to(
        tl.float32
    )
    tl.store(out_ptr + first_offsets, x0 * cos0 - x1 * sin0, mask=mask)
    tl.store(out_ptr + second_offsets, x1 * cos1 + x0 * sin1, mask=mask)


@triton.jit
def _pairwise_rope_backward_kernel(
    grad_ptr,
    rope_ptr,
    out_ptr,
    total_pairs: tl.constexpr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair_offsets < total_pairs
    half = head_dim // 2
    d = pair_offsets % half
    row = pair_offsets // half
    s = (row // num_heads) % seq_len
    first_offsets = row * head_dim + d
    second_offsets = first_offsets + half
    grad0 = tl.load(grad_ptr + first_offsets, mask=mask).to(tl.float32)
    grad1 = tl.load(grad_ptr + second_offsets, mask=mask).to(tl.float32)
    rope_base = s * (head_dim * 2)
    cos0 = tl.load(rope_ptr + rope_base + d, mask=mask).to(tl.float32)
    sin0 = tl.load(rope_ptr + rope_base + head_dim + d, mask=mask).to(tl.float32)
    cos1 = tl.load(rope_ptr + rope_base + d + half, mask=mask).to(tl.float32)
    sin1 = tl.load(rope_ptr + rope_base + head_dim + d + half, mask=mask).to(
        tl.float32
    )
    tl.store(out_ptr + first_offsets, grad0 * cos0 + grad1 * sin0, mask=mask)
    tl.store(out_ptr + second_offsets, grad1 * cos1 - grad0 * sin1, mask=mask)


@triton.jit
def _fused_qk_pairwise_rope_forward_kernel(
    q_ptr,
    k_ptr,
    rope_ptr,
    q_out_ptr,
    k_out_ptr,
    q_total_pairs: tl.constexpr,
    k_total_pairs: tl.constexpr,
    q_num_heads: tl.constexpr,
    k_num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    q_blocks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    is_q = pid < q_blocks
    local_pid = tl.where(is_q, pid, pid - q_blocks)
    pair_offsets = local_pid * BLOCK + tl.arange(0, BLOCK)
    half = head_dim // 2
    d = pair_offsets % half
    q_row = pair_offsets // half
    k_row = pair_offsets // half
    q_s = (q_row // q_num_heads) % seq_len
    k_s = (k_row // k_num_heads) % seq_len
    q_first_offsets = q_row * head_dim + d
    k_first_offsets = k_row * head_dim + d
    q_second_offsets = q_first_offsets + half
    k_second_offsets = k_first_offsets + half
    q_mask = is_q & (pair_offsets < q_total_pairs)
    k_mask = (~is_q) & (pair_offsets < k_total_pairs)

    q_x0 = tl.load(q_ptr + q_first_offsets, mask=q_mask).to(tl.float32)
    q_x1 = tl.load(q_ptr + q_second_offsets, mask=q_mask).to(tl.float32)
    q_rope_base = q_s * (head_dim * 2)
    q_cos0 = tl.load(rope_ptr + q_rope_base + d, mask=q_mask).to(tl.float32)
    q_sin0 = tl.load(rope_ptr + q_rope_base + head_dim + d, mask=q_mask).to(
        tl.float32
    )
    q_cos1 = tl.load(rope_ptr + q_rope_base + d + half, mask=q_mask).to(tl.float32)
    q_sin1 = tl.load(
        rope_ptr + q_rope_base + head_dim + d + half,
        mask=q_mask,
    ).to(tl.float32)
    tl.store(q_out_ptr + q_first_offsets, q_x0 * q_cos0 - q_x1 * q_sin0, mask=q_mask)
    tl.store(q_out_ptr + q_second_offsets, q_x1 * q_cos1 + q_x0 * q_sin1, mask=q_mask)

    k_x0 = tl.load(k_ptr + k_first_offsets, mask=k_mask).to(tl.float32)
    k_x1 = tl.load(k_ptr + k_second_offsets, mask=k_mask).to(tl.float32)
    k_rope_base = k_s * (head_dim * 2)
    k_cos0 = tl.load(rope_ptr + k_rope_base + d, mask=k_mask).to(tl.float32)
    k_sin0 = tl.load(rope_ptr + k_rope_base + head_dim + d, mask=k_mask).to(
        tl.float32
    )
    k_cos1 = tl.load(rope_ptr + k_rope_base + d + half, mask=k_mask).to(tl.float32)
    k_sin1 = tl.load(
        rope_ptr + k_rope_base + head_dim + d + half,
        mask=k_mask,
    ).to(tl.float32)
    tl.store(k_out_ptr + k_first_offsets, k_x0 * k_cos0 - k_x1 * k_sin0, mask=k_mask)
    tl.store(k_out_ptr + k_second_offsets, k_x1 * k_cos1 + k_x0 * k_sin1, mask=k_mask)


@triton.jit
def _fused_qk_pairwise_rope_backward_kernel(
    q_grad_ptr,
    k_grad_ptr,
    rope_ptr,
    q_out_ptr,
    k_out_ptr,
    q_total_pairs: tl.constexpr,
    k_total_pairs: tl.constexpr,
    q_num_heads: tl.constexpr,
    k_num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    q_blocks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    is_q = pid < q_blocks
    local_pid = tl.where(is_q, pid, pid - q_blocks)
    pair_offsets = local_pid * BLOCK + tl.arange(0, BLOCK)
    half = head_dim // 2
    d = pair_offsets % half
    q_row = pair_offsets // half
    k_row = pair_offsets // half
    q_s = (q_row // q_num_heads) % seq_len
    k_s = (k_row // k_num_heads) % seq_len
    q_first_offsets = q_row * head_dim + d
    k_first_offsets = k_row * head_dim + d
    q_second_offsets = q_first_offsets + half
    k_second_offsets = k_first_offsets + half
    q_mask = is_q & (pair_offsets < q_total_pairs)
    k_mask = (~is_q) & (pair_offsets < k_total_pairs)

    q_grad0 = tl.load(q_grad_ptr + q_first_offsets, mask=q_mask).to(tl.float32)
    q_grad1 = tl.load(q_grad_ptr + q_second_offsets, mask=q_mask).to(tl.float32)
    q_rope_base = q_s * (head_dim * 2)
    q_cos0 = tl.load(rope_ptr + q_rope_base + d, mask=q_mask).to(tl.float32)
    q_sin0 = tl.load(rope_ptr + q_rope_base + head_dim + d, mask=q_mask).to(
        tl.float32
    )
    q_cos1 = tl.load(rope_ptr + q_rope_base + d + half, mask=q_mask).to(tl.float32)
    q_sin1 = tl.load(
        rope_ptr + q_rope_base + head_dim + d + half,
        mask=q_mask,
    ).to(tl.float32)
    tl.store(q_out_ptr + q_first_offsets, q_grad0 * q_cos0 + q_grad1 * q_sin0, mask=q_mask)
    tl.store(
        q_out_ptr + q_second_offsets,
        q_grad1 * q_cos1 - q_grad0 * q_sin1,
        mask=q_mask,
    )

    k_grad0 = tl.load(k_grad_ptr + k_first_offsets, mask=k_mask).to(tl.float32)
    k_grad1 = tl.load(k_grad_ptr + k_second_offsets, mask=k_mask).to(tl.float32)
    k_rope_base = k_s * (head_dim * 2)
    k_cos0 = tl.load(rope_ptr + k_rope_base + d, mask=k_mask).to(tl.float32)
    k_sin0 = tl.load(rope_ptr + k_rope_base + head_dim + d, mask=k_mask).to(
        tl.float32
    )
    k_cos1 = tl.load(rope_ptr + k_rope_base + d + half, mask=k_mask).to(tl.float32)
    k_sin1 = tl.load(
        rope_ptr + k_rope_base + head_dim + d + half,
        mask=k_mask,
    ).to(tl.float32)
    tl.store(k_out_ptr + k_first_offsets, k_grad0 * k_cos0 + k_grad1 * k_sin0, mask=k_mask)
    tl.store(
        k_out_ptr + k_second_offsets,
        k_grad1 * k_cos1 - k_grad0 * k_sin1,
        mask=k_mask,
    )


@triton.jit
def _swiglu_forward_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offsets, gate * sigmoid * up, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    grad_out_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sigmoid
    grad_silu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    tl.store(grad_gate_ptr + offsets, grad_out * up * grad_silu, mask=mask)
    tl.store(grad_up_ptr + offsets, grad_out * silu, mask=mask)


class _TritonSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate = gate.contiguous()
        up = up.contiguous()
        out = torch.empty_like(gate)
        block = 1024
        grid = (triton.cdiv(gate.numel(), block),)
        _swiglu_forward_kernel[grid](
            gate,
            up,
            out,
            gate.numel(),
            BLOCK=block,
        )
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        block = 1024
        grid = (triton.cdiv(gate.numel(), block),)
        _swiglu_backward_kernel[grid](
            grad_out,
            gate,
            up,
            grad_gate,
            grad_up,
            gate.numel(),
            BLOCK=block,
        )
        return grad_gate, grad_up


def _triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _TritonSwiGLU.apply(gate, up)


class _SequentialRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        rope_cache = rope_cache.contiguous()
        out = torch.empty_like(x)
        _, seq_len, num_heads, head_dim = x.shape
        block = 256
        grid = (triton.cdiv(x.numel(), block),)
        _sequential_rope_forward_kernel[grid](
            x,
            rope_cache,
            out,
            x.numel(),
            seq_len,
            num_heads,
            head_dim,
            BLOCK=block,
        )
        ctx.save_for_backward(rope_cache)
        ctx.shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rope_cache,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(grad_out)
        _, seq_len, num_heads, head_dim = ctx.shape
        block = 256
        grid = (triton.cdiv(grad_out.numel(), block),)
        _sequential_rope_backward_kernel[grid](
            grad_out,
            rope_cache,
            grad_x,
            grad_out.numel(),
            seq_len,
            num_heads,
            head_dim,
            BLOCK=block,
        )
        return grad_x, None


def _sequential_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    return _SequentialRoPE.apply(x, rope_cache)


class _PairwiseRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        rope_cache = rope_cache.contiguous()
        out = torch.empty_like(x)
        _, seq_len, num_heads, head_dim = x.shape
        block = _PAIRWISE_ROPE_BLOCK
        total_pairs = x.numel() // 2
        grid = (triton.cdiv(total_pairs, block),)
        _pairwise_rope_forward_kernel[grid](
            x,
            rope_cache,
            out,
            total_pairs,
            seq_len,
            num_heads,
            head_dim,
            BLOCK=block,
        )
        ctx.save_for_backward(rope_cache)
        ctx.shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rope_cache,) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(grad_out)
        _, seq_len, num_heads, head_dim = ctx.shape
        block = _PAIRWISE_ROPE_BLOCK
        total_pairs = grad_out.numel() // 2
        grid = (triton.cdiv(total_pairs, block),)
        _pairwise_rope_backward_kernel[grid](
            grad_out,
            rope_cache,
            grad_x,
            total_pairs,
            seq_len,
            num_heads,
            head_dim,
            BLOCK=block,
        )
        return grad_x, None


def _pairwise_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    return _PairwiseRoPE.apply(x, rope_cache)


class _FusedQKPairwiseRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xq = xq.contiguous()
        xk = xk.contiguous()
        rope_cache = rope_cache.contiguous()
        q_out = torch.empty_like(xq)
        k_out = torch.empty_like(xk)
        _, seq_len, q_num_heads, head_dim = xq.shape
        k_num_heads = xk.shape[2]
        block = 256
        q_total_pairs = xq.numel() // 2
        k_total_pairs = xk.numel() // 2
        q_blocks = triton.cdiv(q_total_pairs, block)
        k_blocks = triton.cdiv(k_total_pairs, block)
        _fused_qk_pairwise_rope_forward_kernel[(q_blocks + k_blocks,)](
            xq,
            xk,
            rope_cache,
            q_out,
            k_out,
            q_total_pairs,
            k_total_pairs,
            q_num_heads,
            k_num_heads,
            seq_len,
            head_dim,
            q_blocks,
            BLOCK=block,
        )
        ctx.save_for_backward(rope_cache)
        ctx.q_shape = xq.shape
        ctx.k_shape = xk.shape
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor, grad_k: torch.Tensor):
        (rope_cache,) = ctx.saved_tensors
        grad_q = grad_q.contiguous()
        grad_k = grad_k.contiguous()
        q_out = torch.empty_like(grad_q)
        k_out = torch.empty_like(grad_k)
        _, seq_len, q_num_heads, head_dim = ctx.q_shape
        k_num_heads = ctx.k_shape[2]
        block = 256
        q_total_pairs = grad_q.numel() // 2
        k_total_pairs = grad_k.numel() // 2
        q_blocks = triton.cdiv(q_total_pairs, block)
        k_blocks = triton.cdiv(k_total_pairs, block)
        _fused_qk_pairwise_rope_backward_kernel[(q_blocks + k_blocks,)](
            grad_q,
            grad_k,
            rope_cache,
            q_out,
            k_out,
            q_total_pairs,
            k_total_pairs,
            q_num_heads,
            k_num_heads,
            seq_len,
            head_dim,
            q_blocks,
            BLOCK=block,
        )
        return q_out, k_out, None


def _fused_qk_pairwise_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FusedQKPairwiseRoPE.apply(xq, xk, rope_cache)


def _qk_norm_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, q_eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, k_eps)
    return _sequential_rope(xq, rope_cache), _sequential_rope(xk, rope_cache)


def _qk_norm_pairwise_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, q_eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, k_eps)
    return _pairwise_rope(xq, rope_cache), _pairwise_rope(xk, rope_cache)


def _qk_norm_fused_pairwise_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq = F.rms_norm(xq, (xq.shape[-1],), q_weight, q_eps)
    xk = F.rms_norm(xk, (xk.shape[-1],), k_weight, k_eps)
    return _fused_qk_pairwise_rope(xq, xk, rope_cache)


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _enable_shared_mxfp8_gate_up_input_cast(
    model: Qwen3Model,
    *,
    gate_up_backward_mode: str = "",
    use_triton_swiglu: bool = False,
) -> None:
    patched_count = 0
    gate_up_impls = {
        "": mxfp8_shared_input_gate_up,
        "forward": mxfp8_shared_input_gate_up_forward,
        "forward_grad_input": mxfp8_shared_input_gate_up_forward_grad_input_concat,
        "grad_input": mxfp8_shared_input_gate_up_grad_input_concat,
    }
    shared_gate_up = gate_up_impls[gate_up_backward_mode]
    swiglu = _triton_swiglu if use_triton_swiglu else lambda gate, up: F.silu(gate) * up

    def _shared_mxfp8_feed_forward(x: torch.Tensor, *, feed_forward):
        w1_out, w3_out = shared_gate_up(
            x,
            feed_forward.w1.weight,
            feed_forward.w3.weight,
        )
        return feed_forward.w2(swiglu(w1_out, w3_out))

    for layer in model.layers.values():
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            continue
        if not (
            hasattr(feed_forward.w1.weight, "config")
            and hasattr(feed_forward.w3.weight, "config")
        ):
            continue
        feed_forward.forward = lambda x, feed_forward=feed_forward: (
            _shared_mxfp8_feed_forward(x, feed_forward=feed_forward)
        )
        patched_count += 1

    logger.info(
        "Enabled shared MXFP8 input casts for %s Qwen3 feed-forward modules%s%s",
        patched_count,
        f" with {gate_up_backward_mode} gate/up backward concat"
        if gate_up_backward_mode
        else "",
        " with Triton SwiGLU" if use_triton_swiglu else "",
    )


def _enable_shared_mxfp8_qkv_input_cast(
    model: Qwen3Model, *, qkv_backward_mode: str = ""
) -> None:
    patched_count = 0
    qkv_impls = {
        "": mxfp8_shared_input_qkv,
        "all": mxfp8_shared_input_qkv_backward_concat,
        "forward_grad_input": mxfp8_shared_input_qkv_forward_grad_input_concat,
        "grad_input": mxfp8_shared_input_qkv_grad_input_concat,
        "grad_weight": mxfp8_shared_input_qkv_grad_weight_concat,
    }
    shared_qkv = qkv_impls[qkv_backward_mode]

    def _shared_mxfp8_qkv(x: torch.Tensor, *, qkv_linear):
        bs, seqlen, _ = x.shape
        xq, xk, xv = shared_qkv(
            x,
            qkv_linear.wq.weight,
            qkv_linear.wk.weight,
            qkv_linear.wv.weight,
        )
        xq = xq.view(bs, seqlen, -1, qkv_linear.head_dim)
        xk = xk.view(bs, seqlen, -1, qkv_linear.head_dim)
        xv = xv.view(bs, seqlen, -1, qkv_linear.head_dim)
        return xq, xk, xv

    for layer in model.layers.values():
        attention = getattr(layer, "attention", None)
        qkv_linear = getattr(attention, "qkv_linear", None)
        if qkv_linear is None:
            continue
        if not all(
            hasattr(proj.weight, "config")
            for proj in (qkv_linear.wq, qkv_linear.wk, qkv_linear.wv)
        ):
            continue
        qkv_linear.forward = lambda x, qkv_linear=qkv_linear: _shared_mxfp8_qkv(
            x,
            qkv_linear=qkv_linear,
        )
        patched_count += 1

    logger.info(
        "Enabled shared MXFP8 input casts for %s Qwen3 attention Q/K/V modules%s",
        patched_count,
        f" with {qkv_backward_mode} QKV backward concat"
        if qkv_backward_mode
        else "",
    )


def _compile_qwen3_feed_forward(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    compiled_count = 0
    for layer in model.layers.values():
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            continue
        feed_forward.compile(backend=compile_config.backend, fullgraph=True)
        compiled_count += 1

    logger.info(
        "Compiling %s Qwen3 feed-forward modules with torch.compile",
        compiled_count,
    )


def _compile_qwen3_qkv_linear(model: Qwen3Model, compile_config: CompileConfig) -> None:
    compiled_count = 0
    for layer in model.layers.values():
        attention = getattr(layer, "attention", None)
        qkv_linear = getattr(attention, "qkv_linear", None)
        if qkv_linear is None:
            continue
        qkv_linear.compile(backend=compile_config.backend, fullgraph=True)
        compiled_count += 1

    logger.info(
        "Compiling %s Qwen3 attention Q/K/V projection modules",
        compiled_count,
    )


def _enable_compiled_block_norms(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    compiled_rms_norm = torch.compile(
        _rms_norm,
        backend=compile_config.backend,
        fullgraph=True,
    )
    patched_count = 0

    def _forward_with_compiled_block_norms(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input = compiled_rms_norm(
            x,
            self.attention_norm.weight,
            self.attention_norm.eps,
        )
        attention_output = self.attention(
            attention_input, freqs_cis, attention_masks, positions
        )
        x = x + attention_output
        ffn_input = compiled_rms_norm(
            x,
            self.ffn_norm.weight,
            self.ffn_norm.eps,
        )

        if self.moe_enabled:
            x = x + self.moe(ffn_input)
        else:
            x = x + self.feed_forward(ffn_input)
        return x

    for layer in model.layers.values():
        layer.forward = types.MethodType(_forward_with_compiled_block_norms, layer)
        patched_count += 1

    logger.info(
        "Compiling Qwen3 block RMSNorm helpers for %s layers",
        patched_count,
    )


def _enable_compiled_norm_modules(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    compiled_rms_norm = torch.compile(
        _rms_norm,
        backend=compile_config.backend,
        fullgraph=True,
    )
    patched_count = 0

    def _compiled_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
        return compiled_rms_norm(x, self.weight, self.eps)

    for layer in model.layers.values():
        for norm in (layer.attention_norm, layer.ffn_norm):
            norm.forward = types.MethodType(_compiled_norm_forward, norm)
            patched_count += 1

    logger.info(
        "Compiling %s Qwen3 outer RMSNorm modules with torch.compile",
        patched_count,
    )


def _enable_triton_sequential_rope(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    global _PAIRWISE_ROPE_BLOCK
    patched_count = 0
    compiled_qk_norm_rope = None
    use_pairwise_rope = compile_config.enable and "pairwise_rope" in compile_config.components
    if compile_config.enable and "pairwise_rope_block128" in compile_config.components:
        _PAIRWISE_ROPE_BLOCK = 128
    elif compile_config.enable and "pairwise_rope_block512" in compile_config.components:
        _PAIRWISE_ROPE_BLOCK = 512
    else:
        _PAIRWISE_ROPE_BLOCK = 256
    use_fused_qk_rope = (
        use_pairwise_rope
        and compile_config.enable
        and "fused_qk_rope" in compile_config.components
    )
    if compile_config.enable and "qk_norm_rope" in compile_config.components:
        if use_fused_qk_rope:
            qk_norm_rope = _qk_norm_fused_pairwise_rope
        elif use_pairwise_rope:
            qk_norm_rope = _qk_norm_pairwise_rope
        else:
            qk_norm_rope = _qk_norm_rope
        compiled_qk_norm_rope = torch.compile(
            qk_norm_rope,
            backend=compile_config.backend,
            fullgraph=True,
            dynamic=False
            if "qk_norm_rope_dynamic_false" in compile_config.components
            else None,
        )

    def _forward_with_triton_rope(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.qkv_linear(x)
        rope_applied = False

        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            if compiled_qk_norm_rope is not None and self.use_rope:
                xq, xk = compiled_qk_norm_rope(
                    xq,
                    xk,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    self.q_norm.eps,
                    self.k_norm.eps,
                    rope_cache,
                )
                rope_applied = True
            else:
                xq = self.q_norm(xq)
                xk = self.k_norm(xk)

        if self.use_rope and not rope_applied:
            if self.rope_backend != "cos_sin":
                raise NotImplementedError(
                    "Triton sequential RoPE only supports cos_sin"
                )
            elif use_fused_qk_rope:
                xq, xk = _fused_qk_pairwise_rope(xq, xk, rope_cache)
            else:
                rope = _pairwise_rope if use_pairwise_rope else _sequential_rope
                xq = rope(xq, rope_cache)
                xk = rope(xk, rope_cache)

        if isinstance(attention_masks, dict):
            mask_key = "rope" if self.use_rope else "nope"
            attention_masks = attention_masks[mask_key]

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        output = output.reshape(bs, seqlen, -1)
        return self.wo(output)

    for layer in model.layers.values():
        attention = getattr(layer, "attention", None)
        if attention is None:
            continue
        attention.forward = types.MethodType(_forward_with_triton_rope, attention)
        patched_count += 1

    logger.info(
        "Enabled Triton %s RoPE for %s Qwen3 attention modules with block=%s",
        "fused Q/K pairwise"
        if use_fused_qk_rope
        else "pairwise"
        if use_pairwise_rope
        else "sequential",
        patched_count,
        _PAIRWISE_ROPE_BLOCK,
    )
    if compiled_qk_norm_rope is not None:
        logger.info("Compiling Qwen3 Q/K RMSNorm plus RoPE helper with torch.compile")


def _compile_qwen3_model_blocks_dynamic_false(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )
    for transformer_block in model.layers.values():
        transformer_block.compile(
            backend=compile_config.backend,
            fullgraph=True,
            dynamic=False,
        )
    logger.info("Compiling each TransformerBlock with torch.compile(dynamic=False)")


def _compile_qwen3_model_blocks_non_fullgraph(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )
    for transformer_block in model.layers.values():
        transformer_block.compile(
            backend=compile_config.backend,
            fullgraph=False,
        )
    logger.info("Compiling each TransformerBlock with torch.compile(fullgraph=False)")


def _set_qwen3_sdpa_backends(model: Qwen3Model, backends: list[SDPBackend]) -> None:
    patched_count = 0
    for layer in model.layers.values():
        attention = getattr(layer, "attention", None)
        inner_attention = getattr(attention, "inner_attention", None)
        if isinstance(inner_attention, ScaledDotProductAttention):
            inner_attention.sdpa_backends = backends
            patched_count += 1
    logger.info("Set %s Qwen3 SDPA modules to backends=%s", patched_count, backends)


def parallelize_qwen3(
    model: Qwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    """Generated machine-specific Qwen3 parallelization entry point.

    A valid implementation may make narrow assumptions about the train command,
    mesh shape, hardware topology, memory budget, model flavor, and enabled
    TorchTitan features. It does not need to be a universal implementation.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support TP.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support CP.")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support PP.")
    if parallel_dims.ep_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support EP.")

    if skip_dp or not parallel_dims.dp_enabled:
        return model

    if parallel_dims.dp_replicate != 1:
        raise NotImplementedError("Qwen3 baseline FSDP bootstrap does not support HSDP.")
    if training.enable_cpu_offload:
        raise NotImplementedError(
            "Qwen3 baseline FSDP bootstrap does not support CPU offload."
        )

    _enable_triton_sequential_rope(model, compile_config)
    if compile_config.enable and "sdpa_flash_only" in compile_config.components:
        _set_qwen3_sdpa_backends(model, [SDPBackend.FLASH_ATTENTION])
    elif compile_config.enable and "sdpa_cudnn_only" in compile_config.components:
        _set_qwen3_sdpa_backends(model, [SDPBackend.CUDNN_ATTENTION])
    if compile_config.enable and "compiled_autograd" in compile_config.components:
        torch._dynamo.config.compiled_autograd = True
        logger.info("Enabled torch._dynamo compiled autograd")
    if compile_config.enable and "model" in compile_config.components:
        if "model_dynamic_false" in compile_config.components:
            _compile_qwen3_model_blocks_dynamic_false(model, compile_config)
        elif "model_non_fullgraph" in compile_config.components:
            _compile_qwen3_model_blocks_non_fullgraph(model, compile_config)
        else:
            apply_compile(model, compile_config)
    qkv_backward_mode = ""
    if compile_config.enable:
        if "qkv_backward_concat" in compile_config.components:
            qkv_backward_mode = "all"
        elif "qkv_forward_grad_input_concat" in compile_config.components:
            qkv_backward_mode = "forward_grad_input"
        elif "qkv_grad_input_concat" in compile_config.components:
            qkv_backward_mode = "grad_input"
        elif "qkv_grad_weight_concat" in compile_config.components:
            qkv_backward_mode = "grad_weight"
    _enable_shared_mxfp8_qkv_input_cast(model, qkv_backward_mode=qkv_backward_mode)
    gate_up_backward_mode = ""
    if compile_config.enable:
        if "gate_up_forward_grad_input_concat" in compile_config.components:
            gate_up_backward_mode = "forward_grad_input"
        elif "gate_up_forward_concat" in compile_config.components:
            gate_up_backward_mode = "forward"
        elif "gate_up_grad_input_concat" in compile_config.components:
            gate_up_backward_mode = "grad_input"
    _enable_shared_mxfp8_gate_up_input_cast(
        model,
        gate_up_backward_mode=gate_up_backward_mode,
        use_triton_swiglu=compile_config.enable
        and "triton_swiglu" in compile_config.components,
    )
    if compile_config.enable and "qkv_linear" in compile_config.components:
        _compile_qwen3_qkv_linear(model, compile_config)
    if compile_config.enable and "feed_forward" in compile_config.components:
        _compile_qwen3_feed_forward(model, compile_config)
    if compile_config.enable and "block_norms" in compile_config.components:
        _enable_compiled_block_norms(model, compile_config)
    if compile_config.enable and "norm_modules" in compile_config.components:
        _enable_compiled_norm_modules(model, compile_config)

    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, training.mixed_precision_param),
        reduce_dtype=getattr(torch, training.mixed_precision_reduce),
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    fsdp_config = {
        "mesh": fsdp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
    }

    layers = list(model.layers.values())
    reshard_last_layer = (
        compile_config.enable and "reshard_last_layer" in compile_config.components
    )
    shard_attention_ffn = (
        compile_config.enable and "fsdp_attention_ffn" in compile_config.components
    )
    for idx, layer in enumerate(layers):
        layer_fsdp_config = fsdp_config
        if idx == len(layers) - 1 and not reshard_last_layer:
            layer_fsdp_config = {
                **fsdp_config,
                "reshard_after_forward": False,
            }
        if shard_attention_ffn:
            attention = getattr(layer, "attention", None)
            feed_forward = getattr(layer, "feed_forward", None)
            if attention is not None:
                fully_shard(attention, **fsdp_config)
            if feed_forward is not None:
                fully_shard(feed_forward, **fsdp_config)
        fully_shard(layer, **layer_fsdp_config)
    fully_shard(model.lm_head, **fsdp_config)
    fully_shard(model, **fsdp_config)
    if layers:
        no_forward_prefetch = (
            compile_config.enable
            and "fsdp_no_forward_prefetch" in compile_config.components
        )
        no_backward_prefetch = (
            compile_config.enable
            and "fsdp_no_backward_prefetch" in compile_config.components
        )
        forward_prefetch_distance = (
            2
            if compile_config.enable
            and "fsdp_forward_prefetch2" in compile_config.components
            else 1
        )
        backward_prefetch_distance = (
            2
            if compile_config.enable
            and "fsdp_backward_prefetch2" in compile_config.components
            else 1
        )
        if not no_forward_prefetch:
            for idx, layer in enumerate(layers):
                prefetch_modules = layers[
                    idx + 1 : idx + 1 + forward_prefetch_distance
                ]
                if idx == len(layers) - 1:
                    prefetch_modules = [model.lm_head]
                if prefetch_modules:
                    layer.set_modules_to_forward_prefetch(prefetch_modules)
        if not no_backward_prefetch:
            model.lm_head.set_modules_to_backward_prefetch(
                layers[-backward_prefetch_distance:]
            )
            for idx in range(len(layers) - 1, -1, -1):
                prefetch_modules = layers[
                    max(0, idx - backward_prefetch_distance) : idx
                ]
                if prefetch_modules:
                    layers[idx].set_modules_to_backward_prefetch(prefetch_modules)

    disable_fsdp_gradient_division(model)
    logger.info(
        "Applied baseline Qwen3 FSDP with dp_shard=%s, reshard_after_forward=%s",
        parallel_dims.dp_shard,
        reshard_after_forward,
    )

    return model
