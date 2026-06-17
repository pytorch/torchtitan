# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
torch.library custom-op wrappers around vLLM's fused _C kernels, used by the
inference kernel ablation. Wrapping them as custom ops keeps them opaque to
torch.compile(aot_eager) (the cudagraph rung) -- dynamo uses the register_fake
shape function and the real kernel runs on concrete tensors at execution /
cudagraph-capture time. The underlying torch.ops._C.* ops are registered when
vLLM is imported (the rl package import chain does this).
"""
from __future__ import annotations

import torch

import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.* kernels)


@torch.library.custom_op("ttablation::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """vLLM fused SwiGLU activation: silu(x[..., :d]) * x[..., d:], d = D/2.

    ``x`` is the concatenated gate||up projection of shape (..., 2*d).
    """
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out.view(-1, d), x.contiguous().view(-1, 2 * d))
    return out


@silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)


@torch.library.custom_op("ttablation::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM fused RMSNorm over the last dim. ``x`` is (..., d), weight (d,)."""
    d = x.shape[-1]
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out.view(-1, d), x.contiguous().view(-1, d), weight, eps)
    return out


@rms_norm.register_fake
def _rms_norm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


# vLLM's rotary_embedding wants a (max_pos, head_dim) cache laid out as
# [cos(head_dim/2) | sin(head_dim/2)]. CosSinRoPE.cache is (max_seq_len,
# head_dim*2) = [cos(head_dim) | sin(head_dim)] with cos/sin duplicated, so the
# vLLM cache is the first half of each block. Convert once per cache buffer
# (keyed by data_ptr) inside the op -- the op is eager/opaque to torch.compile.
_VLLM_ROPE_CACHE: dict[tuple[int, int], torch.Tensor] = {}


@torch.library.custom_op("ttablation::vllm_rotary", mutates_args=())
def vllm_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    rope_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply vLLM's fused neox rotary_embedding (in-place op, returned
    functionally). ``query``/``key`` are (num_tokens, n_heads*head_size);
    ``rope_cache`` is the full CosSinRoPE cache (max_pos, head_size*2)."""
    cache_key = (rope_cache.data_ptr(), head_size)
    vc = _VLLM_ROPE_CACHE.get(cache_key)
    if vc is None:
        d = head_size
        half = d // 2
        vc = torch.cat(
            [rope_cache[:, :half], rope_cache[:, d : d + half]], dim=-1
        ).contiguous()
        _VLLM_ROPE_CACHE[cache_key] = vc
    q = query.contiguous().clone()
    k = key.contiguous().clone()
    torch.ops._C.rotary_embedding(positions, q, k, head_size, vc.to(q.dtype), True)
    return q, k


@vllm_rotary.register_fake
def _vllm_rotary_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    rope_cache: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key)


# Merged projection GEMMs. Native vLLM uses a single QKV (3->1) and gate_up
# (2->1) GEMM; torchtitan keeps them separate. We concatenate the (already
# TP-local) weight shards ONCE -- cached by the constituent weights' data_ptrs
# inside the op -- so cudagraph replays run a single F.linear, not a per-step
# weight cat.
_MERGED_WEIGHT_CACHE: dict[tuple, torch.Tensor] = {}


@torch.library.custom_op("ttablation::merged_linear3", mutates_args=())
def merged_linear3(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    key = (w0.data_ptr(), w1.data_ptr(), w2.data_ptr())
    w = _MERGED_WEIGHT_CACHE.get(key)
    if w is None:
        w = torch.cat([w0, w1, w2], dim=0).contiguous()
        _MERGED_WEIGHT_CACHE[key] = w
    return torch.nn.functional.linear(x, w)


@merged_linear3.register_fake
def _merged_linear3_fake(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    out = w0.shape[0] + w1.shape[0] + w2.shape[0]
    return torch.empty(*x.shape[:-1], out, dtype=x.dtype, device=x.device)


@torch.library.custom_op("ttablation::merged_linear2", mutates_args=())
def merged_linear2(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    key = (w0.data_ptr(), w1.data_ptr())
    w = _MERGED_WEIGHT_CACHE.get(key)
    if w is None:
        w = torch.cat([w0, w1], dim=0).contiguous()
        _MERGED_WEIGHT_CACHE[key] = w
    return torch.nn.functional.linear(x, w)


@merged_linear2.register_fake
def _merged_linear2_fake(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
) -> torch.Tensor:
    out = w0.shape[0] + w1.shape[0]
    return torch.empty(*x.shape[:-1], out, dtype=x.dtype, device=x.device)


@torch.library.custom_op("ttablation::fused_add_rmsnorm", mutates_args=())
def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM fused residual-add + RMSNorm. Returns (normed, new_residual) where
    new_residual = x + residual and normed = rms_norm(new_residual). Native
    vLLM fuses the residual add into the norm; torchtitan otherwise runs a
    standalone add + standalone norm per layer."""
    new_residual = x.contiguous().clone()
    res = residual.contiguous().clone()
    torch.ops._C.fused_add_rms_norm(new_residual, res, weight, eps)
    # after the in-place op: new_residual = rms(x+residual), res = x+residual
    return new_residual, res


@fused_add_rmsnorm.register_fake
def _fused_add_rmsnorm_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)
