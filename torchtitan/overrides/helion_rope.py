# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Override: rotary embeddings applied with fused Helion kernels.

This swaps :class:`CosSinRoPE` and :class:`ComplexRoPE` for versions that fuse
the cache gather and rotation into a single Helion kernel (forward and
backward), without touching core.

    torchtitan_train ... --override.imports torchtitan.overrides.helion_rope

Scope and fallbacks (the kernel is opt-in and never changes default behavior):

* **Concrete cache contracts only.** Targets exactly ``CosSinRoPE.Config`` and
  ``ComplexRoPE.Config``; subclasses with different cache contracts (e.g.
  Qwen3.5 ``MRoPE``) are untouched.
* ``helion`` is an optional dependency, but explicitly selecting this override
  requires it to be installed. Unsupported tensor inputs (any tensor on CPU or
  split across devices, a cache with an unexpected shape, or non-integer or
  oddly shaped position ids) fall back to the PyTorch path. Non-contiguous q/k
  views are accepted like the stock modules: the complex RoPE kernel loads q/k
  inputs directly when adjacent real/imag pairs are contiguous in the last
  dimension and returns fresh contiguous outputs; the cos/sin kernel
  materializes contiguous local inputs at the Helion boundary. Position ids are
  bounds-checked exactly as the PyTorch path does, so out-of-range ids fail
  cleanly rather than as an illegal kernel access.

The kernels match the stock RoPE modules numerically (all upcast to fp32 and
reuse the same cache conventions), so they are interoperable with stock
checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override
from torchtitan.models.common.rope import _maybe_check_max_pos, ComplexRoPE, CosSinRoPE
from torchtitan.tools.logging import logger, warn_once

if TYPE_CHECKING:
    # The type checker always sees helion as importable (resolved to Any via
    # ``replace-imports-with-any`` in pyproject.toml); the runtime import-error
    # sentinel below is what makes it a genuine optional dependency.
    import helion
    import helion.language as hl

    _HELION_IMPORT_ERROR: ImportError | None = None
else:
    try:
        import helion
        import helion.language as hl

        _HELION_IMPORT_ERROR = None
    except ImportError as e:
        _HELION_IMPORT_ERROR = e

__all__ = ["HelionComplexRoPE", "HelionRoPE"]


if _HELION_IMPORT_ERROR is None:
    # A reasonable default config so a raw (unbound) kernel call is deterministic
    # rather than triggering autotune; production calls override it per shape via
    # ``_run_tuned`` below.
    _DEFAULT_CONFIG = helion.Config(block_sizes=[1, 16, 8, 1], num_warps=8)

    @helion.kernel(config=_DEFAULT_CONFIG, static_shapes=True)
    def _rope_cos_sin_fwd(
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fuses the per-position cos/sin gather (``rope_cache[positions]``) with
        # the rotate-half rotation. ``rope_cache`` is ``(max_seq, 2 * head_dim)``
        # (cos concatenated with sin); ``*_lo``/``*_hi`` are the first/second
        # head-dim halves that rotate-half mixes.
        _, seqlen, n_heads, head_dim = xq.size()
        _, _, n_kv_heads, _ = xk.size()
        head_dim = hl.specialize(head_dim)

        xq_out = torch.empty(xq.size(), device=xq.device, dtype=xq.dtype)
        xk_out = torch.empty(xk.size(), device=xk.device, dtype=xk.dtype)

        for tile_b, tile_s in hl.tile([xq.size(0), seqlen]):
            position = positions[tile_b, tile_s]
            rope_tile = rope_cache[position, :].to(torch.float32)
            cos, sin = hl.split(
                rope_tile.reshape([tile_b, tile_s, 2, head_dim]).permute(0, 1, 3, 2)
            )
            cos_lo, cos_hi = hl.split(
                cos.reshape([tile_b, tile_s, 2, head_dim // 2]).permute(0, 1, 3, 2)
            )
            sin_lo, sin_hi = hl.split(
                sin.reshape([tile_b, tile_s, 2, head_dim // 2]).permute(0, 1, 3, 2)
            )

            cos_lo = cos_lo[:, :, None, :]
            cos_hi = cos_hi[:, :, None, :]
            sin_lo = sin_lo[:, :, None, :]
            sin_hi = sin_hi[:, :, None, :]

            for tile_h in hl.tile(n_heads):
                xq_tile = xq[tile_b, tile_s, tile_h, :].to(torch.float32)
                xq_lo, xq_hi = hl.split(
                    xq_tile.reshape([tile_b, tile_s, tile_h, 2, head_dim // 2]).permute(
                        0, 1, 2, 4, 3
                    )
                )
                xq_out_lo = xq_lo * cos_lo - xq_hi * sin_lo
                xq_out_hi = xq_hi * cos_hi + xq_lo * sin_hi
                xq_out[tile_b, tile_s, tile_h, :] = (
                    hl.join(xq_out_lo, xq_out_hi)
                    .permute(0, 1, 2, 4, 3)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(xq.dtype)
                )

            for tile_h in hl.tile(n_kv_heads):
                xk_tile = xk[tile_b, tile_s, tile_h, :].to(torch.float32)
                xk_lo, xk_hi = hl.split(
                    xk_tile.reshape([tile_b, tile_s, tile_h, 2, head_dim // 2]).permute(
                        0, 1, 2, 4, 3
                    )
                )
                xk_out_lo = xk_lo * cos_lo - xk_hi * sin_lo
                xk_out_hi = xk_hi * cos_hi + xk_lo * sin_hi
                xk_out[tile_b, tile_s, tile_h, :] = (
                    hl.join(xk_out_lo, xk_out_hi)
                    .permute(0, 1, 2, 4, 3)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(xk.dtype)
                )

        return xq_out, xk_out

    @helion.kernel(config=_DEFAULT_CONFIG, static_shapes=True)
    def _rope_cos_sin_bwd(
        grad_xq_out: torch.Tensor,
        grad_xk_out: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backward of ``_rope_cos_sin_fwd``: same loop structure, with the
        # rotation transposed (the sign pattern is mirrored across lo/hi).
        _, seqlen, n_heads, head_dim = grad_xq_out.size()
        _, _, n_kv_heads, _ = grad_xk_out.size()
        head_dim = hl.specialize(head_dim)

        grad_xq = torch.empty(
            grad_xq_out.size(), device=grad_xq_out.device, dtype=grad_xq_out.dtype
        )
        grad_xk = torch.empty(
            grad_xk_out.size(), device=grad_xk_out.device, dtype=grad_xk_out.dtype
        )
        half_head_dim = head_dim // 2

        for tile_b, tile_s in hl.tile([grad_xq_out.size(0), seqlen]):
            position = positions[tile_b, tile_s]
            rope_tile = rope_cache[position, :].to(torch.float32)
            cos, sin = hl.split(
                rope_tile.reshape([tile_b, tile_s, 2, head_dim]).permute(0, 1, 3, 2)
            )
            cos_lo, cos_hi = hl.split(
                cos.reshape([tile_b, tile_s, 2, half_head_dim]).permute(0, 1, 3, 2)
            )
            sin_lo, sin_hi = hl.split(
                sin.reshape([tile_b, tile_s, 2, half_head_dim]).permute(0, 1, 3, 2)
            )

            cos_lo = cos_lo[:, :, None, :]
            cos_hi = cos_hi[:, :, None, :]
            sin_lo = sin_lo[:, :, None, :]
            sin_hi = sin_hi[:, :, None, :]

            for tile_h in hl.tile(n_heads):
                grad_tile = grad_xq_out[tile_b, tile_s, tile_h, :].to(torch.float32)
                grad_lo, grad_hi = hl.split(
                    grad_tile.reshape(
                        [tile_b, tile_s, tile_h, 2, half_head_dim]
                    ).permute(0, 1, 2, 4, 3)
                )
                grad_xq_lo = grad_lo * cos_lo + grad_hi * sin_hi
                grad_xq_hi = grad_hi * cos_hi - grad_lo * sin_lo
                grad_xq[tile_b, tile_s, tile_h, :] = (
                    hl.join(grad_xq_lo, grad_xq_hi)
                    .permute(0, 1, 2, 4, 3)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(grad_xq.dtype)
                )

            for tile_h in hl.tile(n_kv_heads):
                grad_tile = grad_xk_out[tile_b, tile_s, tile_h, :].to(torch.float32)
                grad_lo, grad_hi = hl.split(
                    grad_tile.reshape(
                        [tile_b, tile_s, tile_h, 2, half_head_dim]
                    ).permute(0, 1, 2, 4, 3)
                )
                grad_xk_lo = grad_lo * cos_lo + grad_hi * sin_hi
                grad_xk_hi = grad_hi * cos_hi - grad_lo * sin_lo
                grad_xk[tile_b, tile_s, tile_h, :] = (
                    hl.join(grad_xk_lo, grad_xk_hi)
                    .permute(0, 1, 2, 4, 3)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(grad_xk.dtype)
                )

        return grad_xq, grad_xk

    @helion.kernel(config=_DEFAULT_CONFIG, static_shapes=True)
    def _rope_complex_fwd(
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ComplexRoPE applies adjacent-dim complex multiplication:
        # (x0 + i*x1) * (cos + i*sin). Helion does not need complex dtype
        # support here; ``rope_cache_real`` is ``view_as_real(cache)`` with shape
        # ``(max_seq, head_dim / 2, 2)``.
        _, seqlen, n_heads, head_dim = xq.size()
        _, _, n_kv_heads, _ = xk.size()
        head_dim = hl.specialize(head_dim)
        half_head_dim = head_dim // 2

        xq_out = torch.empty(xq.size(), device=xq.device, dtype=xq.dtype)
        xk_out = torch.empty(xk.size(), device=xk.device, dtype=xk.dtype)

        for tile_b, tile_s in hl.tile([xq.size(0), seqlen]):
            position = positions[tile_b, tile_s]
            cache_tile = rope_cache_real[position, :, :].to(torch.float32)
            cos, sin = hl.split(cache_tile)

            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]

            for tile_h in hl.tile(n_heads):
                xq_tile = xq[tile_b, tile_s, tile_h, :].to(torch.float32)
                xq_real, xq_imag = hl.split(
                    xq_tile.reshape([tile_b, tile_s, tile_h, half_head_dim, 2])
                )
                xq_out_real = xq_real * cos - xq_imag * sin
                xq_out_imag = xq_imag * cos + xq_real * sin
                xq_out[tile_b, tile_s, tile_h, :] = (
                    hl.join(xq_out_real, xq_out_imag)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(xq.dtype)
                )

            for tile_h in hl.tile(n_kv_heads):
                xk_tile = xk[tile_b, tile_s, tile_h, :].to(torch.float32)
                xk_real, xk_imag = hl.split(
                    xk_tile.reshape([tile_b, tile_s, tile_h, half_head_dim, 2])
                )
                xk_out_real = xk_real * cos - xk_imag * sin
                xk_out_imag = xk_imag * cos + xk_real * sin
                xk_out[tile_b, tile_s, tile_h, :] = (
                    hl.join(xk_out_real, xk_out_imag)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(xk.dtype)
                )

        return xq_out, xk_out

    @helion.kernel(config=_DEFAULT_CONFIG, static_shapes=True)
    def _rope_complex_bwd(
        grad_xq_out: torch.Tensor,
        grad_xk_out: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backward of adjacent-dim complex multiplication:
        # grad_x = grad_out * conj(cos + i*sin).
        _, seqlen, n_heads, head_dim = grad_xq_out.size()
        _, _, n_kv_heads, _ = grad_xk_out.size()
        head_dim = hl.specialize(head_dim)
        half_head_dim = head_dim // 2

        grad_xq = torch.empty(
            grad_xq_out.size(), device=grad_xq_out.device, dtype=grad_xq_out.dtype
        )
        grad_xk = torch.empty(
            grad_xk_out.size(), device=grad_xk_out.device, dtype=grad_xk_out.dtype
        )

        for tile_b, tile_s in hl.tile([grad_xq_out.size(0), seqlen]):
            position = positions[tile_b, tile_s]
            cache_tile = rope_cache_real[position, :, :].to(torch.float32)
            cos, sin = hl.split(cache_tile)

            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]

            for tile_h in hl.tile(n_heads):
                grad_tile = grad_xq_out[tile_b, tile_s, tile_h, :].to(torch.float32)
                grad_real, grad_imag = hl.split(
                    grad_tile.reshape([tile_b, tile_s, tile_h, half_head_dim, 2])
                )
                grad_xq_real = grad_real * cos + grad_imag * sin
                grad_xq_imag = grad_imag * cos - grad_real * sin
                grad_xq[tile_b, tile_s, tile_h, :] = (
                    hl.join(grad_xq_real, grad_xq_imag)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(grad_xq.dtype)
                )

            for tile_h in hl.tile(n_kv_heads):
                grad_tile = grad_xk_out[tile_b, tile_s, tile_h, :].to(torch.float32)
                grad_real, grad_imag = hl.split(
                    grad_tile.reshape([tile_b, tile_s, tile_h, half_head_dim, 2])
                )
                grad_xk_real = grad_real * cos + grad_imag * sin
                grad_xk_imag = grad_imag * cos - grad_real * sin
                grad_xk[tile_b, tile_s, tile_h, :] = (
                    hl.join(grad_xk_real, grad_xk_imag)
                    .reshape([tile_b, tile_s, tile_h, head_dim])
                    .to(grad_xk.dtype)
                )

        return grad_xq, grad_xk

    @cache
    def _config(
        block_sizes: tuple[int, ...],
        num_warps: int,
        num_stages: int | None = None,
        *,
        l2_groupings: tuple[int, ...] | None = None,
        load_eviction_policies: tuple[str, ...] | None = None,
        pid_type: str | None = None,
    ) -> helion.Config:
        config_kwargs: dict[str, Any] = {
            "block_sizes": list(block_sizes),
            "num_warps": num_warps,
            "load_eviction_policies": (
                list(load_eviction_policies)
                if load_eviction_policies is not None
                else ["", "", "first", "first"]
            ),
        }
        if num_stages is not None:
            config_kwargs["num_stages"] = num_stages
        if l2_groupings is not None:
            config_kwargs["l2_groupings"] = list(l2_groupings)
        if pid_type is not None:
            config_kwargs["pid_type"] = pid_type
        return helion.Config(**config_kwargs)

    def _fwd_config(query: torch.Tensor) -> helion.Config:
        # Pick a forward config by token count (batch * seq_len), the dominant
        # work size for this bandwidth-bound kernel. These buckets are tuned on
        # Qwen3 TP=8 local training shapes and still cover all supported shapes.
        num_tokens = query.shape[0] * query.shape[1]
        seq_len = query.shape[1]
        if seq_len >= 1024:
            return _config(
                (1, 4, 4, 1),
                num_warps=2,
                num_stages=1,
                l2_groupings=(64,),
                pid_type="flat",
            )
        if num_tokens <= 512:
            return _config((1, 16, 4, 1), num_warps=2)
        if num_tokens <= 1024:
            return _config((1, 4, 4, 1), num_warps=4)
        if num_tokens <= 2048:
            return _config((1, 4, 8, 1), num_warps=8)
        if num_tokens <= 4096:
            return _config((1, 16, 2, 1), num_warps=8, num_stages=2)
        if num_tokens <= 8192:
            return _config((1, 16, 4, 1), num_warps=2, num_stages=2)
        if num_tokens <= 16384:
            return _config((1, 4, 4, 1), num_warps=2, num_stages=2)
        return _config(
            (1, 8, 4, 1),
            num_warps=2,
            num_stages=2,
            load_eviction_policies=("", "", "last", "last"),
        )

    def _bwd_config(grad_query: torch.Tensor) -> helion.Config:
        # Backward gets its own token-count buckets because the transposed
        # rotation changes which tile shapes are best.
        num_tokens = grad_query.shape[0] * grad_query.shape[1]
        if num_tokens <= 512:
            return _config((1, 16, 4, 1), num_warps=2)
        if num_tokens <= 1024:
            return _config((1, 4, 8, 1), num_warps=2)
        if num_tokens <= 2048:
            return _config((1, 8, 2, 1), num_warps=8)
        if num_tokens <= 4096:
            return _config((1, 8, 4, 1), num_warps=2)
        if num_tokens <= 8192:
            return _config((1, 4, 4, 1), num_warps=2)
        if num_tokens <= 16384:
            return _config((1, 8, 4, 1), num_warps=2, num_stages=2)
        return _config(
            (1, 4, 4, 1),
            num_warps=2,
            num_stages=2,
            load_eviction_policies=("", "last", "", ""),
        )

    def _complex_fwd_config(query: torch.Tensor) -> helion.Config:
        # GB200 finite-search over the same bucket candidates as CosSinRoPE.
        num_tokens = query.shape[0] * query.shape[1]
        seq_len = query.shape[1]
        if seq_len >= 1024:
            return _config((1, 16, 2, 1), num_warps=8, num_stages=2)
        if num_tokens <= 512:
            return _config(
                (1, 4, 4, 1),
                num_warps=2,
                num_stages=1,
                l2_groupings=(64,),
                pid_type="flat",
            )
        if num_tokens <= 1024:
            return _config(
                (1, 8, 4, 1),
                num_warps=2,
                num_stages=2,
                load_eviction_policies=("", "", "last", "last"),
            )
        if num_tokens <= 2048:
            return _config(
                (1, 4, 4, 1),
                num_warps=2,
                num_stages=1,
                l2_groupings=(64,),
                pid_type="flat",
            )
        if num_tokens <= 4096:
            return _config(
                (1, 8, 4, 1),
                num_warps=2,
                num_stages=2,
                load_eviction_policies=("", "", "last", "last"),
            )
        if num_tokens <= 8192:
            return _config((1, 4, 8, 1), num_warps=8)
        if num_tokens <= 16384:
            return _config((1, 4, 4, 1), num_warps=2, num_stages=2)
        return _config((1, 16, 4, 1), num_warps=2, num_stages=2)

    def _complex_bwd_config(grad_query: torch.Tensor) -> helion.Config:
        # Keep ComplexRoPE backward on one shape-only selector, matching
        # CosSinRoPE's contract. Autograd may pass different row-major grad
        # output layouts, but the kernel accepts all layouts with last-dim
        # stride 1, so layout should not fork the bucket policy.
        num_tokens = grad_query.shape[0] * grad_query.shape[1]
        if num_tokens <= 512:
            return _config((1, 4, 8, 1), num_warps=2)
        if num_tokens <= 1024:
            return _config((1, 8, 2, 1), num_warps=8)
        if num_tokens <= 2048:
            return _config((1, 8, 4, 1), num_warps=2)
        return _config((1, 8, 4, 1), num_warps=2, num_stages=2)

    # Cache of bound kernels keyed by (kernel, config, shapes, strides, dtypes,
    # device).
    # Re-binding a config on every call costs ~20us of CPU dispatch;
    # training/inference use a handful of layouts, so this stays tiny. Dict ops
    # are atomic under the GIL, so the forward (main thread) and backward
    # (autograd thread) sharing it is safe.
    _BOUND_KERNELS: dict[tuple, Any] = {}

    def _run_tuned(
        kernel: Any, config: helion.Config, *args: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            id(kernel),
            repr(config),
            tuple(tuple(a.shape) for a in args),
            tuple(tuple(a.stride()) for a in args),
            tuple(a.dtype for a in args),
            args[0].device,
        )
        bound = _BOUND_KERNELS.get(key)
        if bound is None:
            bound = kernel.bind(args)
            bound.set_config(config)
            _BOUND_KERNELS[key] = bound
        return bound(*args)

    def _run_complex_bwd(
        grad_xq_out: torch.Tensor,
        grad_xk_out: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if grad_xq_out.stride(-1) != 1:
            grad_xq_out = grad_xq_out.contiguous()
        if grad_xk_out.stride(-1) != 1:
            grad_xk_out = grad_xk_out.contiguous()
        return _run_tuned(
            _rope_complex_bwd,
            _complex_bwd_config(grad_xq_out),
            grad_xq_out,
            grad_xk_out,
            rope_cache_real,
            positions,
        )

    @torch.library.custom_op(
        "torchtitan::helion_rope_fwd", mutates_args=(), device_types="cuda"
    )
    def _helion_rope_fwd(
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_tuned(
            _rope_cos_sin_fwd, _fwd_config(xq), xq, xk, rope_cache, positions
        )

    @_helion_rope_fwd.register_fake
    def _helion_rope_fwd_fake(
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(xq.size(), device=xq.device, dtype=xq.dtype),
            torch.empty(xk.size(), device=xk.device, dtype=xk.dtype),
        )

    @torch.library.custom_op(
        "torchtitan::helion_rope_bwd", mutates_args=(), device_types="cuda"
    )
    def _helion_rope_bwd(
        grad_xq_out: torch.Tensor,
        grad_xk_out: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The pointer-indexed kernel assumes contiguous gradients (static_shapes);
        # autograd may hand us non-contiguous ones, so make them contiguous.
        grad_xq_out = grad_xq_out.contiguous()
        grad_xk_out = grad_xk_out.contiguous()
        return _run_tuned(
            _rope_cos_sin_bwd,
            _bwd_config(grad_xq_out),
            grad_xq_out,
            grad_xk_out,
            rope_cache,
            positions,
        )

    @_helion_rope_bwd.register_fake
    def _helion_rope_bwd_fake(
        grad_xq_out: torch.Tensor,
        grad_xk_out: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(
                grad_xq_out.size(),
                device=grad_xq_out.device,
                dtype=grad_xq_out.dtype,
            ),
            torch.empty(
                grad_xk_out.size(),
                device=grad_xk_out.device,
                dtype=grad_xk_out.dtype,
            ),
        )

    def _fwd_setup_context(ctx, inputs, output) -> None:
        xq, xk, rope_cache, positions = inputs
        ctx.save_for_backward(rope_cache, positions)
        ctx.xq_shape = xq.shape
        ctx.xk_shape = xk.shape

    def _fwd_backward(ctx, grad_xq_out, grad_xk_out):
        rope_cache, positions = ctx.saved_tensors
        # RoPE feeds both q and k into attention, so both grads are normally
        # present; zero-fill defensively if autograd drops one.
        if grad_xq_out is None:
            grad_xq_out = torch.zeros(
                ctx.xq_shape, device=grad_xk_out.device, dtype=grad_xk_out.dtype
            )
        if grad_xk_out is None:
            grad_xk_out = torch.zeros(
                ctx.xk_shape, device=grad_xq_out.device, dtype=grad_xq_out.dtype
            )
        grad_xq, grad_xk = _helion_rope_bwd(
            grad_xq_out, grad_xk_out, rope_cache, positions
        )
        if not ctx.needs_input_grad[0]:
            grad_xq = None
        if not ctx.needs_input_grad[1]:
            grad_xk = None
        return grad_xq, grad_xk, None, None

    _helion_rope_fwd.register_autograd(_fwd_backward, setup_context=_fwd_setup_context)

    @torch.library.custom_op(
        "torchtitan::helion_complex_rope_fwd", mutates_args=(), device_types="cuda"
    )
    def _helion_complex_rope_fwd(
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_tuned(
            _rope_complex_fwd,
            _complex_fwd_config(xq),
            xq,
            xk,
            rope_cache_real,
            positions,
        )

    @_helion_complex_rope_fwd.register_fake
    def _helion_complex_rope_fwd_fake(
        xq: torch.Tensor,
        xk: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(xq.size(), device=xq.device, dtype=xq.dtype),
            torch.empty(xk.size(), device=xk.device, dtype=xk.dtype),
        )

    @torch.library.custom_op(
        "torchtitan::helion_complex_rope_bwd", mutates_args=(), device_types="cuda"
    )
    def _helion_complex_rope_bwd(
        grad_xq_out: torch.Tensor,
        grad_xk_out: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_complex_bwd(grad_xq_out, grad_xk_out, rope_cache_real, positions)

    @_helion_complex_rope_bwd.register_fake
    def _helion_complex_rope_bwd_fake(
        grad_xq_out: torch.Tensor,
        grad_xk_out: torch.Tensor,
        rope_cache_real: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(
                grad_xq_out.size(),
                device=grad_xq_out.device,
                dtype=grad_xq_out.dtype,
            ),
            torch.empty(
                grad_xk_out.size(),
                device=grad_xk_out.device,
                dtype=grad_xk_out.dtype,
            ),
        )

    def _complex_fwd_setup_context(ctx, inputs, output) -> None:
        xq, xk, rope_cache_real, positions = inputs
        ctx.save_for_backward(rope_cache_real, positions)
        ctx.xq_shape = xq.shape
        ctx.xk_shape = xk.shape

    def _complex_fwd_backward(ctx, grad_xq_out, grad_xk_out):
        rope_cache_real, positions = ctx.saved_tensors
        if grad_xq_out is None:
            grad_xq_out = torch.zeros(
                ctx.xq_shape, device=grad_xk_out.device, dtype=grad_xk_out.dtype
            )
        if grad_xk_out is None:
            grad_xk_out = torch.zeros(
                ctx.xk_shape, device=grad_xq_out.device, dtype=grad_xq_out.dtype
            )
        grad_xq, grad_xk = _helion_complex_rope_bwd(
            grad_xq_out, grad_xk_out, rope_cache_real, positions
        )
        if not ctx.needs_input_grad[0]:
            grad_xq = None
        if not ctx.needs_input_grad[1]:
            grad_xk = None
        return grad_xq, grad_xk, None, None

    _helion_complex_rope_fwd.register_autograd(
        _complex_fwd_backward, setup_context=_complex_fwd_setup_context
    )

    class _HelionComplexRoPEFunction(torch.autograd.Function):
        # The torch.library custom op is needed for compile/fake-tensor support,
        # but eager training pays extra dispatcher overhead through that path.
        # The direct autograd.Function keeps the same saved-tensor contract while
        # calling the bound Helion kernels without the custom-op dispatcher.
        @staticmethod
        def forward(
            ctx,
            xq: torch.Tensor,
            xk: torch.Tensor,
            rope_cache_real: torch.Tensor,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx.save_for_backward(rope_cache_real, positions)
            ctx.xq_shape = xq.shape
            ctx.xk_shape = xk.shape
            return _run_tuned(
                _rope_complex_fwd,
                _complex_fwd_config(xq),
                xq,
                xk,
                rope_cache_real,
                positions,
            )

        @staticmethod
        def backward(ctx, grad_xq_out, grad_xk_out):
            rope_cache_real, positions = ctx.saved_tensors
            if grad_xq_out is None:
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
            grad_xq, grad_xk = _run_complex_bwd(
                grad_xq_out, grad_xk_out, rope_cache_real, positions
            )
            if not ctx.needs_input_grad[0]:
                grad_xq = None
            if not ctx.needs_input_grad[1]:
                grad_xk = None
            return grad_xq, grad_xk, None, None


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _from_local(local: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
    if isinstance(spec, DTensor):
        return DTensor.from_local(
            local, spec.device_mesh, spec.placements, run_check=False
        )
    return local


def _resolve_positions(
    positions: torch.Tensor | None, query_local: torch.Tensor
) -> torch.Tensor:
    # ``positions=None`` means "0, 1, ..., seq_len-1" per row (what the PyTorch
    # path does via ``cache[0:seq_len]``); the kernel gathers by index, so make
    # those explicit, contiguous (batch, seq_len) ids.
    if positions is not None:
        return _to_local(positions)
    batch, seqlen = query_local.shape[0], query_local.shape[1]
    pos = torch.arange(seqlen, device=query_local.device, dtype=torch.int32)
    return pos.unsqueeze(0).expand(batch, -1).contiguous()


def _eligible(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> bool:
    # The kernel gathers ``rope_cache[positions]`` and reshapes the row to
    # ``[..., 2, head_dim]`` on device, so every tensor must be CUDA, the cache a
    # 2D ``(max_seq, 2 * head_dim)`` table, and positions integer ids -- otherwise
    # it would be an illegal access / shape error instead of a clean fallback.
    # (Stock ``CosSinRoPE`` even moves a mismatched-device cache via ``.to``; we
    # fall back rather than gather across devices.) q/k must both be 4D
    # ``(batch, seq, heads, head_dim)`` sharing batch/seq: the kernel unpacks both
    # as 4D and indexes ``xk`` with ``xq``'s batch/seq tiles, so a mismatch would
    # read out of bounds.
    return (
        xq.is_cuda
        and xk.is_cuda
        and rope_cache.is_cuda
        and positions.is_cuda
        and xq.device == xk.device == rope_cache.device == positions.device
        and xq.ndim == 4
        and xk.ndim == 4
        and rope_cache.ndim == 2
        and rope_cache.shape[-1] == 2 * xq.shape[-1]
        and positions.ndim == 2
        and positions.dtype in (torch.int32, torch.int64)
        and tuple(positions.shape) == tuple(xq.shape[:2])
        and tuple(xk.shape[:2]) == tuple(xq.shape[:2])
        and xq.is_contiguous()
        and xk.is_contiguous()
        and rope_cache.is_contiguous()
        and positions.is_contiguous()
        and xq.shape[-1] == xk.shape[-1]
        and xq.shape[-1] % 2 == 0
    )


def _complex_eligible(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache_real: torch.Tensor,
    positions: torch.Tensor,
) -> bool:
    return (
        xq.is_cuda
        and xk.is_cuda
        and rope_cache_real.is_cuda
        and positions.is_cuda
        and xq.device == xk.device == rope_cache_real.device == positions.device
        and xq.ndim == 4
        and xk.ndim == 4
        and rope_cache_real.ndim == 3
        and rope_cache_real.shape[-2] * 2 == xq.shape[-1]
        and rope_cache_real.shape[-1] == 2
        and positions.ndim == 2
        and positions.dtype in (torch.int32, torch.int64)
        and tuple(positions.shape) == tuple(xq.shape[:2])
        and tuple(xk.shape[:2]) == tuple(xq.shape[:2])
        and xq.stride(-1) == 1
        and xk.stride(-1) == 1
        and rope_cache_real.is_contiguous()
        and positions.is_contiguous()
        and xq.shape[-1] == xk.shape[-1]
        and xq.shape[-1] % 2 == 0
    )


if _HELION_IMPORT_ERROR is None:

    def _apply_helion_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Run the fused kernel, or return ``None`` if the inputs are unsupported.

        The custom op is valid in eager and compiled execution. Fallback is
        reserved for inputs the kernel cannot safely handle; the caller uses the
        PyTorch path for those cases.
        """
        xq = _to_local(query)
        xk = _to_local(key)
        cache = _to_local(rope_cache)
        pos = _resolve_positions(positions, xq)
        xq = xq.contiguous()
        xk = xk.contiguous()
        cache = cache.contiguous()
        pos = pos.contiguous()
        if not _eligible(xq, xk, cache, pos):
            warn_once(
                logger,
                "HelionRoPE: inputs unsupported by the fused kernel (need "
                "CUDA q/k/cache/positions on one device, a 2D cache of width "
                "2 * head_dim, and integer (batch, seq_len) position ids); "
                "falling back to the PyTorch cos/sin RoPE.",
            )
            return None

        # Bounds parity with stock CosSinRoPE: async-assert position ids are in
        # range so an out-of-range id fails cleanly instead of triggering an
        # illegal memory access inside the kernel's cache gather. No host sync,
        # and skipped under torch.compile -- same semantics as the PyTorch path.
        _maybe_check_max_pos(pos, max_valid_pos=cache.shape[0] - 1)

        xq_out, xk_out = _helion_rope_fwd(xq, xk, cache, pos)
        return _from_local(xq_out, query), _from_local(xk_out, key)

    def _apply_helion_complex_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        xq = _to_local(query)
        xk = _to_local(key)
        cache = _to_local(rope_cache)
        pos = _resolve_positions(positions, xq)
        if not cache.is_complex():
            return None
        cache_real = torch.view_as_real(cache)
        cache_real = cache_real.contiguous()
        pos = pos.contiguous()
        if not _complex_eligible(xq, xk, cache_real, pos):
            warn_once(
                logger,
                "HelionComplexRoPE: inputs unsupported by the fused kernel "
                "(need CUDA q/k/cache/positions on one device, a complex cache of "
                "width head_dim / 2, and integer (batch, seq_len) position ids); "
                "falling back to the PyTorch complex RoPE.",
            )
            return None

        _maybe_check_max_pos(pos, max_valid_pos=cache.shape[0] - 1)

        if torch.compiler.is_compiling():
            xq_out, xk_out = _helion_complex_rope_fwd(xq, xk, cache_real, pos)
        else:
            xq_out, xk_out = _HelionComplexRoPEFunction.apply(xq, xk, cache_real, pos)
        return _from_local(xq_out, query), _from_local(xk_out, key)

else:

    def _apply_helion_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        raise ImportError(
            "HelionRoPE override is active but `helion` is not installed; "
            "install helion to use torchtitan.overrides.helion_rope."
        )

    def _apply_helion_complex_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        raise ImportError(
            "HelionComplexRoPE override is active but `helion` is not installed; "
            "install helion to use torchtitan.overrides.helion_rope."
        )


class HelionRoPE(CosSinRoPE):
    """Cos/sin RoPE that applies the rotation with a fused Helion kernel.

    A drop-in for :class:`CosSinRoPE`: it reuses the same cache and rotate-half
    math, so it stays checkpoint-compatible, and transparently falls back to the
    PyTorch implementation for unsupported tensor inputs (see the module
    docstring). Registered as an override on ``CosSinRoPE.Config``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = _apply_helion_rope(query, key, self.cache, positions)
        if out is None:
            return super().forward(query, key, positions)
        return out


class HelionComplexRoPE(ComplexRoPE):
    """Complex RoPE applied with a fused real-valued Helion kernel.

    ``ComplexRoPE`` stores ``cos + i*sin`` as a complex cache and rotates
    adjacent dimension pairs via complex multiplication. Helion does not need to
    compile complex tensors for this path: the forward pass presents the cache
    as ``torch.view_as_real(cache)`` and the kernel performs the same real-valued
    multiply. Registered as an exact override on ``ComplexRoPE.Config``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = _apply_helion_complex_rope(query, key, self.cache, positions)
        if out is None:
            return super().forward(query, key, positions)
        return out


@override(
    "helion_rope",
    target=CosSinRoPE.Config,
    exact=True,
    description="Fused Helion cos/sin rotary embedding (CUDA).",
)
def helion_rope(cfg: CosSinRoPE.Config) -> HelionRoPE.Config:
    if _HELION_IMPORT_ERROR is not None:
        raise ImportError(
            "HelionRoPE override was requested but `helion` is not installed; "
            "install helion to use torchtitan.overrides.helion_rope."
        ) from _HELION_IMPORT_ERROR
    return derive(cfg, HelionRoPE.Config)


@override(
    "helion_complex_rope",
    target=ComplexRoPE.Config,
    exact=True,
    description="Fused Helion complex rotary embedding (CUDA).",
)
def helion_complex_rope(cfg: ComplexRoPE.Config) -> HelionComplexRoPE.Config:
    if _HELION_IMPORT_ERROR is not None:
        raise ImportError(
            "HelionComplexRoPE override was requested but `helion` is not "
            "installed; install helion to use torchtitan.overrides.helion_rope."
        ) from _HELION_IMPORT_ERROR
    return derive(cfg, HelionComplexRoPE.Config)
