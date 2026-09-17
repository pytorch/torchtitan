# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module


# ---------------------------------------------------------------------------
# The HC math as plain tensor functions, compiled individually.
#
# Why here and not through the trainer's ``compile.enable``: compiling the
# whole transformer block never fused anything on the flash recipe -- Dynamo
# hit a graph break inside an SPMD typecheck context manager in the attention
# path and left the block body eager (four attempts, profile-verified). These
# functions contain nothing but tensor ops, so each is its own small graph:
# the fp32 upcast, RMS statistic, mixing linear, sinkhorn and the
# broadcast-multiply-sum become a handful of fused kernels instead of ~20
# separate passes over a [T, hc_mult*D] fp32 tensor per call.
#
# ``fullgraph=True`` makes a graph break an error rather than a silent
# fallback to eager. ``dynamic=False``: shapes are fixed per rank.
# HC_COMPILE=0 runs the same functions eagerly (numerics reference).
# ---------------------------------------------------------------------------


def _sinkhorn_split(mixes, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = (
        torch.sigmoid(
            pre * hc_scale[0]
            + hc_base[:hc_mult].view(*([1] * (pre.ndim - 1)), hc_mult)
        )
        + eps
    )
    post = 2 * torch.sigmoid(
        post * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult].view(*([1] * (post.ndim - 1)), hc_mult)
    )
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(
        *([1] * (comb.ndim - 2)), hc_mult, hc_mult
    )

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _hc_pre_math(x, hc_fn, hc_scale, hc_base, *, hc_mult, sinkhorn_iters, eps, norm_eps):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre, post, comb = _sinkhorn_split(
        mixes.float(), hc_scale.float(), hc_base.float(),
        hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps,
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
    return y.to(dtype), post, comb


def _hc_post_math(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def _hc_head_math(x, hc_fn, hc_scale, hc_base, *, norm_eps, eps):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
    return y.to(dtype)


_HC_COMPILE = os.environ.get("HC_COMPILE", "1") == "1"


def _maybe_compile(fn):
    if not _HC_COMPILE:
        return fn
    return torch.compile(fn, fullgraph=True, dynamic=False)


_hc_pre = _maybe_compile(_hc_pre_math)
_hc_post = _maybe_compile(_hc_post_math)
_hc_head = _maybe_compile(_hc_head_math)



class HcSplitSinkhorn(Module):
    """Convert HC mix logits into pre, post, and combination weights."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        sinkhorn_iters: int = 20
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.sinkhorn_iters
        self.eps = config.eps

    def forward(self, mixes, hc_scale, hc_base):
        """Split and normalize HC mixing logits.

        Args:
            mixes: HC logits of shape ``[B, L, (2 + hc_mult) * hc_mult]``.
            hc_scale: Scale tensor of shape ``[3]``.
            hc_base: Bias tensor of shape ``[(2 + hc_mult) * hc_mult]``.

        Returns:
            ``pre`` and ``post`` tensors of shape ``[T, hc_mult]`` and
            ``comb`` of shape ``[T, hc_mult, hc_mult]``.
        """
        return _sinkhorn_split(
            mixes, hc_scale, hc_base,
            hc_mult=self.hc_mult, sinkhorn_iters=self.sinkhorn_iters, eps=self.eps,
        )


class HcPre(Module):
    """Reduce HC branches before attention or FFN computation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        sinkhorn_iters: int = 20
        eps: float = 1e-6
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_mult = config.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.dim
        self.hc_mult = config.hc_mult
        self.norm_eps = config.norm_eps
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_scale = nn.Parameter(torch.empty(3))
        self.sinkhorn = HcSplitSinkhorn.Config(
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.sinkhorn_iters,
            eps=config.eps,
        ).build()

    def forward(self, x):
        """Project multi-branch hidden states into a single branch.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.

        Returns:
            Tuple ``(y, post, comb)`` where ``y`` has shape ``[T, D]`` and
            ``post``/``comb`` are consumed by ``HcPost``.
        """
        return _hc_pre(
            x, self.hc_fn, self.hc_scale, self.hc_base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.sinkhorn.sinkhorn_iters,
            eps=self.sinkhorn.eps,
            norm_eps=self.norm_eps,
        )


class HcPost(Module):
    """Expand a single-branch output back to HC branches with residual mixing."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x, residual, post, comb):
        """Apply HC post mixing.

        Args:
            x: Single-branch output of shape ``[T, D]``.
            residual: Residual branches of shape ``[T, hc_mult, D]``.
            post: Post weights of shape ``[T, hc_mult]``.
            comb: Branch combination weights of shape ``[T, hc_mult, hc_mult]``.

        Returns:
            Hidden states of shape ``[T, hc_mult, D]``.
        """
        return _hc_post(x, residual, post, comb)


class HcHead(Module):
    """Merge final HC branches before the output norm and LM head."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        norm_eps: float = 1e-6
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_dim = config.hc_mult * config.dim
        self.norm_eps = config.norm_eps
        self.eps = config.eps
        self.hc_fn = nn.Parameter(
            torch.empty(config.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, x):
        """Merge HC branches.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.

        Returns:
            Hidden states of shape ``[T, D]``.
        """
        return _hc_head(
            x, self.hc_fn, self.hc_scale, self.hc_base,
            norm_eps=self.norm_eps, eps=self.eps,
        )
