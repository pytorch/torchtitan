# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module


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
        hc_mult = self.hc_mult
        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = (
            torch.sigmoid(
                pre * hc_scale[0]
                + hc_base[:hc_mult].view(*([1] * (pre.ndim - 1)), hc_mult)
            )
            + self.eps
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
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb


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
        shape, dtype = x.size(), x.dtype
        x = x.flatten(-2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_fn.float()) * rsqrt
        pre, post, comb = self.sinkhorn(
            mixes.float(), self.hc_scale.float(), self.hc_base.float()
        )
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
        return y.to(dtype), post, comb


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
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)


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
        shape, dtype = x.size(), x.dtype
        x = x.flatten(-2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
        return y.to(dtype)
