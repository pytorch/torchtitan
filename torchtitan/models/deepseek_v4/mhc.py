# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import operator
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.hi_mid_lo_linear import hi_mid_lo_linear
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
        if torch.compiler.is_compiling():
            return pre, post, self._sinkhorn_per_entry(comb)

        row_max = comb.max(dim=-1, keepdim=True).values
        comb = torch.exp(comb - row_max)
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb

    def _sinkhorn_per_entry(self, comb: torch.Tensor) -> torch.Tensor:
        """``forward``'s softmax + Sinkhorn on ``comb`` ``[..., hc, hc]``, one tensor per entry.

        Compiled, the ``[..., hc, hc]`` form costs one kernel per normalization step:
        each column sum reads the previous step at transposed indices, so Inductor
        cannot fuse consecutive steps. With each entry a separate ``[...]`` tensor,
        every read is at the same index and the whole loop fuses.

        Example:
            comb [T, 4, 4] -> 16 tensors [T] -> 20 row + column passes -> [T, 4, 4]
        """
        n = self.hc_mult
        entries = [[comb[..., i, j] for j in range(n)] for i in range(n)]
        row_max = [functools.reduce(torch.maximum, row) for row in entries]
        entries = [[torch.exp(x - m) for x in row] for row, m in zip(entries, row_max)]
        # forward runs one row + column pass, then sinkhorn_iters - 1 more.
        for _ in range(max(self.sinkhorn_iters, 1)):
            row_sums = [
                functools.reduce(operator.add, row) + self.eps for row in entries
            ]
            entries = [[x / s for x in row] for row, s in zip(entries, row_sums)]
            col_sums = [
                functools.reduce(operator.add, col) + self.eps for col in zip(*entries)
            ]
            entries = [[x / s for x, s in zip(row, col_sums)] for row in entries]
        return torch.stack([torch.stack(row, dim=-1) for row in entries], dim=-2)


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

    @local_compile("mhc", batch_invariant=False)
    def forward(self, x):
        """Project multi-branch hidden states into a single branch.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.

        Returns:
            Tuple ``(y, post, comb)`` where ``y`` has shape ``[T, D]`` and
            ``post``/``comb`` are consumed by ``HcPost``.
        """
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(-2)
        x = x_flat.float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        # bf16 operands (FSDP mixed precision) take a bf16 GEMM with fp32 output
        # instead of an fp32 GEMM on upcast copies; fp32 operands fall back to fp32.
        mixes = hi_mid_lo_linear(x_flat, self.hc_fn) * rsqrt
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

    @local_compile("mhc", batch_invariant=False)
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

    @local_compile("mhc", batch_invariant=False)
    def forward(self, x):
        """Merge HC branches.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.

        Returns:
            Hidden states of shape ``[T, D]``.
        """
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(-2)
        x = x_flat.float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        # bf16 operands (FSDP mixed precision) take a bf16 GEMM with fp32 output
        # instead of an fp32 GEMM on upcast copies; fp32 operands fall back to fp32.
        mixes = hi_mid_lo_linear(x_flat, self.hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=-2)
        return y.to(dtype)
