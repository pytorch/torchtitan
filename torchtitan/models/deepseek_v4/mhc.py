# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.protocols.module import Module


class HcSplitSinkhorn(Module):
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
        hc_mult = self.hc_mult
        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = torch.sigmoid(
            pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)
        ) + self.eps
        post = 2 * torch.sigmoid(
            post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult].unsqueeze(0).unsqueeze(0)
        )
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(
            hc_mult, hc_mult
        ).unsqueeze(0).unsqueeze(0)

        comb = comb.softmax(-1) + self.eps
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            comb = comb / (row_sum + self.eps)
            col_sum = comb.sum(-2, keepdim=True)
            comb = comb / (col_sum + self.eps)
        return pre, post, comb


class HcPre(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        sinkhorn_iters: int = 20
        eps: float = 1e-6
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.norm_eps
        self.sinkhorn = HcSplitSinkhorn.Config(
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.sinkhorn_iters,
            eps=config.eps,
        ).build()

    def forward(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2)
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = self.sinkhorn(mixes, hc_scale, hc_base)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb


class HcPost(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)


class HcHead(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        norm_eps: float = 1e-6
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.norm_eps = config.norm_eps
        self.eps = config.eps

    def forward(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2)
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)
