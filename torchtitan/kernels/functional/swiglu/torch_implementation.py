# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def _swiglu_torch(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    dtype = g.dtype

    g = g.float()
    u = u.float()

    y = u * F.silu(g)
    y = y.to(dtype)

    return y


def _swiglu_packed_torch(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()

    u = x[..., 1::2]
    g = x[..., ::2]

    x = u * F.silu(g)

    return x.to(dtype)
