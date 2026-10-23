# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""make_fx (GraphTrainer's tracer) specializes Python ints in the inputs: a graph traced with n=3
silently returns 3 elements when called with n=5. DeepSeek-V4 sizes tensors from the Python int
VarlenMetadata.max_k and FA4 varlen gets max_seqlen as an int, so traced graphs freeze them. CPU only.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, Known behaviors):
traced with n=3, called with n=5 -> [0.0, 2.0, 4.0] (CPU).
"""
import torch
from torch.fx.experimental.proxy_tensor import make_fx


def head(x, n: int):
    return x[:n] * 2


gm = make_fx(head)(torch.arange(10.0), 3)
print("traced with n=3, called with n=5 ->", gm(torch.arange(10.0), 5).tolist())
