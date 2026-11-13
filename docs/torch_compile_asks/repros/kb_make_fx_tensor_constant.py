# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""make_fx (GraphTrainer's tracer) captures a tensor reached through a class attribute (not an
input) as a graph constant that aliases the trace-time tensor: an in-place update of that tensor
is seen by the traced graph, but rebinding the attribute to a new tensor is silently ignored.
TorchTitan's AuxLoss keeps its step denominator this way. CPU only.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, Known behaviors):
traced 2.0; after fill_(4) 1.0 (eager 1.0); after rebinding to tensor(8) 1.0 (eager 0.5) (CPU).
"""
import torch
from torch.fx.experimental.proxy_tensor import make_fx


class AuxLoss:
    denominator = torch.tensor(2.0)


def loss(x):
    return x.sum() / AuxLoss.denominator


gm = make_fx(loss)(torch.ones(4))
print("graph constants:", [n.target for n in gm.graph.nodes if n.op == "get_attr"])
print("traced, denominator 2:          ", gm(torch.ones(4)).item())
AuxLoss.denominator.fill_(4.0)
print(
    "after in-place fill_(4):         ",
    gm(torch.ones(4)).item(),
    " eager",
    loss(torch.ones(4)).item(),
)
AuxLoss.denominator = torch.tensor(8.0)
print(
    "after rebinding to tensor(8):    ",
    gm(torch.ones(4)).item(),
    " eager",
    loss(torch.ones(4)).item(),
)
