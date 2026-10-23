# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grad mode is part of Dynamo's global-state guard, so a `torch.no_grad()` validation pass
recompiles every region once more. With many shapes per region this doubles graph counts toward
recompile_limit=8, and under fullgraph=True exceeding it is a hard FailOnRecompileLimitHit.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, ask 1):
2 graphs after training, 3 after a no_grad pass; FailOnRecompileLimitHit at signature 9.
"""
import torch
from _common import num_graphs
from torch._dynamo.utils import guard_failures


def region(x, w):
    return torch.nn.functional.silu(x) * w


f = torch.compile(region, fullgraph=True)
w = torch.randn(64, device="cuda", requires_grad=True)
for T in (128, 256, 512):  # static, then dynamic T
    f(torch.randn(T, 64, device="cuda", requires_grad=True), w).sum().backward()
print("after training at 3 token counts: graphs =", num_graphs(region))
with torch.no_grad():
    for T in (128, 384):
        f(torch.randn(T, 64, device="cuda"), w)
print("after a no_grad validation pass: graphs =", num_graphs(region))
print(
    "last guard failure:", str(guard_failures[region.__code__][-1]).split("\n")[0][:100]
)

# The same doubling with a region that sees 5 signatures in training (tensor rank always
# specializes) hits the default limit of 8 once a no_grad pass runs:
torch._dynamo.reset()


def region2(x):
    return torch.nn.functional.silu(x) * 2


g = torch.compile(region2, fullgraph=True)
graphs = 0
try:
    for mode in (torch.enable_grad, torch.no_grad):
        with mode():
            for rank in range(1, 6):
                g(torch.randn(*([4] * rank), device="cuda", requires_grad=True))
                graphs += 1
    print("no failure")
except Exception as e:
    print(
        f"5 ranks x 2 grad modes: {type(e).__name__} at signature {graphs + 1} "
        f"(recompile_limit={torch._dynamo.config.recompile_limit})"
    )
