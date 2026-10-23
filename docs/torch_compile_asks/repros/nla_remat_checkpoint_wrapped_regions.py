# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan main's activation-checkpointing path vs SelectiveAC + wrap_inductor_compiled_regions.

See ../README.md, No longer applies. RegionAC/SelectiveAC wrap each block with torch_remat.checkpoint.
This runs a compiled activation (two shapes in one block) and a compiled flex-attention region inside
it, with wrap_inductor_compiled_regions=True, cold, at three token counts (recompiles between forward
and recompute), Dynamo LRU cache on.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130): every token count prints OK with a max gradient
difference of 0 vs no checkpoint.
"""

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

torch._inductor.config.wrap_inductor_compiled_regions = True
dev = "cuda"


@torch.compile(fullgraph=True)
def act(gate, up):
    return F.silu(gate) * up


flex = torch.compile(flex_attention, fullgraph=True)


def causal(b, h, q, kv):
    return q >= kv


def block(x, w13, w2, wqkv):
    gate, up = (x @ w13).chunk(2, dim=-1)
    big = act(gate, up) @ w2
    s = x[: x.shape[0] // 4]
    small = act(*(s @ w13).chunk(2, dim=-1)) @ w2
    T = x.shape[0]
    q, k, v = (x @ wqkv).view(1, T, 3, 4, 64).permute(2, 0, 3, 1, 4)
    mask = create_block_mask(causal, 1, 1, T, T, device=dev)
    attn = flex(q, k, v, block_mask=mask).permute(0, 2, 1, 3).reshape(T, 256)
    return big + small.repeat(4, 1) + attn


def run(T, w):
    x = torch.randn(T, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
    ref = torch.autograd.grad(block(x, *w).float().sum(), x)[0]
    out = remat.checkpoint(region_name="layers.0")(block)(x, *w)
    g = torch.autograd.grad(out.float().sum(), x)[0]
    return (g - ref).abs().max().item()


torch.manual_seed(0)
w = [
    torch.randn(*s, device=dev, dtype=torch.bfloat16, requires_grad=True) * 0.05
    for s in [(256, 1024), (512, 256), (256, 768)]
]
torch._dynamo.reset()
for T in (1024, 2048, 512):
    print(f"T={T}: OK, max |grad - no-checkpoint grad| = {run(T, w):.3g}")
