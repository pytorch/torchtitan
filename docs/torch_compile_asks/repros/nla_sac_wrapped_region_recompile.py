# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-torch repro: SelectiveAC + wrap_inductor_compiled_regions returns another call's cached outputs.

SAC keeps one FIFO of saved ``inductor_compiled_code`` outputs per compiled callable
(``torch/utils/checkpoint.py``, ``_sac_storage_key``). When one compiled function runs at two
shapes inside a checkpointed region and its second call triggers a recompile (automatic dynamic), the
recompute serves the first call from the newer graph, so it pops the second call's outputs. Depending on
what runs next, that is the RuntimeError below, a downstream shape assert (a grouped GEMM in torchtitan's
MoE, where the routed and shared experts call one activation region at [T*K, F] and [T, F]), or a silent
wrong tensor if the recompute stops before the second call.

Usage: python nla_sac_wrapped_region_recompile.py [cold|warm]  (add --cpu to run on CPU)
    cold: the first call happens inside the checkpoint (as in a training step that compiles)
    warm: run the block once without checkpointing first, so both graphs exist before SAC

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, No longer applies):
    cold: RuntimeError: inductor_compiled_code invocation index 1 encountered during backward but not found in storage
    warm: OK, grad matches no-checkpoint: True
"""

import sys

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

DEVICE = "cpu" if "--cpu" in sys.argv else "cuda"
torch._inductor.config.wrap_inductor_compiled_regions = True


@torch.compile(fullgraph=True)
def act(gate, up):
    return F.silu(gate) * up


def block(x, w13, w2):
    gate, up = (x @ w13).chunk(2, dim=-1)
    big = act(gate, up) @ w2
    small_gate, small_up = (x[: x.shape[0] // 4] @ w13).chunk(2, dim=-1)
    small = act(small_gate, small_up) @ w2
    return big + small.repeat(4, 1)


def policy(ctx, func, *args, **kwargs):
    if func is torch._higher_order_ops.inductor_compiled_code:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def main():
    args = [a for a in sys.argv[1:] if a != "--cpu"]
    mode = args[0] if args else "cold"
    torch.manual_seed(0)
    x = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    w13 = torch.randn(
        256, 1024, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    w2 = torch.randn(512, 256, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    if mode == "warm":
        block(x, w13, w2).sum().backward()
    ref = torch.autograd.grad(block(x, w13, w2).float().sum(), x)[0]
    torch._dynamo.reset() if mode == "cold" else None
    out = checkpoint(
        block,
        x,
        w13,
        w2,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
    )
    grad = torch.autograd.grad(out.float().sum(), x)[0]
    print(f"{mode}: OK, grad matches no-checkpoint: {torch.equal(grad, ref)}")


if __name__ == "__main__":
    main()
