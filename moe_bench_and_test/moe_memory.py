# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from torchtitan.components.metrics import DeviceMemoryMonitor
from torchtitan.models.moe.moe import MoE, MoEArgs, MoEOld

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cls", choices=["moe", "moe_old"])
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--bsz", type=int, default=4)
    args = parser.parse_args()

    # DSv3-16B-like layers
    device = "cuda"
    dim = 2048
    is_moe_list = None
    moe_inter_dim = 1408
    num_experts = 64
    num_shared_experts = 2
    route_norm = False
    score_before_experts = False
    seqlen = 64
    top_k = 6
    use_grouped_mm = True

    score_before_experts = False
    moe_args = MoEArgs(
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        score_func="softmax",
        route_norm=route_norm,
        score_before_experts=score_before_experts,
        top_k=top_k,
        use_grouped_mm=use_grouped_mm,
    )

    if args.cls == "moe":
        cls = MoE
    elif args.cls == "moe_old":
        cls = MoEOld
    else:
        raise ValueError

    torch.manual_seed(42)
    moe = cls(moe_args, dim=dim, hidden_dim=moe_inter_dim).to(
        device=device, dtype=torch.bfloat16
    )
    moe.init_weights(1 / dim**0.5, device)
    inputs = torch.randn(
        args.bsz,
        args.seqlen,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )
    mem_monitor = DeviceMemoryMonitor(device="cuda:0")
    for _ in range(args.iters):
        moe(inputs).sum().backward()
        moe.zero_grad()
    print(f"\n{args=}")
    peak_stats = mem_monitor.get_peak_stats()
    print(f"{peak_stats.max_active_gib=}")
    print(f"{peak_stats.max_reserved_gib=}")
