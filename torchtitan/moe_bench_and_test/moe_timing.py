# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from triton.testing import do_bench

from torchtitan.models.deepseek_v3 import deepseekv3_args
from torchtitan.models.moe.moe import MoE
from torchtitan.moe_bench_and_test import apply_old_moe_monkey_patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cls", choices=["moe", "moe_old"])
    parser.add_argument("--perf_reps", type=int, default=1000)
    parser.add_argument("--perf_warmups", type=int, default=100)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--bsz", type=int, default=4)
    args = parser.parse_args()

    # DSv3-16B layers
    device = "cuda"
    dim = 2048
    moe_inter_dim = 1408
    seqlen = 64

    torch.manual_seed(42)

    moe_args = deepseekv3_args["16B"].moe_args
    moe = MoE(moe_args, dim=dim, hidden_dim=moe_inter_dim).to(
        device=device, dtype=torch.bfloat16
    )
    if args.cls == "moe_old":
        apply_old_moe_monkey_patches(moe)
    moe.init_weights(1 / dim**0.5, device)
    inputs = torch.randn(
        args.bsz,
        args.seqlen,
        dim,
        device=device,
        dtype=torch.bfloat16,
    )
    moe_time_ms = do_bench(
        lambda: moe(inputs).sum().backward(),
        warmup=args.perf_warmups,
        rep=args.perf_reps,
    )
    print(f"\n{args=}")
    print(f"{moe_time_ms=}")
