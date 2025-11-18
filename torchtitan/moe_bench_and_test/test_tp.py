# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.deepseek_v3 import (
    DeepSeekV3Model,
    deepseekv3_args,
    parallelize_deepseekv3,
)
from torchtitan.moe_bench_and_test import (
    apply_old_moe_monkey_patches,
    assert_close,
    get_err_ratio,
)


def check_grads_close(
    model0: nn.Module,
    model1: nn.Module,
    tol=1e-2,
) -> None:
    fails = []
    for (n, p0), (_, p1) in zip(
        model0.named_parameters(), model1.named_parameters(), strict=True
    ):
        if p0.grad is None:
            assert p1.grad is None
        else:
            g0 = p0.grad.full_tensor() if isinstance(p0.grad, DTensor) else p0.grad
            g1 = p1.grad.full_tensor() if isinstance(p1.grad, DTensor) else p1.grad
            try:
                torch.testing.assert_close(g0, g1, atol=tol, rtol=tol)
            except AssertionError:
                fails.append(f"Failed on {n=}")
    if fails:
        raise AssertionError("\n".join(fails))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-2)
    args = parser.parse_args()
    try:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

        model_args = deepseekv3_args["debugmodel"]
        torch.manual_seed(42)

        # Create an FSDP, and two EP models, one which uses the old MoE impl:
        model_args.max_seq_len = args.seqlen
        model_fsdp = DeepSeekV3Model(model_args)
        model_fsdp.init_weights(buffer_device=device)
        model_ep = deepcopy(model_fsdp)
        model_ep_old = deepcopy(model_fsdp)

        # Monkey patch the moe and reorderer forward methods to use the old impl
        apply_old_moe_monkey_patches(model_ep_old)

        pd_kwargs = {
            "dp_replicate": 1,
            "cp": 1,
            "pp": 1,
            "world_size": world_size,
        }
        parallel_dims_fsdp = ParallelDims(
            **pd_kwargs, dp_shard=world_size, tp=1, ep=1, etp=1
        )
        tp = ep = world_size
        etp = 1
        parallel_dims_ep = ParallelDims(
            **pd_kwargs,
            dp_shard=1,
            tp=tp,
            ep=ep,
            etp=etp,
        )

        # Default JobConfig is fine for parallelization.
        job_config = JobConfig()
        model_fsdp = parallelize_deepseekv3(model_fsdp, parallel_dims_fsdp, job_config)
        model_ep = parallelize_deepseekv3(model_ep, parallel_dims_ep, job_config)
        model_ep_old = parallelize_deepseekv3(
            model_ep_old, parallel_dims_ep, job_config
        )

        # Run backwards:
        inputs = torch.randint(
            model_args.vocab_size, size=(args.bsz, args.seqlen), device=device
        )
        out_fsdp = model_fsdp(inputs)
        out_ep = model_ep(inputs)
        out_ep_old = model_ep_old(inputs)

        loss_fsdp = F.cross_entropy(
            out_fsdp.view(-1, out_fsdp.shape[-1]), inputs.view(-1)
        )
        loss_fsdp.backward()

        # TP requires loss-parallel by default
        with dist_utils.get_train_context(enable_loss_parallel=True)(cp_context=None):
            loss_ep = F.cross_entropy(
                out_ep.view(-1, out_ep.shape[-1]), inputs.view(-1)
            )
            loss_ep.backward()
            loss_ep_old = F.cross_entropy(
                out_ep_old.view(-1, out_ep_old.shape[-1]), inputs.view(-1)
            )
            loss_ep_old.backward()

        check_grads_close(model_fsdp, model_ep, tol=args.tol)
        check_grads_close(model_fsdp, model_ep_old, tol=args.tol)

        # All-gather TP loss-parallel outputs for easier comparison.
        out_fsdp = out_fsdp.detach()
        out_ep = out_ep.detach().full_tensor()
        out_ep_old = out_ep_old.detach().full_tensor()

        assert_close("out fsdp vs ep", out_fsdp, out_ep, args.tol)
        assert_close("out fsdp vs ep_old", out_fsdp, out_ep_old, args.tol)

        # Compute error ratios
        err_ratio_fsdp_ep = get_err_ratio(out_fsdp, out_ep)
        err_ratio_fsdp_ep_old = get_err_ratio(out_fsdp, out_ep_old)
        err_ratio_ep_ep_old = get_err_ratio(out_ep, out_ep_old)

        # And compute KL divergences
        kl_fsdp_ep = F.kl_div(
            out_fsdp.reshape(-1, args.seqlen).log_softmax(dim=-1),
            out_ep.reshape(-1, args.seqlen).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        kl_fsdp_ep_old = F.kl_div(
            out_fsdp.reshape(-1, args.seqlen).log_softmax(dim=-1),
            out_ep_old.reshape(-1, args.seqlen).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        kl_ep_ep_old = F.kl_div(
            out_ep.reshape(-1, args.seqlen).log_softmax(dim=-1),
            out_ep_old.reshape(-1, args.seqlen).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )

        if not rank:
            print(f"\n{args=}, {world_size=}, {tp=}, {ep=}, {etp=}")
            print(f"\n{err_ratio_fsdp_ep_old=}")
            print(f"{err_ratio_fsdp_ep=}")
            print(f"{err_ratio_ep_ep_old=}")
            print(f"\n{kl_fsdp_ep_old=}")
            print(f"{kl_fsdp_ep=}")
            print(f"{kl_ep_ep_old=}")
    finally:
        dist.destroy_process_group()
