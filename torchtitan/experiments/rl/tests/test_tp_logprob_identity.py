# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test that TP=2 trainer produces bit-identical logprobs to TP=1 trainer
for the same input, under batch-invariant mode.

Strategy: Build at TP=1, capture logprobs. Then parallelize to TP=2
using the SAME weights, capture logprobs. Compare.

Since we can't easily go TP=1 -> TP=2 in the same process, we use the
TP forward-only test approach: build at TP=2 from scratch, init weights
the same way, and compare batch-of-3 vs batch-of-5 outputs (batch
invariance under TP=2).

Requires 2 GPUs.

Run with:
    torchrun --nproc_per_node=2 torchtitan/experiments/rl/tests/test_tp_logprob_identity.py
"""

import os
import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.rl.batch_invariant import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.models.qwen3 import model_registry


def main():
    # ---- NCCL deterministic settings ----
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"This test requires exactly 2 GPUs, got {world_size}"

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    enable_batch_invariant_mode()

    # ---- Build model at TP=2 ----
    model_spec = model_registry("debugmodel")
    vocab_size = 2048
    seq_len = 32

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = model_spec.model.build()
    torch.set_default_dtype(old_dtype)

    with patch("torchtitan.distributed.parallel_dims.device_type", "cuda"):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            etp=1,
            world_size=2,
        )

    parallelize_qwen3(
        model,
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(tensor_parallel_degree=2),
    )

    model.to_empty(device=device)
    torch.manual_seed(42)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    model.eval()

    # ---- Create test sequences ----
    torch.manual_seed(123)
    sequences = [
        torch.randint(0, vocab_size, (1, seq_len), device=device) for _ in range(5)
    ]

    # ---- Forward: batch of 3 vs batch of 5 ----
    with torch.no_grad():
        batch_3 = torch.cat(sequences[:3], dim=0)
        logits_3 = model(batch_3)
        lps_3 = F.log_softmax(logits_3.float(), dim=-1)

        batch_5 = torch.cat(sequences, dim=0)
        logits_5 = model(batch_5)
        lps_5 = F.log_softmax(logits_5.float(), dim=-1)

    # ---- Compare first 3 sequences ----
    total_mismatches = 0
    total_tokens = 0
    max_diff_all = 0.0
    details = []

    for i in range(3):
        lp_from_3 = lps_3[i]
        lp_from_5 = lps_5[i]
        num_mismatch = (lp_from_3 != lp_from_5).sum().item()
        max_diff = (lp_from_3 - lp_from_5).abs().max().item()
        total_mismatches += num_mismatch
        total_tokens += lp_from_3.numel()
        max_diff_all = max(max_diff_all, max_diff)
        details.append((i, num_mismatch, max_diff))

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("TP=2 batch invariance: batch-of-3 vs batch-of-5 logprobs")
        print(f"{'=' * 60}")
        for i, num_mismatch, max_diff in details:
            status = "PASS" if num_mismatch == 0 else "FAIL"
            print(
                f"  Seq {i}: [{status}] mismatches={num_mismatch}, "
                f"max_diff={max_diff:.6e}"
            )
        print(f"Total tokens:      {total_tokens}")
        print(f"Total mismatches:  {total_mismatches}")
        print(f"Max absolute diff: {max_diff_all:.6e}")

        if total_mismatches == 0:
            print("PASSED: TP=2 forward logprobs are batch-invariant")
        else:
            print("FAILED: TP=2 forward logprobs are NOT batch-invariant")
        print(f"{'=' * 60}")

    # ---- Now test backward batch invariance at TP=2 ----
    model.train()

    torch.manual_seed(789)
    targets = [
        torch.randint(0, vocab_size, (1, seq_len), device=device) for _ in range(3)
    ]

    # Run 1: sequence 0 alone
    model.zero_grad()
    out_single = model(sequences[0])
    loss_single = F.cross_entropy(
        out_single.view(-1, vocab_size),
        targets[0].view(-1),
        reduction="mean",
    )
    loss_single.backward()

    grads_single = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads_single[name] = p.grad.clone()

    # Run 2: all 3 sequences batched, but loss only on sequence 0
    model.zero_grad()
    batch_input = torch.cat(sequences[:3], dim=0)
    out_batch = model(batch_input)
    loss_batch = F.cross_entropy(
        out_batch[0:1].reshape(-1, vocab_size),
        targets[0].view(-1),
        reduction="mean",
    )
    loss_batch.backward()

    grads_batch = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads_batch[name] = p.grad.clone()

    # Compare gradients
    grad_mismatches = []
    for name in grads_single:
        if name in grads_batch:
            if not torch.equal(grads_single[name], grads_batch[name]):
                max_diff = (grads_single[name] - grads_batch[name]).abs().max().item()
                grad_mismatches.append((name, max_diff))

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("TP=2 backward batch invariance: batch=1 vs batch=3 gradients")
        print(f"{'=' * 60}")
        print(f"Parameters compared: {len(grads_single)}")
        print(f"Parameters with mismatches: {len(grad_mismatches)}")
        if grad_mismatches:
            print("FAILED: gradients differ")
            for name, diff in grad_mismatches[:15]:
                print(f"  {name}: max_diff={diff:.6e}")
        else:
            print("PASSED: TP=2 backward gradients are batch-invariant")
        print(f"{'=' * 60}\n")

    # Cleanup
    disable_batch_invariant_mode()
    dist.destroy_process_group()

    exit_code = 0 if (total_mismatches == 0 and len(grad_mismatches) == 0) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
