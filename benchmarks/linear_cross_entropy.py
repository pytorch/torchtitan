# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare ordinary and joint-compiled chunked projection/CE in fresh processes."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.linear import Linear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--vocab", type=int, default=163840)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numerics", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    head = (
        Linear.Config(in_features=args.hidden, out_features=args.vocab)
        .build()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    torch.nn.init.normal_(head.weight, std=0.02)
    x = torch.randn(
        args.tokens,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    labels = torch.randint(args.vocab, (args.tokens,), device="cuda")
    labels[::17] = -100
    # Global denominator differs from this microbatch's local valid-token count.
    denominator = (labels != -100).sum() * 16
    loss_fn = ChunkedLossWrapper.Config(
        num_chunks=args.chunks, linear_cross_entropy=args.joint
    ).build()
    loss_fn.set_lm_head(head)
    apply_local_compile([] if args.eager else ["loss"])
    rows = []
    loss = None
    for iteration in range(args.steps + 3):
        head.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        if args.snapshot and iteration == 3:
            torch.cuda.memory._record_memory_history(stacks="python", max_entries=20000)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        loss, _ = loss_fn(x, labels, denominator)
        # The wrapper executes the projection backward in its forward call.
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        rows.append(
            {
                "iteration": iteration,
                "baseline": baseline,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_above_baseline": torch.cuda.max_memory_allocated() - baseline,
                "reserved": torch.cuda.memory_reserved(),
                "device_used_end": total - free,
                "ms": start.elapsed_time(end),
                "loss": loss.item(),
            }
        )
        if args.snapshot and iteration == 3:
            torch.cuda.memory._dump_snapshot(str(args.output.with_suffix(".pickle")))
            torch.cuda.memory._record_memory_history(enabled=None)
    if args.numerics:
        assert loss is not None and x.grad is not None
        torch.save(
            {
                "loss": loss.detach().cpu(),
                "hidden_grad": x.grad.cpu(),
                "weight_grad": head.weight.grad.cpu(),
            },
            args.output.with_suffix(".pt"),
        )
    result = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cold": rows[0],
        "warm": rows[3:],
        # Report absolute allocation as well: compilation can retain different
        # amounts across variants, so baseline subtraction alone hides savings.
        "median_peak_allocated": statistics.median(
            r["peak_allocated"] for r in rows[3:]
        ),
        "median_baseline": statistics.median(r["baseline"] for r in rows[3:]),
        "median_peak_above_baseline": statistics.median(
            r["peak_above_baseline"] for r in rows[3:]
        ),
        "median_ms": statistics.median(r["ms"] for r in rows[3:]),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("warm", "cold")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
