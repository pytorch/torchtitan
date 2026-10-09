# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure the checked-out Kimi attention residual, one case per process.

For an original-formula control, retain the local-compile decorator and restore
only the scoring expression. FullAC wraps a residual-only module using the
production policy; this is not a full transformer or pipeline benchmark. Keep
snapshot and timing runs separate.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3 import model as kimi_model
from torchtitan.observability.profiler import MemoryProfiler


class Residual(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = Linear.Config(in_features=dim, out_features=1).build()
        self.norm = RMSNorm.Config(normalized_shape=dim, eps=1e-6).build()
        with torch.no_grad():
            self.projection.weight.normal_(std=dim**-0.5)
            self.norm.weight.uniform_(0.5, 1.5)

    def forward(self, partial, stack):
        return kimi_model._apply_attention_residual(
            partial, stack, self.projection, self.norm
        )


class CheckpointedResidual(nn.Module):
    def __init__(self, residual: Residual) -> None:
        super().__init__()
        self.layers = nn.ModuleList([residual])
        FullAC.Config().build().apply(self)

    def forward(self, partial, stack):
        return self.layers[0](partial, stack)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--entries", type=int, default=9, help="Includes partial.")
    parser.add_argument("--dim", type=int, default=7168)
    parser.add_argument("--no-partial", action="store_true")
    parser.add_argument("--strided", action="store_true")
    parser.add_argument("--full-ac", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--save-numerics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens <= 0 or args.entries <= 0 or args.dim <= 0 or args.steps <= 0:
        parser.error("tokens, entries, dim and steps must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    # Explicit precision settings make comparison between revisions reproducible.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    apply_local_compile(["kimi_attention_residual"] if args.mode == "compiled" else [])
    residual = Residual(args.dim).to(device="cuda", dtype=torch.bfloat16)
    module = CheckpointedResidual(residual) if args.full_ac else residual
    num_committed = args.entries - int(not args.no_partial)
    stride = 2 if args.strided else 1
    stack = torch.randn(
        args.tokens,
        num_committed,
        args.dim * stride,
        device="cuda",
        dtype=torch.bfloat16,
    )[..., ::stride].requires_grad_()
    partial = (
        None
        if args.no_partial
        else torch.randn(
            args.tokens,
            args.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
    )
    upstream = torch.randn(args.tokens, args.dim, device="cuda", dtype=torch.bfloat16)

    def iteration():
        module.zero_grad(set_to_none=True)
        stack.grad = None
        if partial is not None:
            partial.grad = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = module(partial, stack)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - start
        forward_peak = torch.cuda.max_memory_allocated()
        forward_end = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output.backward(upstream)
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - start
        backward_peak = torch.cuda.max_memory_allocated()
        stats = {
            "baseline_allocated_bytes": baseline,
            "forward_peak_allocated_bytes": forward_peak,
            "forward_end_allocated_bytes": forward_end,
            "backward_peak_allocated_bytes": backward_peak,
            "peak_over_baseline_bytes": max(forward_peak, backward_peak) - baseline,
            "end_allocated_bytes": torch.cuda.memory_allocated(),
            "end_reserved_bytes": torch.cuda.memory_reserved(),
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
        }
        if args.save_numerics:
            # Serialize after peak/timing collection, never retain GPU outputs.
            assert stack.grad is not None
            assert residual.projection.weight.grad is not None
            assert residual.norm.weight.grad is not None
            partial_gradient = None
            if partial is not None:
                assert partial.grad is not None
                partial_gradient = partial.grad.cpu()
            torch.save(
                {
                    "output": output.detach().cpu(),
                    "stack_grad": stack.grad.cpu(),
                    "partial_grad": partial_gradient,
                    "projection_grad": residual.projection.weight.grad.cpu(),
                    "norm_grad": residual.norm.weight.grad.cpu(),
                },
                args.output.with_suffix(".pt"),
            )
        del output
        return stats

    cold = iteration()
    for _ in range(args.warmup):
        iteration()
    gc.collect()
    memory_profiler = (
        MemoryProfiler(
            step_num=0,
            freq=args.steps + 1,
            snapshot_dir=str(args.output.with_suffix(".snapshots")),
            leaf_folder="",
            rank=0,
            max_entries=200_000,
        )
        if args.snapshot
        else None
    )
    try:
        measurements = [iteration() for _ in range(args.steps)]
    except torch.OutOfMemoryError:
        if memory_profiler is not None:
            memory_profiler.step(exit_ctx=True)
        raise
    if memory_profiler is not None:
        memory_profiler.step(exit_ctx=True)
        torch.cuda.memory._record_memory_history(enabled=None)
    source = Path(kimi_model.__file__)
    result = {
        "config": {**vars(args), "output": str(args.output)},
        "scope": "residual-only module; FullAC uses production policy",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "stack_shape": list(stack.shape),
        "stack_stride": list(stack.stride()),
        "fp32_stack_bytes": args.tokens * args.entries * args.dim * 4,
        "cold": cold,
        "measurements": measurements,
        "median": {
            name: statistics.median(row[name] for row in measurements)
            for name in measurements[0]
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median"]), flush=True)


if __name__ == "__main__":
    main()
