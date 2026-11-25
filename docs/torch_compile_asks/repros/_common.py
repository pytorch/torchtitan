# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the compile-team repros (pure torch)."""
import collections

import torch


def kernel_us(fn, steps=5, warmup=3):
    """Per-call GPU kernel time (sum of profiler kernel durations) in us, kernel count, top kernels."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    per = collections.Counter()
    count = 0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            per[e.name] += e.device_time / steps
            count += 1
    top = [(k[:70], round(v, 1)) for k, v in per.most_common(3)]
    return round(sum(per.values()), 1), count // steps, top


def fwd_bwd(fn, *inputs):
    """Closure running fn(*inputs) forward + backward with a fixed random upstream grad."""
    out = fn(*inputs)
    grad = torch.randn_like(out)

    def step():
        for x in inputs:
            if isinstance(x, torch.Tensor) and x.requires_grad:
                x.grad = None
        fn(*inputs).backward(grad)

    return step


def num_graphs(fn):
    """Number of compiled graphs cached for a function's code object."""
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    return len(_debug_get_cache_entry_list(fn.__code__))


def kernel_stats(
    fn, steps: int = 3
) -> tuple[int, float, dict[str, float], dict[str, int]]:
    """Kernels per call, summed kernel ms per call, per-kernel ms and per-kernel counts."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
    per_kernel = collections.Counter()
    counts = collections.Counter()
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            per_kernel[event.name] += event.device_time / steps / 1000
            counts[event.name] += 1
    total_count = sum(counts.values()) // steps
    return (
        total_count,
        sum(per_kernel.values()),
        dict(per_kernel),
        {k: v // steps for k, v in counts.items()},
    )
