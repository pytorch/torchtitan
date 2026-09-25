# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark tracing speed with vs without aot_nested_region.

Measures wall-clock time of trace_module() on Llama3 debugmodel with
varying n_layers. Verifies bitwise correctness at each layer count.

Usage:
    python -m torchtitan.experiments.graph_trainer.tests.bench_nested_region

Example output on H100 (median of 3 runs):
    Layers     Baseline (s)    Nested (s)      Speedup    Bitwise
    6          0.288           0.116           2.5x       True
    16         0.809           0.166           4.9x       True
    32         1.871           0.234           8.0x       True
    64         3.585           0.411           8.7x       True
"""

import dataclasses
import time

import torch

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    run_traced_module,
    trace_module,
)
from torchtitan.experiments.graph_trainer.nested_region import (
    _subgraph_cache,
    aot_nested_region,
)
from torchtitan.models.llama3 import llama3_configs, Llama3Model

CONST_HASH = lambda *args: "block"
DEVICE = "cuda"
DTYPE = torch.float32
LAYER_COUNTS = [6, 16, 32, 64]
NUM_RUNS = 3


def make_model(n_layers):
    cfg = dataclasses.replace(llama3_configs["debugmodel"], n_layers=n_layers)
    model = Llama3Model(cfg)
    model.to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(DEVICE))
    return model, cfg


def median_trace_time(model, args):
    """Return median tracing time over NUM_RUNS."""
    times = []
    for _ in range(NUM_RUNS):
        _subgraph_cache.clear()
        t0 = time.perf_counter()
        trace_module(model, args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[NUM_RUNS // 2]


def main():
    assert torch.cuda.is_available(), "CUDA required"

    # Warmup: pay one-time JIT/import costs
    warmup, cfg = make_model(4)
    tokens = torch.randint(0, cfg.vocab_size, (2, 128), device=DEVICE)
    trace_module(warmup, (tokens,))
    _subgraph_cache.clear()
    for layer in warmup.layers.values():
        aot_nested_region(layer, hash_fn=CONST_HASH)
    trace_module(warmup, (tokens,))

    print(f"{'Layers':<10} {'Baseline (s)':<15} {'Nested (s)':<15} {'Speedup':<10} {'Bitwise':<10}")
    print("-" * 60)

    for n_layers in LAYER_COUNTS:
        model, cfg = make_model(n_layers)
        tokens = torch.randint(0, cfg.vocab_size, (2, 128), device=DEVICE)
        params = {
            **dict(model.named_parameters(remove_duplicate=False)),
            **dict(model.named_buffers(remove_duplicate=False)),
        }

        # Eager reference
        out_eager = model(tokens)

        # Baseline: trace without nested region, run, verify
        t_baseline = median_trace_time(model, (tokens,))
        traced_baseline = trace_module(model, (tokens,))
        out_baseline = run_traced_module(traced_baseline, params, (tokens,))
        baseline_bitwise = torch.equal(out_eager, out_baseline[0])

        # Nested: patch, trace, run, verify
        for layer in model.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)
        t_nested = median_trace_time(model, (tokens,))
        _subgraph_cache.clear()
        traced_nested = trace_module(model, (tokens,))
        out_nested = run_traced_module(traced_nested, params, (tokens,))
        nested_bitwise = torch.equal(out_eager, out_nested[0])

        speedup = t_baseline / t_nested
        print(
            f"{n_layers:<10} {t_baseline:<15.3f} {t_nested:<15.3f} "
            f"{speedup:<10.1f}x baseline={baseline_bitwise} nested={nested_bitwise}"
        )

    print("-" * 60)


if __name__ == "__main__":
    main()
