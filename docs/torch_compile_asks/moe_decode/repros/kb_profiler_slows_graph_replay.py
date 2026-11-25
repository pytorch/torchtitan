# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Known behavior: after a torch.profiler session, CUDA-graph replays in the same process get slower.

16 small regions (each its own torch.compile), compiled static (T=34) and dynamic (T=8, then 64), replayed at
T=34 in one CUDA graph each. Walls are timed before any profiling and again after one profiler session.

Expected on 1x GB300 (see ../README.md, Known behaviors): before profiling static ~25 us, dynamic ~25 us;
after one profiler session ~30 us for both (~0.3 us per kernel).
"""
import torch
from _common import graph_kernels, graph_us, header

NUM_REGIONS, WIDTH = 16, 2048


def make_region(kind, index):
    """A distinct function per region, so each gets its own Dynamo cache entry."""
    namespace = {"torch": torch}
    exec(
        f"def region_{kind}_{index}(x):\n"
        f"    return torch.nn.functional.silu(x.float() * {1 + index / 64}).to(x.dtype)\n",
        namespace,
    )
    return namespace[f"region_{kind}_{index}"]


def build_chain(kind):
    regions = [
        torch.compile(
            make_region(kind, i),
            fullgraph=True,
            dynamic=False if kind == "static" else None,
        )
        for i in range(NUM_REGIONS)
    ]
    for num_tokens in [34] if kind == "static" else [8, 64]:
        y = torch.randn(num_tokens, WIDTH, device="cuda", dtype=torch.bfloat16)
        for region in regions:
            y = region(y)
    x = torch.randn(34, WIDTH, device="cuda", dtype=torch.bfloat16)

    def chain():
        y = x
        for region in regions:
            y = region(y)
        return y

    return chain


header()
with torch.no_grad():
    chains = {kind: build_chain(kind) for kind in ("static", "dynamic")}
    before = {kind: graph_us(chain, replays=200) for kind, chain in chains.items()}
    graph_kernels(chains["static"])  # one profiler session
    after = {kind: graph_us(chain, replays=200) for kind, chain in chains.items()}
for kind in chains:
    print(
        f"{kind:8s} T=34, {NUM_REGIONS} kernels: before profiling {before[kind]:5.1f} us, "
        f"after one profiler session {after[kind]:5.1f} us"
    )
