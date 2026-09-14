# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registration and kernel policy types for CODA graph patterns."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch._inductor.pattern_matcher import Match, PatternExpr
from torch.fx import GraphModule

from torchtitan.experiments.graph_trainer.compile_time_benchmark import (
    BenchmarkCandidateSelection,
    RewriteBenchmarkRegion,
)


CodaMatcher = Callable[
    [
        GraphModule,
        tuple[Match, ...],
        Counter[str],
        list[RewriteBenchmarkRegion] | None,
        BenchmarkCandidateSelection | None,
        bool,
    ],
    None,
]


@dataclass(frozen=True)
class CodaKernel:
    """Backend and tuning policy for one named GEMM emitted by a pattern."""

    backend: str = "QUACK"
    fast_math: bool = False
    supports_autotune: bool = True
    best_configs: dict[int, dict[str, Any]] = field(default_factory=dict)
    best_configs_by_shape: dict[
        int, dict[tuple[int, int, int], dict[str, Any]]
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class CodaPattern:
    """Declarative search, structural rewrite, and kernel policy for a pattern."""

    name: str
    matcher: CodaMatcher
    priority: int
    search: PatternExpr
    kernels: dict[str, CodaKernel]

    def apply(
        self,
        gm: GraphModule,
        counts: Counter[str],
        benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
        selection: BenchmarkCandidateSelection | None = None,
        autotune: bool = False,
    ) -> None:
        """Run this registration's search, then validate and rewrite its matches."""
        candidates = tuple(
            match
            for node in list(gm.graph.nodes)
            if isinstance((match := self.search.match(node)), Match)
        )
        self.matcher(
            gm,
            candidates,
            counts,
            benchmark_regions,
            selection,
            autotune,
        )


_CODA_PATTERNS: dict[str, CodaPattern] = {}


def register_coda_pattern(
    *,
    priority: int,
    search: PatternExpr,
    kernels: dict[str, CodaKernel],
) -> Callable[[CodaMatcher], CodaMatcher]:
    """Register a named handler, its search, and its complete kernel policy."""

    def register(matcher: CodaMatcher) -> CodaMatcher:
        name = matcher.__name__
        if name in _CODA_PATTERNS:
            raise ValueError(f"CODA pattern {name!r} is already registered")
        if any(pattern.priority == priority for pattern in _CODA_PATTERNS.values()):
            raise ValueError(f"CODA pattern priority {priority} is already registered")
        if not kernels:
            raise ValueError(f"CODA pattern {name!r} must declare a kernel site")
        _CODA_PATTERNS[name] = CodaPattern(
            name,
            matcher,
            priority,
            search,
            kernels,
        )
        return matcher

    return register


def _quack_config(
    tile_m: int,
    tile_n: int,
    *,
    dynamic: bool,
    cluster_m: int = 2,
    cluster_n: int = 1,
    swap_ab: bool = False,
) -> dict[str, Any]:
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": None,
        "num_warps": None,
        "pingpong": False,
        "is_dynamic_persistent": dynamic,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "cluster_k": 1,
        "swap_ab": swap_ab,
        "max_swizzle_size": 8,
        "device_capacity": 10,
        "use_tma_gather": False,
    }
