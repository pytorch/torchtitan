# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-time benchmarking for isolated FX graph rewrite regions."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import triton
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import optimization_hint
from triton.testing import do_bench

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileTimeBenchmarkResult:
    baseline_ms: float
    candidate_ms: float
    cache_hit: bool = False

    @property
    def speedup(self) -> float:
        return self.baseline_ms / self.candidate_ms


@dataclass(frozen=True)
class RewriteBenchmarkRegion:
    """Equivalent baseline and candidate graphs with positional tensor interfaces."""

    baseline: GraphModule
    baseline_inputs: tuple[Node, ...]
    candidate: GraphModule
    candidate_inputs: tuple[Node, ...]
    signature: tuple[Any, ...]


@dataclass(frozen=True)
class RewriteCandidate:
    """Prepared benchmark regions and application for one rewrite occurrence."""

    name: str
    regions: tuple[RewriteBenchmarkRegion, ...]
    apply: Callable[[], None]


@dataclass(frozen=True)
class _BenchmarkApplication:
    name: str
    status: str
    regions: tuple[CompileTimeBenchmarkResult, ...] = ()
    reason: str | None = None


def _hint_int(value: int | torch.SymInt) -> int:
    return int(optimization_hint(value))


def _tensor_signature(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, (tuple, list)):
        signatures = tuple(_tensor_signature(item) for item in value)
        return (
            None if any(signature is None for signature in signatures) else signatures
        )
    if not isinstance(value, torch.Tensor):
        return None
    device = value.fake_device if isinstance(value, FakeTensor) else value.device
    return (
        tuple(_hint_int(dim) for dim in value.shape),
        tuple(_hint_int(stride) for stride in value.stride()),
        value.dtype,
        device.type,
        device.index,
    )


def _region_inout_signature(
    inputs: tuple[Node, ...],
    outputs: tuple[Node, ...],
) -> tuple[Any, ...] | None:
    input_signatures = [_tensor_signature(node.meta.get("val")) for node in inputs]
    output_signatures = [_tensor_signature(node.meta.get("val")) for node in outputs]
    if any(signature is None for signature in (*input_signatures, *output_signatures)):
        return None
    return (
        tuple(input_signatures),
        tuple(output_signatures),
    )


def _graph_outputs(gm: GraphModule) -> tuple[Node, ...]:
    output = next(node for node in gm.graph.nodes if node.op == "output")
    outputs = []

    def collect(node: Node) -> Node:
        outputs.append(node)
        return node

    torch.fx.node.map_arg(output.args[0], collect)
    return tuple(outputs)


def make_rewrite_benchmark_region(
    baseline: GraphModule,
    candidate: GraphModule,
) -> RewriteBenchmarkRegion:
    """Create a benchmark region from graphs with matching tensor interfaces."""
    baseline_inputs = tuple(
        node for node in baseline.graph.nodes if node.op == "placeholder"
    )
    candidate_inputs = tuple(
        node for node in candidate.graph.nodes if node.op == "placeholder"
    )
    baseline_signature = _region_inout_signature(
        baseline_inputs,
        _graph_outputs(baseline),
    )
    candidate_signature = _region_inout_signature(
        candidate_inputs,
        _graph_outputs(candidate),
    )
    if baseline_signature is None or candidate_signature is None:
        raise TypeError("benchmark graphs require tensor metadata on their interfaces")
    if baseline_signature != candidate_signature:
        raise RuntimeError("baseline and candidate benchmark-region signatures differ")
    return RewriteBenchmarkRegion(
        baseline,
        baseline_inputs,
        candidate,
        candidate_inputs,
        baseline_signature,
    )


def _target_signature(gm: GraphModule, node: Node) -> Any:
    if node.op not in {"call_module", "get_attr"}:
        return repr(node.target)
    value: Any = gm
    for component in str(node.target).split("."):
        value = getattr(value, component)
    if isinstance(value, GraphModule):
        return ("graph_module", _graph_fingerprint(value))
    if isinstance(value, torch.Tensor):
        return ("tensor", _tensor_signature(value))
    return (type(value).__qualname__, repr(value))


def _indexed_argument_signature(value: Any, indices: dict[Node, int]) -> Any:
    if isinstance(value, Node):
        return ("node", indices[value])
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(_indexed_argument_signature(item, indices) for item in value),
        )
    if isinstance(value, list):
        return (
            "list",
            tuple(_indexed_argument_signature(item, indices) for item in value),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (repr(key), _indexed_argument_signature(item, indices))
                    for key, item in value.items()
                )
            ),
        )
    return ("value", repr(value))


def _graph_fingerprint(gm: GraphModule) -> tuple[Any, ...]:
    indices = {node: index for index, node in enumerate(gm.graph.nodes)}
    return tuple(
        (
            node.op,
            None
            if node.op in {"placeholder", "output"}
            else _target_signature(gm, node),
            _indexed_argument_signature(node.args, indices),
            _indexed_argument_signature(node.kwargs, indices),
        )
        for node in gm.graph.nodes
    )


def _runtime_fingerprint() -> tuple[Any, ...]:
    versions = (torch.__version__, torch.version.cuda, triton.__version__)
    if not torch.cuda.is_initialized():
        return (*versions, "cuda-not-initialized")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (
        *versions,
        properties.name,
        properties.major,
        properties.minor,
        properties.total_memory,
        getattr(properties, "uuid", None),
    )


def _realize_input(node: Node) -> torch.Tensor:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"benchmark input {node.name} has no tensor metadata")
    tensor = torch.empty_strided(
        tuple(_hint_int(dim) for dim in value.shape),
        tuple(_hint_int(stride) for stride in value.stride()),
        dtype=value.dtype,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if value.dtype.is_floating_point:
        return tensor.normal_(std=0.02)
    if value.dtype == torch.bool:
        return tensor.random_(0, 2)
    return tensor.zero_()


def _realize_paired_inputs(
    baseline_nodes: tuple[Node, ...],
    candidate_nodes: tuple[Node, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    baseline_inputs = tuple(_realize_input(node) for node in baseline_nodes)
    by_name = {
        node.name: (tensor, _tensor_signature(node.meta.get("val")))
        for node, tensor in zip(baseline_nodes, baseline_inputs, strict=True)
    }
    by_signature: defaultdict[tuple[Any, ...], list[torch.Tensor]] = defaultdict(list)
    for node, tensor in zip(baseline_nodes, baseline_inputs, strict=True):
        signature = _tensor_signature(node.meta.get("val"))
        if signature is not None:
            by_signature[signature].append(tensor)

    occurrences: defaultdict[tuple[Any, ...], int] = defaultdict(int)
    candidate_inputs = []
    for node in candidate_nodes:
        signature = _tensor_signature(node.meta.get("val"))
        if signature is None:
            raise TypeError(f"benchmark input {node.name} has no tensor metadata")
        named = by_name.get(node.name)
        if named is not None and named[1] == signature:
            candidate_inputs.append(named[0])
            continue
        occurrence = occurrences[signature]
        occurrences[signature] += 1
        if occurrence >= len(by_signature[signature]):
            raise RuntimeError("baseline and candidate benchmark inputs differ")
        candidate_inputs.append(by_signature[signature][occurrence])
    return baseline_inputs, tuple(candidate_inputs)


class CompileTimeBenchmarker:
    """Measure explicit rewrite regions and cache structurally equivalent runs."""

    def __init__(
        self,
        *,
        duration_ms: int = 20,
        atol: float | None = None,
        rtol: float | None = None,
        minimum_speedup: float = 1.01,
    ) -> None:
        if (atol is None) != (rtol is None):
            raise ValueError("atol and rtol must both be specified or both omitted")
        self.duration_ms = duration_ms
        self.atol = atol
        self.rtol = rtol
        self.minimum_speedup = minimum_speedup
        self._cache: dict[tuple[Any, ...], CompileTimeBenchmarkResult] = {}

    def clear(self) -> None:
        self._cache.clear()

    def accepts(self, result: CompileTimeBenchmarkResult) -> bool:
        return result.speedup >= self.minimum_speedup

    def benchmark_region(
        self,
        baseline: GraphModule,
        baseline_input_nodes: tuple[Node, ...],
        candidate: GraphModule,
        candidate_input_nodes: tuple[Node, ...],
        *,
        process_baseline: Callable[
            [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
        ]
        | None = None,
        process_candidate: Callable[
            [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
        ]
        | None = None,
    ) -> CompileTimeBenchmarkResult:
        """Measure baseline and candidate in their configured runtime forms."""
        baseline_inputs, candidate_inputs = _realize_paired_inputs(
            baseline_input_nodes, candidate_input_nodes
        )
        processed_baseline = (
            baseline
            if process_baseline is None
            else process_baseline(baseline, baseline_inputs)
        )
        processed_candidate = (
            candidate
            if process_candidate is None
            else process_candidate(candidate, candidate_inputs)
        )
        expected = processed_baseline(*baseline_inputs)
        actual = processed_candidate(*candidate_inputs)
        torch.testing.assert_close(
            actual,
            expected,
            atol=self.atol,
            rtol=self.rtol,
        )
        baseline_ms = do_bench(
            lambda: processed_baseline(*baseline_inputs),
            rep=self.duration_ms,
            return_mode="median",
        )
        candidate_ms = do_bench(
            lambda: processed_candidate(*candidate_inputs),
            rep=self.duration_ms,
            return_mode="median",
        )
        return CompileTimeBenchmarkResult(baseline_ms, candidate_ms)

    def benchmark_regions(
        self,
        regions: Iterable[RewriteBenchmarkRegion],
        *,
        cache_key: Hashable,
        benchmark_region: Callable[
            [GraphModule, tuple[Node, ...], GraphModule, tuple[Node, ...]],
            CompileTimeBenchmarkResult,
        ]
        | None = None,
        process_baseline: Callable[
            [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
        ]
        | None = None,
        process_candidate: Callable[
            [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
        ]
        | None = None,
    ) -> tuple[CompileTimeBenchmarkResult, ...]:
        """Benchmark regions and cache structurally equivalent measurements."""
        results = []
        for region in regions:
            # Processing semantics, tensor interfaces, graph behavior, runtime
            # target, and measurement settings must all match for cache reuse.
            measurement_cache_key = (
                cache_key,
                region.signature,
                _graph_fingerprint(region.baseline),
                _graph_fingerprint(region.candidate),
                _runtime_fingerprint(),
                self.duration_ms,
                self.atol,
                self.rtol,
            )
            result = self._cache.get(measurement_cache_key)
            if result is None:
                if benchmark_region is None:
                    result = self.benchmark_region(
                        region.baseline,
                        region.baseline_inputs,
                        region.candidate,
                        region.candidate_inputs,
                        process_baseline=process_baseline,
                        process_candidate=process_candidate,
                    )
                else:
                    result = benchmark_region(
                        region.baseline,
                        region.baseline_inputs,
                        region.candidate,
                        region.candidate_inputs,
                    )
                self._cache[measurement_cache_key] = result
            else:
                result = CompileTimeBenchmarkResult(
                    result.baseline_ms,
                    result.candidate_ms,
                    cache_hit=True,
                )
            results.append(result)
        return tuple(results)


_COMPILE_TIME_BENCHMARKER = CompileTimeBenchmarker()


def clear_compile_time_benchmark_cache() -> None:
    """Clear measurements cached by the shared compile-time benchmarker."""
    _COMPILE_TIME_BENCHMARKER.clear()


def _log_benchmark_summary(
    *,
    name: str,
    applications: list[_BenchmarkApplication],
) -> None:
    counts = Counter(application.status for application in applications)
    rejected = counts["rejected"] + counts["failed"]
    lines = [
        f"Compile-time benchmark results for {name}: "
        f"candidates={len(applications)}, "
        f"applied={counts['applied']}, rejected={rejected} "
        f"(slower={counts['rejected']}, failed={counts['failed']})"
    ]
    for status in ("applied", "rejected", "failed"):
        selected = [item for item in applications if item.status == status]
        if not selected:
            continue
        lines.append(f"  {status.upper()} ({len(selected)}):")
        for application in selected:
            candidate = application.name.removeprefix(f"{name}:")
            details = [
                f"region {index}: baseline="
                f"{result.baseline_ms * 1000:.1f} us, "
                f"candidate={result.candidate_ms * 1000:.1f} us, "
                f"speedup={result.speedup:.3f}x, "
                f"cache={'hit' if result.cache_hit else 'miss'}"
                for index, result in enumerate(application.regions)
            ]
            if application.reason is not None:
                details.append(application.reason)
            lines.append(f"    candidate {candidate}: " + "; ".join(details))
    report = "\n".join(lines)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": f"compile_time_benchmark_{name}",
            "encoding": "string",
        },
        payload_fn=lambda: report,
        expect_trace_id=False,
    )
    log = logger.warning if counts["failed"] else logger.info
    log(report)


def apply_benchmarked_rewrites(
    gm: GraphModule,
    *,
    name: str,
    prepare_candidates: Callable[[GraphModule], Sequence[RewriteCandidate]],
    cache_key: Hashable,
    strict: bool = False,
    benchmarker: CompileTimeBenchmarker | None = None,
    benchmark_region: Callable[
        [GraphModule, tuple[Node, ...], GraphModule, tuple[Node, ...]],
        CompileTimeBenchmarkResult,
    ]
    | None = None,
    process_baseline: Callable[
        [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
    ]
    | None = None,
    process_candidate: Callable[
        [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
    ]
    | None = None,
) -> GraphModule:
    """Prepare candidates once and apply only benchmarked improvements."""
    if not torch.cuda.is_available():
        logger.warning(
            f"{name} compile-time benchmark requires CUDA; "
            "keeping the original graph"
        )
        return gm

    if benchmarker is None:
        benchmarker = _COMPILE_TIME_BENCHMARKER
    applications: list[_BenchmarkApplication] = []
    for candidate in prepare_candidates(gm):
        if not candidate.regions:
            applications.append(
                _BenchmarkApplication(
                    candidate.name,
                    "failed",
                    reason="rewrite declared no benchmark regions",
                )
            )
            continue

        results: tuple[CompileTimeBenchmarkResult, ...] = ()
        try:
            results = benchmarker.benchmark_regions(
                candidate.regions,
                cache_key=cache_key,
                benchmark_region=benchmark_region,
                process_baseline=process_baseline,
                process_candidate=process_candidate,
            )
            use_candidate = all(benchmarker.accepts(result) for result in results)
        except Exception as error:
            if strict:
                raise RuntimeError(
                    f"{name} candidate {candidate.name} benchmark failed"
                ) from error
            applications.append(
                _BenchmarkApplication(
                    candidate.name,
                    "failed",
                    results,
                    f"benchmark failed with {type(error).__name__}: {error}",
                )
            )
            use_candidate = False
        else:
            applications.append(
                _BenchmarkApplication(
                    candidate.name,
                    "applied" if use_candidate else "rejected",
                    results,
                    None
                    if use_candidate
                    else "candidate was not faster for every changed region",
                )
            )

        if use_candidate:
            try:
                candidate.apply()
            except Exception as error:
                raise RuntimeError(
                    f"{name} candidate {candidate.name} apply failed after "
                    "successful benchmarking"
                ) from error

    _log_benchmark_summary(
        name=name,
        applications=applications,
    )
    return gm
