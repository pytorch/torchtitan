# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-time benchmarking for isolated FX graph rewrite regions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import triton
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule
from triton.testing import do_bench

from torchtitan.tools.logging import logger


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
    """Equivalent eager and candidate graphs with positional tensor interfaces."""

    baseline: GraphModule
    baseline_inputs: tuple[Node, ...]
    candidate: GraphModule
    candidate_inputs: tuple[Node, ...]
    signature: tuple[Any, ...]


@dataclass
class BenchmarkCandidateSelection:
    """Select one rewrite occurrence while preparing and applying it."""

    rejected: set[str]
    selected: str | None = None


@dataclass(frozen=True)
class _BenchmarkApplication:
    name: str
    status: str
    regions: tuple[CompileTimeBenchmarkResult, ...] = ()
    reason: str | None = None


BenchmarkRegionFn = Callable[
    [GraphModule, tuple[Node, ...], GraphModule, tuple[Node, ...]],
    CompileTimeBenchmarkResult,
]
BenchmarkCandidateFn = Callable[
    [
        GraphModule,
        BenchmarkCandidateSelection,
        list[RewriteBenchmarkRegion] | None,
    ],
    GraphModule,
]
BenchmarkGraphProcessorFn = Callable[
    [GraphModule, tuple[torch.Tensor, ...]],
    Callable[..., Any],
]


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


def _argument_signature(value: Any) -> Any:
    if isinstance(value, Node):
        return ("node", value.name)
    if isinstance(value, tuple):
        return ("tuple", tuple(_argument_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_argument_signature(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (repr(key), _argument_signature(item))
                    for key, item in value.items()
                )
            ),
        )
    return ("value", repr(value))


def _node_signature(node: Node) -> tuple[Any, ...]:
    return (
        node.op,
        repr(node.target),
        _argument_signature(node.args),
        _argument_signature(node.kwargs),
        tuple(sorted(user.name for user in node.users)),
    )


def _nodes_chain(gm: GraphModule, nodes: Iterable[Node]) -> list[Node]:
    selected = set(nodes)
    if not selected:
        return []

    descendants = set(selected)
    frontier = list(selected)
    while frontier:
        node = frontier.pop()
        for user in node.users:
            if user not in descendants:
                descendants.add(user)
                frontier.append(user)

    ancestors = set(selected)
    frontier = list(selected)
    while frontier:
        node = frontier.pop()
        for input_node in node.all_input_nodes:
            if input_node not in ancestors:
                ancestors.add(input_node)
                frontier.append(input_node)

    selected.update(descendants & ancestors)
    return [node for node in gm.graph.nodes if node in selected]


def changed_nodes(
    baseline: GraphModule,
    candidate: GraphModule,
) -> tuple[list[Node], list[Node]]:
    """Return convex regions containing every structural graph difference."""
    baseline_by_name = {node.name: node for node in baseline.graph.nodes}
    candidate_by_name = {node.name: node for node in candidate.graph.nodes}
    changed_names = {
        name
        for name in baseline_by_name.keys() & candidate_by_name.keys()
        if _node_signature(baseline_by_name[name])
        != _node_signature(candidate_by_name[name])
    }
    baseline_nodes = [
        node
        for node in baseline.graph.nodes
        if node.op not in ("placeholder", "output")
        and (node.name not in candidate_by_name or node.name in changed_names)
    ]
    candidate_nodes = [
        node
        for node in candidate.graph.nodes
        if node.op not in ("placeholder", "output")
        and (node.name not in baseline_by_name or node.name in changed_names)
    ]
    return (
        _nodes_chain(baseline, baseline_nodes),
        _nodes_chain(candidate, candidate_nodes),
    )


def _connected_components(nodes: Iterable[Node]) -> list[list[Node]]:
    ordered_nodes = list(nodes)
    selected = set(ordered_nodes)
    remaining = set(ordered_nodes)
    components = []
    while remaining:
        root = next(node for node in ordered_nodes if node in remaining)
        stack = [root]
        component = set()
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            component.add(node)
            stack.extend(
                neighbor
                for neighbor in (*node.all_input_nodes, *node.users)
                if neighbor in selected
            )
        components.append([node for node in ordered_nodes if node in component])
    return components


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


def _graph_inputs(gm: GraphModule) -> tuple[Node, ...]:
    return tuple(node for node in gm.graph.nodes if node.op == "placeholder")


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
    baseline_inputs = _graph_inputs(baseline)
    candidate_inputs = _graph_inputs(candidate)
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


def extract_regions(
    gm: GraphModule,
    nodes: list[Node],
    prefix: str,
) -> dict[tuple[Any, ...], list[tuple[GraphModule, tuple[Node, ...]]]]:
    regions: dict[tuple[Any, ...], list[tuple[GraphModule, tuple[Node, ...]]]] = {}
    for index, component in enumerate(_connected_components(nodes)):
        region, inputs, outputs = fuse_as_graphmodule(
            gm,
            component,
            f"{prefix}_{index}",
            always_return_tuple=True,
        )
        signature = _region_inout_signature(inputs, outputs)
        if signature is None:
            missing = [
                node.name
                for node in (*inputs, *outputs)
                if _tensor_signature(node.meta.get("val")) is None
            ]
            raise TypeError(
                f"benchmark region {prefix}_{index} lacks tensor metadata for "
                f"{missing}"
            )
        regions.setdefault(signature, []).append((region, inputs))
    return regions


def infer_rewrite_regions(
    baseline: GraphModule,
    candidate: GraphModule,
) -> tuple[RewriteBenchmarkRegion, ...]:
    """Infer convex regions when a rewrite cannot declare them explicitly."""
    baseline_nodes, candidate_nodes = changed_nodes(baseline, candidate)
    baseline_regions = extract_regions(baseline, baseline_nodes, "BenchmarkBaseline")
    candidate_regions = extract_regions(
        candidate, candidate_nodes, "BenchmarkCandidate"
    )
    if baseline_regions.keys() != candidate_regions.keys() or any(
        len(regions) != len(candidate_regions[signature])
        for signature, regions in baseline_regions.items()
    ):
        raise RuntimeError("baseline and candidate benchmark-region signatures differ")

    result = []
    for signature, regions in baseline_regions.items():
        for (baseline_region, baseline_inputs), (
            candidate_region,
            candidate_inputs,
        ) in zip(regions, candidate_regions[signature], strict=True):
            result.append(
                RewriteBenchmarkRegion(
                    baseline_region,
                    baseline_inputs,
                    candidate_region,
                    candidate_inputs,
                    signature,
                )
            )
    return tuple(result)


def _resolve_attr(module: torch.nn.Module, target: str) -> Any:
    value: Any = module
    for component in target.split("."):
        value = getattr(value, component)
    return value


def _target_signature(gm: GraphModule, node: Node) -> Any:
    if node.op not in {"call_module", "get_attr"}:
        return repr(node.target)
    value = _resolve_attr(gm, str(node.target))
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
        atol: float = 0.15,
        rtol: float = 0.05,
        minimum_speedup: float = 1.01,
    ) -> None:
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
        process_baseline: BenchmarkGraphProcessorFn | None = None,
        process_candidate: BenchmarkGraphProcessorFn | None = None,
    ) -> CompileTimeBenchmarkResult:
        """Measure baseline and candidate in their configured runtime forms.

        Each optional processor converts its graph and realized inputs into the
        callable representation that will execute at runtime. Without a
        processor, that side executes as generated Python ``GraphModule`` code.
        """
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

    def benchmark_rewrite(
        self,
        baseline: GraphModule,
        candidate: GraphModule,
        *,
        cache_key: Hashable,
        benchmark_region: BenchmarkRegionFn | None = None,
        process_baseline: BenchmarkGraphProcessorFn | None = None,
        process_candidate: BenchmarkGraphProcessorFn | None = None,
    ) -> tuple[CompileTimeBenchmarkResult, ...]:
        return self.benchmark_regions(
            infer_rewrite_regions(baseline, candidate),
            cache_key=cache_key,
            benchmark_region=benchmark_region,
            process_baseline=process_baseline,
            process_candidate=process_candidate,
        )

    def benchmark_regions(
        self,
        regions: Iterable[RewriteBenchmarkRegion],
        *,
        cache_key: Hashable,
        benchmark_region: BenchmarkRegionFn | None = None,
        process_baseline: BenchmarkGraphProcessorFn | None = None,
        process_candidate: BenchmarkGraphProcessorFn | None = None,
    ) -> tuple[CompileTimeBenchmarkResult, ...]:
        """Benchmark regions and cache structurally equivalent measurements.

        ``cache_key`` is the caller-owned part of the assembled cache key. It
        must distinguish behavior not represented by the graphs, such as graph
        processors and their compiler settings. Equal keys allow measurements
        to be shared across patterns and occurrences.

        The remaining cache key describes the region's input/output tensor
        shapes, strides, dtypes, and devices; both graph structures; the
        PyTorch, Triton, CUDA, and GPU environment; and benchmark duration and
        tolerances. The cache is in-memory and lasts for the current process.
        """
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


def benchmark_region(
    baseline: GraphModule,
    baseline_input_nodes: tuple[Node, ...],
    candidate: GraphModule,
    candidate_input_nodes: tuple[Node, ...],
    *,
    process_baseline: BenchmarkGraphProcessorFn | None = None,
    process_candidate: BenchmarkGraphProcessorFn | None = None,
) -> CompileTimeBenchmarkResult:
    """Benchmark one explicit baseline/candidate region pair."""
    return _COMPILE_TIME_BENCHMARKER.benchmark_region(
        baseline,
        baseline_input_nodes,
        candidate,
        candidate_input_nodes,
        process_baseline=process_baseline,
        process_candidate=process_candidate,
    )


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
    apply_candidate: BenchmarkCandidateFn,
    cache_key: Hashable,
    strict: bool = False,
    benchmarker: CompileTimeBenchmarker | None = None,
    benchmark_region: BenchmarkRegionFn | None = None,
    process_baseline: BenchmarkGraphProcessorFn | None = None,
    process_candidate: BenchmarkGraphProcessorFn | None = None,
) -> GraphModule:
    """Independently retain candidates faster than their processed baseline.

    ``apply_candidate`` is first called with a region list and must describe
    one candidate without mutating ``gm``. If every region is faster, it is
    called again with ``None`` to apply that same selected candidate in place.
    """
    if not torch.cuda.is_available():
        logger.warning(
            f"{name} compile-time benchmark requires CUDA; "
            "keeping the original graph"
        )
        return gm

    if benchmarker is None:
        benchmarker = _COMPILE_TIME_BENCHMARKER
    applications: list[_BenchmarkApplication] = []
    rejected: set[str] = set()
    while True:
        selection = BenchmarkCandidateSelection(rejected)
        declared_regions: list[RewriteBenchmarkRegion] = []
        try:
            apply_candidate(gm, selection, declared_regions)
        except Exception as error:
            candidate_name = selection.selected
            if strict:
                raise RuntimeError(
                    f"{name} candidate {candidate_name or 'unknown'} rewrite failed"
                ) from error
            applications.append(
                _BenchmarkApplication(
                    candidate_name or "unknown",
                    "failed",
                    reason=f"rewrite failed with {type(error).__name__}: {error}",
                )
            )
            if candidate_name is None:
                break
            rejected.add(candidate_name)
            continue

        candidate_name = selection.selected
        if candidate_name is None:
            break

        if not declared_regions:
            applications.append(
                _BenchmarkApplication(
                    candidate_name,
                    "failed",
                    reason="rewrite declared no benchmark regions",
                )
            )
            rejected.add(candidate_name)
            continue

        results: tuple[CompileTimeBenchmarkResult, ...] = ()
        try:
            results = benchmarker.benchmark_regions(
                declared_regions,
                cache_key=cache_key,
                benchmark_region=benchmark_region,
                process_baseline=process_baseline,
                process_candidate=process_candidate,
            )
            use_candidate = bool(results) and all(
                benchmarker.accepts(result) for result in results
            )
        except Exception as error:
            if strict:
                raise RuntimeError(
                    f"{name} candidate {candidate_name} benchmark failed"
                ) from error
            applications.append(
                _BenchmarkApplication(
                    candidate_name,
                    "failed",
                    results,
                    f"benchmark failed with {type(error).__name__}: {error}",
                )
            )
            use_candidate = False
        else:
            applications.append(
                _BenchmarkApplication(
                    candidate_name,
                    "applied" if use_candidate else "rejected",
                    results,
                    None
                    if use_candidate
                    else "candidate was not faster for every changed region",
                )
            )

        if use_candidate:
            try:
                gm = apply_candidate(gm, selection, None)
            except Exception as error:
                raise RuntimeError(
                    f"{name} candidate {candidate_name} apply failed after "
                    "successful benchmarking"
                ) from error
        else:
            rejected.add(candidate_name)

    _log_benchmark_summary(
        name=name,
        applications=applications,
    )
    return gm
