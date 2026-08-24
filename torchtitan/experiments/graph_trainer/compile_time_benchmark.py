# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-time benchmarking for transactional FX graph rewrites."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import triton
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.node import _get_qualified_name
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
    """Selection state shared with a transactional rewrite callback."""

    rejected: set[str]
    selected: str | None = None
    defer_finalize: bool = False
    collect_all: bool = False
    accepted: set[str] | None = None
    candidates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _BenchmarkApplication:
    name: str
    status: str
    regions: tuple[CompileTimeBenchmarkResult, ...] = ()
    reason: str | None = None


class _BenchmarkRegionCollector(list[RewriteBenchmarkRegion]):
    def __init__(self, selection: BenchmarkCandidateSelection) -> None:
        super().__init__()
        self.selection = selection
        self.by_candidate: defaultdict[str, list[RewriteBenchmarkRegion]] = (
            defaultdict(list)
        )

    def append(self, region: RewriteBenchmarkRegion) -> None:
        candidate = self.selection.selected
        if candidate is None:
            raise AssertionError("benchmark region was declared without a candidate")
        super().append(region)
        self.by_candidate[candidate].append(region)

    def extend(self, regions: Iterable[RewriteBenchmarkRegion]) -> None:
        for region in regions:
            self.append(region)


BenchmarkRegionFn = Callable[
    [GraphModule, tuple[Node, ...], GraphModule, tuple[Node, ...]],
    CompileTimeBenchmarkResult,
]
BenchmarkCandidateFn = Callable[
    [GraphModule, BenchmarkCandidateSelection, list[RewriteBenchmarkRegion]],
    GraphModule,
]
RewriteFinalizerFn = Callable[[GraphModule], GraphModule]


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
    if isinstance(value, torch.SymInt):
        return ("symint", _hint_int(value))
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


def _convex_closure(gm: GraphModule, nodes: Iterable[Node]) -> list[Node]:
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
        _convex_closure(baseline, baseline_nodes),
        _convex_closure(candidate, candidate_nodes),
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


def _region_signature(
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
    baseline_signature = _region_signature(
        baseline_inputs,
        _graph_outputs(baseline),
    )
    candidate_signature = _region_signature(
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
        signature = _region_signature(inputs, outputs)
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


def _preserve_get_attr_identity(source: GraphModule, target: GraphModule) -> None:
    """Restore shared attributes copied by transactional graph rewrites."""
    source_targets = {
        str(node.target) for node in source.graph.nodes if node.op == "get_attr"
    }
    for node in target.graph.nodes:
        if node.op != "get_attr" or str(node.target) not in source_targets:
            continue
        path, _, name = str(node.target).rpartition(".")
        parent = _resolve_attr(target, path) if path else target
        setattr(parent, name, _resolve_attr(source, str(node.target)))


def _target_signature(gm: GraphModule, node: Node) -> Any:
    if node.op == "call_function":
        try:
            return _get_qualified_name(node.target)
        except Exception:
            return repr(node.target)
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
        minimum_speedup: float = 1.02,
        cache_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.duration_ms = duration_ms
        self.atol = atol
        self.rtol = rtol
        self.minimum_speedup = minimum_speedup
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[tuple[Any, ...], CompileTimeBenchmarkResult] = {}

    def clear(self) -> None:
        self._cache.clear()

    def accepts(self, result: CompileTimeBenchmarkResult) -> bool:
        return result.speedup >= self.minimum_speedup

    def _persistent_cache_path(self) -> Path | None:
        if self.cache_path is not None:
            return self.cache_path
        path = os.environ.get("TORCHTITAN_COMPILE_TIME_BENCHMARK_CACHE")
        return Path(path) if path else None

    @staticmethod
    def _persistent_key(cache_key: tuple[Any, ...]) -> str:
        def normalize(value: Any) -> Any:
            if (
                isinstance(value, tuple)
                and len(value) == 5
                and isinstance(value[0], tuple)
                and isinstance(value[1], tuple)
                and isinstance(value[2], torch.dtype)
                and isinstance(value[3], str)
                and (value[4] is None or isinstance(value[4], int))
            ):
                return (*value[:-1], None)
            if isinstance(value, tuple):
                return tuple(normalize(item) for item in value)
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value

        return hashlib.sha256(repr(normalize(cache_key)).encode()).hexdigest()

    @staticmethod
    def _legacy_persistent_key(cache_key: tuple[Any, ...]) -> str:
        return hashlib.sha256(repr(cache_key).encode()).hexdigest()

    @staticmethod
    def _read_persistent_cache(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"version": 1, "results": {}}
        try:
            cache = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Ignoring invalid compile-time benchmark cache: {error}")
            return {"version": 1, "results": {}}
        if cache.get("version") != 1 or not isinstance(cache.get("results"), dict):
            logger.warning(f"Ignoring unsupported compile-time benchmark cache {path}")
            return {"version": 1, "results": {}}
        return cache

    @staticmethod
    def _write_persistent_cache(path: Path, cache: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def _persistent_lookup_or_benchmark(
        self,
        cache_key: tuple[Any, ...],
        benchmark: Callable[[], CompileTimeBenchmarkResult],
    ) -> tuple[CompileTimeBenchmarkResult, bool]:
        path = self._persistent_cache_path()
        if path is None:
            return benchmark(), False

        path.parent.mkdir(parents=True, exist_ok=True)
        key = self._persistent_key(cache_key)
        legacy_key = self._legacy_persistent_key(cache_key)
        lock_path = path.with_name(f"{path.name}.lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cache = self._read_persistent_cache(path)
            cached = cache["results"].get(key)
            if cached is None:
                cached = cache["results"].get(legacy_key)
            if cached is not None:
                if key not in cache["results"]:
                    cache["results"][key] = {**cached, "key": repr(cache_key)}
                    self._write_persistent_cache(path, cache)
                return (
                    CompileTimeBenchmarkResult(
                        float(cached["baseline_ms"]),
                        float(cached["candidate_ms"]),
                    ),
                    True,
                )

            result = benchmark()
            cache["results"][key] = {
                "baseline_ms": result.baseline_ms,
                "candidate_ms": result.candidate_ms,
                "key": repr(cache_key),
            }
            self._write_persistent_cache(path, cache)
            return result, False

    def benchmark_region(
        self,
        baseline: GraphModule,
        baseline_input_nodes: tuple[Node, ...],
        candidate: GraphModule,
        candidate_input_nodes: tuple[Node, ...],
    ) -> CompileTimeBenchmarkResult:
        baseline_inputs, candidate_inputs = _realize_paired_inputs(
            baseline_input_nodes, candidate_input_nodes
        )
        compiled_candidate = torch.compile(
            candidate,
            backend="inductor",
            fullgraph=True,
            mode="max-autotune-no-cudagraphs",
        )
        try:
            expected = baseline(*baseline_inputs)
            actual = compiled_candidate(*candidate_inputs)
            torch.testing.assert_close(
                actual,
                expected,
                atol=self.atol,
                rtol=self.rtol,
            )
            baseline_ms = do_bench(
                lambda: baseline(*baseline_inputs),
                rep=self.duration_ms,
                return_mode="median",
            )
            candidate_ms = do_bench(
                lambda: compiled_candidate(*candidate_inputs),
                rep=self.duration_ms,
                return_mode="median",
            )
            return CompileTimeBenchmarkResult(baseline_ms, candidate_ms)
        finally:
            torch.cuda.empty_cache()

    def benchmark_rewrite(
        self,
        baseline: GraphModule,
        candidate: GraphModule,
        *,
        namespace: Hashable,
        benchmark_region: BenchmarkRegionFn | None = None,
    ) -> tuple[CompileTimeBenchmarkResult, ...]:
        return self.benchmark_regions(
            infer_rewrite_regions(baseline, candidate),
            namespace=namespace,
            benchmark_region=benchmark_region,
        )

    def benchmark_regions(
        self,
        regions: Iterable[RewriteBenchmarkRegion],
        *,
        namespace: Hashable,
        benchmark_region: BenchmarkRegionFn | None = None,
    ) -> tuple[CompileTimeBenchmarkResult, ...]:
        """Benchmark explicitly declared baseline/candidate region pairs."""
        benchmark = benchmark_region or self.benchmark_region
        results = []
        for region in regions:
            cache_key = (
                namespace,
                region.signature,
                _graph_fingerprint(region.baseline),
                _graph_fingerprint(region.candidate),
                _runtime_fingerprint(),
                self.duration_ms,
                self.atol,
                self.rtol,
            )
            result = self._cache.get(cache_key)
            if result is None:
                result, cache_hit = self._persistent_lookup_or_benchmark(
                    cache_key,
                    lambda: benchmark(
                        region.baseline,
                        region.baseline_inputs,
                        region.candidate,
                        region.candidate_inputs,
                    ),
                )
                self._cache[cache_key] = result
                if cache_hit:
                    result = CompileTimeBenchmarkResult(
                        result.baseline_ms,
                        result.candidate_ms,
                        cache_hit=True,
                    )
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
) -> CompileTimeBenchmarkResult:
    """Benchmark one explicit eager/candidate region pair."""
    return _COMPILE_TIME_BENCHMARKER.benchmark_region(
        baseline,
        baseline_input_nodes,
        candidate,
        candidate_input_nodes,
    )


def _log_benchmark_summary(
    *,
    report_title: str,
    artifact_name: str,
    candidate_prefix: str,
    candidate_label: str,
    applications: list[_BenchmarkApplication],
) -> None:
    counts = Counter(application.status for application in applications)
    rejected = counts["rejected"] + counts["failed"]
    header = (
        f"{report_title}: candidates={len(applications)}, "
        f"applied={counts['applied']}, rejected={rejected} "
        f"(slower={counts['rejected']}, failed={counts['failed']})"
    )

    def details(application: _BenchmarkApplication, *, cache: str | None = None):
        result_details = [
            f"region {index}: eager={result.baseline_ms * 1000:.1f} us, "
            f"{candidate_label}={result.candidate_ms * 1000:.1f} us, "
            f"speedup={result.speedup:.3f}x, "
            f"cache={cache or ('hit' if result.cache_hit else 'miss')}"
            for index, result in enumerate(application.regions)
        ]
        if application.reason is not None:
            result_details.append(application.reason)
        return "; ".join(result_details)

    lines = [header]
    for status in ("applied", "rejected", "failed"):
        selected = [item for item in applications if item.status == status]
        if not selected:
            continue
        lines.append(f"  {status.upper()} ({len(selected)}):")
        for application in selected:
            candidate = application.name.removeprefix(candidate_prefix)
            lines.append(f"    candidate {candidate}: {details(application)}")
    report = "\n".join(lines)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: report,
        expect_trace_id=False,
    )
    log = logger.warning if counts["failed"] else logger.info
    if len(applications) <= 20:
        log(report)
        return

    compact_lines = [header]
    for status in ("applied", "rejected", "failed"):
        selected = [item for item in applications if item.status == status]
        if not selected:
            continue
        compact_lines.append(f"  {status.upper()} ({len(selected)}):")
        groups: defaultdict[tuple[Any, ...], list[_BenchmarkApplication]] = defaultdict(
            list
        )
        for application in selected:
            key = (
                tuple(
                    (result.baseline_ms, result.candidate_ms)
                    for result in application.regions
                ),
                application.reason,
            )
            groups[key].append(application)
        for equivalent in groups.values():
            application = equivalent[0]
            candidate = application.name.removeprefix(candidate_prefix)
            suffix = (
                ""
                if len(equivalent) == 1
                else f" (+{len(equivalent) - 1} equivalent)"
            )
            total_results = sum(len(item.regions) for item in equivalent)
            cache_hits = sum(
                result.cache_hit for item in equivalent for result in item.regions
            )
            cache = f"{cache_hits}/{total_results} hits" if total_results else None
            compact_lines.append(
                f"    candidate {candidate}{suffix}: {details(application, cache=cache)}"
            )
    log("\n".join(compact_lines))


def apply_benchmarked_rewrites(
    gm: GraphModule,
    *,
    rewrite_name: str,
    apply_candidate: BenchmarkCandidateFn,
    namespace: Hashable,
    strict: bool = False,
    benchmarker: CompileTimeBenchmarker | None = None,
    benchmark_region: BenchmarkRegionFn | None = None,
    report_title: str | None = None,
    artifact_name: str | None = None,
    candidate_prefix: str = "",
    candidate_label: str = "candidate",
    finalize: RewriteFinalizerFn | None = None,
    batch_candidates: bool = False,
) -> GraphModule:
    """Independently retain each rewrite candidate that is faster than eager."""
    if not torch.cuda.is_available():
        logger.warning(
            f"{rewrite_name} compile-time benchmark requires CUDA; "
            "keeping the original graph"
        )
        return gm

    if benchmarker is None:
        benchmarker = _COMPILE_TIME_BENCHMARKER

    if batch_candidates:
        selection = BenchmarkCandidateSelection(
            set(),
            defer_finalize=finalize is not None,
            collect_all=True,
        )
        declared_regions = _BenchmarkRegionCollector(selection)
        try:
            candidate = apply_candidate(gm, selection, declared_regions)
            _preserve_get_attr_identity(gm, candidate)
        except Exception as error:
            if strict:
                raise RuntimeError(
                    f"{rewrite_name} candidate {selection.selected or 'unknown'} "
                    "rewrite failed"
                ) from error
        else:
            candidate_names = selection.candidates
            if candidate_names:
                applications = []
                accepted = set()
                for candidate_name in candidate_names:
                    candidate_regions = declared_regions.by_candidate[candidate_name]
                    if not candidate_regions:
                        applications.append(
                            _BenchmarkApplication(
                                candidate_name,
                                "rejected",
                                reason="rewrite produced no benchmark region",
                            )
                        )
                        continue
                    results: tuple[CompileTimeBenchmarkResult, ...] = ()
                    try:
                        results = benchmarker.benchmark_regions(
                            candidate_regions,
                            namespace=namespace,
                            benchmark_region=benchmark_region,
                        )
                        use_candidate = all(
                            benchmarker.accepts(result) for result in results
                        )
                    except Exception as error:
                        if strict:
                            raise RuntimeError(
                                f"{rewrite_name} candidate {candidate_name} "
                                "benchmark failed"
                            ) from error
                        applications.append(
                            _BenchmarkApplication(
                                candidate_name,
                                "failed",
                                results,
                                "benchmark failed with "
                                f"{type(error).__name__}: {error}",
                            )
                        )
                        continue

                    applications.append(
                        _BenchmarkApplication(
                            candidate_name,
                            "applied" if use_candidate else "rejected",
                            results,
                            None
                            if use_candidate
                            else f"{candidate_label} was not faster for every "
                            "changed region",
                        )
                    )
                    if use_candidate:
                        accepted.add(candidate_name)

                if not accepted:
                    result = gm
                elif len(accepted) == len(candidate_names):
                    result = candidate
                else:
                    accepted_selection = BenchmarkCandidateSelection(
                        set(),
                        defer_finalize=finalize is not None,
                        accepted=accepted,
                    )
                    result = apply_candidate(gm, accepted_selection, [])
                    _preserve_get_attr_identity(gm, result)
                if accepted and finalize is not None:
                    result = finalize(result)
                _log_benchmark_summary(
                    report_title=report_title
                    or f"Compile-time benchmark results for {rewrite_name}",
                    artifact_name=artifact_name
                    or f"compile_time_benchmark_{rewrite_name}",
                    candidate_prefix=candidate_prefix,
                    candidate_label=candidate_label,
                    applications=applications,
                )
                return result

    current = gm
    accepted_any = False
    applications: list[_BenchmarkApplication] = []
    rejected: set[str] = set()
    while True:
        selection = BenchmarkCandidateSelection(
            rejected,
            defer_finalize=finalize is not None,
        )
        declared_regions: list[RewriteBenchmarkRegion] = []
        try:
            candidate = apply_candidate(current, selection, declared_regions)
            _preserve_get_attr_identity(current, candidate)
        except Exception as error:
            candidate_name = selection.selected
            if strict:
                raise RuntimeError(
                    f"{rewrite_name} candidate {candidate_name or 'unknown'} "
                    "rewrite failed"
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
            if finalize is not None:
                candidate = finalize(candidate)
            baseline_nodes, candidate_nodes = changed_nodes(current, candidate)
            if not baseline_nodes and not candidate_nodes:
                applications.append(
                    _BenchmarkApplication(
                        candidate_name,
                        "rejected",
                        reason="rewrite produced no changed nodes",
                    )
                )
                rejected.add(candidate_name)
                continue

        results: tuple[CompileTimeBenchmarkResult, ...] = ()
        try:
            if declared_regions:
                results = benchmarker.benchmark_regions(
                    declared_regions,
                    namespace=namespace,
                    benchmark_region=benchmark_region,
                )
            else:
                results = benchmarker.benchmark_rewrite(
                    current,
                    candidate,
                    namespace=namespace,
                    benchmark_region=benchmark_region,
                )
            use_candidate = bool(results) and all(
                benchmarker.accepts(result) for result in results
            )
        except Exception as error:
            if strict:
                raise RuntimeError(
                    f"{rewrite_name} candidate {candidate_name} benchmark failed"
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
                    else f"{candidate_label} was not faster for every changed region",
                )
            )

        if use_candidate:
            if declared_regions and finalize is not None:
                candidate = finalize(candidate)
            current = candidate
            accepted_any = True
        else:
            rejected.add(candidate_name)

    _log_benchmark_summary(
        report_title=report_title
        or f"Compile-time benchmark results for {rewrite_name}",
        artifact_name=artifact_name or f"compile_time_benchmark_{rewrite_name}",
        candidate_prefix=candidate_prefix,
        candidate_label=candidate_label,
        applications=applications,
    )
    return current if accepted_any else gm
