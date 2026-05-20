# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused kernel pass: extract, replace, and optionally accelerate in one step.

Single-pass design:
  1. Discover fusible regions in the FX graph (fqn-based segmentation)
  2. Replace each region with a ``call_module`` wrapping the original subgraph
     as a nested ``GraphModule`` (zero-overhead eager fallback)
  3. Compute a stable hash for each region (ops + shapes)
  4. If ``{fused_kernel_dir}/{hash}/kernel.py`` exists, swap the module's
     forward to use the optimized kernel
  5. If not, write ``problem.py`` for offline kernel generation

The hash ensures that the same region always maps to the same directory,
regardless of extraction order or run-to-run variation.

Usage:
  # First run: extracts problems, trains with eager fallback
  ./run_train.sh --compile.fused_kernel_dir /tmp/kernels

  # Offline: generate kernels
  python -m torchtitan.experiments.graph_trainer.kernel_gen.generate

  # Next run: same command, auto-picks up kernels
  ./run_train.sh --compile.fused_kernel_dir /tmp/kernels
"""

from __future__ import annotations

import hashlib
import importlib.util
import operator
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Op classification
# ---------------------------------------------------------------------------

# str(node.target) produces e.g. "aten.mm.default" (no "torch.ops." prefix)
_UNFUSABLE_OPS = {
    "bucketing._pre_bucket_all_gather.default",
    "bucketing._pre_bucket_reduce_scatter.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_flash_attention_backward.default",
    "aten.embedding.default",
    "aten.embedding_dense_backward.default",
}

# Namespaces where ALL ops are unfusable
_UNFUSABLE_NAMESPACES = {"_c10d_functional", "ao", "bucketing"}

_ALREADY_OPTIMIZED_OPS = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
}

_METADATA_OPS = {
    "aten.reshape.default",
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.squeeze.dims",
    "aten.slice.Tensor",
    "aten.expand.default",
    "aten.permute.default",
    "aten.clone.default",
    "aten.view.dtype",
    "<built-in function getitem>",
}

_DTYPE_MAP = {
    torch.bfloat16: "torch.bfloat16",
    torch.float16: "torch.float16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.complex64: "torch.complex64",
    torch.complex128: "torch.complex128",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.bool: "torch.bool",
}

_RANDN_DTYPES = {
    torch.bfloat16, torch.float16, torch.float32, torch.float64,
    torch.complex64, torch.complex128,
}


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _get_node_fqn(node: torch.fx.Node) -> str:
    custom = node.meta.get("custom", {})
    return custom.get("module_fqn", "") if isinstance(custom, dict) else ""


def _normalize_fqn(fqn: str) -> str:
    return re.sub(r"layers\.\d+", "layers.*", fqn)


def _get_tensor_info(node: torch.fx.Node) -> tuple[tuple[int, ...], torch.dtype | None]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape), val.dtype
    if isinstance(val, (tuple, list)) and val and isinstance(val[0], torch.Tensor):
        return tuple(val[0].shape), val[0].dtype
    return (), None


def _is_unfusable(node: torch.fx.Node) -> bool:
    target_str = str(node.target)
    if target_str in _UNFUSABLE_OPS:
        return True
    # Block entire namespaces (collectives, offload, bucketing)
    ns = target_str.split(".")[0] if "." in target_str else ""
    return ns in _UNFUSABLE_NAMESPACES


def _iter_node_args(node: torch.fx.Node):
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            yield arg
        elif isinstance(arg, (list, tuple)):
            for a in arg:
                if isinstance(a, torch.fx.Node):
                    yield a
    for v in node.kwargs.values():
        if isinstance(v, torch.fx.Node):
            yield v


# ---------------------------------------------------------------------------
# Region: a fusible subgraph with inputs and outputs
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """A fusible subgraph discovered by a RegionExtractor."""

    nodes: list[torch.fx.Node]
    external_inputs: list[torch.fx.Node]
    output_nodes: list[torch.fx.Node]
    norm_fqn: str = ""

    @property
    def num_compute_ops(self) -> int:
        return sum(1 for n in self.nodes if str(n.target) not in _METADATA_OPS)

    @property
    def estimated_bytes(self) -> int:
        max_bytes = 0
        for n in self.nodes:
            shape, dtype = _get_tensor_info(n)
            if shape and dtype and hasattr(dtype, "itemsize"):
                size = 1
                for d in shape:
                    size *= d
                max_bytes = max(max_bytes, size * dtype.itemsize)
        return max_bytes


def _compute_region_hash(region: Region) -> str:
    """Stable hash from op targets + input/output shapes."""
    parts = []
    for n in region.nodes:
        parts.append(str(n.target))
        shape, dtype = _get_tensor_info(n)
        parts.append(f"{dtype}:{shape}")
    for inp in region.external_inputs:
        shape, dtype = _get_tensor_info(inp)
        parts.append(f"in:{dtype}:{shape}")
    for out in region.output_nodes:
        shape, dtype = _get_tensor_info(out)
        parts.append(f"out:{dtype}:{shape}")
    key = "|".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Region extractors (pluggable)
# ---------------------------------------------------------------------------


def _split_connected(nodes: list[torch.fx.Node]) -> list[list[torch.fx.Node]]:
    """Split into connected components via union-find."""
    if not nodes:
        return []
    name_to_idx = {n.name: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, n in enumerate(nodes):
        for arg in _iter_node_args(n):
            if arg.name in name_to_idx:
                union(i, name_to_idx[arg.name])

    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)

    return [[nodes[i] for i in sorted(idxs)] for idxs in groups.values()]


def _build_region(comp: list[torch.fx.Node], norm_fqn: str = "") -> Region:
    """Build a Region from a list of connected nodes."""
    node_names = {n.name for n in comp}

    external_inputs: list[torch.fx.Node] = []
    seen: set[str] = set()
    for n in comp:
        for arg in _iter_node_args(n):
            if arg.name not in node_names and arg.name not in seen:
                seen.add(arg.name)
                external_inputs.append(arg)

    output_nodes: list[torch.fx.Node] = []
    for n in comp:
        for user in n.users:
            if user.name not in node_names:
                if n not in output_nodes:
                    output_nodes.append(n)
                break
    if not output_nodes:
        output_nodes = [comp[-1]]

    return Region(
        nodes=comp, norm_fqn=norm_fqn,
        external_inputs=external_inputs,
        output_nodes=output_nodes,
    )


class RegionExtractor(ABC):
    """Base class for fusible region extractors."""

    @abstractmethod
    def extract(self, gm: torch.fx.GraphModule) -> list[Region]:
        """Discover all fusible regions in the graph."""
        ...


class FqnRegionExtractor(RegionExtractor):
    """Segments at module_fqn boundaries, splits connected components,
    filters by handcoded op lists."""

    def extract(self, gm: torch.fx.GraphModule) -> list[Region]:
        raw_regions: list[tuple[list[torch.fx.Node], str]] = []
        current: list[torch.fx.Node] = []
        current_fqn = ""

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            fqn = _normalize_fqn(_get_node_fqn(node))
            if fqn != current_fqn:
                if current:
                    raw_regions.append((current, current_fqn))
                current = []
                current_fqn = fqn
            current.append(node)
        if current:
            raw_regions.append((current, current_fqn))

        all_regions: list[Region] = []
        for chunk, norm_fqn in raw_regions:
            for comp in _split_connected(chunk):
                if any(_is_unfusable(n) for n in comp):
                    continue
                compute = {str(n.target) for n in comp if str(n.target) not in _METADATA_OPS}
                if compute and compute <= _ALREADY_OPTIMIZED_OPS:
                    continue
                all_regions.append(_build_region(comp, norm_fqn))

        return all_regions


class InductorRegionExtractor(RegionExtractor):
    """Uses inductor's ``is_fusible_node`` and ``CapabilityBasedPartitioner``
    to discover fusible regions — matches inductor's actual fusion decisions."""

    def extract(self, gm: torch.fx.GraphModule) -> list[Region]:
        try:
            from torch._inductor.fx_passes.fusion_regions import (
                is_fusible_node,
                is_view_node,
            )
            from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
            from torch.fx.passes.operator_support import create_op_support
        except ImportError:
            logger.warning(
                "InductorRegionExtractor: inductor partitioner not available, "
                "falling back to FqnRegionExtractor"
            )
            return FqnRegionExtractor().extract(gm)

        def _is_supported(_submodules, node):
            return is_fusible_node(node)

        # CapabilityBasedPartitioner.propose_partitions() is O(n²) due to
        # DFS cycle checks on every merge attempt. On 10K+ node graphs it
        # hangs. Workaround: segment by fqn first (fast, O(n)), then run
        # the partitioner on each segment separately.
        import time as _time
        _t0 = _time.time()

        # CapabilityBasedPartitioner.propose_partitions() is O(n²) due to
        # DFS cycle checks on every merge. Instead, group consecutive
        # fusible nodes (split at non-fusible boundaries) — this matches
        # inductor's fusion behavior without the expensive merge logic.
        n_nodes = len(list(gm.graph.nodes))

        # Classify each node as fusible using inductor's heuristic
        fusible_nodes: set[str] = set()
        for node in gm.graph.nodes:
            if node.op == "call_function":
                try:
                    if is_fusible_node(node):
                        fusible_nodes.add(node.name)
                except Exception:
                    pass

        # Group consecutive fusible nodes, split at non-fusible boundaries
        partitions: list[tuple[list[torch.fx.Node], str]] = []
        current_partition: list[torch.fx.Node] = []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.name in fusible_nodes:
                current_partition.append(node)
            else:
                if current_partition:
                    norm_fqn = _normalize_fqn(_get_node_fqn(current_partition[0]))
                    partitions.append((current_partition, norm_fqn))
                    current_partition = []
        if current_partition:
            norm_fqn = _normalize_fqn(_get_node_fqn(current_partition[0]))
            partitions.append((current_partition, norm_fqn))

        logger.info(
            f"InductorRegionExtractor: {n_nodes} nodes, "
            f"{len(fusible_nodes)} fusible, "
            f"{len(partitions)} partitions in {_time.time()-_t0:.1f}s"
        )

        all_regions: list[Region] = []
        for comp, norm_fqn in partitions:
            # Skip view-only partitions
            if all(is_view_node(n) or n.op != "call_function" for n in comp):
                continue
            # Filter unfusable ops that slipped through
            if any(_is_unfusable(n) for n in comp):
                continue
            # Split disconnected components
            for connected in _split_connected(comp):
                all_regions.append(_build_region(connected, norm_fqn))

        return all_regions


_EXTRACTORS: dict[str, type[RegionExtractor]] = {
    "fqn": FqnRegionExtractor,
    "inductor": InductorRegionExtractor,
}


def _filter_regions(
    regions: list[Region],
    *,
    min_ops: int = 2,
    min_compute_ops: int = 1,
    min_count: int = 2,
) -> tuple[list[Region], Counter[str]]:
    """Filter and deduplicate regions. Returns (unique regions, hash counts)."""
    hash_counts: Counter[str] = Counter()
    hash_to_region: dict[str, Region] = {}
    for r in regions:
        h = _compute_region_hash(r)
        hash_counts[h] += 1
        if h not in hash_to_region:
            hash_to_region[h] = r

    result: list[Region] = []
    for h, count in hash_counts.items():
        r = hash_to_region[h]
        if len(r.nodes) < min_ops:
            continue
        if r.num_compute_ops < min_compute_ops:
            continue
        if count < min_count:
            continue
        result.append(r)

    return result, hash_counts


# ---------------------------------------------------------------------------
# Subgraph extraction → GraphModule
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Kernel loading + backend selection
# ---------------------------------------------------------------------------

def _load_kernel_fn(kernel_dir: Path) -> Callable | None:
    """Load kernel_function from kernel.py if it exists."""
    kernel_path = kernel_dir / "kernel.py"
    if not kernel_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"_kernel_{kernel_dir.name}", str(kernel_path)
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "kernel_function", None)
    except Exception as e:
        logger.debug(f"Failed to load kernel from {kernel_path}: {e}")
        return None


def _try_compile(fn: Callable) -> Callable | None:
    """Try to torch.compile a function. Returns None on failure."""
    try:
        return torch.compile(fn, fullgraph=True)
    except Exception:
        return None


def _select_best_backend(
    region_dir: Path,
    eager_fn: Callable | None = None,
) -> tuple[Callable | None, str]:
    """Select the fastest backend from offline benchmark results.

    Reads benchmark.json (produced by benchmark.py) and picks the
    winner. Only returns a non-eager backend if it's proven faster
    by the offline benchmark. Returns (None, "eager") when eager wins
    or no benchmark data exists.

    Returns (best_fn_or_None, backend_name).
    """
    import json

    triton_fn = _load_kernel_fn(region_dir)

    bench_path = region_dir / "benchmark.json"
    if not bench_path.exists():
        return None, "eager"

    try:
        bench_data = json.loads(bench_path.read_text())
    except Exception:
        return None, "eager"

    eager_ms = bench_data.get("eager_ms", float("inf"))

    # Check triton
    triton_ms = bench_data.get("triton_ms", float("inf"))
    if triton_fn is not None and triton_ms < eager_ms:
        return triton_fn, "triton"

    # Check compile (only if it beats eager AND triton)
    compile_ms = bench_data.get("compile_ms", float("inf"))
    if compile_ms < eager_ms and compile_ms < triton_ms:
        if eager_fn is not None:
            compiled_fn = _try_compile(eager_fn)
            if compiled_fn is not None:
                return compiled_fn, "compile"

    return None, "eager"


# ---------------------------------------------------------------------------
# Problem file generation
# ---------------------------------------------------------------------------

def _format_arg_for_problem(
    arg: object,
    input_map: dict[str, str],
    var_map: dict[str, str],
) -> str:
    if isinstance(arg, torch.fx.Node):
        return input_map.get(arg.name, var_map.get(arg.name, arg.name))
    if isinstance(arg, torch.dtype):
        return str(arg)
    if isinstance(arg, torch.device):
        # Normalize to generic 'cuda' — problem.py runs on any GPU
        if arg.type == "cuda":
            return "torch.device('cuda')"
        return f"torch.device('{arg}')"
    if isinstance(arg, torch.memory_format):
        return str(arg)
    if isinstance(arg, torch.layout):
        return str(arg)
    if isinstance(arg, (int, float, bool)):
        return repr(arg)
    if arg is None:
        return "None"
    if isinstance(arg, (list, tuple)):
        elems = [_format_arg_for_problem(a, input_map, var_map) for a in arg]
        if isinstance(arg, tuple):
            return f"({', '.join(elems)}{',' if len(elems) == 1 else ''})"
        return f"[{', '.join(elems)}]"
    return repr(arg)


def _write_problem(
    region: Region,
    output_dir: Path,
    count: int,
) -> None:
    """Write a KernelAgent-compatible problem.py."""
    nodes = region.nodes
    external_inputs = region.external_inputs
    output_nodes = region.output_nodes

    input_map = {inp.name: f"input_{i}" for i, inp in enumerate(external_inputs)}
    var_map = {n.name: f"v_{i}" for i, n in enumerate(nodes)}

    body_lines = []
    for n in nodes:
        var = var_map[n.name]
        if n.target is operator.getitem:
            src = _format_arg_for_problem(n.args[0], input_map, var_map)
            idx = n.args[1]
            shape, dtype = _get_tensor_info(n)
            comment = f"  # {dtype} {list(shape)}" if shape else ""
            body_lines.append(f"        {var} = {src}[{idx}]{comment}")
        else:
            fargs = [_format_arg_for_problem(a, input_map, var_map) for a in n.args]
            fkwargs = {k: _format_arg_for_problem(v, input_map, var_map) for k, v in n.kwargs.items()}
            all_args = list(fargs) + [f"{k}={v}" for k, v in fkwargs.items()]
            shape, dtype = _get_tensor_info(n)
            comment = f"  # {dtype} {list(shape)}" if shape else ""
            body_lines.append(
                f"        {var} = torch.ops.{n.target}({', '.join(all_args)}){comment}"
            )

    # Return
    num_outputs = len(output_nodes)
    if num_outputs == 1:
        body_lines.append(f"        return {var_map[output_nodes[0].name]}")
        ret_type = "torch.Tensor"
    else:
        out_vars = ", ".join(var_map[n.name] for n in output_nodes)
        body_lines.append(f"        return ({out_vars})")
        ret_type = f"tuple[{', '.join(['torch.Tensor'] * num_outputs)}]"

    # Params and inputs
    param_parts = []
    input_lines = []
    return_parts = []
    for inp in external_inputs:
        pname = input_map[inp.name]
        param_parts.append(f"{pname}: torch.Tensor")
        return_parts.append(pname)
        shape, dtype = _get_tensor_info(inp)
        if shape and dtype:
            torch_dtype = _DTYPE_MAP.get(dtype, "torch.float32")
            if dtype in _RANDN_DTYPES:
                input_lines.append(f"    {pname} = torch.randn({shape!r}, dtype={torch_dtype}, device='cuda')")
            elif dtype == torch.bool:
                input_lines.append(f"    {pname} = torch.randint(0, 2, {shape!r}, dtype={torch_dtype}, device='cuda')")
            else:
                input_lines.append(f"    {pname} = torch.randint(0, 10, {shape!r}, dtype={torch_dtype}, device='cuda')")
        else:
            input_lines.append(f"    {pname} = torch.randn((1,), dtype=torch.float32, device='cuda')")

    compute_ops = [
        str(n.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")
        for n in nodes if str(n.target) not in _METADATA_OPS
    ]
    desc = f"Fused region ({region.norm_fqn}): {' -> '.join(compute_ops) if compute_ops else 'reshape chain'}\n"
    desc += f"Instances: {count}. Ops: {len(nodes)}, compute: {len(compute_ops)}, outputs: {num_outputs}.\n"

    problem = desc + f"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, {', '.join(param_parts)}) -> {ret_type}:
{chr(10).join(body_lines)}

def get_inputs():
{chr(10).join(input_lines) if input_lines else '    pass'}
    return [{', '.join(return_parts)}]

def get_init_inputs():
    return []
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "problem.py").write_text(problem)


# ---------------------------------------------------------------------------
# Graph replacement
# ---------------------------------------------------------------------------

def _replace_region_with_kernel(
    gm: torch.fx.GraphModule,
    region: Region,
    kernel_fn: Callable,
) -> None:
    """Replace a region's nodes with a direct call_function to kernel_fn.

    No call_module, no subgraph module — just inserts
    ``kernel_fn(input_0, input_1, ...)`` directly into the parent graph.
    """
    graph = gm.graph
    nodes = region.nodes
    external_inputs = region.external_inputs
    output_nodes = region.output_nodes

    first_node = nodes[0]
    with graph.inserting_before(first_node):
        call_node = graph.call_function(
            kernel_fn, args=tuple(external_inputs)
        )

        if len(output_nodes) == 1:
            call_node.meta = output_nodes[0].meta.copy()
        else:
            call_node.meta = {}

    if len(output_nodes) == 1:
        output_nodes[0].replace_all_uses_with(call_node)
    else:
        with graph.inserting_after(call_node):
            for out_idx, out_node in enumerate(output_nodes):
                gi = graph.call_function(
                    operator.getitem, args=(call_node, out_idx)
                )
                gi.meta = out_node.meta.copy()
                out_node.replace_all_uses_with(gi)

    for old in reversed(nodes):
        if not old.users:
            graph.erase_node(old)


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def fused_kernel_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    kernel_dir: str = "",
    extractor: str = "fqn",
    min_ops: int = 2,
    min_compute_ops: int = 1,
    min_count: int = 2,
) -> torch.fx.GraphModule:
    """Single-pass extract + replace + accelerate.

    Discovers fusible regions using the selected extractor, replaces each
    with a ``call_module`` wrapping the original subgraph (zero-overhead
    eager fallback). If a kernel exists in ``{kernel_dir}/{hash}/kernel.py``,
    swaps the module's forward. If not, writes ``problem.py`` for offline
    kernel generation.

    Args:
        extractor: Region extraction strategy. Options:
            ``"fqn"``: segments at module_fqn boundaries (default)
            ``"inductor"``: uses inductor's ``is_fusible_node`` partitioner

    No-op when ``kernel_dir`` is empty.
    """
    if not kernel_dir:
        return gm

    kdir = Path(kernel_dir)
    kdir.mkdir(parents=True, exist_ok=True)

    # Discover all regions using the selected extractor
    extractor_cls = _EXTRACTORS.get(extractor)
    if extractor_cls is None:
        logger.warning(
            f"Unknown extractor '{extractor}', available: {list(_EXTRACTORS.keys())}. "
            "Falling back to 'fqn'."
        )
        extractor_cls = FqnRegionExtractor
    all_regions = extractor_cls().extract(gm)
    unique_regions, hash_counts = _filter_regions(
        all_regions, min_ops=min_ops, min_compute_ops=min_compute_ops,
        min_count=min_count,
    )

    if not unique_regions:
        logger.info("Fused kernel pass: no fusible regions found")
        return gm

    valid_hashes = {_compute_region_hash(r) for r in unique_regions}

    # Collect ALL instances (not just unique) for replacement
    node_order = {n: i for i, n in enumerate(gm.graph.nodes)}
    instances: list[tuple[int, str, Region]] = []
    for r in all_regions:
        h = _compute_region_hash(r)
        if h in valid_hashes:
            pos = node_order.get(r.nodes[0], 0)
            instances.append((pos, h, r))

    # Sort by position DESCENDING — replace from end of graph to start
    # so earlier node references remain valid
    instances.sort(key=lambda x: x[0], reverse=True)

    problems_written = 0

    # Phase 1: Write problem.py for every unique region (no graph modification)
    seen_hashes: set[str] = set()
    for _, h, r in instances:
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        region_dir = kdir / h
        if not (region_dir / "problem.py").exists():
            count = hash_counts.get(h, 1)
            _write_problem(r, region_dir, count)
            problems_written += 1

    # Phase 2: Only replace regions where a non-eager backend wins.
    # Regions without kernels or where eager is fastest stay untouched.
    hash_best: dict[str, tuple[Callable, str]] = {}
    backend_counts: Counter[str] = Counter()

    for h in seen_hashes:
        region_dir = kdir / h
        eager_fn = None  # placeholder, not needed if we skip eager
        best_fn, backend_name = _select_best_backend(region_dir, eager_fn)
        if backend_name != "eager" and best_fn is not None:
            hash_best[h] = (best_fn, backend_name)
            backend_counts[backend_name] += 1
        else:
            backend_counts["eager"] += 1

    # Only replace instances whose hash has a non-eager winner.
    # Inserts call_function(kernel_fn) directly — no call_module wrapper.
    replaced = 0
    for pos, h, r in instances:
        if h not in hash_best:
            continue
        best_fn, _ = hash_best[h]
        _replace_region_with_kernel(gm, r, best_fn)
        replaced += 1

    if replaced > 0:
        gm.graph.lint()
        gm.recompile()

    backend_summary = ", ".join(f"{k}={v}" for k, v in sorted(backend_counts.items()))
    logger.info(
        f"Fused kernel pass: {replaced} replacements "
        f"({len(hash_best)} patterns with kernels, "
        f"{len(seen_hashes) - len(hash_best)} eager-only skipped), "
        f"backends: [{backend_summary}], "
        f"{problems_written} new problems written to {kdir}"
    )

    return gm
