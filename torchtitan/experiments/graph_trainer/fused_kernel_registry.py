# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused kernel pass: extract fusible regions and replace with optimized kernels.

Two-phase design with the same config:

  Step 1 (no kernels): discovers fusible regions via a pluggable extractor,
  writes ``problem.py`` for each unique region (keyed by stable hash).
  No graph modification — zero overhead.

  Step 3 (kernels exist): only replaces regions where ``benchmark.json``
  proves a non-eager backend wins. Inserts ``call_function(kernel_fn)``
  directly into the FX graph — no ``call_module`` wrapper.

Usage:
  # Step 1: extract problems, train unchanged
  ./run_train.sh --compile.fused_kernel_dir /tmp/kernels

  # Step 2: generate + benchmark (offline)
  python -m torchtitan.experiments.graph_trainer.kernel_gen.generate --dir /tmp/kernels

  # Step 3: same command, auto-picks up proven kernels
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
    """Applies inductor decompositions, then classifies nodes with
    ``is_fusible_node`` to discover regions that match inductor's
    actual fusion decisions.

    Steps:
      1. Decompose the graph (aten.t → aten.permute, etc.) using
         inductor's ``select_decomp_table``
      2. Classify each node as fusible/non-fusible via ``is_fusible_node``
      3. Group consecutive fusible nodes (split at non-fusible boundaries)
      4. Split disconnected components via union-find
    """

    def __init__(self, example_inputs: tuple | None = None):
        self.example_inputs = example_inputs

    def extract(self, gm: torch.fx.GraphModule) -> list[Region]:
        import time as _time

        try:
            from torch._inductor.fx_passes.fusion_regions import (
                is_fusible_node,
                is_view_node,
            )
        except ImportError:
            logger.warning(
                "InductorRegionExtractor: inductor not available, "
                "falling back to FqnRegionExtractor"
            )
            return FqnRegionExtractor().extract(gm)

        _t0 = _time.time()

        # Step 1: Apply inductor decompositions
        work_gm = gm
        if self.example_inputs is not None:
            try:
                from torch._inductor.decomposition import select_decomp_table
                from torch._subclasses.fake_tensor import FakeTensor
                from torch.fx.experimental.proxy_tensor import make_fx

                decomp_table = select_decomp_table()
                fake_mode = None
                for inp in self.example_inputs:
                    if isinstance(inp, FakeTensor):
                        fake_mode = inp.fake_mode
                        break

                if fake_mode is not None:
                    with fake_mode:
                        work_gm = make_fx(
                            gm,
                            decomposition_table=decomp_table,
                            _allow_non_fake_inputs=True,
                        )(*self.example_inputs)
                    logger.info(
                        f"InductorRegionExtractor: decomposed "
                        f"{len(list(gm.graph.nodes))} → "
                        f"{len(list(work_gm.graph.nodes))} nodes"
                    )
            except Exception as e:
                logger.warning(f"InductorRegionExtractor: decomposition failed: {e}")

        n_nodes = len(list(work_gm.graph.nodes))

        # Step 2: Classify each node as fusible
        fusible_nodes: set[str] = set()
        for node in work_gm.graph.nodes:
            if node.op == "call_function":
                try:
                    if is_fusible_node(node):
                        fusible_nodes.add(node.name)
                except Exception:
                    pass

        # Step 3: Group consecutive fusible nodes
        partitions: list[tuple[list[torch.fx.Node], str]] = []
        current_partition: list[torch.fx.Node] = []
        for node in work_gm.graph.nodes:
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

        # Step 4: Split disconnected components, filter
        all_regions: list[Region] = []
        for comp, norm_fqn in partitions:
            if all(is_view_node(n) or n.op != "call_function" for n in comp):
                continue
            if any(_is_unfusable(n) for n in comp):
                continue
            for connected in _split_connected(comp):
                all_regions.append(_build_region(connected, norm_fqn))

        return all_regions


class SchedulerRegionExtractor(RegionExtractor):
    """Hooks into inductor's post-fusion scheduler to capture actual kernel groups.

    Runs a full inductor compile on the graph, intercepts the scheduler's
    fused kernel decisions via ``_post_fusion_custom_pass``, and maps each
    fused group's origin FX nodes back to ``Region`` objects.

    This produces regions that exactly match what inductor would compile
    into individual Triton kernels, including reduction + pointwise epilog
    fusion and shared-input reduction merging.

    Expensive (runs full inductor compile) but only needs to run once —
    results are cached via the hash-based directory.
    """

    def __init__(self, example_inputs: tuple | None = None):
        self.example_inputs = example_inputs

    def extract(self, gm: torch.fx.GraphModule) -> list[Region]:
        import time as _time

        try:
            import torch._inductor.config as inductor_config
            from torch._inductor.compile_fx import compile_fx_inner
            from torch._inductor.scheduler import (
                BaseSchedulerNode,
                FusedSchedulerNode,
                SchedulerNode,
            )
            from torch._inductor.virtualized import V
        except ImportError:
            logger.warning(
                "SchedulerRegionExtractor: inductor scheduler not available, "
                "falling back to FqnRegionExtractor"
            )
            return FqnRegionExtractor().extract(gm)

        _t0 = _time.time()
        captured_regions: list[Region] = []
        seen_hashes: set[str] = set()

        def _get_all_origins(node: BaseSchedulerNode) -> list[torch.fx.Node]:
            """Get all FX origin nodes from all subnodes in a fused group."""
            origins = []
            seen_ids: set[int] = set()
            snodes = node.get_nodes() if hasattr(node, "get_nodes") else [node]
            for snode in snodes:
                if not isinstance(snode, SchedulerNode):
                    continue
                irnode = snode.node
                if irnode is None:
                    continue
                try:
                    for o in irnode.get_origins():
                        if id(o) not in seen_ids:
                            seen_ids.add(id(o))
                            origins.append(o)
                except Exception:
                    pass
            return origins

        def post_fusion_hook(
            nodes: list[BaseSchedulerNode],
        ) -> list[BaseSchedulerNode]:
            """Intercept inductor's post-fusion pass to capture kernel groups."""
            try:
                orig_gm = V.graph.orig_gm
            except Exception:
                return nodes

            for node in nodes:
                # Get all FX origin nodes for this fused group
                all_origins = _get_all_origins(node)
                if not all_origins:
                    continue

                # Skip single-node non-fused groups (not worth extracting)
                n_subnodes = (
                    len(node.snodes)
                    if isinstance(node, FusedSchedulerNode)
                    else 1
                )
                if n_subnodes < 2 and len(all_origins) < 2:
                    continue

                # Dedup by origin ops hash
                origin_ops = sorted(
                    str(o.target) for o in all_origins if hasattr(o, "target")
                )
                origin_shapes = sorted(
                    f"{o.meta.get('val', '?')}" for o in all_origins
                    if hasattr(o, "meta") and "val" in o.meta
                )
                hash_input = "|".join(origin_ops + origin_shapes)
                h = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                # Filter unfusable origins
                if any(_is_unfusable(o) for o in all_origins if o.op == "call_function"):
                    continue

                # Build region from origin nodes
                norm_fqn = _normalize_fqn(_get_node_fqn(all_origins[0]))
                region = _build_region(all_origins, norm_fqn)
                captured_regions.append(region)

            return nodes

        # Run inductor compile with our hook
        old_pass = getattr(inductor_config, "_post_fusion_custom_pass", None)
        inductor_config._post_fusion_custom_pass = post_fusion_hook
        old_force_disable = getattr(inductor_config, "force_disable_caches", False)
        inductor_config.force_disable_caches = True

        try:
            logger.info("SchedulerRegionExtractor: running inductor compile...")

            # Build inputs from graph placeholders
            if self.example_inputs is not None:
                fake_inputs = list(self.example_inputs)
            else:
                fake_inputs = []
                for node in gm.graph.nodes:
                    if node.op == "placeholder":
                        val = node.meta.get("val")
                        fake_inputs.append(val if val is not None else 0)

            # Apply inductor decompositions (aten.t → aten.permute, etc.)
            # same as full_inductor_compilation_pass does
            from torch._inductor.decomposition import select_decomp_table
            from torch._subclasses.fake_tensor import FakeTensor
            from torch.fx.experimental.proxy_tensor import make_fx

            decomp_table = select_decomp_table()
            fake_mode = None
            for inp in fake_inputs:
                if isinstance(inp, FakeTensor):
                    fake_mode = inp.fake_mode
                    break

            decomposed_gm = gm
            if fake_mode is not None:
                with fake_mode:
                    decomposed_gm = make_fx(
                        gm,
                        decomposition_table=decomp_table,
                        _allow_non_fake_inputs=True,
                    )(*fake_inputs)

            # Run inductor compile — triggers the post_fusion_hook
            compile_fx_inner(decomposed_gm, fake_inputs)
        except Exception as e:
            logger.warning(
                f"SchedulerRegionExtractor: inductor compile failed: {e}. "
                f"Captured {len(captured_regions)} regions before failure."
            )
        finally:
            inductor_config._post_fusion_custom_pass = old_pass
            inductor_config.force_disable_caches = old_force_disable

        logger.info(
            f"SchedulerRegionExtractor: captured {len(captured_regions)} regions "
            f"in {_time.time()-_t0:.1f}s"
        )
        return captured_regions


_EXTRACTORS: dict[str, type[RegionExtractor]] = {
    "fqn": FqnRegionExtractor,
    "inductor": InductorRegionExtractor,
    "scheduler": SchedulerRegionExtractor,
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
    if extractor_cls in (SchedulerRegionExtractor, InductorRegionExtractor):
        all_regions = extractor_cls(example_inputs=example_inputs).extract(gm)
    else:
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
