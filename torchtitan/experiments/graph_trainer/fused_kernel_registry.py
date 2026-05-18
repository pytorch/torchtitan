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
  python -m autoresearch.kernel_agent.run_all

  # Next run: same command, auto-picks up kernels
  ./run_train.sh --compile.fused_kernel_dir /tmp/kernels
"""

from __future__ import annotations

import hashlib
import importlib.util
import operator
import re
import types
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
# Region discovery
# ---------------------------------------------------------------------------

@dataclass
class _Region:
    nodes: list[torch.fx.Node]
    norm_fqn: str
    external_inputs: list[torch.fx.Node]
    output_nodes: list[torch.fx.Node]

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


def _compute_region_hash(region: _Region) -> str:
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


def _split_connected(nodes: list[torch.fx.Node]) -> list[list[torch.fx.Node]]:
    """Split into connected components via union-find."""
    if not nodes:
        return []
    node_set = {n.name for n in nodes}
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


def _discover_regions(
    gm: torch.fx.GraphModule,
    *,
    min_ops: int = 2,
    min_compute_ops: int = 1,
    min_count: int = 2,
) -> list[_Region]:
    """Discover fusible regions in the graph."""
    # Segment at fqn boundaries
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

    # Split connected components, filter unfusable
    all_regions: list[_Region] = []
    for chunk, norm_fqn in raw_regions:
        for comp in _split_connected(chunk):
            if any(_is_unfusable(n) for n in comp):
                continue
            compute = {str(n.target) for n in comp if str(n.target) not in _METADATA_OPS}
            if compute and compute <= _ALREADY_OPTIMIZED_OPS:
                continue

            node_names = {n.name for n in comp}

            # External inputs
            external_inputs: list[torch.fx.Node] = []
            seen: set[str] = set()
            for n in comp:
                for arg in _iter_node_args(n):
                    if arg.name not in node_names and arg.name not in seen:
                        seen.add(arg.name)
                        external_inputs.append(arg)

            # Output nodes: consumed outside the region
            output_nodes: list[torch.fx.Node] = []
            for n in comp:
                for user in n.users:
                    if user.name not in node_names:
                        if n not in output_nodes:
                            output_nodes.append(n)
                        break
            if not output_nodes:
                output_nodes = [comp[-1]]

            all_regions.append(_Region(
                nodes=comp, norm_fqn=norm_fqn,
                external_inputs=external_inputs,
                output_nodes=output_nodes,
            ))

    return all_regions


def _filter_regions(
    regions: list[_Region],
    *,
    min_ops: int = 2,
    min_compute_ops: int = 1,
    min_count: int = 2,
) -> tuple[list[_Region], Counter[str]]:
    """Filter and deduplicate regions. Returns (unique regions, hash counts)."""
    hash_counts: Counter[str] = Counter()
    hash_to_region: dict[str, _Region] = {}
    for r in regions:
        h = _compute_region_hash(r)
        hash_counts[h] += 1
        if h not in hash_to_region:
            hash_to_region[h] = r

    result: list[_Region] = []
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

def _create_subgraph_module(
    region: _Region,
) -> torch.fx.GraphModule:
    """Create a GraphModule that executes the region's ops."""
    nodes = region.nodes
    external_inputs = region.external_inputs
    output_nodes = region.output_nodes
    node_names = {n.name for n in nodes}

    # Build new graph
    new_graph = torch.fx.Graph()

    # Map from original node → new node
    node_map: dict[torch.fx.Node, torch.fx.Node] = {}

    # Placeholders for external inputs
    for i, inp in enumerate(external_inputs):
        ph = new_graph.placeholder(f"input_{i}")
        ph.meta = inp.meta.copy()
        node_map[inp] = ph

    # Copy region nodes
    def _map_arg(arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node):
            if arg in node_map:
                return node_map[arg]
            return arg
        if isinstance(arg, (list, tuple)):
            mapped = [_map_arg(a) for a in arg]
            return type(arg)(mapped)
        return arg

    for n in nodes:
        new_args = tuple(_map_arg(a) for a in n.args)
        new_kwargs = {k: _map_arg(v) for k, v in n.kwargs.items()}
        new_node = new_graph.call_function(n.target, new_args, new_kwargs)
        new_node.meta = n.meta.copy()
        node_map[n] = new_node

    # Output
    if len(output_nodes) == 1:
        new_graph.output(node_map[output_nodes[0]])
    else:
        new_graph.output(tuple(node_map[o] for o in output_nodes))

    new_graph.lint()
    return torch.fx.GraphModule(torch.nn.Module(), new_graph)


# ---------------------------------------------------------------------------
# Kernel loading
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
    region: _Region,
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

def _replace_region(
    gm: torch.fx.GraphModule,
    region: _Region,
    module_name: str,
    subgraph_module: torch.fx.GraphModule,
) -> None:
    """Replace a region's nodes with a call_module to the subgraph module."""
    gm.add_module(module_name, subgraph_module)

    graph = gm.graph
    nodes = region.nodes
    external_inputs = region.external_inputs
    output_nodes = region.output_nodes
    node_names = {n.name for n in nodes}

    # Insert call_module right before the first node in the region
    # (all external inputs are defined before this point)
    first_node = nodes[0]
    with graph.inserting_before(first_node):
        call_node = graph.call_module(
            module_name, args=tuple(external_inputs)
        )

        if len(output_nodes) == 1:
            call_node.meta = output_nodes[0].meta.copy()
        else:
            call_node.meta = {}

    # Redirect users of output nodes to the call_module (or getitems)
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

    # Erase old nodes in reverse topological order
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
    min_ops: int = 2,
    min_compute_ops: int = 1,
    min_count: int = 2,
) -> torch.fx.GraphModule:
    """Single-pass extract + replace + accelerate.

    Discovers fusible regions, replaces each with a ``call_module``
    wrapping the original subgraph (zero-overhead eager fallback).
    If a kernel exists in ``{kernel_dir}/{hash}/kernel.py``, swaps
    the module's forward. If not, writes ``problem.py`` for offline
    kernel generation.

    No-op when ``kernel_dir`` is empty.
    """
    if not kernel_dir:
        return gm

    kdir = Path(kernel_dir)
    kdir.mkdir(parents=True, exist_ok=True)

    # Discover all regions, then filter
    all_regions = _discover_regions(gm)
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
    instances: list[tuple[int, str, _Region]] = []
    for r in all_regions:
        h = _compute_region_hash(r)
        if h in valid_hashes:
            pos = node_order.get(r.nodes[0], 0)
            instances.append((pos, h, r))

    # Sort by position DESCENDING — replace from end of graph to start
    # so earlier node references remain valid
    instances.sort(key=lambda x: x[0], reverse=True)

    replaced = 0
    kernels_loaded = 0
    problems_written = 0

    # Pre-build one template module + kernel per hash (shared across instances)
    hash_kernel: dict[str, Callable | None] = {}
    for _, h, r in instances:
        if h in hash_kernel:
            continue
        region_dir = kdir / h

        # Write problem.py once per hash
        if not (region_dir / "problem.py").exists():
            count = hash_counts.get(h, 1)
            _write_problem(r, region_dir, count)
            problems_written += 1

        hash_kernel[h] = _load_kernel_fn(region_dir)
        if hash_kernel[h] is not None:
            kernels_loaded += 1

    for pos, h, r in instances:
        # Each instance gets its own subgraph module (same structure,
        # different FX nodes), but shares the kernel if available
        subgraph_module = _create_subgraph_module(r)
        kernel_fn = hash_kernel.get(h)
        if kernel_fn is not None:
            subgraph_module.forward = kernel_fn  # type: ignore[assignment]

        module_name = f"_fused_{h}_{replaced}"
        _replace_region(gm, r, module_name, subgraph_module)
        replaced += 1

    if replaced > 0:
        gm.graph.lint()
        gm.recompile()

    logger.info(
        f"Fused kernel pass: {replaced} replacements, "
        f"{kernels_loaded} kernels loaded, "
        f"{problems_written} problems written to {kdir}"
    )

    return gm
