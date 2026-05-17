#!/usr/bin/env python3
"""Extract fusible subgraphs from a text FX graph dump into KernelAgent problems.

Discovery-based extraction with two key improvements:
  1. Uses module_fqn annotations to find repeated layer patterns (not heavy-op boundaries)
  2. Includes shape constraints in signatures so kernels only match same-shaped subgraphs

Algorithm:
  1. Parse the dump into nodes, attaching fqn from preceding comment annotations
  2. Normalize fqn (layers.0 → layers.*) to identify repeated structure
  3. Segment into connected subgraphs within each fqn transition boundary
  4. Group by (normalized_fqn, op_signature, shape_signature) — deduplicate across layers
  5. Rank by frequency × size, generate problem.py for top-N

Usage:
  python -m autoresearch.kernel_agent.extract_from_dump /tmp/final_graph_after_all_passes.txt
  python -m autoresearch.kernel_agent.extract_from_dump graph.txt --dry-run
  python -m autoresearch.kernel_agent.extract_from_dump graph.txt --min-ops 2 --min-count 4
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


GENERATED_DIR = Path(__file__).parent / "generated"

# Ops that are just metadata (zero compute, zero memory traffic)
_METADATA_OPS = {
    "torch.ops.aten.reshape.default",
    "torch.ops.aten.view.default",
    "torch.ops.aten._unsafe_view.default",
    "torch.ops.aten.t.default",
    "torch.ops.aten.transpose.int",
    "torch.ops.aten.unsqueeze.default",
    "torch.ops.aten.squeeze.dim",
    "torch.ops.aten.squeeze.dims",
    "torch.ops.aten.slice.Tensor",
    "torch.ops.aten.expand.default",
    "torch.ops.aten.permute.default",
    "operator.getitem",
}

# Ops that must NOT appear in a fused region.
# Collectives require distributed communication — a local Triton kernel can't replace them.
# Matmuls are already heavily optimized (cuBLAS/nvjet) — replacing them is counterproductive.
# Bucketing ops are custom FSDP scheduling primitives, not compute.
_UNFUSABLE_OPS = {
    # Bucketing (FSDP scheduling)
    "torch.ops.bucketing._pre_bucket_all_gather.default",
    "torch.ops.bucketing._pre_bucket_reduce_scatter.default",
    # Offload/reload (CPU↔GPU transfers)
    "torch.ops.ao.offload.default",
    "torch.ops.ao.reload.default",
    "torch.ops.ao.wait_tensor.default",
    # Flash attention (already optimized, opaque HOP)
    "torch.ops.aten._scaled_dot_product_flash_attention.default",
    "torch.ops.aten._scaled_dot_product_flash_attention_backward.default",
    # Embedding (sparse + irregular memory access)
    "torch.ops.aten.embedding.default",
    "torch.ops.aten.embedding_dense_backward.default",
}

# Ops that are already heavily optimized. A region containing ONLY these
# (plus metadata) is not worth fusing. But a region with these + other
# compute (epilog fusion like mm→cast, mm→silu) IS worth extracting.
_ALREADY_OPTIMIZED_OPS = {
    "torch.ops.aten.mm.default",
    "torch.ops.aten.bmm.default",
    "torch.ops.aten.addmm.default",
}

_DTYPE_MAP = {
    "bf16": "torch.bfloat16",
    "f32": "torch.float32",
    "f16": "torch.float16",
    "c64": "torch.complex64",
    "c128": "torch.complex128",
    "i64": "torch.int64",
    "i32": "torch.int32",
    "b8": "torch.bool",
}


# ---------------------------------------------------------------------------
# Parsed node
# ---------------------------------------------------------------------------

@dataclass
class ParsedNode:
    name: str
    type_str: str
    dtype: str
    shape: tuple[int, ...]
    target: str
    raw_args: str
    line_no: int
    fqn: str = ""          # module_fqn from annotation
    norm_fqn: str = ""     # normalized fqn (layers.0 → layers.*)
    is_getitem: bool = False
    getitem_src: str = ""
    getitem_idx: int = 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_OP_RE = re.compile(r"^\s+(\w+):\s+\"([^\"]+)\"\s+=\s+(torch\.ops\.\S+)\((.+)")
_OP_NO_TYPE_RE = re.compile(r"^\s+(\w+)\s+=\s+(torch\.ops\.\S+)\((.+)")
_GETITEM_RE = re.compile(r"^\s+(\w+):\s+\"([^\"]+)\"\s+=\s+(\w+)\[(\d+)\]")
_GETITEM_NO_TYPE_RE = re.compile(r"^\s+(\w+)\s+=\s+(\w+)\[(\d+)\]")
_TYPE_RE = re.compile(r"^(\w+)\[([^\]]*)\]\[([^\]]*)\](\S+)$")
_FQN_RE = re.compile(r"module_fqn': '([^']+)'")
_PARAM_RE = re.compile(r"^\s+(\w+):\s+\"([^\"]+)\"")


def _parse_type(type_str: str) -> tuple[str, tuple[int, ...]]:
    m = _TYPE_RE.match(type_str)
    if not m:
        return "unknown", ()
    dtype = m.group(1)
    shape_str = m.group(2)
    if not shape_str:
        return dtype, ()
    shape = tuple(int(x.strip()) for x in shape_str.split(","))
    return dtype, shape


def _normalize_fqn(fqn: str) -> str:
    """layers.0.attention → layers.*.attention"""
    return re.sub(r"layers\.\d+", "layers.*", fqn)


def _extract_arg_names(raw_args: str) -> list[str]:
    raw = raw_args.split(";")[0]
    tokens = re.findall(r'\b([a-zA-Z_]\w*)\b', raw)
    exclude = {
        "None", "True", "False", "torch", "device", "type", "cuda", "cpu",
        "memory_format", "contiguous_format", "preserve_format",
        "dtype", "layout", "strided", "pin_memory", "out",
        "bfloat16", "float16", "float32", "float64", "int32", "int64",
        "bool", "complex64", "complex128",
        "index", "dim", "keepdim", "eps", "size", "start", "end", "step",
    }
    return [t for t in tokens if t not in exclude and not t.startswith("torch")]


def parse_graph(text: str) -> tuple[list[ParsedNode], dict[str, ParsedNode]]:
    """Parse the dump into nodes with fqn annotations, plus graph params."""
    nodes = []
    params: dict[str, ParsedNode] = {}
    current_fqn = ""
    in_sig = False

    for line_no, line in enumerate(text.splitlines(), 1):
        # Parse function signature for param types
        if "def forward(" in line:
            in_sig = True
            continue
        if in_sig:
            if "= torch.ops." in line or (line.strip().startswith("#") and "Annotation" in line):
                in_sig = False
            else:
                m = _PARAM_RE.match(line)
                if m:
                    name, type_str = m.groups()
                    dtype, shape = _parse_type(type_str)
                    params[name] = ParsedNode(
                        name=name, type_str=type_str, dtype=dtype, shape=shape,
                        target="param", raw_args="", line_no=line_no,
                    )
                continue

        # Track fqn from annotation comments
        m = _FQN_RE.search(line)
        if m:
            current_fqn = m.group(1)
            continue

        norm_fqn = _normalize_fqn(current_fqn)

        # Parse op with type annotation
        m = _OP_RE.match(line)
        if m:
            name, type_str, target, raw_args = m.groups()
            dtype, shape = _parse_type(type_str)
            nodes.append(ParsedNode(
                name=name, type_str=type_str, dtype=dtype, shape=shape,
                target=target, raw_args=raw_args, line_no=line_no,
                fqn=current_fqn, norm_fqn=norm_fqn,
            ))
            continue

        # Op without type annotation
        m = _OP_NO_TYPE_RE.match(line)
        if m:
            name, target, raw_args = m.groups()
            nodes.append(ParsedNode(
                name=name, type_str="", dtype="unknown", shape=(),
                target=target, raw_args=raw_args, line_no=line_no,
                fqn=current_fqn, norm_fqn=norm_fqn,
            ))
            continue

        # Getitem with type
        m = _GETITEM_RE.match(line)
        if m:
            name, type_str, src, idx = m.groups()
            dtype, shape = _parse_type(type_str)
            nodes.append(ParsedNode(
                name=name, type_str=type_str, dtype=dtype, shape=shape,
                target="operator.getitem", raw_args=f"{src}, {idx}",
                line_no=line_no, is_getitem=True,
                getitem_src=src, getitem_idx=int(idx),
                fqn=current_fqn, norm_fqn=norm_fqn,
            ))
            continue

        # Getitem without type
        m = _GETITEM_NO_TYPE_RE.match(line)
        if m:
            name, src, idx = m.groups()
            nodes.append(ParsedNode(
                name=name, type_str="", dtype="unknown", shape=(),
                target="operator.getitem", raw_args=f"{src}, {idx}",
                line_no=line_no, is_getitem=True,
                getitem_src=src, getitem_idx=int(idx),
                fqn=current_fqn, norm_fqn=norm_fqn,
            ))
            continue

    return nodes, params


# ---------------------------------------------------------------------------
# Region: a connected subgraph within an fqn boundary
# ---------------------------------------------------------------------------

@dataclass
class Region:
    nodes: list[ParsedNode]
    norm_fqn: str
    # Signature: op targets (for matching in FX graph)
    op_signature: tuple[str, ...]
    # Shape signature: (dtype, shape) per node (for ensuring same-shaped matches)
    shape_signature: tuple[tuple[str, tuple[int, ...]], ...]
    num_compute_ops: int

    @property
    def key(self) -> tuple:
        """Deduplication key: same ops on same shapes in same module type."""
        return (self.norm_fqn, self.op_signature, self.shape_signature)

    @property
    def slug(self) -> str:
        compute = [
            n.target.replace("torch.ops.aten.", "").replace(".default", "")
                    .replace(".Tensor", "").replace(".Scalar", "")
            for n in self.nodes if n.target not in _METADATA_OPS
        ]
        if not compute:
            compute = ["reshape_chain"]
        return "_".join(compute[:4])


def _is_connected(nodes: list[ParsedNode]) -> bool:
    """Check that every node (except the first) has at least one arg from the node set."""
    if len(nodes) <= 1:
        return True
    node_names = {n.name for n in nodes}
    for n in nodes[1:]:
        args = _extract_arg_names(n.raw_args)
        if n.is_getitem:
            args.append(n.getitem_src)
        if not any(a in node_names for a in args):
            return False
    return True


def _split_connected(nodes: list[ParsedNode]) -> list[list[ParsedNode]]:
    """Split a list of nodes into connected components using union-find."""
    if not nodes:
        return []

    name_to_idx = {n.name: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, n in enumerate(nodes):
        args = _extract_arg_names(n.raw_args)
        if n.is_getitem:
            args.append(n.getitem_src)
        for arg in args:
            if arg in name_to_idx:
                union(i, name_to_idx[arg])

    # Group by root
    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Return in original order
    return [[nodes[i] for i in sorted(idxs)] for idxs in groups.values()]


def segment_regions(nodes: list[ParsedNode]) -> list[Region]:
    """Segment the graph into regions at fqn transition boundaries,
    then split disconnected components within each region."""
    regions: list[Region] = []
    current: list[ParsedNode] = []
    current_norm_fqn = ""

    def flush(chunk: list[ParsedNode], norm_fqn: str):
        if not chunk:
            return
        # Split into connected components
        components = _split_connected(chunk)
        for comp in components:
            # Skip components containing unfusable ops (collectives, etc.)
            if any(
                n.target in _UNFUSABLE_OPS
                or n.target.startswith("torch.ops._c10d_functional.")
                for n in comp
            ):
                continue
            # Skip components where the only compute ops are already-optimized
            # (e.g., a bare mm with only reshapes). But keep mm + epilog fusions.
            compute_targets = {n.target for n in comp if n.target not in _METADATA_OPS}
            if compute_targets and compute_targets <= _ALREADY_OPTIMIZED_OPS:
                continue
            op_sig = tuple(n.target for n in comp)
            shape_sig = tuple((n.dtype, n.shape) for n in comp)
            num_compute = sum(1 for n in comp if n.target not in _METADATA_OPS)
            regions.append(Region(
                nodes=comp,
                norm_fqn=norm_fqn,
                op_signature=op_sig,
                shape_signature=shape_sig,
                num_compute_ops=num_compute,
            ))

    for node in nodes:
        if node.norm_fqn != current_norm_fqn:
            flush(current, current_norm_fqn)
            current = []
            current_norm_fqn = node.norm_fqn
        current.append(node)

    flush(current, current_norm_fqn)
    return regions


def _estimate_bytes(region: Region) -> int:
    dtype_bytes = {"bf16": 2, "f16": 2, "f32": 4, "f64": 8, "c64": 8, "c128": 16, "i64": 8, "i32": 4, "b8": 1}
    max_bytes = 0
    for n in region.nodes:
        if n.shape:
            elem_size = dtype_bytes.get(n.dtype, 4)
            size = 1
            for d in n.shape:
                size *= d
            max_bytes = max(max_bytes, size * elem_size)
    return max_bytes


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def generate_problem(region: Region, all_nodes: dict[str, ParsedNode], count: int) -> str:
    nodes = region.nodes
    node_names = {n.name for n in nodes}

    # External inputs
    input_names: list[str] = []
    seen: set[str] = set()
    for n in nodes:
        for arg_name in _extract_arg_names(n.raw_args):
            if arg_name not in node_names and arg_name not in seen:
                seen.add(arg_name)
                input_names.append(arg_name)

    input_map = {name: f"input_{i}" for i, name in enumerate(input_names)}
    var_map = {n.name: f"v_{i}" for i, n in enumerate(nodes)}

    # Forward body
    body_lines = []
    for n in nodes:
        var = var_map[n.name]
        if n.is_getitem:
            src = var_map.get(n.getitem_src, input_map.get(n.getitem_src, n.getitem_src))
            body_lines.append(f"        {var} = {src}[{n.getitem_idx}]")
        else:
            raw = n.raw_args.split(";")[0].rstrip()
            if raw.endswith(")"):
                raw = raw[:-1]
            raw = re.sub(r'(?<!\w)device\(', 'torch.device(', raw)
            for orig, repl in {**input_map, **var_map}.items():
                raw = re.sub(rf'(?<!\.)(?<!\w){re.escape(orig)}\b', repl, raw)
            comment = f"  # {n.type_str}" if n.type_str else ""
            body_lines.append(f"        {var} = {n.target}({raw}){comment}")

    last_node = nodes[-1]
    out_var = var_map[last_node.name]
    body_lines.append(f"        return {out_var}")
    forward_body = "\n".join(body_lines)

    # get_inputs
    param_parts = []
    input_lines = []
    return_parts = []
    for inp_name in input_names:
        pname = input_map[inp_name]
        param_parts.append(f"{pname}: torch.Tensor")
        return_parts.append(pname)
        node = all_nodes.get(inp_name)
        if node and node.shape:
            torch_dtype = _DTYPE_MAP.get(node.dtype, "torch.float32")
            shape_str = repr(node.shape)
            if node.dtype in ("bf16", "f16", "f32", "f64", "c64", "c128"):
                input_lines.append(f"    {pname} = torch.randn({shape_str}, dtype={torch_dtype}, device='cuda')")
            elif node.dtype == "b8":
                input_lines.append(f"    {pname} = torch.randint(0, 2, {shape_str}, dtype={torch_dtype}, device='cuda')")
            else:
                input_lines.append(f"    {pname} = torch.randint(0, 10, {shape_str}, dtype={torch_dtype}, device='cuda')")
        else:
            input_lines.append(f"    {pname} = torch.randn((1,), dtype=torch.float32, device='cuda')")

    param_str = ", ".join(param_parts)
    inputs_body = "\n".join(input_lines) if input_lines else "    pass"
    return_list = ", ".join(return_parts)

    compute_ops = [
        n.target.replace("torch.ops.aten.", "").replace(".default", "").replace(".Tensor", "")
        for n in nodes if n.target not in _METADATA_OPS
    ]
    shapes = set()
    for n in nodes:
        if n.shape:
            shapes.add(f"{n.dtype}{list(n.shape)}")
    shape_info = ", ".join(sorted(shapes)[:3])

    desc = f"Fused region ({region.norm_fqn}): {' -> '.join(compute_ops) if compute_ops else 'reshape chain'}\n"
    desc += f"Instances: {count}. Ops: {len(nodes)}, compute: {len(compute_ops)}. Shapes: {shape_info}\n"

    return f"""{desc}
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, {param_str}) -> torch.Tensor:
{forward_body}

def get_inputs():
{inputs_body}
    return [{return_list}]

def get_init_inputs():
    return []
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract fusible regions from FX graph dump."
    )
    parser.add_argument("graph_dump", type=Path)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-ops", type=int, default=2)
    parser.add_argument("--min-compute-ops", type=int, default=1)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    text = args.graph_dump.read_text()
    print(f"Parsing {args.graph_dump} ({len(text.splitlines())} lines)...")
    nodes, params = parse_graph(text)
    all_nodes = {n.name: n for n in nodes}
    all_nodes.update(params)
    print(f"Parsed {len(nodes)} nodes, {len(params)} params")

    # Op frequency
    op_counts = Counter(n.target for n in nodes)
    print(f"\nTop 20 ops:")
    for op, count in op_counts.most_common(20):
        print(f"  {op:<60} {count:>5}")

    # Unique fqn patterns
    fqn_counts = Counter(n.norm_fqn for n in nodes if n.norm_fqn)
    print(f"\nTop 10 module fqn patterns:")
    for fqn, count in fqn_counts.most_common(10):
        print(f"  {fqn:<50} {count:>5} nodes")

    # Segment
    regions = segment_regions(nodes)
    print(f"\nTotal regions (after fqn split + connectivity): {len(regions)}")

    # Group by key (norm_fqn + op_sig + shape_sig)
    by_key: dict[tuple, list[Region]] = {}
    for r in regions:
        by_key.setdefault(r.key, []).append(r)

    print(f"Unique (fqn, ops, shapes) signatures: {len(by_key)}")

    # Filter and rank
    candidates: list[tuple[int, int, int, Region, str]] = []
    for key, instances in by_key.items():
        rep = instances[0]
        count = len(instances)
        if len(rep.nodes) < args.min_ops:
            continue
        if rep.num_compute_ops < args.min_compute_ops:
            continue
        if count < args.min_count:
            continue
        est_bytes = _estimate_bytes(rep)
        candidates.append((count, rep.num_compute_ops, est_bytes, rep, rep.slug))

    candidates.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
    top = candidates[:args.top_n]

    print(f"\nTop {len(top)} fusible regions:\n")
    print(f"  {'#':<4} {'Count':>5} {'Ops':>4} {'Compute':>7} {'Bytes':>12} {'FQN':<35} {'Slug'}")
    print(f"  {'-'*4} {'-'*5} {'-'*4} {'-'*7} {'-'*12} {'-'*35} {'-'*30}")

    written = 0
    for i, (count, num_compute, est_bytes, region, slug) in enumerate(top):
        dir_name = f"{i:02d}_{slug}"
        fqn_short = region.norm_fqn[:35] if region.norm_fqn else "(none)"
        print(f"  {i:<4} {count:>5} {len(region.nodes):>4} {num_compute:>7} {est_bytes:>12,} {fqn_short:<35} {dir_name}")

        problem = generate_problem(region, all_nodes, count)

        if args.dry_run:
            print(f"\n--- {dir_name} ---")
            print(problem)
            print("---\n")
        else:
            out_dir = args.output_dir / dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "problem.py").write_text(problem)
            written += 1

    if not args.dry_run and written:
        print(f"\nWrote {written} problem files to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
