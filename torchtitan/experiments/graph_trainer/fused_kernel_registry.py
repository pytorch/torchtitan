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
from typing import Callable

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
    # Matmul: dedicated kernels (cublas/cutlass), not part of fused triton.
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    # Complex-tensor reinterpret views: triton can't handle conjugate views
    # or change-of-dtype views of complex tensors.
    "aten.view_as_complex.default",
    "aten.view_as_real.default",
    "aten._conj.default",
}

# Namespaces where ALL ops are unfusable
_UNFUSABLE_NAMESPACES = {"_c10d_functional", "ao", "bucketing"}

# Kept for FqnRegionExtractor's "already optimized" filter (not for classifier).
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

# Pointwise/elementwise ops — fully fusible into a single triton kernel.
_POINTWISE_OPS = frozenset({
    "aten.add.Tensor", "aten.add.Scalar",
    "aten.sub.Tensor", "aten.sub.Scalar",
    "aten.mul.Tensor", "aten.mul.Scalar",
    "aten.div.Tensor", "aten.div.Scalar",
    "aten.rsub.Scalar",
    "aten.neg.default",
    "aten.abs.default",
    "aten.exp.default", "aten.log.default", "aten.log1p.default",
    "aten.sin.default", "aten.cos.default", "aten.tan.default",
    "aten.tanh.default", "aten.sigmoid.default",
    "aten.silu.default", "aten.silu_backward.default",
    "aten.gelu.default", "aten.gelu_backward.default",
    "aten.relu.default", "aten.threshold_backward.default",
    "aten.rsqrt.default", "aten.sqrt.default",
    "aten.reciprocal.default",
    "aten.pow.Tensor_Scalar", "aten.pow.Scalar", "aten.pow.Tensor_Tensor",
    "aten.where.self",
    "aten.maximum.default", "aten.minimum.default",
    "aten.clamp.default", "aten.clamp_min.default", "aten.clamp_max.default",
    "aten.eq.Tensor", "aten.eq.Scalar",
    "aten.ne.Tensor", "aten.ne.Scalar",
    "aten.lt.Tensor", "aten.lt.Scalar",
    "aten.gt.Tensor", "aten.gt.Scalar",
    "aten.le.Tensor", "aten.le.Scalar",
    "aten.ge.Tensor", "aten.ge.Scalar",
    "aten.bitwise_and.Tensor", "aten.bitwise_or.Tensor",
    "aten.bitwise_not.default", "aten.logical_not.default",
    "aten.fill.Scalar", "aten.fill.Tensor",
    "aten.zeros_like.default", "aten.ones_like.default",
    "aten.full.default", "aten.full_like.default",
    "aten._to_copy.default",
    "aten.copy.default",
    "aten.clone.default",
    "prims.convert_element_type.default",
    "prims.broadcast_in_dim.default",
    # Indexing/gather — inductor fuses these with pointwise ops.
    "aten.index.Tensor",
    "aten.gather.default",
    "aten.scatter.value", "aten.scatter.src",
    "aten.embedding.default",
    "aten.embedding_dense_backward.default",
})

# Reduction ops — fusible (inductor groups them with pointwise epilogs).
_REDUCTION_OPS = frozenset({
    "aten.sum.default", "aten.sum.dim_IntList",
    "aten.mean.default", "aten.mean.dim",
    "aten.amax.default", "aten.amax.dim",
    "aten.amin.default", "aten.amin.dim",
    "aten.max.default", "aten.max.dim",
    "aten.min.default", "aten.min.dim",
    "aten.prod.default", "aten.prod.dim_int",
    "aten.std.default", "aten.std.dim",
    "aten.var.default", "aten.var.dim",
    "aten.argmax.default", "aten.argmin.default",
    "aten.any.default", "aten.any.dim",
    "aten.all.default", "aten.all.dim",
})

# Norm/softmax ops — decompose to pointwise+reduction, treat as fusible.
_FUSIBLE_DECOMPOSABLE_OPS = frozenset({
    "aten._fused_rms_norm.default",
    "aten._fused_rms_norm_backward.default",
    "aten.rms_norm.default",
    "aten.native_layer_norm.default",
    "aten.native_layer_norm_backward.default",
    "aten.layer_norm.default",
    "aten._softmax.default",
    "aten._softmax_backward_data.default",
    "aten._log_softmax.default",
    "aten._log_softmax_backward_data.default",
    "aten.softmax.int",
    "aten.log_softmax.int",
    "aten.nll_loss_forward.default",
    "aten.nll_loss_backward.default",
})

# View ops — don't block fusion, but a partition of only views isn't a kernel.
_VIEW_OPS = frozenset({
    "aten.view.default",
    "aten.view.dtype",
    "aten._unsafe_view.default",
    "aten.reshape.default",
    "aten.expand.default",
    "aten.slice.Tensor",
    "aten.select.int",
    "aten.transpose.int",
    "aten.t.default",
    "aten.permute.default",
    "aten.unsqueeze.default",
    "aten.squeeze.default",
    "aten.squeeze.dim", "aten.squeeze.dims",
    "aten.alias.default",
    "aten.detach.default",
    "aten.contiguous.default",
    # Tuple-returning view: returns a list of slices (no data movement).
    # The downstream getitem nodes index into the tuple to get tensors.
    "aten.split_with_sizes.default",
    "aten.split.Tensor",
    "aten.chunk.default",
    "aten.unbind.int",
    "<built-in function getitem>",
})

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


def _classify_node(node: torch.fx.Node) -> str:
    """Classify a node as one of:

      ``"unfusable"`` — collectives, matmul, flash attention, complex
                        tensor views — never fuse.
      ``"view"``      — pure view/reshape (doesn't block fusion, isn't a kernel)
      ``"pointwise"`` — elementwise ops, dtype convert, embedding/gather, etc.
      ``"reduction"`` — sum/mean/amax/etc.
      ``"decomposable"`` — _fused_rms_norm/layernorm/softmax/nll (decompose
                          into pointwise+reduction and are fusible as a whole)
      ``"other"``     — anything not on our allow-list (treated as unfusable)

    Graph_trainer-specific classifier, deliberately independent of
    inductor's ``is_fusible_node`` which expects a post-decomposition
    graph and miscategorizes many of our pre-decomp nodes.
    """
    if node.op != "call_function":
        return "other"
    if _is_unfusable(node):
        return "unfusable"
    target_str = str(node.target)
    if target_str in _POINTWISE_OPS:
        return "pointwise"
    if target_str in _REDUCTION_OPS:
        return "reduction"
    if target_str in _FUSIBLE_DECOMPOSABLE_OPS:
        return "decomposable"
    if target_str in _VIEW_OPS:
        return "view"
    return "other"


def is_fusible_node(node: torch.fx.Node) -> bool:
    """Whether a node can be part of a fused triton kernel.

    Pointwise, reductions, decomposable norms/softmax/loss, and views
    are fusible. Matmul, attention, collectives, embedding, and other
    ops are not.
    """
    return _classify_node(node) in ("pointwise", "reduction", "decomposable", "view")


def is_view_node(node: torch.fx.Node) -> bool:
    return _classify_node(node) == "view"


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
    """A fusible subgraph discovered by a RegionExtractor.

    ``nodes``/``external_inputs``/``output_nodes`` reference the parent
    graph (for replacement). ``subgraph_gm`` is an optional standalone
    fx.GraphModule containing copies of the same nodes — used for
    problem.py generation via ``print_readable`` and for op-set hashing.
    ``inputs_info`` captures shape/stride/dtype/device per placeholder
    so the generated problem.py recreates exact-stride inputs.
    """

    nodes: list[torch.fx.Node]
    external_inputs: list[torch.fx.Node]
    output_nodes: list[torch.fx.Node]
    norm_fqn: str = ""

    subgraph_gm: torch.fx.GraphModule | None = None
    inputs_info: list[dict] | None = None

    @property
    def num_compute_ops(self) -> int:
        if self.subgraph_gm is not None:
            return sum(
                1 for n in self.subgraph_gm.graph.nodes
                if n.op == "call_function" and str(n.target) not in _METADATA_OPS
            )
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
    """Stable hash from op targets + input/output shapes.

    When ``subgraph_gm`` is set, hashes a position-independent op set
    + per-input shape/stride/dtype (more robust across runs). Otherwise
    hashes the original-graph nodes positionally.
    """
    parts = []
    if region.subgraph_gm is not None:
        # Position-independent op set + per-input shape/stride/dtype
        ops = sorted(
            str(n.target) for n in region.subgraph_gm.graph.nodes
            if n.op == "call_function"
        )
        parts.extend(ops)
        for info in region.inputs_info or []:
            parts.append(
                f"in:{info.get('dtype')}:{info.get('shape')}:{info.get('stride')}"
            )
    else:
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


def tag_fusible_nodes_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Tag each call_function node with its fusibility classification.

    Uses ``_classify_node`` (graph_trainer-specific) which is more
    reliable than inductor's ``is_fusible_node`` for pre-decomp graphs.
    Stores the result under ``node.meta["custom"]["fusion_class"]`` and
    ``["is_fusible"]`` so it shows up in tlparse dumps.
    """
    counts: dict[str, int] = {}
    # Track meta["custom"] dict identity to detect shared references
    custom_id_to_nodes: dict[int, list[str]] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        cls = _classify_node(node)
        counts[cls] = counts.get(cls, 0) + 1
        # Always allocate a FRESH custom dict — if the bucketing pass (or
        # similar) inserted nodes that share a meta["custom"] dict by
        # reference, our per-node update would clobber siblings.
        existing = node.meta.get("custom", {})
        node.meta["custom"] = dict(existing)
        node.meta["custom"]["fusion_class"] = cls
        node.meta["custom"]["is_fusible"] = cls in (
            "pointwise", "reduction", "decomposable", "view"
        )
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    logger.info(f"tag_fusible_nodes: {summary}")
    return gm


class RegionExtractor(ABC):
    """Base class for fusible region extractors.

    ``extract`` returns a graph and the regions discovered in it. Both
    current extractors return the input ``gm`` unchanged; the tuple
    return is kept for backward compatibility with prior extractors
    that transformed the graph.
    """

    @abstractmethod
    def extract(
        self, gm: torch.fx.GraphModule
    ) -> tuple[torch.fx.GraphModule, list[Region]]:
        """Discover regions. Returns (work_gm, regions)."""
        ...


class FqnRegionExtractor(RegionExtractor):
    """Segments at module_fqn boundaries, splits connected components,
    filters by handcoded op lists."""

    def extract(
        self, gm: torch.fx.GraphModule
    ) -> tuple[torch.fx.GraphModule, list[Region]]:
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

        return gm, all_regions


def _has_reduction(nodes: list[torch.fx.Node]) -> bool:
    """Check if any node in the set is a reduction op via torch.Tag.reduction."""
    for n in nodes:
        if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
            if torch.Tag.reduction in n.target.tags:
                return True
    return False


def _is_activation_tensor(node: torch.fx.Node) -> bool:
    """True if node looks like an activation (multi-dim, large).

    Rejects 1D tensors (weights/biases) and scalars, which cause
    spurious merges when weights are shared across layers.
    """
    val = node.meta.get("val", None)
    if isinstance(val, torch.Tensor):
        return val.dim() >= 2 and val.numel() >= 64
    return False


def _merge_shared_input_reductions(
    components: list[list[torch.fx.Node]],
) -> list[list[torch.fx.Node]]:
    """Merge reduction partitions that share a common activation input.

    Models inductor's mix-order reduction: if two partitions both contain
    reductions and read from the same activation (e.g. LayerNorm backward
    has inner reductions and outer reductions reading from tangents_1),
    they would be fused into one kernel.  Only merges when one partition
    is small (≤5 ops) to avoid over-merging in weight-sharing models.
    """
    if len(components) <= 1:
        return components

    reduction_idxs = [i for i, c in enumerate(components) if _has_reduction(c)]
    if len(reduction_idxs) <= 1:
        return components

    def _external_inputs(comp_nodes: list[torch.fx.Node]) -> set[torch.fx.Node]:
        node_set = set(comp_nodes)
        inputs = set()
        for n in comp_nodes:
            for inp in n.all_input_nodes:
                if inp not in node_set:
                    inputs.add(inp)
        return inputs

    parent = list(range(len(components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    inputs_by_comp = {i: _external_inputs(components[i]) for i in reduction_idxs}

    for i in range(len(reduction_idxs)):
        for j in range(i + 1, len(reduction_idxs)):
            ci, cj = reduction_idxs[i], reduction_idxs[j]
            if min(len(components[ci]), len(components[cj])) > 5:
                continue
            shared = inputs_by_comp[ci] & inputs_by_comp[cj]
            if any(_is_activation_tensor(inp) for inp in shared):
                union(ci, cj)

    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(components)):
        groups[find(i)].append(i)

    merged = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(components[idxs[0]])
        else:
            combined: list[torch.fx.Node] = []
            seen: set[int] = set()
            for idx in idxs:
                for n in components[idx]:
                    if id(n) not in seen:
                        seen.add(id(n))
                        combined.append(n)
            merged.append(combined)
    return merged


def _extract_subgraph(
    origin_nodes: list[torch.fx.Node], gm: torch.fx.GraphModule
) -> tuple[torch.fx.GraphModule, list[dict], list[torch.fx.Node]] | None:
    """Extract a minimal FX subgraph containing only the origin nodes.

    Dependencies outside the origin set become placeholders. Returns:
      - new_gm: the standalone subgraph
      - placeholder_info: shape/stride/dtype/device per placeholder
      - output_nodes: the source-graph nodes that are subgraph outputs
        (in order, matching new_gm's output tuple)
    """
    if not origin_nodes:
        return None

    import copy as _copy

    seen_ids: set[int] = set()
    unique: list[torch.fx.Node] = []
    for n in origin_nodes:
        if id(n) not in seen_ids:
            seen_ids.add(id(n))
            unique.append(n)
    origin_nodes = unique

    needed: set[torch.fx.Node] = set(origin_nodes)
    output_nodes: list[torch.fx.Node] = []
    for n in origin_nodes:
        has_internal_user = any(
            u in needed and u.op != "output" for u in n.users
        )
        if not has_internal_user:
            output_nodes.append(n)
    if not output_nodes:
        output_nodes = list(origin_nodes)

    new_graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    placeholder_info: list[dict] = []
    placeholder_names: set[str] = set()

    def _record(name: str, meta: dict, orig_node: torch.fx.Node) -> None:
        val = meta.get("val", None)
        if isinstance(val, torch.Tensor):
            stride = list(val.stride()) if not val.is_contiguous() else []
            placeholder_info.append({
                "name": name,
                "shape": list(val.shape),
                "stride": stride,
                "dtype": str(val.dtype),
                "device": str(val.device),
                "orig_node": orig_node,
            })

    def _ensure(x):
        if isinstance(x, torch.fx.Node):
            if x in env:
                return env[x]
            ph = new_graph.placeholder(x.name)
            ph.meta = _copy.copy(x.meta) if x.meta else {}
            env[x] = ph
            if x.name not in placeholder_names:
                placeholder_names.add(x.name)
                _record(x.name, x.meta or {}, x)
            return ph
        return x

    all_nodes = list(gm.graph.nodes)
    order = {n: i for i, n in enumerate(all_nodes)}
    sorted_needed = sorted(needed, key=lambda n: order.get(n, 0))

    for node in sorted_needed:
        if node.op == "placeholder":
            new_n = new_graph.placeholder(node.name)
            new_n.meta = _copy.copy(node.meta) if node.meta else {}
            env[node] = new_n
            if node.name not in placeholder_names:
                placeholder_names.add(node.name)
                _record(node.name, node.meta or {}, node)
        elif node.op == "get_attr":
            new_n = new_graph.get_attr(node.target)
            new_n.meta = _copy.copy(node.meta) if node.meta else {}
            env[node] = new_n
        elif node.op in ("call_function", "call_method"):
            new_args = torch.fx.map_arg(node.args, _ensure)
            new_kwargs = torch.fx.map_arg(node.kwargs, _ensure)
            if node.op == "call_function":
                new_n = new_graph.call_function(
                    node.target, args=new_args, kwargs=new_kwargs
                )
            else:
                new_n = new_graph.call_method(
                    node.target, args=new_args, kwargs=new_kwargs
                )
            new_n.meta = _copy.copy(node.meta) if node.meta else {}
            env[node] = new_n

    final_output_nodes = [n for n in output_nodes if n in env]
    if not final_output_nodes:
        return None
    mapped_outputs = [env[n] for n in final_output_nodes]
    if len(mapped_outputs) == 1:
        new_graph.output(mapped_outputs[0])
    else:
        new_graph.output(tuple(mapped_outputs))

    new_graph.lint()
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm, placeholder_info, final_output_nodes


class InductorRegionExtractor(RegionExtractor):
    """Discover fusible regions matching inductor's fusion groups.

    Closely mirrors compile-utils' ATen-level extractor, but operates
    directly on the pre-decomposition graph (no make_fx step):
      1. Partition with ``CapabilityBasedPartitioner`` + our own
         ``is_fusible_node`` classifier (graph_trainer-specific, see
         ``_classify_node``).
      2. Merge reduction partitions that share a common activation input
         (mix-order reduction fusion).
      3. For each partition, extract a standalone ``fx.GraphModule``
         subgraph; its placeholders are direct references to the
         original graph's nodes, so replacement is trivial.

    Working pre-decomposition means: no from_node back-mapping, no
    dtype/shape mismatch checks, and problem.py uses semantic ops
    (e.g. ``_fused_rms_norm`` instead of the decomposed pow/mean/rsqrt
    chain).
    """

    def __init__(self, example_inputs: tuple | None = None):
        # example_inputs kept for signature compat; unused now.
        self.example_inputs = example_inputs

    def extract(
        self, gm: torch.fx.GraphModule
    ) -> tuple[torch.fx.GraphModule, list[Region]]:
        import time as _time

        try:
            from torch.fx.passes.infra.partitioner import (
                CapabilityBasedPartitioner,
            )
            from torch.fx.passes.operator_support import create_op_support
        except ImportError:
            logger.warning(
                "InductorRegionExtractor: partitioner not available, "
                "falling back to FqnRegionExtractor"
            )
            return FqnRegionExtractor().extract(gm)

        _t0 = _time.time()

        # Partition the original graph using our classifier
        def _is_supported(_subm, node):
            return is_fusible_node(node)

        support = create_op_support(_is_supported)
        partitioner = CapabilityBasedPartitioner(
            gm, support, allows_single_node_partition=True,
        )
        partitions = partitioner.propose_partitions()
        components: list[list[torch.fx.Node]] = [
            list(p.nodes.keys()) for p in partitions
        ]

        n_before = len(components)
        components = _merge_shared_input_reductions(components)
        merged_msg = (
            f" (merged {n_before} → {len(components)})"
            if n_before != len(components) else ""
        )
        logger.info(
            f"InductorRegionExtractor: {len(list(gm.graph.nodes))} nodes, "
            f"{len(components)} partitions{merged_msg} in {_time.time()-_t0:.1f}s"
        )

        all_regions: list[Region] = []
        for comp in components:
            # Skip view-only and any unfusable partitions
            if all(is_view_node(n) or n.op != "call_function" for n in comp):
                continue
            if any(_is_unfusable(n) for n in comp if n.op == "call_function"):
                continue

            result = _extract_subgraph(comp, gm)
            if result is None:
                continue
            sub_gm, placeholder_info, output_nodes = result

            # External inputs: the original-graph nodes referenced by the
            # subgraph's placeholders.  Already recorded in placeholder_info.
            ext_inputs = [info["orig_node"] for info in placeholder_info]

            order = {n: i for i, n in enumerate(gm.graph.nodes)}
            nodes_sorted = sorted(comp, key=lambda n: order.get(n, 0))
            norm_fqn = _normalize_fqn(_get_node_fqn(nodes_sorted[0]))

            region = Region(
                nodes=nodes_sorted,
                external_inputs=ext_inputs,
                output_nodes=output_nodes,
                norm_fqn=norm_fqn,
                subgraph_gm=sub_gm,
                inputs_info=placeholder_info,
            )
            all_regions.append(region)

        return gm, all_regions


_EXTRACTORS: dict[str, type[RegionExtractor]] = {
    "fqn": FqnRegionExtractor,
    "inductor": InductorRegionExtractor,
}


def _filter_regions(
    regions: list[Region],
    *,
    min_ops: int = 2,
    min_compute_ops: int = 1,
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
    *,
    min_speedup: float = 1.1,
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

    # Check triton (must beat eager by min_speedup threshold)
    triton_ms = bench_data.get("triton_ms", float("inf"))
    if triton_fn is not None and triton_ms * min_speedup < eager_ms:
        return triton_fn, "triton"

    # Check compile (must beat eager by min_speedup AND beat triton)
    compile_ms = bench_data.get("compile_ms", float("inf"))
    if compile_ms * min_speedup < eager_ms and compile_ms < triton_ms:
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


def _input_creation_line(name: str, info: dict) -> str:
    """Generate input creation code that preserves shape, stride, dtype, device.

    Uses ``.as_strided()`` when the input is non-contiguous so the kernel
    receives exactly the same memory layout as in production.
    """
    shape = info.get("shape", [])
    stride = info.get("stride", [])
    dtype = info.get("dtype", "torch.float32")
    device = info.get("device", "cuda")
    if "cuda" in device:
        device = "cuda"

    is_int = "int" in dtype
    is_bool = "bool" in dtype

    if stride and shape:
        # Allocate minimum storage that covers the strided view
        storage_size = sum(
            s * (d - 1) for s, d in zip(stride, shape) if d > 1
        ) + 1
        if is_int:
            return (
                f"    {name} = torch.randint(0, 100, ({storage_size},), "
                f"dtype={dtype}, device='{device}').as_strided({shape}, {stride})"
            )
        if is_bool:
            return (
                f"    {name} = torch.randint(0, 2, ({storage_size},), "
                f"dtype={dtype}, device='{device}').as_strided({shape}, {stride})"
            )
        return (
            f"    {name} = torch.randn(({storage_size},), "
            f"dtype={dtype}, device='{device}').as_strided({shape}, {stride})"
        )
    # Contiguous
    if is_int:
        return f"    {name} = torch.randint(0, 100, {shape}, dtype={dtype}, device='{device}')"
    if is_bool:
        return f"    {name} = torch.randint(0, 2, {shape}, dtype={dtype}, device='{device}')"
    return f"    {name} = torch.randn({shape}, dtype={dtype}, device='{device}')"


def _format_subgraph_as_model(subgraph_gm: torch.fx.GraphModule) -> str:
    """Format the extracted subgraph as ``class Model(nn.Module)``.

    Uses ``GraphModule.print_readable`` for a faithful repro (same
    flags as the tlparse graph dump: include stride, include device,
    expanded def, autograd_backward meta), then renames the class.
    """
    code = subgraph_gm.print_readable(
        print_output=False,
        include_stride=True,
        include_device=True,
        expanded_def=True,
        additional_meta=["autograd_backward"],
    )
    code = code.replace("class GraphModule(", "class Model(", 1)
    return code


def _write_problem(
    region: Region,
    output_dir: Path,
    count: int,
) -> None:
    """Write a KernelAgent-compatible problem.py.

    When ``region.subgraph_gm`` is set (InductorRegionExtractor),
    serializes the subgraph via ``print_readable`` and uses exact
    strides for inputs. Otherwise falls back to building code from
    ``region.nodes`` directly.
    """
    if region.subgraph_gm is not None:
        _write_problem_from_subgraph(region, output_dir, count)
        return
    _write_problem_from_orig(region, output_dir, count)


def _write_problem_from_subgraph(
    region: Region,
    output_dir: Path,
    count: int,
) -> None:
    subgraph_gm = region.subgraph_gm
    info_list = region.inputs_info or []

    model_code = _format_subgraph_as_model(subgraph_gm)

    # Get placeholder names in declaration order
    ph_names = [
        n.name for n in subgraph_gm.graph.nodes if n.op == "placeholder"
    ]
    info_by_name = {info["name"]: info for info in info_list}
    input_lines = [
        _input_creation_line(name, info_by_name.get(name, {}))
        for name in ph_names
    ]

    compute_ops = [
        str(n.target).split(".")[-2]
        for n in subgraph_gm.graph.nodes
        if n.op == "call_function" and str(n.target) not in _METADATA_OPS
    ]
    n_ops = sum(1 for n in subgraph_gm.graph.nodes if n.op == "call_function")
    n_outputs = 1
    out_node = next(n for n in subgraph_gm.graph.nodes if n.op == "output")
    outs = out_node.args[0]
    if isinstance(outs, (tuple, list)):
        n_outputs = len(outs)

    desc = (
        f"# Fused region ({region.norm_fqn}): "
        f"{' -> '.join(compute_ops) if compute_ops else 'reshape chain'}\n"
        f"# Instances: {count}. Ops: {n_ops}, compute: {len(compute_ops)}, "
        f"outputs: {n_outputs}.\n"
    )

    problem = desc + f"""
import torch
import torch.nn as nn

{model_code}

def get_inputs():
{chr(10).join(input_lines) if input_lines else '    pass'}
    return [{', '.join(ph_names)}]

def get_init_inputs():
    return []
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "problem.py").write_text(problem)


def _write_problem_from_orig(
    region: Region,
    output_dir: Path,
    count: int,
) -> None:
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

    num_outputs = len(output_nodes)
    if num_outputs == 1:
        body_lines.append(f"        return {var_map[output_nodes[0].name]}")
        ret_type = "torch.Tensor"
    else:
        out_vars = ", ".join(var_map[n.name] for n in output_nodes)
        body_lines.append(f"        return ({out_vars})")
        ret_type = f"tuple[{', '.join(['torch.Tensor'] * num_outputs)}]"

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
    desc = f"# Fused region ({region.norm_fqn}): {' -> '.join(compute_ops) if compute_ops else 'reshape chain'}\n"
    desc += f"# Instances: {count}. Ops: {len(nodes)}, compute: {len(compute_ops)}, outputs: {num_outputs}.\n"

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

    # Find a valid insertion point:
    #  - AFTER all external inputs are defined
    #  - BEFORE the earliest external user of any output_node
    # If no valid point exists, raise to skip this region.
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    region_set = set(nodes)

    latest_input_pos = max(
        (node_order.get(inp, -1) for inp in external_inputs),
        default=-1,
    )
    # Earliest external user position across all output_nodes
    earliest_user_pos = float("inf")
    for out_node in output_nodes:
        for u in out_node.users:
            if u in region_set:
                continue
            u_pos = node_order.get(u, -1)
            if u_pos >= 0 and u_pos < earliest_user_pos:
                earliest_user_pos = u_pos

    if latest_input_pos >= earliest_user_pos:
        raise RuntimeError(
            f"Cannot insert kernel: latest input at pos {latest_input_pos} >= "
            f"earliest external user at pos {earliest_user_pos}"
        )

    if external_inputs and latest_input_pos >= 0:
        latest_input_node = max(external_inputs, key=lambda n: node_order.get(n, -1))
        with graph.inserting_after(latest_input_node):
            call_node = graph.call_function(
                kernel_fn, args=tuple(external_inputs)
            )
    else:
        with graph.inserting_before(nodes[0]):
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
        last = call_node
        for out_idx, out_node in enumerate(output_nodes):
            with graph.inserting_after(last):
                gi = graph.call_function(
                    operator.getitem, args=(call_node, out_idx)
                )
                gi.meta = out_node.meta.copy()
            out_node.replace_all_uses_with(gi)
            last = gi

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
    if extractor_cls is InductorRegionExtractor:
        work_gm, all_regions = extractor_cls(example_inputs=example_inputs).extract(gm)
    else:
        work_gm, all_regions = extractor_cls().extract(gm)

    unique_regions, hash_counts = _filter_regions(
        all_regions, min_ops=min_ops, min_compute_ops=min_compute_ops,
    )

    if not unique_regions:
        logger.info("Fused kernel pass: no fusible regions found")
        return work_gm

    valid_hashes = {_compute_region_hash(r) for r in unique_regions}

    # Collect ALL instances (not just unique) for replacement
    node_order = {n: i for i, n in enumerate(work_gm.graph.nodes)}
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
    # Track live nodes — previous replacements may invalidate later regions.
    replaced = 0
    skipped_invalid = 0
    live_nodes: set[str] = {n.name for n in work_gm.graph.nodes}
    for pos, h, r in instances:
        if h not in hash_best:
            continue
        if not all(n.name in live_nodes for n in r.nodes):
            skipped_invalid += 1
            continue
        if not all(inp.name in live_nodes for inp in r.external_inputs):
            skipped_invalid += 1
            continue
        best_fn, _ = hash_best[h]
        try:
            _replace_region_with_kernel(work_gm, r, best_fn)
        except RuntimeError as e:
            # Topological-order conflict: external_inputs and external
            # users straddle the region.  Skip safely.
            logger.debug(f"Skipping {h}: {e}")
            skipped_invalid += 1
            continue
        live_nodes = {n.name for n in work_gm.graph.nodes}
        replaced += 1

    if replaced > 0:
        work_gm.graph.lint()
        work_gm.recompile()

    if skipped_invalid:
        logger.info(
            f"Fused kernel pass: skipped {skipped_invalid} regions "
            f"(nodes invalidated by prior replacements)"
        )

    backend_summary = ", ".join(f"{k}={v}" for k, v in sorted(backend_counts.items()))
    logger.info(
        f"Fused kernel pass: {replaced} replacements "
        f"({len(hash_best)} patterns with kernels, "
        f"{len(seen_hashes) - len(hash_best)} eager-only skipped), "
        f"backends: [{backend_summary}], "
        f"{problems_written} new problems written to {kdir}"
    )

    return work_gm
