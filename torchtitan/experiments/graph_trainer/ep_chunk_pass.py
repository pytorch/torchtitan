# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chunk selected graph_trainer module regions for scheduling experiments.

This file implements the internal chunking transform used by the public
``ep_overlap`` pass. The contract is:

```
chunking = correctness-preserving dataflow transform
ep_overlap = scheduling transform over the chunked graph
```

Pass order is part of the contract. No-op/view cleanup runs before chunking, so
the chunk pass sees normalized tensor plumbing. Later memory and communication
passes run after chunking, so they operate on the same chunked structure that
eager chunking would expose to the tracer. EP overlap scheduling runs after
bucketing and only reorders ready chunk-local body nodes; it does not change
tensor values.

Chunk dimensions are explicit. Before tracing, TorchTitan marks the selected
input dimension with an unbacked ``SymInt`` and stores chunk metadata describing
the logical mode (``batch`` or ``seq``) plus the concrete full-size hint.
graph_trainer traces are config-specialized, so the mark also records an
equality specialization for the current extent. Unlike a singleton min/max
range, that preserves symbolic expressions such as ``u0`` and ``batch * u0``
for reshape/view metadata while documenting the exact runtime extent expected by
distributed sharding code. The graph pass imports and propagates the metadata;
it must not infer batch/sequence from tensor rank. A tensor live-in is
chunkable only if it is not static model state, carries the selected-mode
metadata, has an even full-size hint, and can produce coherent half-size
symbolic metadata.

Core terms:
- Region: one concrete forward or backward module root matched from the user
  FQN pattern, such as ``layers.0`` or ``layers.0.moe``. Forward and backward
  regions are planned independently even when they share a root.
- Search space: same-direction local graph slice for the root. It is built by
  reverse traversal from matched module nodes and stops at placeholders, static
  model state, already chunked nodes, nodes outside the root, saved forward
  activations for backward planning, adjacent fqn-less backward accumulators,
  and opposite-direction nodes.
- Copied body: descendant closure from chunkable and per-chunk live-ins inside
  the search space. Only this body is duplicated per chunk.
- Live-in: any value consumed by the copied body but produced outside it.
  Chunkable live-ins are split; per-chunk live-ins come from earlier per-chunk
  live-outs; shared full live-ins such as parameters, gathered/casted params,
  constants, and full activations are passed unchanged to both chunks.
- Live-out: any value produced by the copied body and consumed outside it.
  Live-outs with only chunked consumers are kept as per-chunk values. A
  value also counts as consumed by a chunked region when it reaches that
  region's copied body through that same region's local search space.
- Per-chunk value table: semantic state ``full_value -> (chunk0, chunk1)``.
  It is required for non-invertible values such as MoE token counts: a summed
  full count may be correct for a full consumer, but it cannot recover the
  original per-chunk counts for later chunked consumers.

Live-out materialization rules:
- Chunked consumers only: do not materialize a full value; rewrite consumers to
  matching per-chunk values.
- Non-chunked consumer with the selected chunk dimension: materialize with
  ``torch.cat([chunk0, chunk1], dim=annotated_dim)`` and validate metadata
  against the original full live-out.
- Non-chunked consumer without the selected chunk dimension: materialize with
  ``add`` only for proven accumulations, e.g. parameter gradients, graph-output
  gradients, reduce-scatter/all-reduce gradient consumers, ``histc``/``bincount``
  token counts, reductions over the chunk dimension, or reviewed additive
  accumulation paths. Unsupported reductions such as max/argmax/nonlinear
  reductions must error rather than guess.
- Symbolic scalar live-outs follow the same rule: per-chunk for chunked
  consumers, reconstruct full scalars only when proven, otherwise error.

Collectives are classified by their dataflow relation to chunked values, not by
mesh name. Chunk-local collectives that consume/produce chunk-local activation
values may be duplicated inside the copied body. Shared full collectives such as
parameter all-gather/wait/cast chains are shared live-ins and must not be
duplicated. Full-gradient reduce-scatter/all-reduce collectives are non-chunked
consumers of accumulated backward live-outs and must not be duplicated.

Canonical body order is also contractual. Chunking rewrites the original body
nodes in place as the canonical first chunk, then copies the peer chunk as one
contiguous block after the final first-chunk body node. Forward uses chunk0 then
chunk1; backward uses chunk1 then chunk0. This keeps chunking a pure dataflow
transform, while EP scheduling remains the only pass allowed to interleave
dispatch/combine phases.

Numerical correctness is defined against eager chunking, not unchunked eager.
For the selected module and dimension, eager reference execution splits inputs
into two chunks, runs the original module on each chunk, and cats outputs back
on the same dimension. Because autograd sees the eager wrapper, backward is
chunked by construction. The graph pass is expected to be bitwise equivalent to
that eager-chunked reference under the same seed, parallelism, and deterministic
settings.

TODO: The dataflow contract can be generalized beyond two chunks, but v1 keeps
exactly two because EP overlap scheduling is pairwise. N-way chunks would need
N-way per-chunk value tracking, materialization, scalar handling, eager
references, and launch/wait scheduling.
"""

from __future__ import annotations

import fnmatch
import operator
from collections import defaultdict
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Literal

import torch
import torch.fx as fx
from torch.utils._pytree import tree_leaves

from torchtitan.experiments.graph_trainer.common_utils import (
    _dynamic_dim_symbols,
    _earliest_node,
    _free_symbols,
    _get_module_fqn,
    _is_backward_node,
    _is_module_fqn_inside_root,
    _ordered_nodes,
    _tensor_meta,
)
from torchtitan.experiments.graph_trainer.configs import validate_ep_overlap_config
from torchtitan.experiments.graph_trainer.registry import (
    register_trace_input_preparer,
)
from torchtitan.tools.logging import logger


aten = torch.ops.aten

ChunkMode = Literal["batch", "seq"]
_CHUNK_DIMS_ATTR = "_torchtitan_chunk_dims"
_CHUNK_DIMS_META = "torchtitan_chunk_dims"


@dataclass(frozen=True)
class _Region:
    root_fqn: str
    is_backward: bool
    nodes: tuple[fx.Node, ...]


class _ChunkNodeRole(Enum):
    SPLIT_BOUNDARY = auto()
    CHUNK_INPUT = auto()
    BODY = auto()
    MATERIALIZATION = auto()


@dataclass(frozen=True)
class _RegionPlan:
    region: _Region
    search_space: frozenset[fx.Node]
    chunkable_live_ins: frozenset[fx.Node]
    per_chunk_live_ins: frozenset[fx.Node]
    region_nodes: frozenset[fx.Node]
    region_nodes_tuple: tuple[fx.Node, ...]
    region_live_ins: frozenset[fx.Node]
    live_out_users: dict[fx.Node, tuple[fx.Node, ...]]


def _is_excluded_node(node: fx.Node) -> bool:
    """Nodes that must not be duplicated by chunking."""
    return node.op != "call_function"


def _is_reverse_closure_boundary(
    node: fx.Node, *, root_fqn: str, is_backward: bool
) -> bool:
    # Step 2.1: reverse closure only discovers a local search space. Boundary
    # nodes may become live-ins, but they must not be duplicated.
    if node.op == "output":
        raise ValueError(
            f"Chunk pass unexpectedly reached graph output while building "
            f"reverse closure for region {root_fqn!r}."
        )
    if node.op == "placeholder":
        return True
    if "chunked_region_fqn" in node.meta:
        return True
    if any(user.op == "output" for user in node.users) and not _fqn(node):
        # Step 2.2: backward closures can reach the graph-owned loss through
        # autograd's ones_like(loss) seed. Keep that scalar outside the region
        # so loss reduction order remains unchanged.
        return True
    if _is_excluded_node(node):
        return True
    fqn = _fqn(node)
    if fqn and not _is_module_fqn_inside_root(fqn, root_fqn):
        return True
    if is_backward and not fqn:
        user_fqns = [_fqn(user) for user in node.users if _fqn(user)]
        if user_fqns and not any(
            _is_module_fqn_inside_root(user_fqn, root_fqn) for user_fqn in user_fqns
        ):
            # Step 2.5: backward graphs contain fqn-less gradient accumulation
            # nodes between module regions. The accumulator belongs to the
            # region that has an immediate module-fqn user; earlier regions see
            # it as an incoming gradient live-in.
            return True
    if (
        is_backward
        and "autograd_backward" not in node.meta
        and any(not _is_backward_node(user) for user in node.users)
    ):
        # Step 2.3: saved forward activations can have module FQN metadata but
        # no backward marker. If they also feed forward users, they are shared
        # boundary values for backward, not backward compute to duplicate.
        return True
    if "autograd_backward" in node.meta and _is_backward_node(node) != is_backward:
        # Step 2.3: backward regions normally consume saved forward activations;
        # those values are live-ins. A forward region depending on backward
        # nodes is structurally invalid for this transform.
        if is_backward:
            return True
        direction = "backward" if is_backward else "forward"
        raise ValueError(
            f"Chunk pass crossed into the opposite graph direction while "
            f"building {direction} reverse closure for region {root_fqn!r}: "
            f"{node.name}."
        )
    return False


def _fqn(node: fx.Node) -> str:
    return _get_module_fqn(node)


def _pattern_root(pattern: str, fqn: str) -> str | None:
    """Return the concrete region root if ``pattern`` matches ``fqn``.

    Patterns are matched segment-by-segment so ``layers.*`` maps
    ``layers.0.attention.wq`` to the concrete root ``layers.0`` instead of
    making one giant region containing every layer.
    """
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    if len(fqn_parts) < len(pattern_parts):
        return None
    for pattern_part, fqn_part in zip(pattern_parts, fqn_parts):
        if not fnmatch.fnmatchcase(fqn_part, pattern_part):
            return None
    return ".".join(fqn_parts[: len(pattern_parts)])


def _reverse_closure_from_boundary_nodes(
    boundary_nodes: list[fx.Node], *, root_fqn: str, is_backward: bool
) -> set[fx.Node]:
    """Collect same-direction dependencies used to discover chunkable sources."""
    closure: set[fx.Node] = set()
    stack = list(boundary_nodes)
    order = {n: i for i, n in enumerate(boundary_nodes[0].graph.nodes)}
    first_boundary_idx = min(order[node] for node in boundary_nodes)

    while stack:
        node = stack.pop()
        if (
            is_backward
            and "autograd_backward" not in node.meta
            and order[node] < first_boundary_idx
        ):
            # Step 2.4: nodes before the first tagged backward node are saved
            # forward values from the original forward pass. Backward chunking
            # can consume them as live-ins but must not duplicate the same
            # original forward node in the backward region.
            continue
        if node in closure or _is_reverse_closure_boundary(
            node, root_fqn=root_fqn, is_backward=is_backward
        ):
            continue
        closure.add(node)
        stack.extend(node.all_input_nodes)

    return closure


def _forward_closure_from_sources(
    sources: set[fx.Node],
    *,
    search_space: set[fx.Node],
) -> set[fx.Node]:
    """Collect nodes in ``search_space`` that data-depend on chunked sources."""
    closure: set[fx.Node] = set()
    stack = [
        user for source in sources for user in source.users if user in search_space
    ]

    while stack:
        node = stack.pop()
        if node in closure:
            continue
        closure.add(node)
        stack.extend(user for user in node.users if user in search_space)

    return closure


def _is_symbolic_shape_scalar(node: fx.Node) -> bool:
    return _tensor_meta(node) is None and bool(_free_symbols(node.meta.get("val")))


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _ep_region(node: fx.Node) -> str | None:
    ep = _custom_meta(node).get("EP")
    return ep if ep in ("dispatch", "combine") else None


def _find_regions(gm: fx.GraphModule, patterns: list[str]) -> list[_Region]:
    matched_boundaries: dict[tuple[str, bool], list[fx.Node]] = defaultdict(list)

    for node in gm.graph.nodes:
        if _is_excluded_node(node):
            continue
        fqn = _fqn(node)
        if not fqn:
            continue
        roots = [root for p in patterns if (root := _pattern_root(p, fqn))]
        if not roots:
            continue
        if len(set(roots)) > 1:
            raise ValueError(
                f"Chunk pass patterns match node {node.name!r} ambiguously: {roots}"
            )
        root = roots[0]
        # Step 1: FQN matching identifies user-selected anchors only. The copied
        # region is computed later from activation live-ins, which avoids
        # duplicating parameter-only prep such as weight all-gathers or casts.
        matched_boundaries[(root, _is_backward_node(node))].append(node)

    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    regions = [
        _Region(root, is_backward, tuple(sorted(boundary_nodes, key=order.__getitem__)))
        for (root, is_backward), boundary_nodes in matched_boundaries.items()
    ]
    return sorted(regions, key=lambda r: min(order[n] for n in r.nodes))


def _annotated_chunk_dim(node: fx.Node, mode: ChunkMode) -> int | None:
    chunk_dims = node.meta.get(_CHUNK_DIMS_META)
    if not isinstance(chunk_dims, dict):
        return None
    spec = chunk_dims.get(mode)
    if isinstance(spec, dict) and isinstance(spec.get("dim"), int):
        return spec["dim"]
    return None


def _annotated_chunk_hint(node: fx.Node, mode: ChunkMode) -> int | None:
    chunk_dims = node.meta.get(_CHUNK_DIMS_META)
    if not isinstance(chunk_dims, dict):
        return None
    spec = chunk_dims.get(mode)
    if isinstance(spec, dict) and isinstance(spec.get("hint"), int):
        return spec["hint"]
    return None


def _range_upper_bound(value: object) -> int | None:
    sym_node = getattr(value, "node", None)
    hint = getattr(sym_node, "hint", None)
    if hint is not None:
        try:
            return int(hint)
        except (TypeError, ValueError, OverflowError):
            pass
    shape_env = getattr(sym_node, "shape_env", None)
    expr = getattr(sym_node, "expr", None)
    if shape_env is None or expr is None:
        return None
    value_range = getattr(shape_env, "var_to_range", {}).get(expr)
    upper = getattr(value_range, "upper", None)
    try:
        return int(upper) if upper is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _dynamic_dims_with_hints(val: torch.Tensor) -> list[tuple[int, int]]:
    dynamic_dims: list[tuple[int, int]] = []
    for dim, size in enumerate(val.shape):
        if not _free_symbols(size):
            continue
        hint = _range_upper_bound(size)
        if hint is None:
            raise ValueError(
                f"Chunk pass could not infer a hint for dynamic dimension {dim} "
                f"of shape {tuple(val.shape)}."
            )
        dynamic_dims.append((dim, hint))
    return dynamic_dims


def import_chunk_dim_metadata_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
) -> fx.GraphModule:
    """Import chunk dimension metadata onto placeholders before chunk planning.

    The tracer stays pass-agnostic: this pass reads explicit TorchTitan metadata
    when present, otherwise it derives the selected chunk dimension from the
    single dynamic placeholder dimension created by ``mark_chunk_dynamic_dims``.
    """
    if mode not in ("batch", "seq"):
        raise ValueError(f"Unknown chunk mode: {mode!r}")

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    for idx, node in enumerate(placeholders):
        chunk_dims = dict(node.meta.get(_CHUNK_DIMS_META, {}))
        if example_inputs is not None and idx < len(example_inputs):
            explicit = getattr(example_inputs[idx], _CHUNK_DIMS_ATTR, None)
            if explicit is not None:
                chunk_dims.update(dict(explicit))

        if mode not in chunk_dims:
            val = _tensor_meta(node)
            if val is None:
                continue
            dynamic_dims = _dynamic_dims_with_hints(val)
            if not dynamic_dims:
                continue
            if len(dynamic_dims) != 1:
                raise ValueError(
                    f"Chunk pass expected one dynamic placeholder dimension for "
                    f"{node.name}, found {dynamic_dims}."
                )
            dim, hint = dynamic_dims[0]
            chunk_dims[mode] = {"dim": dim, "hint": hint}

        node.meta[_CHUNK_DIMS_META] = chunk_dims
    return gm


def _static_placeholders(gm: fx.GraphModule, num_static_inputs: int) -> set[fx.Node]:
    return {
        node
        for idx, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")
        if idx < num_static_inputs
    }


def _static_derived_nodes(
    gm: fx.GraphModule, static_placeholders: set[fx.Node]
) -> set[fx.Node]:
    static_nodes = set(static_placeholders)
    for node in gm.graph.nodes:
        if node in static_nodes:
            continue
        inputs = node.all_input_nodes
        if inputs and all(inp in static_nodes for inp in inputs):
            static_nodes.add(node)
    return static_nodes


def _expr(value: object) -> object:
    return getattr(getattr(value, "node", None), "expr", value)


def _expr_matches(lhs: object, rhs: object) -> bool:
    lhs_expr = _expr(lhs)
    rhs_expr = _expr(rhs)
    if lhs_expr == rhs_expr:
        return True
    try:
        return bool((lhs_expr - rhs_expr) == 0)
    except (TypeError, ValueError):
        return False


def _chunk_size_for_live_in(
    node: fx.Node,
    mode: ChunkMode,
    val: torch.Tensor,
    dim: int,
    *,
    dim_chunk_sizes: list[tuple[object, object]] | None = None,
) -> int | torch.SymInt:
    hint = _annotated_chunk_hint(node, mode)
    if hint is not None:
        if hint % 2 != 0:
            raise ValueError(
                f"Cannot split annotated {mode} dimension with hint {hint} "
                "into two equal chunks."
            )
    full_dim = val.shape[dim]
    for candidate_full_dim, candidate_chunk_dim in dim_chunk_sizes or ():
        if _expr_matches(full_dim, candidate_full_dim):
            return candidate_chunk_dim
    if isinstance(full_dim, torch.SymInt):
        return full_dim // 2
    full_size = int(full_dim)
    if full_size % 2 != 0:
        raise ValueError(
            f"Cannot split dimension {dim} of shape {tuple(val.shape)} into two "
            "equal chunks."
        )
    return full_size // 2


def _chunk_meta(
    val: torch.Tensor,
    dim: int,
    chunk_size: int | torch.SymInt,
    *,
    preserve_stride: bool,
) -> torch.Tensor:
    shape = list(val.shape)
    shape[dim] = chunk_size
    if preserve_stride:
        # torch.split returns views, so a chunk along dim 0 keeps the original
        # stride order. Preserving this here is required for regional Inductor
        # input assertions on transposed attention tensors.
        return val.new_empty_strided(shape, val.stride())
    return val.new_empty(shape)


def _propagate_chunk_dim_metadata(gm: fx.GraphModule) -> None:
    symbol_specs: dict[object, dict[ChunkMode, int]] = defaultdict(dict)
    for node in gm.graph.nodes:
        val = _tensor_meta(node)
        if val is None:
            continue
        for mode in ("batch", "seq"):
            dim = _annotated_chunk_dim(node, mode)
            hint = _annotated_chunk_hint(node, mode)
            if dim is None or hint is None:
                continue
            for symbol in _dynamic_dim_symbols(val, dim):
                if _expr_matches(val.shape[dim], symbol):
                    symbol_specs[symbol][mode] = hint

    if not symbol_specs:
        return

    for node in gm.graph.nodes:
        val = _tensor_meta(node)
        if val is None:
            continue
        inferred: dict[ChunkMode, tuple[int, int] | None] = {}
        for dim in range(val.dim()):
            for symbol in _dynamic_dim_symbols(val, dim):
                for mode, hint in symbol_specs.get(symbol, {}).items():
                    spec = inferred.get(mode)
                    if spec is None and mode in inferred:
                        continue
                    dim_hint = _evaluate_with_hints(val.shape[dim], {symbol: hint})
                    if dim_hint is None:
                        continue
                    new_spec = (dim, dim_hint)
                    inferred[mode] = new_spec if spec in (None, new_spec) else None

        chunk_dims = dict(node.meta.get(_CHUNK_DIMS_META, {}))
        for mode, spec in inferred.items():
            if spec is None:
                continue
            dim, hint = spec
            existing = chunk_dims.get(mode)
            if isinstance(existing, dict):
                if existing.get("dim") != dim:
                    continue
            else:
                chunk_dims[mode] = {"dim": dim, "hint": hint}
        if chunk_dims:
            node.meta[_CHUNK_DIMS_META] = chunk_dims


def _collect_symbol_full_sizes(
    gm: fx.GraphModule, mode: ChunkMode
) -> dict[object, int]:
    symbol_full_sizes: dict[object, int] = {}
    for node in gm.graph.nodes:
        val = _tensor_meta(node)
        if val is None:
            continue
        dim = _annotated_chunk_dim(node, mode)
        hint = _annotated_chunk_hint(node, mode)
        if dim is None or hint is None:
            continue
        for symbol in _dynamic_dim_symbols(val, dim):
            if _expr_matches(val.shape[dim], symbol):
                existing = symbol_full_sizes.get(symbol)
                if existing is not None and existing != hint:
                    raise ValueError(
                        f"Chunk pass found conflicting hints for symbol {symbol}: "
                        f"{existing} and {hint}."
                    )
                symbol_full_sizes[symbol] = hint
    return symbol_full_sizes


def _make_symbol_half(
    symbol: object, *, like_value: object, symbol_full_sizes: dict[object, int]
) -> object:
    shape_env = getattr(getattr(like_value, "node", None), "shape_env", None)
    chunk_expr = symbol // 2
    hint = symbol_full_sizes[symbol] // 2
    if shape_env is None:
        return chunk_expr
    return shape_env.create_symintnode(chunk_expr, hint=hint)


def _replace_chunk_symbols(
    value: object,
    *,
    dim_chunk_sizes: list[tuple[object, object]],
    symbol_chunk_values: dict[object, object],
    symbol_full_sizes: dict[object, int],
) -> object:
    for full_dim, chunk_dim in dim_chunk_sizes:
        if _expr_matches(value, full_dim):
            return chunk_dim

    symbols = _free_symbols(value)
    if not (symbols & set(symbol_chunk_values)):
        return value

    expr = _expr(value)
    shape_env = getattr(getattr(value, "node", None), "shape_env", None)
    replacements = {
        symbol: _expr(symbol_chunk_values[symbol])
        for symbol in symbols
        if symbol in symbol_chunk_values
    }
    try:
        chunk_expr = expr.subs(replacements)
    except AttributeError:
        return value

    if shape_env is None:
        evaluated = _evaluate_with_hints(chunk_expr, symbol_full_sizes)
        return evaluated if evaluated is not None else value

    hint = _evaluate_with_hints(chunk_expr, symbol_full_sizes)
    return shape_env.create_symintnode(chunk_expr, hint=hint)


def _chunked_meta_from_original(
    original: fx.Node,
    *,
    dim_chunk_sizes: list[tuple[object, object]],
    symbol_chunk_values: dict[object, object],
    symbol_full_sizes: dict[object, int],
) -> torch.Tensor | None:
    val = _tensor_meta(original)
    if val is None:
        return None
    shape = []
    changed = False
    for dim in val.shape:
        chunk_dim = _replace_chunk_symbols(
            dim,
            dim_chunk_sizes=dim_chunk_sizes,
            symbol_chunk_values=symbol_chunk_values,
            symbol_full_sizes=symbol_full_sizes,
        )
        changed = changed or chunk_dim is not dim
        shape.append(chunk_dim)
    if not changed:
        return val
    return val.new_empty(shape)


def _chunk_dim_for_live_in(node: fx.Node, mode: ChunkMode) -> int:
    val = _tensor_meta(node)
    if val is None:
        raise ValueError(f"Chunk live-in {node.name} has no tensor metadata.")
    annotated_dim = _annotated_chunk_dim(node, mode)
    if annotated_dim is not None:
        return annotated_dim

    raise ValueError(
        f"Chunk pass requires an annotated {mode} dimension for live-in "
        f"{node.name} with shape {tuple(val.shape)}."
    )


def _chunk_symbol_dims(
    node: fx.Node, chunk_symbols: frozenset[object], val: torch.Tensor | None = None
) -> list[int]:
    val = _tensor_meta(node) if val is None else val
    if val is None:
        return []
    return [
        dim
        for dim in range(val.dim())
        if _dynamic_dim_symbols(val, dim) & chunk_symbols
    ]


def _combine_kind_and_dim(
    node: fx.Node,
    mode: ChunkMode,
    chunk_symbols: frozenset[object],
    val: torch.Tensor | None = None,
) -> tuple[Literal["cat", "add"], int | None]:
    matching_dims = _chunk_symbol_dims(node, chunk_symbols, val)
    if len(matching_dims) == 1:
        return "cat", matching_dims[0]
    if not matching_dims:
        annotated_dim = _annotated_chunk_dim(node, mode)
        if annotated_dim is not None:
            return "cat", annotated_dim
        return "add", None
    raise ValueError(
        f"Chunk pass found multiple chunk-symbol dims for live-out {node.name}: "
        f"{matching_dims}"
    )


def mark_chunk_dynamic_dims(
    tensor: torch.Tensor,
    *,
    mode: ChunkMode,
) -> None:
    """Mark graph_trainer's main input dimensions used by chunk passes.

    The graph pass uses explicit TorchTitan metadata to distinguish batch and
    sequence dims without relying on rank.
    """
    from torch._dynamo.decorators import mark_unbacked

    dims: dict[str, dict[str, int]] = dict(getattr(tensor, _CHUNK_DIMS_ATTR, {}))

    def mark(dim: int) -> None:
        if tensor.dim() <= dim:
            raise ValueError(
                f"Cannot mark {mode} chunk dim {dim} for input shape "
                f"{tuple(tensor.shape)}."
            )
        hint = int(tensor.shape[dim])
        mark_unbacked(
            tensor,
            dim,
            hint_override=hint,
            min=2,
            max=hint,
            # Keep the dimension symbolic for chunk metadata, while recording
            # that this config-specialized trace is for the concrete extent.
            specialize_on=[lambda extent, hint=hint: extent == hint],
            shape_id=f"torchtitan_chunk_{mode}",
        )
        dims[mode] = {"dim": dim, "hint": hint}

    if mode == "batch":
        mark(0)
    elif mode == "seq":
        mark(1)
    else:
        raise ValueError(f"Unknown chunk mode: {mode!r}")

    setattr(tensor, _CHUNK_DIMS_ATTR, dims)


@register_trace_input_preparer("ep_overlap")
def prepare_ep_overlap_trace_inputs(
    compile_config: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Apply EP-overlap input annotations before make_fx tracing.

    ``minimal_fx_tracer`` calls this hook immediately before tracing so
    overlap-specific trace preparation stays close to the internal chunking
    implementation. The graph pass needs these annotations before tracing
    because make_fx must see a symbolic batch/sequence dimension.
    """
    if "ep_overlap" not in getattr(compile_config, "passes", []):
        return
    mode, chunk_strategy, _module_fqn = validate_ep_overlap_config(compile_config)
    if chunk_strategy == "eager":
        return
    if not args or not isinstance(args[0], torch.Tensor):
        raise ValueError(
            "ep_overlap tracing expects the first user input to be a Tensor"
        )
    if mode == "batch":
        dim = 0
    elif mode == "seq":
        dim = 1
    else:
        raise ValueError(f"Unknown ep_overlap_chunk_dim: {mode!r}")
    main_input = args[0]
    hint = int(main_input.shape[dim])

    # TODO: Replace this shape-based input annotation with an explicit trace
    # input schema. Today we mark every tensor input whose selected dimension
    # matches the main input extent. That is practical for the current train
    # step inputs, where batch/sequence-shaped tensors are the values that can
    # flow into chunked regions. It is still broader than the real contract: an
    # unrelated tensor with the same extent could receive chunk metadata. That
    # tensor is only split if it later becomes a live-in to a selected region,
    # but a schema would be cleaner and more precise, e.g. "arg0 dim 0 is
    # batch", "arg0 dim 1 is sequence", and "this input is never chunkable".
    def mark_if_matching(value: object) -> None:
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > dim
            and int(value.shape[dim]) == hint
        ):
            mark_chunk_dynamic_dims(value, mode=mode)

    for value in [*tree_leaves(args), *tree_leaves(kwargs)]:
        mark_if_matching(value)


def _is_chunkable_live_in(
    node: fx.Node,
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
) -> bool:
    if node in static_nodes:
        return False
    val = _tensor_meta(node)
    if val is None:
        return False
    # Step 7: missing chunk metadata means this boundary value is not an activation
    # source for the selected mode. Invalid annotated sizes should still raise
    # so user configuration mistakes do not become silent no-ops.
    try:
        dim = _chunk_dim_for_live_in(node, mode)
    except ValueError:
        return False
    _chunk_size_for_live_in(node, mode, val, dim)
    return True


def _set_direction_meta(node: fx.Node, *, is_backward: bool) -> None:
    if is_backward:
        node.meta["autograd_backward"] = True
    else:
        node.meta.pop("autograd_backward", None)


def _set_chunk_region_meta(
    node: fx.Node,
    *,
    region: _Region,
    chunk_id: int | None = None,
    is_backward: bool | None = None,
    role: _ChunkNodeRole | None = None,
) -> None:
    node_is_backward = region.is_backward if is_backward is None else is_backward
    _set_direction_meta(node, is_backward=node_is_backward)
    node.meta["chunked_region_fqn"] = region.root_fqn
    node.meta["chunked_region_is_backward"] = node_is_backward
    if role is not None:
        node.meta["chunked_region_role"] = role.name.lower()
    if chunk_id is not None:
        node.meta["chunk_id"] = chunk_id


def _rename(node: fx.Node, candidate: str) -> None:
    node._rename(candidate)


def _infer_node_val(node: fx.Node) -> None:
    def meta_arg(arg: object) -> object:
        if isinstance(arg, fx.Node) and "val" in arg.meta:
            return arg.meta["val"]
        return arg

    try:
        args = torch.fx.map_arg(node.args, meta_arg)
        kwargs = torch.fx.map_arg(node.kwargs, meta_arg)
        node.meta["val"] = node.target(*args, **kwargs)
    except Exception:
        pass


def _dim_matches(
    original_dim: object,
    materialized_dim: object,
    symbol_full_sizes: dict[object, int],
) -> bool:
    original_symbols = _free_symbols(original_dim)
    if original_symbols:
        original_size = _evaluate_with_hints(original_dim, symbol_full_sizes)
        materialized_size = _evaluate_with_hints(materialized_dim, symbol_full_sizes)
        if original_size is not None and materialized_size is not None:
            return original_size == materialized_size
        # Unknown symbolic dims are intentionally left symbolic; avoid forcing a
        # guard just to validate metadata.
        return any(symbol not in symbol_full_sizes for symbol in original_symbols)

    if _free_symbols(materialized_dim):
        original_size = _evaluate_with_hints(original_dim, symbol_full_sizes)
        materialized_size = _evaluate_with_hints(materialized_dim, symbol_full_sizes)
        if original_size is not None and materialized_size is not None:
            return original_size == materialized_size
        return True

    return int(original_dim) == int(materialized_dim)


def _evaluate_with_hints(
    value: object, symbol_full_sizes: dict[object, int]
) -> int | None:
    symbols = _free_symbols(value)
    if not symbols:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if any(symbol not in symbol_full_sizes for symbol in symbols):
        return None

    expr = getattr(getattr(value, "node", None), "expr", value)
    try:
        expr = expr.subs({symbol: symbol_full_sizes[symbol] for symbol in symbols})
    except AttributeError:
        return None
    if _free_symbols(expr):
        return None
    try:
        return int(expr)
    except (TypeError, ValueError):
        return None


def _set_materialized_chunk_dim_meta(
    materialized: fx.Node,
    original: fx.Node,
    *,
    mode: ChunkMode,
    dim: int,
    symbol_full_sizes: dict[object, int],
    original_val: torch.Tensor | None = None,
    original_chunk_dims: dict[str, dict[str, int]] | None = None,
) -> None:
    chunk_dims = dict(
        original.meta.get(_CHUNK_DIMS_META, {})
        if original_chunk_dims is None
        else original_chunk_dims
    )
    hint = _annotated_chunk_hint(original, mode)
    if hint is None:
        val = _tensor_meta(original) if original_val is None else original_val
        if val is not None:
            hint = _evaluate_with_hints(val.shape[dim], symbol_full_sizes)
    if hint is not None:
        chunk_dims[mode] = {"dim": dim, "hint": hint}
        materialized.meta[_CHUNK_DIMS_META] = chunk_dims


def _validate_materialized_meta(
    materialized: fx.Node,
    original: fx.Node,
    symbol_full_sizes: dict[object, int],
    original_val: torch.Tensor | None = None,
) -> None:
    original_val = _tensor_meta(original) if original_val is None else original_val
    materialized_val = _tensor_meta(materialized)
    if not isinstance(original_val, torch.Tensor) or not isinstance(
        materialized_val, torch.Tensor
    ):
        return
    if len(original_val.shape) != len(materialized_val.shape):
        raise RuntimeError(
            f"Chunk pass materialized {materialized.name} with rank "
            f"{len(materialized_val.shape)}, expected {len(original_val.shape)} "
            f"from original {original.name}."
        )
    if not all(
        _dim_matches(original_dim, materialized_dim, symbol_full_sizes)
        for original_dim, materialized_dim in zip(
            original_val.shape, materialized_val.shape
        )
    ):
        raise RuntimeError(
            f"Chunk pass materialized {materialized.name} with shape "
            f"{tuple(materialized_val.shape)}, expected {tuple(original_val.shape)} "
            f"from original {original.name}."
        )


def _validate_chunked_copy_meta(
    copied: fx.Node,
    original: fx.Node,
    expected: torch.Tensor,
    symbol_full_sizes: dict[object, int],
) -> None:
    copied_val = _tensor_meta(copied)
    if copied_val is None:
        raise RuntimeError(
            f"Chunk pass could not infer tensor metadata for copied node "
            f"{copied.name} from original {original.name}."
        )
    if len(copied_val.shape) != len(expected.shape):
        raise RuntimeError(
            f"Chunk pass copied {original.name} to {copied.name} with rank "
            f"{len(copied_val.shape)}, expected {len(expected.shape)}."
        )
    if not all(
        _dim_matches(expected_dim, copied_dim, symbol_full_sizes)
        for expected_dim, copied_dim in zip(expected.shape, copied_val.shape)
    ):
        raise RuntimeError(
            f"Chunk pass copied {original.name} to {copied.name} with shape "
            f"{tuple(copied_val.shape)}, expected {tuple(expected.shape)}."
        )


def _map_arg_for_chunk(
    arg: object,
    *,
    node: fx.Node,
    copied: dict[fx.Node, fx.Node],
    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]],
    chunk_id: int,
) -> object:
    if isinstance(arg, fx.Node):
        if arg in copied:
            return copied[arg]
        chunks = split_live_ins.get(arg)
        if chunks is not None:
            if arg in _full_scalar_live_ins_for_hop(node):
                return arg
            return chunks[chunk_id]
    return arg


def _full_scalar_live_ins_for_hop(node: fx.Node) -> set[fx.Node]:
    # HOP closure scalar inputs describe the full logical problem size for
    # nested FlexAttention graph modules. They are not chunk-local tensors, so
    # copying the HOP body should pass them through unchanged while tensor
    # operands are mapped to the selected chunk.
    if node.op != "call_function":
        return set()

    if node.target is torch.ops.higher_order.flex_attention:
        closure_args = node.args[7:9]
    elif node.target is torch.ops.higher_order.flex_attention_backward:
        closure_args = node.args[12:14]
    else:
        return set()

    full_scalars = set()

    def collect(arg: object) -> object:
        if (
            isinstance(arg, fx.Node)
            and _tensor_meta(arg) is None
            and bool(_free_symbols(arg.meta.get("val")))
        ):
            full_scalars.add(arg)
        return arg

    torch.fx.map_arg(closure_args, collect)
    return full_scalars


def _insert_symbolic_split_size(
    gm: fx.GraphModule,
    live_in: fx.Node,
    *,
    dim: int,
    meta_chunk_size: int | torch.SymInt,
    region: _Region,
) -> fx.Node:
    full_size = gm.graph.call_function(aten.sym_size.int, args=(live_in, dim))
    _set_chunk_region_meta(full_size, region=region, role=_ChunkNodeRole.SPLIT_BOUNDARY)
    full_size.meta["val"] = _tensor_meta(live_in).shape[dim]
    return _insert_symbolic_half_node(
        gm,
        full_size,
        meta_half_size=meta_chunk_size,
        region=region,
        assert_message=f"chunk pass requires dimension {dim} to be even",
    )


def _insert_symbolic_half_node(
    gm: fx.GraphModule,
    size_node: fx.Node,
    *,
    meta_half_size: object,
    region: _Region,
    assert_message: str,
) -> fx.Node:
    remainder = gm.graph.call_function(operator.mod, args=(size_node, 2))
    _set_chunk_region_meta(remainder, region=region, role=_ChunkNodeRole.SPLIT_BOUNDARY)
    is_even = gm.graph.call_function(operator.eq, args=(remainder, 0))
    _set_chunk_region_meta(is_even, region=region, role=_ChunkNodeRole.SPLIT_BOUNDARY)
    assert_node = gm.graph.call_function(
        aten._assert_scalar.default,
        args=(is_even, assert_message),
    )
    _set_chunk_region_meta(
        assert_node, region=region, role=_ChunkNodeRole.SPLIT_BOUNDARY
    )

    half_size = gm.graph.call_function(operator.floordiv, args=(size_node, 2))
    _set_chunk_region_meta(half_size, region=region, role=_ChunkNodeRole.CHUNK_INPUT)
    half_size.meta["val"] = meta_half_size
    return half_size


def _helper_direction_from_users(
    users: list[fx.Node], order: dict[fx.Node, int]
) -> bool:
    """Direction metadata follows the helper's insertion point."""
    return _is_backward_node(_earliest_node(users, order))


def _earliest_region_user(
    live_in: fx.Node, region_nodes: tuple[fx.Node, ...], order: dict[fx.Node, int]
) -> fx.Node:
    users = [node for node in region_nodes if live_in in node.all_input_nodes]
    return _earliest_node(users, order)


def _constant_factor(numerator: object, denominator: object) -> int | None:
    try:
        factor = _expr(numerator) / _expr(denominator)
        if _free_symbols(factor):
            return None
        if int(factor) == factor:
            return int(factor)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def _scalar_matches_split_dim(
    scalar_val: object,
    *,
    mode: ChunkMode,
    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]],
) -> bool:
    for live_in in split_live_ins:
        val = _tensor_meta(live_in)
        if val is None:
            continue
        dim = _annotated_chunk_dim(live_in, mode)
        if dim is None:
            continue
        if (
            _constant_factor(scalar_val, val.shape[dim]) is not None
            or _constant_factor(val.shape[dim], scalar_val) is not None
        ):
            return True
    return False


def _scalar_sum_matches(
    chunk0_val: object, chunk1_val: object, original_val: object
) -> bool:
    try:
        return _expr_matches(_expr(chunk0_val) + _expr(chunk1_val), original_val)
    except (TypeError, ValueError):
        return False


def _scalar_halves_match_guarded_even_original(
    chunk0_val: object, chunk1_val: object, original_val: object
) -> bool:
    try:
        half_original = _expr(original_val) // 2
        return _expr_matches(chunk0_val, half_original) and _expr_matches(
            chunk1_val, half_original
        )
    except (TypeError, ValueError):
        return False


def _materialize_scalar_live_out(
    gm: fx.GraphModule,
    live_out: fx.Node,
    copied0: fx.Node,
    copied1: fx.Node,
    *,
    mode: ChunkMode,
    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]],
    original_val: object | None = None,
) -> fx.Node:
    original_val = live_out.meta.get("val") if original_val is None else original_val
    chunk0_val = copied0.meta.get("val")
    chunk1_val = copied1.meta.get("val")

    if _expr_matches(chunk0_val, original_val) and _expr_matches(
        chunk1_val, original_val
    ):
        return copied0

    if _scalar_sum_matches(chunk0_val, chunk1_val, original_val):
        materialized = gm.graph.call_function(operator.add, args=(copied0, copied1))
        _rename(materialized, f"{live_out.name}_chunk_sum")
        materialized.meta["val"] = original_val
        return materialized

    if _scalar_halves_match_guarded_even_original(
        chunk0_val, chunk1_val, original_val
    ) or (
        _expr_matches(chunk0_val, chunk1_val)
        and _scalar_matches_split_dim(
            original_val,
            mode=mode,
            split_live_ins=split_live_ins,
        )
    ):
        # Step 9.3: split insertion asserted the symbolic dimension is even.
        # Under that guard, two copied sym_size halves reconstruct the original
        # full symbolic size even when general SymPy simplification cannot
        # prove floor(u0 / 2) + floor(u0 / 2) == u0 for an unconstrained symbol.
        materialized = gm.graph.call_function(operator.add, args=(copied0, copied1))
        _rename(materialized, f"{live_out.name}_chunk_sum")
        materialized.meta["val"] = original_val
        return materialized

    raise ValueError(
        f"Chunk pass cannot materialize scalar live-out {live_out.name}: "
        f"chunk values {chunk0_val!r}, {chunk1_val!r} do not reconstruct "
        f"original value {original_val!r}."
    )


def _region_contains_attention(region: _Region) -> bool:
    return any("attention" in _fqn(n).split(".") for n in region.nodes)


def _candidate_live_ins(search_space: set[fx.Node]) -> set[fx.Node]:
    return {
        inp
        for node in search_space
        for inp in node.all_input_nodes
        if inp not in search_space
    }


def _chunkable_live_ins(
    live_ins: set[fx.Node],
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
) -> set[fx.Node]:
    return {
        live_in
        for live_in in live_ins
        if _is_chunkable_live_in(
            live_in,
            mode=mode,
            static_nodes=static_nodes,
        )
    }


def _format_candidate_live_ins(
    live_ins: set[fx.Node],
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
    order: dict[fx.Node, int],
) -> str:
    details = []

    def sort_key(node: fx.Node) -> tuple[int, int]:
        val = _tensor_meta(node)
        has_symbol = int(
            val is not None and any(_free_symbols(dim) for dim in val.shape)
        )
        return (int(node in static_nodes) - has_symbol, order[node])

    for node in sorted(live_ins, key=sort_key):
        val = _tensor_meta(node)
        details.append(
            f"{node.name}:{node.target} "
            f"static={node in static_nodes} "
            f"shape={tuple(val.shape) if val is not None else None} "
            f"chunk_dim={_annotated_chunk_dim(node, mode)} "
            f"chunk_hint={_annotated_chunk_hint(node, mode)}"
        )
    return "; ".join(details[:12])


def _make_region_plan(
    gm: fx.GraphModule,
    region: _Region,
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
    per_chunk_sources: set[fx.Node],
) -> _RegionPlan:
    if mode == "seq" and _region_contains_attention(region):
        raise NotImplementedError(
            f"chunk_seq for attention-containing region {region.root_fqn!r} "
            "requires a full-K/V attention rewrite and is not implemented in v1."
        )

    order = _ordered_nodes(gm)
    search_space = _reverse_closure_from_boundary_nodes(
        list(region.nodes),
        root_fqn=region.root_fqn,
        is_backward=region.is_backward,
    )
    if not search_space:
        raise ValueError(
            f"Chunk pass matched region {region.root_fqn!r}, but found no "
            "same-direction nodes inside that module root."
        )

    candidate_live_ins = _candidate_live_ins(search_space)
    chunkable_live_ins = _chunkable_live_ins(
        candidate_live_ins,
        mode=mode,
        static_nodes=static_nodes,
    )
    per_chunk_live_ins = candidate_live_ins & per_chunk_sources

    # Compute the copied descendant closure to a local fixed point. Some live-ins
    # only become visible after the first activation-dependent slice is selected.
    while True:
        region_nodes = _forward_closure_from_sources(
            chunkable_live_ins | per_chunk_live_ins,
            search_space=search_space,
        )
        region_live_ins = {
            inp
            for node in region_nodes
            for inp in node.all_input_nodes
            if inp not in region_nodes
        }
        extra_chunkable_live_ins = _chunkable_live_ins(
            region_live_ins - chunkable_live_ins,
            mode=mode,
            static_nodes=static_nodes,
        )
        extra_per_chunk_live_ins = (
            region_live_ins & per_chunk_sources
        ) - per_chunk_live_ins
        if not extra_chunkable_live_ins and not extra_per_chunk_live_ins:
            break
        chunkable_live_ins |= extra_chunkable_live_ins
        per_chunk_live_ins |= extra_per_chunk_live_ins

    if not region_nodes:
        raise ValueError(
            f"Chunk pass matched region {region.root_fqn!r}, but found no "
            f"activation-dependent nodes for chunk_{mode}. Candidate live-ins: "
            f"{_format_candidate_live_ins(candidate_live_ins, mode=mode, static_nodes=static_nodes, order=order)}"
        )

    region_nodes_tuple = tuple(sorted(region_nodes, key=order.__getitem__))
    region_live_ins = {
        inp
        for node in region_nodes_tuple
        for inp in node.all_input_nodes
        if inp not in region_nodes
    }
    chunkable_live_ins = (chunkable_live_ins & region_live_ins) - region_nodes
    per_chunk_live_ins = (per_chunk_live_ins & region_live_ins) - region_nodes
    live_outs = [
        node
        for node in region_nodes_tuple
        if any(user not in region_nodes for user in node.users)
    ]
    if not live_outs:
        raise ValueError(
            f"Chunk pass found no live-outs for activation-dependent region "
            f"{region.root_fqn!r}."
        )

    return _RegionPlan(
        region=region,
        search_space=frozenset(search_space),
        chunkable_live_ins=frozenset(chunkable_live_ins),
        per_chunk_live_ins=frozenset(per_chunk_live_ins),
        region_nodes=frozenset(region_nodes),
        region_nodes_tuple=region_nodes_tuple,
        region_live_ins=frozenset(region_live_ins),
        live_out_users={
            live_out: tuple(user for user in live_out.users if user not in region_nodes)
            for live_out in live_outs
        },
    )


def _plan_regions(
    gm: fx.GraphModule,
    regions: list[_Region],
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
) -> list[_RegionPlan]:
    per_chunk_sources: set[fx.Node] = set()
    plans: list[_RegionPlan] = []

    while True:
        plans = [
            _make_region_plan(
                gm,
                region,
                mode=mode,
                static_nodes=static_nodes,
                per_chunk_sources=per_chunk_sources,
            )
            for region in regions
        ]
        chunk_consumer_nodes = _chunk_consumer_nodes(plans)
        next_per_chunk_sources = {
            live_out
            for plan in plans
            for live_out, users in plan.live_out_users.items()
            if any(user in chunk_consumer_nodes for user in users)
            or _live_out_reaches_chunked_body(
                live_out,
                plans=plans,
            )
        }
        if next_per_chunk_sources == per_chunk_sources:
            break
        per_chunk_sources = next_per_chunk_sources

    _validate_disjoint_plans(plans)
    order = _ordered_nodes(gm)
    return _sort_plans_by_per_chunk_deps(
        plans,
        order=order,
    )


def _chunk_consumer_nodes(plans: list[_RegionPlan]) -> set[fx.Node]:
    return {node for plan in plans for node in plan.region_nodes}


def _sort_plans_by_per_chunk_deps(
    plans: list[_RegionPlan],
    *,
    order: dict[fx.Node, int],
) -> list[_RegionPlan]:
    """Transform producer regions before consumers of their per-chunk values."""
    plan_order = [
        min(order[node] for node in plan.region_nodes_tuple) for plan in plans
    ]
    producer: dict[fx.Node, _RegionPlan] = {
        live_out: plan for plan in plans for live_out in plan.live_out_users
    }
    plan_index = {id(plan): idx for idx, plan in enumerate(plans)}
    dep_live_ins: list[dict[int, list[fx.Node]]] = []
    for idx, plan in enumerate(plans):
        by_dep: dict[int, list[fx.Node]] = {}
        for live_in in plan.per_chunk_live_ins:
            dep = producer.get(live_in)
            if dep is None:
                continue
            dep_idx = plan_index[id(dep)]
            if dep_idx != idx:
                by_dep.setdefault(dep_idx, []).append(live_in)
        dep_live_ins.append(by_dep)
    deps: list[set[int]] = [set(by_dep) for by_dep in dep_live_ins]

    ready = sorted(
        (idx for idx, plan_deps in enumerate(deps) if not plan_deps),
        key=lambda idx: plan_order[idx],
    )
    result_indices: list[int] = []
    while ready:
        idx = ready.pop(0)
        result_indices.append(idx)
        for candidate_idx in sorted(range(len(deps)), key=lambda i: plan_order[i]):
            candidate_deps = deps[candidate_idx]
            if idx not in candidate_deps:
                continue
            candidate_deps.remove(idx)
            if (
                not candidate_deps
                and candidate_idx not in result_indices
                and candidate_idx not in ready
            ):
                ready.append(candidate_idx)
        ready.sort(key=lambda i: plan_order[i])

    if len(result_indices) != len(plans):
        cycle = [
            (
                plans[idx].region.root_fqn,
                "backward" if plans[idx].region.is_backward else "forward",
                [
                    (
                        plans[dep_idx].region.root_fqn,
                        (
                            "backward"
                            if plans[dep_idx].region.is_backward
                            else "forward"
                        ),
                        [
                            live_in.name
                            for live_in in sorted(
                                dep_live_ins[idx][dep_idx],
                                key=order.__getitem__,
                            )[:8]
                        ],
                    )
                    for dep_idx in sorted(plan_deps, key=lambda i: plan_order[i])
                ],
            )
            for idx, plan_deps in enumerate(deps)
            if plan_deps
        ]
        raise ValueError(
            "Chunk pass found cyclic per-chunk dependencies between regions: "
            f"{cycle}"
        )
    return [plans[idx] for idx in result_indices]


def _live_out_reaches_chunked_body(
    live_out: fx.Node,
    *,
    plans: list[_RegionPlan],
) -> bool:
    # Check each planned body through its own local search space; using a global
    # search-space union would make per-chunk classification depend on
    # unrelated regions.
    return any(
        _forward_closure_from_sources(
            {live_out}, search_space=set(consumer_plan.search_space)
        )
        & set(consumer_plan.region_nodes)
        for consumer_plan in plans
    )


def _validate_disjoint_plans(plans: list[_RegionPlan]) -> None:
    owner: dict[fx.Node, _RegionPlan] = {}
    for plan in plans:
        for node in plan.region_nodes:
            previous = owner.get(node)
            if previous is not None:
                if _is_symbolic_shape_scalar(node):
                    # Step 3.2: symbolic shape helpers such as sym_size nodes can
                    # be shared by adjacent module regions. They are pure shape
                    # compute, and each region copy rewrites them against that
                    # region's chunked live-ins.
                    continue
                raise ValueError(
                    "Chunk pass planned overlapping regions for node "
                    f"{node.name}: {previous.region.root_fqn!r} "
                    f"({'backward' if previous.region.is_backward else 'forward'}) "
                    f"and {plan.region.root_fqn!r} "
                    f"({'backward' if plan.region.is_backward else 'forward'})."
                )
            owner[node] = plan


# Maintenance note: this allowlist is part of the chunking correctness contract.
# It should contain only operations whose per-chunk values can be summed to
# recover the full value. Add new entries only with a concrete additive-semantics
# argument and a numerics test. Backward gradients are handled separately below:
# we infer them structurally from backward-region live-outs consumed by graph
# outputs or by full-gradient reduce-scatter/all-reduce collectives, not from a
# dedicated gradient metadata tag.
def _is_additive_partial_sum_passthrough(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return node.target in (operator.add, aten.add.Tensor, aten.cumsum.default)


def _is_partial_sum_seed(node: fx.Node, chunk_symbols: frozenset[object]) -> bool:
    if node.op != "call_function":
        return False
    if "histc" in str(node.target) or "bincount" in str(node.target):
        return True
    if node.target in (aten.sum.default, aten.sum.dim_IntList):
        for input_node in node.all_input_nodes:
            val = _tensor_meta(input_node)
            if val is not None and any(
                _dynamic_dim_symbols(val, dim) & chunk_symbols
                for dim in range(val.dim())
            ):
                return True
    return False


def _is_partial_sum_value(
    node: fx.Node,
    *,
    plan_nodes: frozenset[fx.Node],
    chunk_symbols: frozenset[object],
    memo: dict[fx.Node, bool],
) -> bool:
    cached = memo.get(node)
    if cached is not None:
        return cached
    if node not in plan_nodes:
        memo[node] = False
        return False
    if _is_partial_sum_seed(node, chunk_symbols):
        memo[node] = True
        return True
    if not _is_additive_partial_sum_passthrough(node):
        memo[node] = False
        return False
    input_nodes = node.all_input_nodes
    result = bool(input_nodes) and any(
        _is_partial_sum_value(
            input_node,
            plan_nodes=plan_nodes,
            chunk_symbols=chunk_symbols,
            memo=memo,
        )
        for input_node in input_nodes
    )
    memo[node] = result
    return result


def _is_full_gradient_collective_consumer(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = str(node.target)
    return "reduce_scatter" in target or "all_reduce" in target


def _can_materialize_add(
    live_out: fx.Node,
    *,
    region: _Region,
    full_users: list[fx.Node],
    plan_nodes: frozenset[fx.Node],
    chunk_symbols: frozenset[object],
    partial_sum_memo: dict[fx.Node, bool],
) -> bool:
    if region.is_backward:
        return any(user.op == "output" for user in full_users) or any(
            _is_full_gradient_collective_consumer(user) for user in full_users
        )
    return _is_partial_sum_value(
        live_out,
        plan_nodes=plan_nodes,
        chunk_symbols=chunk_symbols,
        memo=partial_sum_memo,
    )


def _transform_region(
    gm: fx.GraphModule,
    plan: _RegionPlan,
    *,
    mode: ChunkMode,
    all_chunk_consumer_nodes: set[fx.Node],
    chunk_value_nodes: dict[fx.Node, tuple[fx.Node, fx.Node]],
) -> int:
    order = _ordered_nodes(gm)
    region = plan.region
    region_nodes_tuple = plan.region_nodes_tuple
    region_live_ins = set(plan.region_live_ins)
    chunkable_live_ins = set(plan.chunkable_live_ins)

    split_live_ins: dict[fx.Node, tuple[fx.Node, fx.Node]] = {}
    dim_chunk_sizes: list[tuple[object, object]] = []
    symbol_chunk_values: dict[object, object] = {}
    symbol_full_sizes = _collect_symbol_full_sizes(gm, mode)

    # Step 6.1: planned per-chunk live-ins are live-outs from another chunked
    # region. The graph still has the original full edge, but copied chunk
    # consumers use the explicit chunk tuple built by the producer region.
    for live_in in sorted(plan.per_chunk_live_ins, key=order.get):
        split_live_ins[live_in] = chunk_value_nodes[live_in]

    # Step 7: record hint-derived sizes for symbolic chunk dims. Half-size
    # metadata flows through copied chunk nodes. A raw symbol always maps only
    # to that symbol's half value; full dimension expressions such as 16*u0 get
    # their own expression-level replacement so flattened batch dimensions do
    # not corrupt the meaning of u0 for later nodes.
    for live_in in chunkable_live_ins:
        val = _tensor_meta(live_in)
        assert val is not None
        hint = _annotated_chunk_hint(live_in, mode)
        if hint is None:
            continue
        dim = _chunk_dim_for_live_in(live_in, mode)
        chunk_size = _chunk_size_for_live_in(live_in, mode, val, dim)
        dim_chunk_sizes.append((val.shape[dim], chunk_size))
        for symbol in _dynamic_dim_symbols(val, dim):
            if symbol not in symbol_full_sizes:
                raise ValueError(
                    f"Chunk pass could not find a root hint for dynamic symbol "
                    f"{symbol} used by live-in {live_in.name}."
                )
            if symbol not in symbol_chunk_values:
                symbol_chunk_values[symbol] = _make_symbol_half(
                    symbol,
                    like_value=val.shape[dim],
                    symbol_full_sizes=symbol_full_sizes,
                )
    chunk_symbols = frozenset(symbol_chunk_values)

    for live_in in sorted(chunkable_live_ins, key=order.get):
        if live_in in split_live_ins:
            continue
        val = _tensor_meta(live_in)
        assert val is not None
        dim = _chunk_dim_for_live_in(live_in, mode)
        chunk_size = _chunk_size_for_live_in(
            live_in, mode, val, dim, dim_chunk_sizes=dim_chunk_sizes
        )
        with gm.graph.inserting_before(
            _earliest_region_user(live_in, region_nodes_tuple, order)
        ):
            split_size = (
                _insert_symbolic_split_size(
                    gm,
                    live_in,
                    dim=dim,
                    meta_chunk_size=chunk_size,
                    region=region,
                )
                if _dynamic_dim_symbols(val, dim)
                else chunk_size
            )
            split = gm.graph.call_function(
                aten.split_with_sizes.default,
                args=(live_in, [split_size, split_size], dim),
            )
            _set_chunk_region_meta(
                split, region=region, role=_ChunkNodeRole.SPLIT_BOUNDARY
            )
            split.meta["chunk_dim"] = dim
            split.meta["chunk_size"] = chunk_size
            raw_getitem0 = gm.graph.call_function(operator.getitem, args=(split, 0))
            raw_getitem1 = gm.graph.call_function(operator.getitem, args=(split, 1))
            _set_chunk_region_meta(
                raw_getitem0,
                region=region,
                chunk_id=0,
                role=_ChunkNodeRole.CHUNK_INPUT,
            )
            _set_chunk_region_meta(
                raw_getitem1,
                region=region,
                chunk_id=1,
                role=_ChunkNodeRole.CHUNK_INPUT,
            )
            getitem0 = raw_getitem0
            getitem1 = raw_getitem1
            if dim != 0:
                getitem0 = gm.graph.call_function(aten.contiguous.default, (getitem0,))
                getitem1 = gm.graph.call_function(aten.contiguous.default, (getitem1,))
                _set_chunk_region_meta(
                    getitem0,
                    region=region,
                    chunk_id=0,
                    role=_ChunkNodeRole.CHUNK_INPUT,
                )
                _set_chunk_region_meta(
                    getitem1,
                    region=region,
                    chunk_id=1,
                    role=_ChunkNodeRole.CHUNK_INPUT,
                )
        getitem0.meta["val"] = _chunk_meta(
            val, dim, chunk_size, preserve_stride=dim == 0
        )
        getitem1.meta["val"] = _chunk_meta(
            val, dim, chunk_size, preserve_stride=dim == 0
        )
        split_live_ins[live_in] = (getitem0, getitem1)

    if not split_live_ins:
        raise ValueError(
            f"Chunk pass found no chunkable activation live-ins for region "
            f"{region.root_fqn!r}."
        )

    for live_in in sorted(region_live_ins, key=order.get):
        if live_in in split_live_ins or _tensor_meta(live_in) is not None:
            continue
        scalar_val = live_in.meta.get("val")
        if not (_free_symbols(scalar_val) & chunk_symbols):
            continue
        if not _scalar_matches_split_dim(
            scalar_val,
            mode=mode,
            split_live_ins=split_live_ins,
        ):
            raise ValueError(
                f"Chunk pass found symbolic scalar live-in {live_in.name}, but "
                "it is not a constant multiple of any split tensor dimension."
            )
        with gm.graph.inserting_before(
            _earliest_region_user(live_in, region_nodes_tuple, order)
        ):
            half_scalar = _insert_symbolic_half_node(
                gm,
                live_in,
                meta_half_size=scalar_val // 2,
                region=region,
                assert_message="chunk pass requires symbolic scalar live-in to be even",
            )
        split_live_ins[live_in] = (half_scalar, half_scalar)

    # Step 8: rewrite the original body in place as the canonical first chunk,
    # then copy the peer chunk as one contiguous block after it. This keeps
    # shared live-ins in their natural graph position and avoids copying nodes
    # that do not depend on chunked activations.
    original_args = {node: node.args for node in region_nodes_tuple}
    original_kwargs = {node: node.kwargs for node in region_nodes_tuple}
    original_names = {node: node.name for node in region_nodes_tuple}
    original_meta = {node: dict(node.meta) for node in region_nodes_tuple}
    expected_meta = {
        node: _chunked_meta_from_original(
            node,
            dim_chunk_sizes=dim_chunk_sizes,
            symbol_chunk_values=symbol_chunk_values,
            symbol_full_sizes=symbol_full_sizes,
        )
        for node in region_nodes_tuple
    }
    copied_by_chunk: list[dict[fx.Node, fx.Node]] = [dict(), dict()]
    first_chunk_id = 1 if region.is_backward else 0
    second_chunk_id = 1 - first_chunk_id
    first_copied = copied_by_chunk[first_chunk_id]
    for node in region_nodes_tuple:
        node.args = torch.fx.map_arg(
            original_args[node],
            lambda arg: _map_arg_for_chunk(
                arg,
                node=node,
                copied=first_copied,
                split_live_ins=split_live_ins,
                chunk_id=first_chunk_id,
            ),
        )
        node.kwargs = torch.fx.map_arg(
            original_kwargs[node],
            lambda arg: _map_arg_for_chunk(
                arg,
                node=node,
                copied=first_copied,
                split_live_ins=split_live_ins,
                chunk_id=first_chunk_id,
            ),
        )
        _rename(node, f"{original_names[node]}_chunk{first_chunk_id}")
        node.meta = dict(original_meta[node])
        if "custom" in node.meta:
            node.meta["custom"] = dict(node.meta["custom"])
        _set_chunk_region_meta(
            node,
            region=region,
            chunk_id=first_chunk_id,
            role=_ChunkNodeRole.BODY,
        )
        _infer_node_val(node)
        expected_val = expected_meta[node]
        if expected_val is not None:
            _validate_chunked_copy_meta(
                node,
                node,
                expected_val,
                symbol_full_sizes,
            )
        first_copied[node] = node

    second_copied = copied_by_chunk[second_chunk_id]
    with gm.graph.inserting_before(region_nodes_tuple[-1].next):
        for node in region_nodes_tuple:
            new_args = torch.fx.map_arg(
                original_args[node],
                lambda arg: _map_arg_for_chunk(
                    arg,
                    node=node,
                    copied=second_copied,
                    split_live_ins=split_live_ins,
                    chunk_id=second_chunk_id,
                ),
            )
            new_kwargs = torch.fx.map_arg(
                original_kwargs[node],
                lambda arg: _map_arg_for_chunk(
                    arg,
                    node=node,
                    copied=second_copied,
                    split_live_ins=split_live_ins,
                    chunk_id=second_chunk_id,
                ),
            )
            new_node = gm.graph.create_node(
                node.op,
                node.target,
                new_args,
                new_kwargs,
                type_expr=node.type,
            )
            _rename(new_node, f"{original_names[node]}_chunk{second_chunk_id}")
            new_node.meta = dict(original_meta[node])
            if "custom" in new_node.meta:
                new_node.meta["custom"] = dict(new_node.meta["custom"])
            _set_chunk_region_meta(
                new_node,
                region=region,
                chunk_id=second_chunk_id,
                role=_ChunkNodeRole.BODY,
            )
            _infer_node_val(new_node)
            expected_val = expected_meta[node]
            if expected_val is not None:
                _validate_chunked_copy_meta(
                    new_node,
                    node,
                    expected_val,
                    symbol_full_sizes,
                )
            second_copied[node] = new_node

    replaced_live_outs = 0
    partial_sum_memo: dict[fx.Node, bool] = {}
    for live_out, outside_users_tuple in plan.live_out_users.items():
        val = original_meta.get(live_out, live_out.meta).get("val")
        tensor_val = val if isinstance(val, torch.Tensor) else None
        outside_users = list(outside_users_tuple)
        if not outside_users:
            continue
        chunk_users = [
            user for user in outside_users if user in all_chunk_consumer_nodes
        ]
        full_users = [
            user for user in outside_users if user not in all_chunk_consumer_nodes
        ]

        if chunk_users:
            # Step 9.1: carry chunk-local live-outs through the planning side
            # table. Do not insert an FX tuple: tuple(get0, get1) depends on
            # both chunks and would create false cross-chunk data dependencies
            # that block backward scheduling.
            chunk_value_nodes[live_out] = (
                copied_by_chunk[0][live_out],
                copied_by_chunk[1][live_out],
            )

        if not full_users:
            continue

        # Step 9.2: materialize only for true full consumers. Forward no-dim
        # values may be added only when explicit accumulation proof proves
        # the full value is the sum of chunk values. Backward no-dim values may
        # be added when they feed graph outputs or gradient collectives.
        helper_is_backward = (
            region.is_backward
            if any(user.op == "output" for user in full_users)
            else _helper_direction_from_users(full_users, order)
        )
        with gm.graph.inserting_before(_earliest_node(full_users, order)):
            if tensor_val is None:
                materialized = _materialize_scalar_live_out(
                    gm,
                    live_out,
                    copied_by_chunk[0][live_out],
                    copied_by_chunk[1][live_out],
                    mode=mode,
                    split_live_ins=split_live_ins,
                    original_val=val,
                )
            else:
                combine_kind, dim = _combine_kind_and_dim(
                    live_out, mode, chunk_symbols, val=tensor_val
                )
                if combine_kind == "add":
                    if not _can_materialize_add(
                        live_out,
                        region=region,
                        full_users=full_users,
                        plan_nodes=plan.region_nodes,
                        chunk_symbols=chunk_symbols,
                        partial_sum_memo=partial_sum_memo,
                    ):
                        live_out_users = ", ".join(
                            f"{user.name}:{user.target}" for user in full_users
                        )
                        raise ValueError(
                            f"Chunk pass cannot materialize live-out "
                            f"{live_out.name}:{live_out.target} with add in "
                            f"{region.root_fqn!r} "
                            f"({'backward' if region.is_backward else 'forward'}); "
                            f"shape={getattr(val, 'shape', None)}, "
                            f"users=[{live_out_users}]. Expected either "
                            "forward accumulation proof or a backward "
                            "parameter-gradient consumer."
                        )
                    materialized = gm.graph.call_function(
                        aten.add.Tensor,
                        args=(
                            copied_by_chunk[0][live_out],
                            copied_by_chunk[1][live_out],
                        ),
                    )
                    _rename(materialized, f"{live_out.name}_chunk_sum")
                else:
                    materialized = gm.graph.call_function(
                        aten.cat.default,
                        args=(
                            [
                                copied_by_chunk[0][live_out],
                                copied_by_chunk[1][live_out],
                            ],
                            dim,
                        ),
                    )
                    _rename(materialized, f"{live_out.name}_chunk_cat")
                    assert dim is not None
                    materialized.meta["chunk_dim"] = dim
                _set_chunk_region_meta(
                    materialized,
                    region=region,
                    is_backward=helper_is_backward,
                    role=_ChunkNodeRole.MATERIALIZATION,
                )
                _infer_node_val(materialized)
                _validate_materialized_meta(
                    materialized, live_out, symbol_full_sizes, original_val=tensor_val
                )
                materialized.meta["val"] = tensor_val
                if combine_kind == "cat":
                    assert dim is not None
                    _set_materialized_chunk_dim_meta(
                        materialized,
                        live_out,
                        mode=mode,
                        dim=dim,
                        symbol_full_sizes=symbol_full_sizes,
                        original_val=tensor_val,
                        original_chunk_dims=original_meta.get(live_out, {}).get(
                            _CHUNK_DIMS_META
                        ),
                    )
        if materialized not in (
            copied_by_chunk[0][live_out],
            copied_by_chunk[1][live_out],
        ):
            _set_chunk_region_meta(
                materialized,
                region=region,
                is_backward=helper_is_backward,
                role=_ChunkNodeRole.MATERIALIZATION,
            )
            materialized.meta.pop("chunk_id", None)
        for user in list(full_users):
            user.replace_input_with(live_out, materialized)
        replaced_live_outs += 1

    return replaced_live_outs


def apply_chunk_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
    module_patterns: list[str],
    num_static_inputs: int = 0,
    require_ep_region: bool = False,
) -> fx.GraphModule:
    """Chunk selected module regions into two chunks.

    Args:
        gm: Joint fwd/loss/bwd graph.
        example_inputs: Unused, accepted for the graph pass interface.
        mode: Selects the annotated dynamic dimension to split.
        module_patterns: Dot-segment FQN patterns selecting module regions.
        num_static_inputs: Number of leading graph placeholders that represent
            model state. These are never split.
    """
    if not module_patterns:
        return gm

    if mode not in ("batch", "seq"):
        raise ValueError(f"Unknown chunk mode: {mode!r}")

    static_nodes = _static_derived_nodes(
        gm, _static_placeholders(gm, num_static_inputs)
    )
    _propagate_chunk_dim_metadata(gm)
    regions = _find_regions(gm, module_patterns)
    if not regions:
        raise ValueError(
            f"No graph regions matched chunk_{mode} patterns: {module_patterns}"
        )
    if require_ep_region:
        num_regions = len(regions)
        regions = [
            region
            for region in regions
            if any(_ep_region(node) for node in region.nodes)
        ]
        if len(regions) != num_regions:
            logger.info(
                "Skipped %d chunk_%s region(s) without EP dispatch/combine metadata",
                num_regions - len(regions),
                mode,
            )
        if not regions:
            raise ValueError(
                f"No EP dispatch/combine regions matched chunk_{mode} patterns: "
                f"{module_patterns}"
            )
    plans = _plan_regions(
        gm,
        regions,
        mode=mode,
        static_nodes=static_nodes,
    )
    all_chunk_consumer_nodes = _chunk_consumer_nodes(plans)
    chunk_value_nodes: dict[fx.Node, tuple[fx.Node, fx.Node]] = {}

    transformed = 0
    for plan in plans:
        transformed += _transform_region(
            gm,
            plan,
            mode=mode,
            all_chunk_consumer_nodes=all_chunk_consumer_nodes,
            chunk_value_nodes=chunk_value_nodes,
        )

    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Applied chunk_%s to %d regions (%d materialized live-outs): %s",
        mode,
        len(regions),
        transformed,
        module_patterns,
    )
    return gm


def ep_overlap_chunk_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
    module_pattern: str,
    num_static_inputs: int = 0,
) -> fx.GraphModule:
    """Internal EP-overlap chunking phase run after cleanup passes."""
    return apply_chunk_pass(
        gm,
        example_inputs,
        mode=mode,
        module_patterns=[module_pattern],
        num_static_inputs=num_static_inputs,
        require_ep_region=True,
    )
