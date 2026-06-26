# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph chunking for EP overlap.

The pass is a graph implementation of the eager wrapper:

    split selected live-ins -> run selected module on each chunk -> materialize
    full live-outs when a non-chunked consumer needs them.

The correctness target is numerical equivalence to eager chunking, not an
identical lowered graph. ``module_fqn`` seeds the candidate region, but body
ownership is dataflow based: only nodes reached by chunk-influenced values are
duplicated. Boundary nodes that the pass creates (split/getitem/scalar
halves/cat/add) use the parent-wrapper metadata, while copied body nodes keep
the selected module metadata and receive ``chunk_id``.

Terms:
- live-in: a value consumed by the selected body and produced outside it.
  Tensor live-ins whose fake shape contains the selected unbacked chunk symbol
  are split. Per-chunk provenance from an earlier region is reused directly.
  Static state, params, buffers, and other full values are shared by both chunks.
- live-out: a value produced by the selected body and consumed outside it.
  Chunked consumers receive the matching per-chunk value through provenance.
  Full consumers receive a materialized value: ``cat`` for values carrying the
  chunk dimension and ``add`` for backward parameter-gradient reductions.
- provenance: the internal ``full_value -> (chunk0, chunk1)`` table. It is not
  materialized as tuple nodes because no downstream pass needs tuple plumbing.

The implementation deliberately stays local. It does not special-case SAC,
activation offload, or bucketing nodes; those interactions are expressed as
ordinary dataflow through live-ins and live-outs. The selected unbacked SymInt is
the only source of truth for chunk influence; graph metadata records only
symbol->hint pairs so cleanup can erase the symbols after scheduling. The
selected dimension is symbolic while this pass and scheduling run, then
``concretize_ep_chunk_symbolic_shapes_pass`` replaces chunk-derived symbolic
shapes with their config hints before backend codegen.

V1 creates two equal chunks because the EP schedule is pairwise. The data
structures use per-chunk maps, so extending to N equal chunks is mechanical, but
the overlap scheduler and tests must be generalized at the same time.

Pseudo-code
===========
1. Mark the requested input dimension dynamic before tracing, then import the
   selected symbol hints onto placeholders after tracing.
2. Discover selected module roots and plan forward/backward body nodes by
   following only values influenced by the selected chunk symbol.
3. Split chunked live-ins, reuse provenance from earlier chunked regions, and
   create half-size symbolic scalars used by copied body nodes.
4. Rewrite one chunk body in place and copy the peer chunk body, preserving
   eager-like metadata, fake values, and mutation alias dependencies.
5. For every live-out, either route per-chunk provenance to chunked consumers or
   materialize a full value with cat/add for full consumers.
6. Erase unused copied nodes, validate selected-symbol shapes, lint, recompile,
   and leave final symbol concretization to the cleanup pass after scheduling.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Literal

import sympy
import torch
import torch.fx as fx
from torch.utils._pytree import tree_leaves, tree_map
from torch.utils._sympy.functions import FloorDiv

from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_EXCHANGE,
    _get_module_fqn,
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.experiments.graph_trainer.configs import validate_ep_overlap_config
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    _eval_hint,
    chunk_symbol_hints_for_mode,
    CHUNK_SYMBOL_HINTS_META,
    depends_on_chunk_symbol,
    expr_key,
    free_symbols,
    is_c10d_functional_node,
    is_module_fqn_inside_root,
    is_view_target,
    ordered_nodes,
    same_shape_with_hints,
    same_symbolic_expr,
    same_tensor_domain,
    statically_equals_hint,
    symbolic_expr,
    tensor_meta,
    view_shape_arg_index,
)
from torchtitan.experiments.graph_trainer.registry import (
    register_trace_call_input_preparer,
    register_trace_input_preparer,
)
from torchtitan.tools.logging import logger

aten = torch.ops.aten
ChunkMode = Literal["batch", "seq"]


# Step 0: Metadata and small graph utilities used by the lowering steps below.


@dataclass(frozen=True)
class _Region:
    root_fqn: str
    synthetic_parent_fqn: str
    is_backward: bool
    nodes: tuple[fx.Node, ...]


@dataclass
class _Plan:
    region: _Region
    body: frozenset[fx.Node]
    live_ins: frozenset[fx.Node]
    live_out_users: dict[fx.Node, tuple[fx.Node, ...]]
    grad_escape_boundaries: frozenset[fx.Node]


@dataclass(frozen=True)
class _ChunkDim:
    dim: int
    hint: int | None


@dataclass(frozen=True)
class _MaterializationBoundary:
    node: fx.Node
    suffix: tuple[fx.Node, ...]


@dataclass(frozen=True)
class _MatchedRoot:
    root_fqn: str
    synthetic_parent_fqn: str


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    """Return mutable custom metadata when present, otherwise an empty dict."""
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _mutates_input(node: fx.Node) -> bool:
    """Return whether a call_function schema writes through an aliased input."""
    return bool(_mutable_arg_indices(node))


def _mutable_arg_indices(node: fx.Node) -> tuple[int, ...]:
    """Return positional schema arg indices written by ``node``."""
    schema = getattr(node.target, "_schema", None)
    if node.op != "call_function" or schema is None:
        return ()
    return tuple(
        idx
        for idx, arg in enumerate(schema.arguments)
        if arg.alias_info is not None and arg.alias_info.is_write
    )


def _is_semantic_user(node: fx.Node) -> bool:
    """Return whether a user must be preserved when crossing a chunk boundary."""
    return node.op == "output" or bool(node.users) or _mutates_input(node)


def _copy_meta(meta: dict[str, Any]) -> dict[str, Any]:
    copied = dict(meta)
    if "custom" in copied:
        copied["custom"] = dict(copied["custom"])
    if "unbacked_bindings" in copied:
        copied["unbacked_bindings"] = dict(copied["unbacked_bindings"])
    return copied


def _set_direction_meta(node: fx.Node, *, is_backward: bool) -> None:
    if is_backward:
        node.meta["autograd_backward"] = True
    else:
        node.meta.pop("autograd_backward", None)


def _set_chunk_meta(
    node: fx.Node,
    *,
    region: _Region,
    role: str,
    chunk_id: int | None = None,
) -> None:
    _set_direction_meta(node, is_backward=region.is_backward)
    node.meta["chunked_region_fqn"] = region.root_fqn
    node.meta["chunked_region_is_backward"] = region.is_backward
    node.meta["chunked_region_producer"] = "graph"
    node.meta["chunked_region_role"] = role
    if chunk_id is None:
        node.meta.pop("chunk_id", None)
    else:
        node.meta["chunk_id"] = chunk_id


def _set_synthetic_meta(
    node: fx.Node,
    *,
    region: _Region,
    role: str,
    chunk_id: int | None = None,
) -> None:
    """Attach parent-wrapper metadata to nodes introduced by graph chunking."""
    custom = dict(_custom_meta(node))
    parent_fqn = region.synthetic_parent_fqn
    if parent_fqn:
        custom[_MODULE_FQN] = parent_fqn
        node.meta["custom"] = custom
    elif custom:
        custom.pop(_MODULE_FQN, None)
        node.meta["custom"] = custom
    else:
        node.meta.pop("custom", None)
    _set_chunk_meta(node, region=region, role=role, chunk_id=chunk_id)


def _pattern_root(pattern: str, fqn: str) -> _MatchedRoot | None:
    """Return the matched root and synthetic-parent FQN for a module pattern."""
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    if len(fqn_parts) < len(pattern_parts):
        return None
    for pattern_part, fqn_part in zip(pattern_parts, fqn_parts):
        if pattern_part != "*" and pattern_part != fqn_part:
            return None
    root_parts = fqn_parts[: len(pattern_parts)]
    return _MatchedRoot(
        root_fqn=".".join(root_parts),
        synthetic_parent_fqn=".".join(root_parts[:-1]),
    )


def _region_roots(gm: fx.GraphModule, patterns: list[str]) -> list[_MatchedRoot]:
    """Find concrete module roots matching the configured chunk patterns."""
    roots: dict[str, _MatchedRoot] = {}
    for node in gm.graph.nodes:
        fqn = _get_module_fqn(node)
        if not fqn:
            continue
        for pattern in patterns:
            if (matched := _pattern_root(pattern, fqn)) is not None:
                roots.setdefault(matched.root_fqn, matched)
    return sorted(
        roots.values(),
        key=lambda matched: (matched.root_fqn.count("."), matched.root_fqn),
    )


def _region_nodes(
    gm: fx.GraphModule,
    root: str,
    *,
    is_backward: bool,
    symbol_hints: dict[object, int],
    grad_escape_boundaries: set[fx.Node],
    per_chunk_sources: set[fx.Node],
) -> tuple[fx.Node, ...]:
    """Return selected-symbol-influenced nodes inside one root/direction."""
    candidates = {
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.meta.get("chunked_region_role") is None
        and _is_backward_node(node) == is_backward
        and (fqn := _get_module_fqn(node))
        and is_module_fqn_inside_root(fqn, root)
    }
    if is_backward:
        changed = True
        while changed:
            changed = False
            for node in gm.graph.nodes:
                neighbors = (*node.all_input_nodes, *node.users)
                if (
                    node in candidates
                    or node.op != "call_function"
                    or _get_module_fqn(node)
                    or not _is_backward_node(node)
                    or is_c10d_functional_node(node)
                ):
                    continue
                anchored_to_root = any(neighbor in candidates for neighbor in neighbors)
                if not anchored_to_root:
                    anchored_to_root = any(
                        neighbor in per_chunk_sources
                        and is_module_fqn_inside_root(_get_module_fqn(neighbor), root)
                        for neighbor in neighbors
                    )
                if not anchored_to_root:
                    continue
                neighbor_fqns = [
                    fqn for neighbor in neighbors if (fqn := _get_module_fqn(neighbor))
                ]
                if not neighbor_fqns or all(
                    is_module_fqn_inside_root(fqn, root) for fqn in neighbor_fqns
                ):
                    candidates.add(node)
                    changed = True
    return _chunk_influenced_region_nodes(
        gm,
        candidates,
        is_backward,
        symbol_hints,
        grad_escape_boundaries,
        per_chunk_sources,
    )


def _node_depends_on_chunk_symbol(
    node: fx.Node, symbol_hints: dict[object, int]
) -> bool:
    """Return whether a node's fake value carries the selected chunk symbol."""
    return any(
        depends_on_chunk_symbol(value, symbol_hints)
        for value in tree_leaves(node.meta.get("val"))
    )


def _is_ep_collective_node(node: fx.Node, body: frozenset[fx.Node]) -> bool:
    """Return whether a c10d node belongs to EP token exchange/count plumbing."""
    custom = _custom_meta(node)
    if "EP" in custom or _EP_TOKEN_EXCHANGE in custom:
        return True
    if (
        node.target == torch.ops._c10d_functional.wait_tensor.default
        and node.all_input_nodes
        and node.all_input_nodes[0] in body
    ):
        waited_custom = _custom_meta(node.all_input_nodes[0])
        return "EP" in waited_custom or _EP_TOKEN_EXCHANGE in waited_custom
    return False


def _validate_no_non_ep_collectives(plan: _Plan) -> None:
    """Graph chunking may only duplicate EP collectives inside chunk bodies."""
    offenders = [
        node
        for node in plan.body
        if is_c10d_functional_node(node) and not _is_ep_collective_node(node, plan.body)
    ]
    if offenders:
        order = {node: idx for idx, node in enumerate(plan.region.nodes)}
        details = ", ".join(
            f"{node.name}:{getattr(node.target, '__name__', node.target)}"
            for node in sorted(offenders, key=order.get)
        )
        raise ValueError(
            "Graph EP chunking requires chunk bodies to contain only EP "
            f"collectives. Found non-EP c10d node(s) in {plan.region.root_fqn}: "
            f"{details}."
        )


def _owned_chunk_dependencies(
    body: set[fx.Node],
    candidates: set[fx.Node],
    symbol_hints: dict[object, int],
) -> set[fx.Node]:
    """Return selected-symbol producers owned by an eager chunk body.

    The selected module root defines the eager chunk body. In flattened FX,
    Python shape computations and shape-derived tensor values can appear as
    ancestors of body nodes even though they do not data-depend on an activation
    live-in. These are part of invoking the module on a chunk, not live-ins to
    split. Ownership is still bounded: a producer is included only when it is in
    the same candidate root and every semantic user is also inside this closure.
    """

    deps: set[fx.Node] = set()
    work = [inp for node in body for inp in node.all_input_nodes]
    while work:
        node = work.pop()
        if (
            node in body
            or node in deps
            or not _node_depends_on_chunk_symbol(node, symbol_hints)
        ):
            continue
        # Shape scalar helpers do not always carry module FQN metadata, but
        # they are still part of invoking the selected body when every escaping
        # user remains in the chunk closure.
        if node not in candidates and not (
            node.op == "call_function" and tensor_meta(node) is None
        ):
            continue
        deps.add(node)
        work.extend(node.all_input_nodes)

    changed = True
    while changed:
        changed = False
        owned = body | deps
        for dep in tuple(deps):
            if any(user not in owned and _is_semantic_user(user) for user in dep.users):
                deps.remove(dep)
                changed = True
    return deps


def _grad_output_suffix_nodes(
    gm: fx.GraphModule, grad_escape_boundaries: set[fx.Node]
) -> set[fx.Node]:
    """Return post-materialization grad-output plumbing outside chunk bodies."""
    suffix_nodes: set[fx.Node] = set()
    for output in _graph_output_nodes(gm)[1:]:
        node = output
        seen: set[fx.Node] = set()
        while (
            node not in seen
            and node not in grad_escape_boundaries
            and tensor_meta(node) is not None
        ):
            seen.add(node)
            suffix_nodes.add(node)
            inputs = _tensor_inputs(node)
            if len(inputs) != 1:
                break
            node = inputs[0]
    return suffix_nodes


def _semantic_body_closure(body: set[fx.Node]) -> set[fx.Node]:
    """Keep only nodes needed to produce chunked tensor outputs.

    Symbolic shape helpers are part of the body only when they feed real tensor
    compute inside the body. Shape-only template chains whose only semantic
    consumer is a full parent-region allocation should remain full; otherwise
    the pass would need to "materialize" scalar metadata that has no meaningful
    chunk reconstruction.
    """
    seeds = {
        node
        for node in body
        if _mutates_input(node)
        or any(user.op == "output" for user in node.users)
        or (
            tensor_meta(node) is not None
            and any(user not in body and _is_semantic_user(user) for user in node.users)
        )
    }
    required: set[fx.Node] = set()
    work = list(seeds)
    while work:
        node = work.pop()
        if node in required:
            continue
        required.add(node)
        work.extend(inp for inp in node.all_input_nodes if inp in body)
    return required


def _chunk_influenced_region_nodes(
    gm: fx.GraphModule,
    candidates: set[fx.Node],
    is_backward: bool,
    symbol_hints: dict[object, int],
    grad_escape_boundaries: set[fx.Node],
    per_chunk_sources: set[fx.Node],
) -> tuple[fx.Node, ...]:
    """Shrink module candidates to the body that must be duplicated per chunk."""
    influenced: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node not in candidates:
            continue
        if not is_backward and any(
            inp not in candidates
            and inp.op == "call_function"
            and tensor_meta(inp) is not None
            and _is_backward_node(inp)
            for inp in node.all_input_nodes
        ):
            raise ValueError(
                f"Chunk pass crossed into the opposite graph direction through {node.name}."
            )
        if any(
            inp in influenced
            or inp in per_chunk_sources
            or (inp not in candidates and _is_tensor_with_chunk_dim(inp, symbol_hints))
            for inp in node.all_input_nodes
        ):
            influenced.add(node)

    body: set[fx.Node] = set()
    work = list(influenced)
    while work:
        node = work.pop()
        if node in body:
            continue
        body.add(node)
        work.extend(
            inp
            for inp in node.all_input_nodes
            if inp in candidates and inp in influenced
        )
    body |= _owned_chunk_dependencies(body, candidates, symbol_hints)
    body = _semantic_body_closure(body)
    if is_backward:
        body -= _grad_output_suffix_nodes(gm, grad_escape_boundaries)
    return tuple(node for node in gm.graph.nodes if node in body)


def _find_regions(
    gm: fx.GraphModule,
    patterns: list[str],
    *,
    symbol_hints: dict[object, int],
    grad_escape_boundaries: set[fx.Node],
    per_chunk_sources: set[fx.Node],
) -> list[_Region]:
    """Build ordered forward/backward regions for all matched roots."""
    regions = []
    for matched in _region_roots(gm, patterns):
        for is_backward in (False, True):
            nodes = _region_nodes(
                gm,
                matched.root_fqn,
                is_backward=is_backward,
                symbol_hints=symbol_hints,
                grad_escape_boundaries=grad_escape_boundaries,
                per_chunk_sources=per_chunk_sources,
            )
            if nodes:
                regions.append(
                    _Region(
                        matched.root_fqn,
                        matched.synthetic_parent_fqn,
                        is_backward,
                        nodes,
                    )
                )
    order = ordered_nodes(gm)
    return sorted(regions, key=lambda region: order[region.nodes[0]])


# Step 1: Populate/prepare the selected dynamic dimension used as chunk truth.


def populate_chunk_dim_metadata_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
) -> fx.GraphModule:
    """Step 1: record selected dynamic symbol hints on placeholders."""
    del example_inputs
    if mode not in ("batch", "seq"):
        raise ValueError(f"Unknown chunk mode: {mode!r}")
    # The chunk pass uses the selected unbacked SymInt as its only source of
    # truth. This populate pass records only symbol->hint pairs on placeholders
    # so later cleanup can concretize exactly the symbols we introduced; it does
    # not propagate per-node chunk-dimension metadata.
    symbol_hints: dict[object, int] = chunk_symbol_hints_for_mode(gm, mode)
    if not symbol_hints:
        raise ValueError(
            "ep_overlap graph chunking expected at least one selected dynamic "
            "placeholder dimension, but none was found."
        )
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = tensor_meta(node)
        node_symbols = (
            {
                symbol: hint
                for symbol, hint in symbol_hints.items()
                if any(symbol in free_symbols(extent) for extent in val.shape)
            }
            if val is not None
            else {}
        )
        if node_symbols:
            node.meta[CHUNK_SYMBOL_HINTS_META] = node_symbols
    return gm


def mark_chunk_dynamic_dims(tensor: torch.Tensor, *, mode: ChunkMode) -> None:
    """Step 1: mark the user-selected tensor dimension unbacked with a hint."""
    from torch._dynamo.decorators import mark_unbacked

    dim = {"batch": 0, "seq": 1}.get(mode)
    if dim is None:
        raise ValueError(f"Unknown chunk mode: {mode!r}")
    if tensor.dim() <= dim:
        raise ValueError(
            f"Cannot mark {mode} dim {dim} for shape {tuple(tensor.shape)}."
        )
    chunk_dim_shape = int(tensor.shape[dim])
    if chunk_dim_shape < 2 or chunk_dim_shape % 2:
        raise ValueError(
            "EP overlap graph chunking requires an even selected dimension, "
            f"got {mode} size {chunk_dim_shape} for shape {tuple(tensor.shape)}."
        )
    mark_unbacked(
        tensor,
        dim,
        hint_override=chunk_dim_shape,
        # Graph chunking splits the selected dimension into two equal pieces.
        # Use the original half-size as the lower bound so downstream shape
        # checks can still prove non-singleton subchunks after further local
        # chunking (for example ChunkedCELoss over sequence).
        min=chunk_dim_shape // 2,
        max=chunk_dim_shape,
        specialize_on=[
            lambda extent, chunk_dim_shape=chunk_dim_shape: extent == chunk_dim_shape
        ],
        shape_id=f"torchtitan_chunk_{mode}",
    )


@register_trace_input_preparer("ep_overlap")
def prepare_ep_overlap_trace_inputs(
    compile_config: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Step 1: prepare trace inputs for graph chunking when EP overlap is enabled."""
    if not compile_config.ep_overlap.enabled:
        return
    mode, chunk_strategy, _ = validate_ep_overlap_config(compile_config.ep_overlap)
    if chunk_strategy == "eager":
        return
    dim = {"batch": 0, "seq": 1}[mode]
    if not args or not isinstance(args[0], torch.Tensor):
        raise ValueError("ep_overlap tracing expects first user input to be a Tensor")
    hint = int(args[0].shape[dim])

    def mark_leaf(value: object) -> None:
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > dim
            and int(value.shape[dim]) == hint
        ):
            # TODO(sanketpurandare): replace this bounded shape-equality
            # heuristic with explicit token-grid input role metadata.
            mark_chunk_dynamic_dims(value, mode=mode)

    # The traced training step is `(inputs, labels, global_valid_tokens,
    # extra_inputs, extra_kwargs)`. Mark only tensors with semantic training
    # roles tied to the model token grid; skip scalar loss bookkeeping.
    model_inputs = [args[0]]
    if len(args) > 1:
        model_inputs.append(args[1])
    if len(args) > 3:
        model_inputs.append(args[3])
    if len(args) > 4:
        model_inputs.append(args[4])
    model_inputs.append(kwargs)

    for leaf in tree_leaves(model_inputs):
        mark_leaf(leaf)


@register_trace_call_input_preparer("ep_overlap")
def prepare_ep_overlap_trace_call_inputs(
    compile_config: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Bind FlexAttention mask lengths to fake token-grid dims during tracing."""
    if not compile_config.ep_overlap.enabled:
        return None
    _, chunk_strategy, _ = validate_ep_overlap_config(compile_config.ep_overlap)
    if chunk_strategy == "eager" or len(args) <= 3 or not isinstance(args[3], dict):
        return None

    extra_kwargs = args[3]
    positions = extra_kwargs.get("positions")
    attention_masks = extra_kwargs.get("attention_masks")
    if not isinstance(positions, torch.Tensor) or attention_masks is None:
        return None
    if positions.dim() < 2:
        return None

    from torch.nn.attention.flex_attention import BlockMask

    seq_len = positions.shape[1]

    def rebind(mask: object) -> object:
        if not isinstance(mask, BlockMask):
            return mask
        return BlockMask(
            seq_lengths=(seq_len, seq_len),
            kv_num_blocks=mask.kv_num_blocks,
            kv_indices=mask.kv_indices,
            full_kv_num_blocks=mask.full_kv_num_blocks,
            full_kv_indices=mask.full_kv_indices,
            q_num_blocks=mask.q_num_blocks,
            q_indices=mask.q_indices,
            full_q_num_blocks=mask.full_q_num_blocks,
            full_q_indices=mask.full_q_indices,
            BLOCK_SIZE=mask.BLOCK_SIZE,
            mask_mod=mask.mask_mod,
            dq_write_order=mask.dq_write_order,
            dq_write_order_full=mask.dq_write_order_full,
            dq_kv_order=mask.dq_kv_order,
            dq_kv_order_spt=mask.dq_kv_order_spt,
        )

    if isinstance(attention_masks, dict):
        rebound_masks = {key: rebind(mask) for key, mask in attention_masks.items()}
    else:
        rebound_masks = rebind(attention_masks)
    if rebound_masks is attention_masks:
        return None

    rebound_extra_kwargs = dict(extra_kwargs)
    rebound_extra_kwargs["attention_masks"] = rebound_masks
    rebound_args = list(args)
    rebound_args[3] = rebound_extra_kwargs
    return tuple(rebound_args), kwargs


# Step 2: Discover selected-symbol body regions and classify their boundaries.


def _chunk_dim_from_symbols(
    val: torch.Tensor | None, symbol_hints: dict[object, int]
) -> _ChunkDim | None:
    """Map the selected unbacked symbol to the tensor dimension it appears in."""
    if val is None or not symbol_hints:
        return None
    matches: list[_ChunkDim] = []
    for dim, size in enumerate(val.shape):
        if not (free_symbols(size) & symbol_hints.keys()):
            continue
        hint = _eval_hint(size, symbol_hints)
        matches.append(_ChunkDim(dim, int(hint) if hint is not None else None))
    if len(matches) > 1:
        raise ValueError(
            f"Chunk pass found selected chunk symbols in multiple dims: {tuple(val.shape)}."
        )
    return matches[0] if matches else None


def _validate_selected_symbol_shapes(gm: fx.GraphModule, *, mode: ChunkMode) -> None:
    """Check every chunk-created selected-symbol tensor is still classifiable."""
    symbol_hints = chunk_symbol_hints_for_mode(gm, mode)
    if not symbol_hints:
        return
    for node in gm.graph.nodes:
        if node.meta.get("chunked_region_role") is None:
            continue
        val = tensor_meta(node)
        if val is None:
            continue
        if not any(free_symbols(size) & symbol_hints.keys() for size in val.shape):
            continue
        if _chunk_dim_from_symbols(val, symbol_hints) is None:
            raise ValueError(
                "Chunk pass could not classify selected-symbol shape for "
                f"{node.name}: {tuple(val.shape)}."
            )


def _chunk_dim_for_live_in(node: fx.Node, symbol_hints: dict[object, int]) -> int:
    dim_hint = _chunk_dim_from_symbols(tensor_meta(node), symbol_hints)
    val = tensor_meta(node)
    if dim_hint is None or val is None:
        raise ValueError(f"{node.name} does not carry the selected chunk symbol.")
    dim, hint = dim_hint.dim, dim_hint.hint
    if dim >= val.dim():
        raise ValueError(f"Selected chunk dim {dim} is invalid for {node.name}.")
    if hint is not None and hint % 2:
        raise ValueError(
            f"Cannot split selected chunk dimension of {node.name}: {hint}."
        )
    return dim


def _split_view_meta(val: torch.Tensor, dim: int) -> torch.Tensor:
    shape = list(val.shape)
    shape[dim] = shape[dim] // 2
    return val.new_empty_strided(tuple(shape), val.stride())


def _contiguous_chunk_meta(val: torch.Tensor, dim: int) -> torch.Tensor:
    shape = list(val.shape)
    shape[dim] = shape[dim] // 2
    return val.new_empty(tuple(shape))


def _is_contiguous_with_hints(
    val: torch.Tensor, symbol_hints: dict[object, int]
) -> bool:
    """Return contiguity using explicit hints instead of symbolic guards."""
    expected_stride = 1
    for size, stride in reversed(tuple(zip(val.shape, val.stride(), strict=True))):
        size_hint = _eval_hint(size, symbol_hints)
        stride_hint = _eval_hint(stride, symbol_hints)
        if size_hint is None or stride_hint is None:
            return False
        if size_hint == 1:
            continue
        if stride_hint != expected_stride:
            return False
        expected_stride *= size_hint
    return True


def _is_chunkable(
    node: fx.Node,
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
    symbol_hints: dict[object, int],
) -> bool:
    if node in static_nodes or tensor_meta(node) is None:
        return False
    if not _is_tensor_with_chunk_dim(node, symbol_hints):
        return False
    _chunk_dim_for_live_in(node, symbol_hints)
    return True


def _is_tensor_with_chunk_dim(node: fx.Node, symbol_hints: dict[object, int]) -> bool:
    return _chunk_dim_from_symbols(tensor_meta(node), symbol_hints) is not None


def _chunk_symbol_subs(symbol_hints: dict[object, int]) -> dict[object, object]:
    return {symbol: FloorDiv(symbol, 2) for symbol in symbol_hints}


def _materialize_symint_args(
    gm: fx.GraphModule,
    values: list[torch.SymInt | int] | tuple[torch.SymInt | int, ...],
    *,
    region: _Region,
    role: str,
    chunk_id: int | None = None,
) -> list[fx.Node | int]:
    """Materialize symbolic scalar args as FX nodes at the current insertion point."""
    before = set(gm.graph.nodes)
    materialized = gm.graph.materialize_symints(values)
    for node in gm.graph.nodes:
        if node not in before:
            _set_synthetic_meta(node, region=region, role=role, chunk_id=chunk_id)
    return materialized


def _materialize_symint_arg(
    gm: fx.GraphModule,
    value: torch.SymInt | int,
    *,
    region: _Region,
    role: str,
    chunk_id: int | None = None,
) -> fx.Node | int:
    return _materialize_symint_args(
        gm, [value], region=region, role=role, chunk_id=chunk_id
    )[0]


def _create_size_or_stride_node(
    gm: fx.GraphModule,
    target: object,
    tensor_node: fx.Node,
    dim: int,
    *,
    region: _Region,
    role: str,
    chunk_id: int | None = None,
) -> fx.Node:
    if target is aten.sym_size.int:
        node = gm.graph.create_size_node(tensor_node, dim)
    elif target is aten.sym_stride.int:
        node = gm.graph.create_stride_node(tensor_node, dim)
    else:
        raise ValueError(f"Unsupported symbolic tensor property target: {target}.")
    _set_synthetic_meta(node, region=region, role=role, chunk_id=chunk_id)
    return node


def _create_half_extent_node(
    gm: fx.GraphModule, live_in: fx.Node, dim: int, *, region: _Region
) -> tuple[fx.Node, fx.Node]:
    size = gm.graph.create_size_node(live_in, dim)
    _set_synthetic_meta(size, region=region, role="split_boundary")
    half = gm.graph.call_function(operator.floordiv, args=(size, 2))
    _set_synthetic_meta(half, region=region, role="chunk_input")
    val = tensor_meta(live_in)
    if val is not None:
        half.meta["val"] = val.shape[dim] // 2
    return size, half


def _rewrite_chunk_symint(
    value: object,
    symbol_hints: dict[object, int],
) -> object:
    return _rewrite_symbolic_value(
        value, _chunk_symbol_subs(symbol_hints), symbol_hints
    )


def _rewrite_symbolic_value(
    value: object,
    symbol_subs: dict[object, object],
    symbol_hints: dict[object, int],
) -> object:
    symbols = free_symbols(value)
    if not symbols or not (symbols & symbol_subs.keys()):
        return value
    if not hasattr(value, "node"):
        return value.subs(symbol_subs) if hasattr(value, "subs") else value

    expr = value.node.expr.subs(symbol_subs)
    shape_env = value.node.shape_env
    hint = _eval_hint(expr, symbol_hints)
    if isinstance(value, torch.SymInt):
        return shape_env.create_symintnode(expr, hint=hint)
    if isinstance(value, torch.SymFloat):
        return shape_env.create_symfloatnode(expr, hint=hint)
    if isinstance(value, torch.SymBool):
        return shape_env.create_symboolnode(expr)
    return value


def _symbolic_scalars(value: object) -> list[object]:
    scalars = []
    for leaf in tree_leaves(value):
        if isinstance(leaf, torch.Tensor):
            scalars.extend((*leaf.shape, *leaf.stride(), leaf.storage_offset()))
        else:
            scalars.append(leaf)
    return scalars


def _shape_env_for_symbol(meta: dict[str, Any], symbol: sympy.Symbol) -> object:
    for scalar in _symbolic_scalars(meta.get("val")):
        if hasattr(scalar, "node") and symbol in free_symbols(scalar):
            return scalar.node.shape_env
    raise ValueError(
        "Chunk pass found an unbacked binding whose symbol is not present in "
        f"the node metadata value: {symbol}."
    )


def _symbol_has_selected_runtime_assert(
    shape_env: object,
    symbol: sympy.Symbol,
    symbol_hints: dict[object, int],
) -> bool:
    for runtime_assert in getattr(shape_env, "deferred_runtime_asserts", {}).get(
        symbol, ()
    ):
        expr = getattr(runtime_assert, "expr", runtime_assert)
        if free_symbols(expr) & symbol_hints.keys():
            return True
    return False


def _hint_for_copied_unbacked_symbol(
    shape_env: object,
    symbol: sympy.Symbol,
    hint: int,
    symbol_hints: dict[object, int],
) -> int:
    if not _symbol_has_selected_runtime_assert(shape_env, symbol, symbol_hints):
        return hint
    # The copied scalar is constrained by a value whose selected dimension was
    # split by this pass, so its representative value must be chunk-local too.
    return max(1, (hint + 1) // 2)


def _fresh_unbacked_symbol_for_copy(
    shape_env: object,
    symbol: sympy.Symbol,
    symbol_hints: dict[object, int],
) -> sympy.Symbol:
    symint = shape_env.create_unbacked_symint()
    new_symbol = symint.node.expr
    # This pass is cloning metadata, not running a fake kernel. The new symbol
    # is immediately bound through copied node.meta["unbacked_bindings"].
    if new_symbol in shape_env.pending_fresh_unbacked_symbols:
        shape_env.pending_fresh_unbacked_symbols.remove(new_symbol)
    if symbol in shape_env.var_to_range:
        shape_env.var_to_range[new_symbol] = shape_env.var_to_range[symbol]
    if symbol in shape_env.var_to_range_sloc:
        shape_env.var_to_range_sloc[new_symbol] = shape_env.var_to_range_sloc[symbol]
    if symbol in shape_env.var_to_hint_override:
        hint = shape_env.var_to_hint_override[symbol]
        shape_env.var_to_hint_override[new_symbol] = _hint_for_copied_unbacked_symbol(
            shape_env, symbol, hint, symbol_hints
        )
    if symbol in shape_env.size_like:
        shape_env.size_like.add(new_symbol)
    return new_symbol


def _rewrite_unbacked_keypath(
    keypath: tuple[object, ...],
    symbol_subs: dict[object, object],
    symbol_hints: dict[object, int],
) -> tuple[object, ...]:
    from torch.fx.experimental.symbolic_shapes import DivideByKey

    rewritten = []
    for key in keypath:
        if isinstance(key, DivideByKey):
            rewritten.append(
                DivideByKey(
                    _rewrite_symbolic_value(key.divisor, symbol_subs, symbol_hints)
                )
            )
        else:
            rewritten.append(key)
    return tuple(rewritten)


def _freshen_copied_unbacked_bindings(
    copied_meta: dict[str, Any],
    symbol_replacements: dict[object, object],
    symbol_hints: dict[object, int],
) -> None:
    bindings = copied_meta.get("unbacked_bindings")
    if not bindings:
        return

    fresh_bindings = {}
    for symbol, keypath in bindings.items():
        shape_env = _shape_env_for_symbol(copied_meta, symbol)
        new_symbol = _fresh_unbacked_symbol_for_copy(shape_env, symbol, symbol_hints)
        symbol_replacements[symbol] = new_symbol
        fresh_bindings[new_symbol] = keypath
    copied_meta["unbacked_bindings"] = fresh_bindings


def _chunk_tensor_meta(
    val: torch.Tensor,
    symbol_hints: dict[object, int],
    symbol_replacements: dict[object, object] | None = None,
) -> torch.Tensor:
    symbol_subs = {
        **(symbol_replacements or {}),
        **_chunk_symbol_subs(symbol_hints),
    }
    if not any(
        free_symbols(size_or_stride) & symbol_subs.keys()
        for size_or_stride in (*val.shape, *val.stride(), val.storage_offset())
    ):
        return val
    shape = tuple(
        _rewrite_symbolic_value(dim, symbol_subs, symbol_hints) for dim in val.shape
    )
    stride = tuple(
        _rewrite_symbolic_value(dim, symbol_subs, symbol_hints) for dim in val.stride()
    )
    return val.new_empty_strided(shape, stride)


def _chunk_meta_value(
    value: object,
    symbol_hints: dict[object, int],
    symbol_replacements: dict[object, object],
) -> object:
    if isinstance(value, torch.Tensor):
        return _chunk_tensor_meta(value, symbol_hints, symbol_replacements)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        symbol_subs = {**symbol_replacements, **_chunk_symbol_subs(symbol_hints)}
        return _rewrite_symbolic_value(value, symbol_subs, symbol_hints)
    return value


def _chunk_copied_meta(
    meta: dict[str, Any],
    symbol_hints: dict[object, int],
    symbol_replacements: dict[object, object] | None = None,
) -> dict[str, Any]:
    symbol_replacements = symbol_replacements if symbol_replacements is not None else {}
    copied = _copy_meta(meta)
    _freshen_copied_unbacked_bindings(copied, symbol_replacements, symbol_hints)
    symbol_subs = {**symbol_replacements, **_chunk_symbol_subs(symbol_hints)}
    if "unbacked_bindings" in copied:
        copied["unbacked_bindings"] = {
            symbol: _rewrite_unbacked_keypath(tuple(keypath), symbol_subs, symbol_hints)
            for symbol, keypath in copied["unbacked_bindings"].items()
        }
    if "val" in copied:
        copied["val"] = tree_map(
            lambda value: _chunk_meta_value(value, symbol_hints, symbol_replacements),
            copied["val"],
        )
    return copied


def _validate_materialized(
    materialized: fx.Node,
    original: fx.Node,
    *,
    original_val: torch.Tensor | None = None,
    symbol_hints: dict[object, int],
) -> None:
    """Verify a materialized full value matches the original fake shape."""
    mat_val, orig_val = tensor_meta(materialized), tensor_meta(original)
    orig_val = original_val if original_val is not None else orig_val
    if (
        mat_val is not None
        and orig_val is not None
        and not same_shape_with_hints(mat_val, orig_val, original, symbol_hints)
    ):
        raise RuntimeError(
            f"Chunk pass materialized {materialized.name} with shape "
            f"{tuple(mat_val.shape)}, expected {tuple(orig_val.shape)} from {original.name}."
        )


def _graph_output_nodes(gm: fx.GraphModule) -> list[fx.Node]:
    output = next(node for node in gm.graph.nodes if node.op == "output")
    return [leaf for leaf in tree_leaves(output.args[0]) if isinstance(leaf, fx.Node)]


def _tensor_inputs(node: fx.Node) -> list[fx.Node]:
    return [inp for inp in node.all_input_nodes if tensor_meta(inp) is not None]


def _is_shape_preserving_reduction(
    node: fx.Node, symbol_hints: dict[object, int]
) -> bool:
    if node.target is torch.ops._c10d_functional.all_reduce.default:
        return True
    if node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default:
        inputs = _tensor_inputs(node)
        node_val = tensor_meta(node)
        input_val = tensor_meta(inputs[0]) if len(inputs) == 1 else None
        return (
            node_val is not None
            and input_val is not None
            and same_shape_with_hints(node_val, input_val, node, symbol_hints)
        )
    return node.target is torch.ops._c10d_functional.wait_tensor.default


def _is_materialization_suffix_node(
    node: fx.Node,
    *,
    optimize_grad_live_out: bool,
    symbol_hints: dict[object, int],
) -> bool:
    del optimize_grad_live_out
    if node.op != "call_function" or len(_tensor_inputs(node)) != 1:
        return False
    if is_view_target(node.target) or _is_shape_preserving_reduction(
        node, symbol_hints
    ):
        return True
    # Dtype/device casts are intentional materialization boundaries.  We still
    # materialize the collective suffix once, but we do not move accumulation before
    # casts because that changes eager chunking's cast/add order.
    return False


def _materialization_boundary(
    live_out: fx.Node,
    *,
    plan: _Plan,
    optimize_grad_live_out: bool,
    symbol_hints: dict[object, int],
) -> _MaterializationBoundary:
    """Choose the earliest safe boundary before materializing a grad suffix once."""
    if not plan.region.is_backward or not optimize_grad_live_out:
        return _MaterializationBoundary(live_out, ())
    suffix: list[fx.Node] = []
    node = live_out
    while node in plan.body and _is_materialization_suffix_node(
        node,
        optimize_grad_live_out=optimize_grad_live_out,
        symbol_hints=symbol_hints,
    ):
        inputs = _tensor_inputs(node)
        assert len(inputs) == 1
        inp = inputs[0]
        # Keep this optimization chain-local. Branches are real compute or
        # accumulation points and must be handled by the normal live-out logic.
        if any(user is not node and user in plan.body for user in inp.users):
            break
        suffix.append(node)
        node = inp
    return _MaterializationBoundary(node, tuple(reversed(suffix)))


def _grad_escape_boundaries(
    gm: fx.GraphModule,
    *,
    mode: ChunkMode,
    symbol_hints: dict[object, int],
    optimize_grad_live_out: bool,
) -> set[fx.Node]:
    """Step 2: identify backward no-chunk-dim grad escape boundaries."""
    del mode, optimize_grad_live_out
    boundaries: set[fx.Node] = set()
    # GraphTrainer returns [loss, *param_grads]. The loss is not a grad escape
    # path; each remaining output is authoritative evidence for one grad escape
    # path. Optimized graph chunking walks through single-input grad plumbing to
    # the first no-chunk-dim producer, so casts and distributed grad
    # materialization remain outside the chunk body. Dtype/device-domain changes
    # are always boundaries; moving accumulation across them changes eager
    # chunking's exact cast/add order.
    for output in _graph_output_nodes(gm)[1:]:
        if tensor_meta(output) is None or _is_tensor_with_chunk_dim(
            output, symbol_hints
        ):
            continue
        node = output
        seen: set[fx.Node] = set()
        while node not in seen and tensor_meta(node) is not None:
            seen.add(node)
            inputs = _tensor_inputs(node)
            if len(inputs) != 1:
                boundaries.add(node)
                break
            inp = inputs[0]
            if _is_tensor_with_chunk_dim(inp, symbol_hints):
                boundaries.add(node)
                break
            if not same_tensor_domain(node, inp):
                boundaries.add(node)
                break
            node = inp
    return boundaries


# Step 2 continued: build concrete region plans.


def _plan_regions(
    gm: fx.GraphModule,
    patterns: list[str],
    *,
    mode: ChunkMode,
    optimize_grad_live_out: bool,
    per_chunk_sources: set[fx.Node] | None = None,
) -> list[_Plan]:
    """Step 2: plan all selected regions before lowering one direction."""
    symbol_hints = chunk_symbol_hints_for_mode(gm, mode)
    grad_escape_boundaries = _grad_escape_boundaries(
        gm,
        mode=mode,
        symbol_hints=symbol_hints,
        optimize_grad_live_out=optimize_grad_live_out,
    )
    per_chunk_sources = per_chunk_sources or set()
    plans = []
    for region in _find_regions(
        gm,
        patterns,
        symbol_hints=symbol_hints,
        grad_escape_boundaries=grad_escape_boundaries,
        per_chunk_sources=per_chunk_sources,
    ):
        plans.append(_build_plan(region, grad_escape_boundaries))
    logger.debug(
        "Chunk pass planned %d region(s): mode=%s patterns=%s "
        "grad_escape_boundaries=%d per_chunk_sources=%d",
        len(plans),
        mode,
        patterns,
        len(grad_escape_boundaries),
        len(per_chunk_sources),
    )
    return plans


def _build_plan(
    region: _Region, grad_escape_boundaries: set[fx.Node] | frozenset[fx.Node]
) -> _Plan:
    """Step 2: compute live-ins and live-outs for the current graph."""
    body = frozenset(region.nodes)
    live_ins = frozenset(
        inp for node in region.nodes for inp in node.all_input_nodes if inp not in body
    )
    live_out_users = {}
    for node in region.nodes:
        users = tuple(
            user for user in node.users if user not in body and _is_semantic_user(user)
        )
        if users:
            live_out_users[node] = users
    return _Plan(
        region, body, live_ins, live_out_users, frozenset(grad_escape_boundaries)
    )


def _map_arg(
    arg: object,
    *,
    copied: dict[fx.Node, fx.Node],
    chunks: dict[fx.Node, tuple[fx.Node, fx.Node]],
    scalars: dict[object, object],
    symbol_replacements: dict[object, object],
    symbol_hints: dict[object, int],
    chunk_id: int,
    map_symbolic_exprs: bool = False,
) -> object:
    """Map original node/scalar arguments to the selected chunk equivalents."""

    def scalar_for_chunk(value: object) -> object:
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(item, fx.Node) for item in value)
        ):
            return value[chunk_id]
        return value

    if isinstance(arg, fx.Node):
        if arg in copied:
            return copied[arg]
        if arg in chunks:
            return chunks[arg][chunk_id]
        if arg in scalars:
            return scalar_for_chunk(scalars[arg])
        return arg
    if not map_symbolic_exprs:
        return arg
    if isinstance(arg, torch.SymInt):
        expr = arg.node.expr
    elif hasattr(arg, "free_symbols"):
        expr = arg
    else:
        return arg
    symbols = free_symbols(arg)
    if len(symbols) == 1:
        if expr_key(expr) in scalars:
            return scalar_for_chunk(scalars[expr_key(expr)])
    symbol_subs = {**symbol_replacements, **_chunk_symbol_subs(symbol_hints)}
    return _rewrite_symbolic_value(arg, symbol_subs, symbol_hints)


# Step 4 helper definitions: shape-copy helpers are defined before Step 3
# lowering because live-in splitting and body copying both use them.


def _linear_extent_coeff(
    value: object, symbols: set[object]
) -> tuple[object, int] | None:
    value_symbols = free_symbols(value)
    if len(value_symbols) != 1 or not value_symbols <= symbols:
        return None
    symbol = next(iter(value_symbols))
    expr = symbolic_expr(value)
    try:
        zero = expr.subs(symbol, 0)
        one = expr.subs(symbol, 1)
        if free_symbols(zero):
            return None
        coeff = int(one - zero)
        if int(zero) != 0 or coeff <= 0:
            return None
        return symbol, coeff
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _is_contiguous_chunk_input_stride(
    value: object,
    split_live_ins: list[fx.Node],
    symbol_hints: dict[object, int],
) -> bool:
    """Return whether ``value`` is a stride from the contiguous chunk input."""
    for live_in in split_live_ins:
        val = tensor_meta(live_in)
        if val is None:
            continue
        dim_hint = _chunk_dim_from_symbols(val, symbol_hints)
        if dim_hint is None:
            continue
        chunk_val = _contiguous_chunk_meta(val, dim_hint.dim)
        if any(same_symbolic_expr(value, stride) for stride in chunk_val.stride()):
            return True
    return False


def _shape_arg_depends_on_chunk_symbol(
    shape_arg: object, symbol_hints: dict[object, int]
) -> bool:
    for item in tree_leaves(shape_arg):
        if isinstance(item, fx.Node):
            if depends_on_chunk_symbol(item.meta.get("val"), symbol_hints):
                return True
        elif depends_on_chunk_symbol(item, symbol_hints):
            return True
    return False


def _source_stack(node: fx.Node) -> str:
    stack = getattr(node, "stack_trace", None) or node.meta.get("stack_trace")
    return stack if isinstance(stack, str) and stack else "<source stack unavailable>"


def _validate_view_shape_uses_symbolic_extent(
    node: fx.Node,
    args: tuple[Any, ...],
    *,
    original_meta: dict[str, Any],
    symbol_hints: dict[object, int],
) -> None:
    """Reject baked Python shape constants in chunk-influenced view ops."""
    shape_arg_idx = view_shape_arg_index(node.target)
    if shape_arg_idx is None or len(args) <= shape_arg_idx:
        return
    size_arg = args[shape_arg_idx]
    original_output_val = original_meta.get("val")
    if (
        not isinstance(original_output_val, torch.Tensor)
        or not isinstance(size_arg, (list, tuple))
        or any(dim == -1 for dim in size_arg)
        or _chunk_dim_from_symbols(original_output_val, symbol_hints) is None
        or _shape_arg_depends_on_chunk_symbol(size_arg, symbol_hints)
    ):
        return

    target = getattr(node.target, "__name__", str(node.target))
    module_fqn = _get_module_fqn(node)
    raise ValueError(
        "Graph EP chunking found a view/reshape shape that baked a Python "
        "constant where the selected unbacked SymInt extent should flow. "
        f"node={node.name}, target={target}, module_fqn={module_fqn!r}, "
        f"shape_arg={tuple(size_arg)}, output_shape={tuple(original_output_val.shape)}.\n"
        "Use tensor-derived shape values such as x.shape[i] / x.size(i), or "
        "use -1 when this dimension should be inferred by view/reshape.\n"
        f"Source stack:\n{_source_stack(node)}"
    )


def _copy_body(
    gm: fx.GraphModule,
    plan: _Plan,
    *,
    chunks: dict[fx.Node, tuple[fx.Node, fx.Node]],
    scalars: dict[object, object],
    symbol_hints: dict[object, int],
    mode: ChunkMode,
    chunk_id: int,
    originals: dict[
        fx.Node, tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], str]
    ],
    in_place: bool,
    prior_chunk_copies: dict[fx.Node, fx.Node] | None = None,
    preserve_originals: set[fx.Node] | None = None,
) -> dict[fx.Node, fx.Node]:
    """Step 4: rewrite/copy one chunk body with remapped args and metadata."""
    preserve_originals = preserve_originals or set()
    copied: dict[fx.Node, fx.Node] = {}
    symbol_replacements: dict[object, object] = {}

    def materialize_raw_chunk_symbol_arg(arg: object, *, chunk_id: int) -> object:
        if not isinstance(arg, torch.SymInt):
            return arg
        if not (free_symbols(arg) & symbol_hints.keys()):
            return arg
        chunked = _rewrite_chunk_symint(arg, symbol_hints)
        if not isinstance(chunked, torch.SymInt):
            return chunked
        return _materialize_symint_arg(
            gm,
            chunked,
            region=plan.region,
            role="chunk_input",
            chunk_id=chunk_id,
        )

    for node in plan.region.nodes:
        args, kwargs, meta, name = originals[node]
        insert_before = (
            node if in_place and node not in preserve_originals else node.next
        )
        with gm.graph.inserting_before(insert_before):
            new_args = tree_map(
                lambda arg: _map_arg(
                    arg,
                    copied=copied,
                    chunks=chunks,
                    scalars=scalars,
                    symbol_replacements=symbol_replacements,
                    symbol_hints=symbol_hints,
                    chunk_id=chunk_id,
                    map_symbolic_exprs=True,
                ),
                args,
            )
            new_kwargs = tree_map(
                lambda arg: _map_arg(
                    arg,
                    copied=copied,
                    chunks=chunks,
                    scalars=scalars,
                    symbol_replacements=symbol_replacements,
                    symbol_hints=symbol_hints,
                    chunk_id=chunk_id,
                    map_symbolic_exprs=True,
                ),
                kwargs,
            )
            new_args = tree_map(
                lambda arg: materialize_raw_chunk_symbol_arg(arg, chunk_id=chunk_id),
                new_args,
            )
            new_kwargs = tree_map(
                lambda arg: materialize_raw_chunk_symbol_arg(arg, chunk_id=chunk_id),
                new_kwargs,
            )
            if isinstance(new_args, tuple):
                _validate_view_shape_uses_symbolic_extent(
                    node,
                    new_args,
                    original_meta=meta,
                    symbol_hints=symbol_hints,
                )
            if (
                prior_chunk_copies is not None
                and node in prior_chunk_copies
                and isinstance(args, tuple)
                and isinstance(new_args, tuple)
            ):
                mutable_indices = [
                    idx
                    for idx in _mutable_arg_indices(node)
                    if idx < len(args)
                    and idx < len(new_args)
                    and isinstance(args[idx], fx.Node)
                    and args[idx] not in copied
                ]
                if mutable_indices:
                    if len(mutable_indices) != 1:
                        raise ValueError(
                            "Graph EP chunking only supports duplicated mutable ops "
                            "with one schema-declared mutable positional input. "
                            f"Found mutable arg indices {mutable_indices} on "
                            f"{node.name} ({node.target})."
                        )
                    # Match eager chunking for shared mutable live-ins such as MoE
                    # token-count buffers: later chunks consume earlier chunks'
                    # mutation results through the schema-declared write aliases.
                    updated_args = list(new_args)
                    for idx in mutable_indices:
                        updated_args[idx] = prior_chunk_copies[node]
                    new_args = tuple(updated_args)
            if in_place and node not in preserve_originals:
                new_node = node
                new_node.args = new_args
                new_node.kwargs = new_kwargs
            else:
                new_node = gm.graph.create_node(
                    node.op, node.target, new_args, new_kwargs, type_expr=node.type
                )
        new_node._rename(f"{name}_chunk{chunk_id}")
        new_node.meta = _chunk_copied_meta(meta, symbol_hints, symbol_replacements)
        _set_chunk_meta(
            new_node,
            region=plan.region,
            role="body",
            chunk_id=chunk_id,
        )
        new_node.meta["chunked_original_name"] = name
        copied[node] = new_node
    return copied


def _validate_no_raw_chunk_symbol_args(
    gm: fx.GraphModule, symbol_hints: dict[object, int]
) -> None:
    """Reject chunk-created executable args that still embed selected SymInts."""
    if not symbol_hints:
        return
    chunk_symbols = symbol_hints.keys()
    for node in gm.graph.nodes:
        if node.meta.get("chunked_region_role") is None:
            continue
        for value in tree_leaves((node.args, node.kwargs)):
            if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)) and (
                free_symbols(value) & chunk_symbols
            ):
                raise ValueError(
                    "Graph EP chunking left a raw selected symbolic scalar in "
                    f"executable args for {node.name}: {value}."
                )


# Step 3: Split live-ins and recover provenance.


def _split_live_ins(
    gm: fx.GraphModule,
    plan: _Plan,
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
    per_chunk: dict[fx.Node, tuple[fx.Node, fx.Node]],
) -> tuple[
    dict[fx.Node, tuple[fx.Node, fx.Node]], dict[object, object], dict[object, int]
]:
    """Step 3: split chunked live-ins or recover existing provenance chunks."""
    order = ordered_nodes(gm)
    chunks = {
        node: values for node, values in per_chunk.items() if node in plan.live_ins
    }
    chunked_by_origin = {
        (
            node.meta.get("chunked_region_fqn"),
            node.meta.get("chunked_region_is_backward"),
            node.meta.get("chunked_original_name"),
            node.meta.get("chunk_id"),
        ): node
        for node in gm.graph.nodes
        if node.meta.get("chunked_original_name") is not None
        and node.meta.get("chunk_id") in (0, 1)
    }
    for live_in in plan.live_ins:
        if live_in in chunks:
            continue
        key = (
            live_in.meta.get("chunked_region_fqn"),
            live_in.meta.get("chunked_region_is_backward"),
            live_in.meta.get("chunked_original_name"),
        )
        if key[0] is not None and key[2] is not None:
            node0 = chunked_by_origin.get((*key, 0))
            node1 = chunked_by_origin.get((*key, 1))
            if isinstance(node0, fx.Node) and isinstance(node1, fx.Node):
                chunks[live_in] = (node0, node1)
    scalars: dict[object, object] = {}
    symbol_hints = chunk_symbol_hints_for_mode(gm, mode)
    selected_symbols: set[object] = set(symbol_hints)
    split_live_ins = [
        live_in
        for live_in in plan.live_ins
        if live_in not in chunks
        and _is_chunkable(
            live_in,
            mode=mode,
            static_nodes=static_nodes,
            symbol_hints=symbol_hints,
        )
    ]
    for live_in in plan.live_ins:
        if (
            not plan.region.is_backward
            and live_in.op == "call_function"
            and live_in not in chunks
            and (live_in_val := tensor_meta(live_in)) is not None
            and live_in_val.dim() > 0
            and _is_backward_node(live_in)
        ):
            raise ValueError(
                "Chunk pass crossed into the opposite graph direction through "
                f"live-in {live_in.name} for region {plan.region.root_fqn!r}."
            )
    if not split_live_ins and not chunks:
        raise ValueError(
            f"Chunk pass found no chunkable activation live-ins for "
            f"{plan.region.root_fqn!r}."
        )
    logger.debug(
        "Chunk pass live-ins: root=%s direction=%s live_ins=%d split=%d provenance=%d",
        plan.region.root_fqn,
        "backward" if plan.region.is_backward else "forward",
        len(plan.live_ins),
        len(split_live_ins),
        len(chunks),
    )

    def first_body_user(node: fx.Node) -> fx.Node:
        return min((user for user in node.users if user in plan.body), key=order.get)

    def first_chunk_symbol_consumer(node: fx.Node) -> fx.Node:
        def value_depends_on_chunk_symbol(value: object) -> bool:
            if isinstance(value, fx.Node):
                return _node_depends_on_chunk_symbol(value, symbol_hints)
            return depends_on_chunk_symbol(value, symbol_hints)

        consumers = [
            body_node
            for body_node in plan.region.nodes
            if body_node in plan.body
            and order[body_node] > order[node]
            and (
                node in body_node.all_input_nodes
                or any(
                    value_depends_on_chunk_symbol(value)
                    for value in tree_leaves((body_node.args, body_node.kwargs))
                )
            )
        ]
        return min(consumers, key=order.get)

    for live_in in sorted(split_live_ins, key=order.get):
        dim = _chunk_dim_for_live_in(live_in, symbol_hints)
        val = tensor_meta(live_in)
        assert val is not None
        with gm.graph.inserting_before(first_chunk_symbol_consumer(live_in)):
            _, half = _create_half_extent_node(gm, live_in, dim, region=plan.region)
            scalars[expr_key(val.shape[dim])] = half
            # Do not promote every symbol in a flattened extent to a chunk
            # symbol. For seq chunking, token extents can contain batch*seq;
            # only the user-selected seq symbol should drive scalar rewrites.
        with gm.graph.inserting_before(first_chunk_symbol_consumer(live_in)):
            split = gm.graph.call_function(
                aten.split_with_sizes.default,
                args=(live_in, [half, half], dim),
            )
            _set_synthetic_meta(split, region=plan.region, role="split_boundary")
            get0 = gm.graph.call_function(operator.getitem, args=(split, 0))
            get1 = gm.graph.call_function(operator.getitem, args=(split, 1))
            for chunk_id, get in ((0, get0), (1, get1)):
                _set_synthetic_meta(
                    get,
                    region=plan.region,
                    role="split_boundary",
                    chunk_id=chunk_id,
                )
                get.meta["val"] = _split_view_meta(val, dim)
            # Match eager ``chunk.contiguous()`` exactly: clone only when the
            # split view is non-contiguous. Batch splits are often already
            # contiguous, and forcing a clone there changes bitwise numerics.
            contig_chunks = []
            for chunk_id, get in ((0, get0), (1, get1)):
                split_val = tensor_meta(get)
                if split_val is not None and _is_contiguous_with_hints(
                    split_val, symbol_hints
                ):
                    contig_chunks.append(get)
                    continue
                contig = gm.graph.call_function(
                    aten.clone.default,
                    args=(get,),
                    kwargs={"memory_format": torch.contiguous_format},
                )
                _set_synthetic_meta(
                    contig,
                    region=plan.region,
                    role="chunk_input",
                    chunk_id=chunk_id,
                )
                contig.meta["val"] = _contiguous_chunk_meta(val, dim)
                contig_chunks.append(contig)
            chunks[live_in] = (contig_chunks[0], contig_chunks[1])

    for live_in in sorted(plan.live_ins, key=order.get):
        if live_in in chunks or live_in in scalars:
            continue
        val = live_in.meta.get("val")
        if isinstance(val, torch.SymInt):
            expr = val.node.expr
        elif hasattr(val, "free_symbols"):
            expr = val
        else:
            continue
        symbols = free_symbols(val)
        if not symbols:
            continue
        if not symbols & selected_symbols:
            continue
        if (
            live_in.op == "call_function"
            and live_in.target in (aten.sym_size.int, aten.sym_stride.int)
            and len(live_in.args) >= 2
            and isinstance(live_in.args[0], fx.Node)
            and live_in.args[0] in chunks
            and isinstance(live_in.args[1], int)
        ):
            source, prop_dim = live_in.args[:2]
            props = []
            with gm.graph.inserting_after(chunks[source][1]):
                for chunk_id, chunk in enumerate(chunks[source]):
                    prop = _create_size_or_stride_node(
                        gm,
                        live_in.target,
                        chunk,
                        prop_dim,
                        region=plan.region,
                        role="chunk_input",
                        chunk_id=chunk_id,
                    )
                    chunk_val = tensor_meta(chunk)
                    if chunk_val is None or prop_dim >= chunk_val.dim():
                        raise ValueError(
                            "Chunk pass could not derive chunk-local tensor "
                            f"metadata scalar for {live_in.name}."
                        )
                    prop.meta["val"] = (
                        chunk_val.shape[prop_dim]
                        if live_in.target is aten.sym_size.int
                        else chunk_val.stride()[prop_dim]
                    )
                    props.append(prop)
            scalars[live_in] = (props[0], props[1])
            scalars[expr_key(expr)] = (props[0], props[1])
            continue
        hint = _eval_hint(val, symbol_hints)
        if _is_contiguous_chunk_input_stride(val, split_live_ins, symbol_hints):
            continue
        if (
            not symbols <= selected_symbols
            or _linear_extent_coeff(val, selected_symbols) is None
        ):
            if hint is not None and statically_equals_hint(val, hint):
                continue
            chunked_val = _rewrite_chunk_symint(val, symbol_hints)
            if isinstance(chunked_val, torch.SymInt):
                with gm.graph.inserting_before(first_body_user(live_in)):
                    chunked_val = _materialize_symint_arg(
                        gm,
                        chunked_val,
                        region=plan.region,
                        role="chunk_input",
                    )
            scalars[live_in] = chunked_val
            scalars[expr_key(expr)] = chunked_val
            continue
        if hint is None:
            raise ValueError(
                "Chunk pass found a symbolic scalar live-in without a chunk "
                f"hint: {live_in.name}={val}."
            )
        if hint % 2:
            raise ValueError(
                "Chunk pass found a symbolic scalar live-in without an even "
                f"chunk hint: {live_in.name}={val}."
            )
        with gm.graph.inserting_before(first_body_user(live_in)):
            half = gm.graph.call_function(operator.floordiv, args=(live_in, 2))
            _set_synthetic_meta(half, region=plan.region, role="chunk_input")
            half.meta["val"] = val // 2
        scalars[live_in] = half
        scalars[expr_key(expr)] = half
    return chunks, scalars, symbol_hints


# Step 5: Route per-chunk provenance or materialize full live-outs.


def _split_live_out_users(
    users: tuple[fx.Node, ...],
    plans: list[_Plan],
    producer_region: _Region,
    symbol_hints: dict[object, int],
) -> tuple[tuple[fx.Node, ...], tuple[fx.Node, ...]]:
    """Step 5: classify live-out consumers as chunked or full."""
    owner_by_node = {node: plan.region for plan in plans for node in plan.body}
    chunked = []
    full = []
    for user in users:
        consumer_region = owner_by_node.get(user)
        same_root_consumer = (
            consumer_region is not None
            and consumer_region.root_fqn == producer_region.root_fqn
        )
        same_root_backward_consumer = (
            same_root_consumer
            and not producer_region.is_backward
            and consumer_region.is_backward
        )
        if same_root_backward_consumer or (
            not producer_region.is_backward
            and _is_backward_node(user)
            and (
                is_module_fqn_inside_root(
                    _get_module_fqn(user), producer_region.root_fqn
                )
                or (
                    not _get_module_fqn(user)
                    and _node_depends_on_chunk_symbol(user, symbol_hints)
                    and all(
                        is_module_fqn_inside_root(fqn, producer_region.root_fqn)
                        for neighbor in (*user.all_input_nodes, *user.users)
                        if (fqn := _get_module_fqn(neighbor))
                    )
                )
            )
        ):
            chunked.append(user)
        else:
            full.append(user)
    return tuple(chunked), tuple(full)


def _add_materialization_proof(
    live_out: fx.Node,
    users: tuple[fx.Node, ...],
    *,
    is_backward: bool,
    grad_escape_boundaries: frozenset[fx.Node],
) -> str | None:
    """Step 5: prove when add reconstructs a no-chunk-dim live-out."""
    if live_out.op == "call_function" and live_out.target is aten.sym_size.int:
        return "symbolic_size"
    if tensor_meta(live_out) is None:
        return None
    if is_backward and users and all(user.op == "output" for user in users):
        return "graph_output_partial_sum"
    if not is_backward:
        return None
    if live_out not in grad_escape_boundaries:
        return None
    logger.debug(
        "Chunk pass add-materialized %s: graph-output grad escape boundary",
        live_out.name,
    )
    return "backward_partial_sum"


def _materialize_live_out(
    gm: fx.GraphModule,
    plan: _Plan,
    live_out: fx.Node,
    copies: tuple[fx.Node, fx.Node],
    users: tuple[fx.Node, ...],
    *,
    mode: ChunkMode,
    symbol_hints: dict[object, int],
    original_meta: dict[str, Any],
) -> fx.Node:
    """Step 5: reconstruct one full live-out from its two chunk values."""
    original_val = original_meta.get("val")
    original_dim = _chunk_dim_from_symbols(
        original_val if isinstance(original_val, torch.Tensor) else None,
        symbol_hints,
    )
    live_out_dim = _chunk_dim_from_symbols(tensor_meta(live_out), symbol_hints)
    if live_out_dim is not None or original_dim is not None:
        dim = live_out_dim.dim if live_out_dim is not None else original_dim.dim
        assert dim is not None
        materialized = gm.graph.call_function(
            aten.cat.default, args=([copies[0], copies[1]], dim)
        )
        materialized.meta["val"] = original_val
    elif (
        proof := _add_materialization_proof(
            live_out,
            users,
            is_backward=plan.region.is_backward,
            grad_escape_boundaries=plan.grad_escape_boundaries,
        )
    ) is not None:
        # Add is semantic reconstruction, not shape reconstruction. Keep the
        # proof explicit so future extensions do not silently broaden it.
        materialized = gm.graph.call_function(
            aten.add.Tensor, args=(copies[0], copies[1])
        )
        materialized.meta["val"] = original_val
        materialized.meta["chunked_materialization_proof"] = proof
    else:
        if tensor_meta(live_out) is None:
            raise ValueError(
                f"Chunk pass cannot materialize scalar live-out {live_out.name}: "
                "the scalar is not proven to reconstruct from chunk values."
            )
        raise ValueError(
            "Chunk pass cannot materialize live-out without chunk dimension; "
            f"{live_out.name} from {plan.region.root_fqn!r} has full users "
            f"{[_describe_node(user) for user in users]} and requires a forward "
            "accumulation proof or a backward parameter-gradient consumer."
        )
    materialized._rename(f"{live_out.name}_chunk_materialized")
    _set_synthetic_meta(materialized, region=plan.region, role="materialization")
    _validate_materialized(
        materialized,
        live_out,
        original_val=original_val if isinstance(original_val, torch.Tensor) else None,
        symbol_hints=symbol_hints,
    )
    return materialized


def _materialize_suffix_from_full_boundary(
    gm: fx.GraphModule,
    plan: _Plan,
    boundary: fx.Node,
    materialized: fx.Node,
    suffix: tuple[fx.Node, ...],
    *,
    originals: dict[
        fx.Node, tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], str]
    ],
) -> fx.Node:
    """Step 5: copy tensor or shape suffixes after a boundary is full."""
    remapped: dict[fx.Node, fx.Node] = {boundary: materialized}
    current = materialized
    for node in suffix:
        args, kwargs, meta, name = originals[node]
        new_args = tree_map(lambda arg: remapped.get(arg, arg), args)
        new_kwargs = tree_map(lambda arg: remapped.get(arg, arg), kwargs)
        original_val = meta.get("val")
        shape_arg_idx = view_shape_arg_index(node.target)
        if (
            shape_arg_idx is not None
            and isinstance(new_args, tuple)
            and len(new_args) > shape_arg_idx
            and isinstance(new_args[shape_arg_idx], (list, tuple))
            and isinstance(original_val, torch.Tensor)
        ):
            # The first chunk is rewritten in place, so saved FX shape args may
            # now refer to half-size values. A full-value suffix must use the
            # original full extent recorded by tensor metadata.
            materialized_shape = _materialize_symint_args(
                gm,
                list(original_val.shape),
                region=plan.region,
                role="materialization",
            )
            new_args = (
                *new_args[:shape_arg_idx],
                materialized_shape,
                *new_args[shape_arg_idx + 1 :],
            )
        with gm.graph.inserting_after(current):
            current = gm.graph.create_node(
                node.op, node.target, new_args, new_kwargs, type_expr=node.type
            )
        current._rename(f"{name}_chunk_materialized")
        current.meta = _copy_meta(meta)
        _set_synthetic_meta(current, region=plan.region, role="materialization")
        remapped[node] = current
    return current


def _materialize_full_live_out(
    gm: fx.GraphModule,
    plan: _Plan,
    live_out: fx.Node,
    users: tuple[fx.Node, ...],
    *,
    mode: ChunkMode,
    symbol_hints: dict[object, int],
    originals: dict[
        fx.Node, tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], str]
    ],
    copied_by_chunk: dict[int, dict[fx.Node, fx.Node]],
    optimize_grad_live_out: bool,
    memo: dict[fx.Node, fx.Node],
) -> fx.Node:
    """Step 5: materialize a full live-out, possibly before a suffix chain."""
    if live_out in memo:
        return memo[live_out]

    boundary = _MaterializationBoundary(live_out, ())
    if _is_tensor_with_chunk_dim(live_out, symbol_hints):
        boundary = _materialization_boundary(
            live_out,
            plan=plan,
            optimize_grad_live_out=optimize_grad_live_out,
            symbol_hints=symbol_hints,
        )
        if boundary.node not in copied_by_chunk[0]:
            boundary = _MaterializationBoundary(live_out, ())
    if boundary.node is not live_out:
        materialized = _materialize_full_live_out(
            gm,
            plan,
            boundary.node,
            users,
            mode=mode,
            symbol_hints=symbol_hints,
            originals=originals,
            copied_by_chunk=copied_by_chunk,
            optimize_grad_live_out=optimize_grad_live_out,
            memo=memo,
        )
        if boundary.suffix:
            materialized = _materialize_suffix_from_full_boundary(
                gm,
                plan,
                boundary.node,
                materialized,
                boundary.suffix,
                originals=originals,
            )
            _validate_materialized(
                materialized,
                live_out,
                original_val=originals[live_out][2].get("val"),
                symbol_hints=symbol_hints,
            )
        memo[live_out] = materialized
        return materialized
    copies = (
        copied_by_chunk[0][boundary.node],
        copied_by_chunk[1][boundary.node],
    )
    materialized = _materialize_live_out(
        gm,
        plan,
        boundary.node,
        copies,
        users,
        mode=mode,
        symbol_hints=symbol_hints,
        original_meta=originals[boundary.node][2],
    )
    if boundary.suffix:
        materialized = _materialize_suffix_from_full_boundary(
            gm,
            plan,
            boundary.node,
            materialized,
            boundary.suffix,
            originals=originals,
        )
        _validate_materialized(
            materialized,
            live_out,
            original_val=originals[live_out][2].get("val"),
            symbol_hints=symbol_hints,
        )
    memo[live_out] = materialized
    return materialized


def _describe_node(node: fx.Node) -> str:
    target = getattr(node.target, "__name__", str(node.target))
    fqn = _get_module_fqn(node)
    return f"{node.name}:{target}:fqn={fqn or '<none>'}"


# Step 6: Clean up, validate, and expose public graph chunking passes.


def _erase_unused_copied_body(
    gm: fx.GraphModule,
    copied_by_chunk: dict[int, dict[fx.Node, fx.Node]],
    *,
    protected: set[fx.Node],
) -> None:
    """Step 6: erase unused copied nodes while preserving side effects/provenance."""
    copied = {
        node for by_origin in copied_by_chunk.values() for node in by_origin.values()
    }
    for node in reversed(tuple(gm.graph.nodes)):
        if (
            node in copied
            and node not in protected
            and not node.users
            and not _mutates_input(node)
        ):
            gm.graph.erase_node(node)


def _transform_region(
    gm: fx.GraphModule,
    plan: _Plan,
    *,
    mode: ChunkMode,
    static_nodes: set[fx.Node],
    plans: list[_Plan],
    per_chunk: dict[fx.Node, tuple[fx.Node, fx.Node]],
    optimize_grad_live_out: bool,
) -> None:
    """Steps 3-6: lower one planned region into two chunks."""
    if mode == "seq" and any(
        "attention" in _get_module_fqn(n).split(".") for n in plan.region.nodes
    ):
        raise NotImplementedError(
            "seq chunking through attention needs full-K/V handling"
        )
    _validate_no_non_ep_collectives(plan)

    chunks, scalars, symbol_hints = _split_live_ins(
        gm,
        plan,
        mode=mode,
        static_nodes=static_nodes,
        per_chunk=per_chunk,
    )
    logger.debug(
        "Chunk pass lowering region: root=%s direction=%s body=%d live_ins=%d "
        "live_outs=%d chunks=%d scalars=%d optimize_grad_live_out=%s",
        plan.region.root_fqn,
        "backward" if plan.region.is_backward else "forward",
        len(plan.body),
        len(plan.live_ins),
        len(plan.live_out_users),
        len(chunks),
        len(scalars),
        optimize_grad_live_out,
    )
    originals = {
        node: (node.args, node.kwargs, _copy_meta(node.meta), node.name)
        for node in plan.region.nodes
    }
    live_out_user_classes = {
        live_out: _split_live_out_users(users, plans, plan.region, symbol_hints)
        for live_out, users in plan.live_out_users.items()
    }
    preserved_scalar_live_outs = {
        live_out
        for live_out, full_users in plan.live_out_users.items()
        if full_users
        and tensor_meta(live_out) is None
        and any(user in plan.body for user in live_out.users)
    }
    first_chunk = 0
    second_chunk = 1
    copied_by_chunk: dict[int, dict[fx.Node, fx.Node]] = {}
    copied_by_chunk[first_chunk] = _copy_body(
        gm,
        plan,
        chunks=chunks,
        scalars=scalars,
        symbol_hints=symbol_hints,
        mode=mode,
        chunk_id=first_chunk,
        originals=originals,
        in_place=True,
        preserve_originals=preserved_scalar_live_outs,
    )
    copied_by_chunk[second_chunk] = _copy_body(
        gm,
        plan,
        chunks=chunks,
        scalars=scalars,
        symbol_hints=symbol_hints,
        mode=mode,
        chunk_id=second_chunk,
        originals=originals,
        in_place=False,
        prior_chunk_copies=copied_by_chunk[first_chunk],
    )
    order = ordered_nodes(gm)
    materialized_live_outs: dict[fx.Node, fx.Node] = {}
    live_out_items = []

    for live_out, (chunked_users, full_users) in live_out_user_classes.items():
        first_full_user = (
            order[min(full_users, key=order.get)] if full_users else len(order)
        )
        live_out_items.append((first_full_user, live_out, chunked_users, full_users))

    for _, live_out, chunked_users, full_users in sorted(
        live_out_items, key=lambda item: item[0]
    ):
        logger.debug(
            "Chunk pass live-out: root=%s node=%s chunked_users=%d full_users=%d",
            plan.region.root_fqn,
            live_out.name,
            len(chunked_users),
            len(full_users),
        )
        if chunked_users:
            copies = (copied_by_chunk[0][live_out], copied_by_chunk[1][live_out])
            per_chunk[live_out] = copies
            per_chunk[copies[0]] = copies
            per_chunk[copies[1]] = copies
        if not full_users:
            continue
        if live_out in preserved_scalar_live_outs:
            # Scalar shape expressions are not reconstructed from chunk values.
            # Keep the original full scalar for full users and route chunked
            # users through the chunk-local scalar copies recorded above.
            continue
        if (
            chunked_users
            and _chunk_dim_from_symbols(
                originals[live_out][2].get("val")
                if isinstance(originals[live_out][2].get("val"), torch.Tensor)
                else None,
                symbol_hints,
            )
            is None
            and not _is_tensor_with_chunk_dim(live_out, symbol_hints)
        ):
            raise ValueError(
                "Chunk pass does not support hybrid live-out consumers without "
                "the selected chunk dimension. "
                f"root={plan.region.root_fqn!r}, live_out={_describe_node(live_out)}, "
                f"chunked_users={[user.name for user in chunked_users]}, "
                f"full_users={[user.name for user in full_users]}, "
                f"live_out_shape={getattr(tensor_meta(live_out), 'shape', None)}, "
                f"original_shape={getattr(originals[live_out][2].get('val'), 'shape', None)}."
            )
        with gm.graph.inserting_before(min(full_users, key=order.get)):
            materialized = _materialize_full_live_out(
                gm,
                plan,
                live_out,
                full_users,
                mode=mode,
                symbol_hints=symbol_hints,
                originals=originals,
                copied_by_chunk=copied_by_chunk,
                optimize_grad_live_out=optimize_grad_live_out and not chunked_users,
                memo=materialized_live_outs,
            )
            _validate_materialized(
                materialized,
                live_out,
                original_val=originals[live_out][2].get("val"),
                symbol_hints=symbol_hints,
            )
        for user in full_users:
            user.replace_input_with(live_out, materialized)
    protected = {chunk for chunks in per_chunk.values() for chunk in chunks}
    _erase_unused_copied_body(gm, copied_by_chunk, protected=protected)


def _static_placeholders(gm: fx.GraphModule, num_static_inputs: int) -> set[fx.Node]:
    """Return placeholders that represent static model state."""
    return {
        node
        for idx, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")
        if idx < num_static_inputs
    }


def _ep_roots(gm: fx.GraphModule, module_pattern: str) -> list[str]:
    """Find selected roots that contain real all-to-all collectives."""
    roots = set()
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target != torch.ops._c10d_functional.all_to_all_single.default
        ):
            continue
        fqn = _get_module_fqn(node)
        if not fqn:
            continue
        matched = _pattern_root(module_pattern, fqn)
        if matched is not None:
            roots.add(matched.root_fqn)
    return sorted(roots)


def _ep_annotated_roots(gm: fx.GraphModule, module_pattern: str) -> list[str]:
    """Find selected roots from EP annotations when no real EP PG is present."""
    roots = set()
    for node in gm.graph.nodes:
        custom = _custom_meta(node)
        if "EP" not in custom and _EP_TOKEN_EXCHANGE not in custom:
            continue
        fqn = _get_module_fqn(node)
        if not fqn:
            continue
        matched = _pattern_root(module_pattern, fqn)
        if matched is not None:
            roots.add(matched.root_fqn)
    return sorted(roots)


def apply_chunk_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
    module_patterns: list[str],
    num_static_inputs: int = 0,
    optimize_grad_live_out: bool = True,
) -> fx.GraphModule:
    """Run graph chunking for the requested mode and module patterns."""
    del example_inputs
    if mode not in ("batch", "seq"):
        raise ValueError(f"Unknown chunk mode: {mode!r}")
    logger.debug(
        "Chunk pass start: mode=%s patterns=%s num_static_inputs=%d "
        "optimize_grad_live_out=%s",
        mode,
        module_patterns,
        num_static_inputs,
        optimize_grad_live_out,
    )
    static_nodes = _static_placeholders(gm, num_static_inputs)
    per_chunk: dict[fx.Node, tuple[fx.Node, fx.Node]] = {}
    num_regions = 0
    for is_backward in (False, True):
        # Step 2: plan regions for the current graph before lowering this
        # direction; forward lowering can create provenance used by backward.
        plans = _plan_regions(
            gm,
            module_patterns,
            mode=mode,
            optimize_grad_live_out=optimize_grad_live_out,
            per_chunk_sources=set(per_chunk),
        )
        directional_plans = [
            plan for plan in plans if plan.region.is_backward == is_backward
        ]
        logger.debug(
            "Chunk pass direction plan: direction=%s regions=%d",
            "backward" if is_backward else "forward",
            len(directional_plans),
        )
        for plan in directional_plans:
            # Steps 3-6: rebuild the region boundary from the current graph,
            # then lower one region completely before moving to the next.
            plan = _build_plan(plan.region, plan.grad_escape_boundaries)
            _transform_region(
                gm,
                plan,
                mode=mode,
                static_nodes=static_nodes,
                plans=plans,
                per_chunk=per_chunk,
                optimize_grad_live_out=optimize_grad_live_out,
            )
            num_regions += 1
    if num_regions == 0:
        raise ValueError(f"Chunk pass found no regions for {module_patterns}.")
    _validate_selected_symbol_shapes(gm, mode=mode)
    _validate_no_raw_chunk_symbol_args(gm, chunk_symbol_hints_for_mode(gm, mode))
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Applied %s chunking to %d region(s): patterns=%s optimize_grad_live_out=%s",
        mode,
        num_regions,
        module_patterns,
        optimize_grad_live_out,
    )
    return gm


def ep_overlap_chunk_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    mode: ChunkMode,
    module_pattern: str,
    num_static_inputs: int = 0,
    optimize_grad_live_out: bool = True,
    require_all_to_all: bool = True,
) -> fx.GraphModule:
    """Resolve EP roots from one pattern, then run graph chunking."""
    roots = _ep_roots(gm, module_pattern)
    if not roots:
        if require_all_to_all:
            raise ValueError(f"No EP all-to-all regions matched {module_pattern!r}.")
        roots = _ep_annotated_roots(gm, module_pattern)
    if not roots:
        raise ValueError(f"No EP regions matched {module_pattern!r}.")
    return apply_chunk_pass(
        gm,
        example_inputs,
        mode=mode,
        module_patterns=roots,
        num_static_inputs=num_static_inputs,
        optimize_grad_live_out=optimize_grad_live_out,
    )
