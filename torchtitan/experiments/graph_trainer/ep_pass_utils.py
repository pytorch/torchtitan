# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared EP chunk/overlap metadata and symbolic-shape helpers.

The EP passes use this module as their common contract layer:

1. Region discovery reads chunk metadata produced by eager or graph chunking and
   groups nodes by ``(module_fqn, direction, chunk_id)``.
2. Symbol helpers treat the unbacked SymInt selected by mark_unbacked as the
   source of truth for chunk influence and use optimization hints only for pass
   decisions that require concrete extents.
3. The final concretization pass removes chunk-introduced SymInts and their
   scalar plumbing after EP scheduling, so downstream compilers see static
   metadata for these temporary symbols.
"""

from __future__ import annotations

import fnmatch

import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from torch.utils._pytree import tree_leaves, tree_map

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.tools.logging import logger

aten = torch.ops.aten
CHUNK_SYMBOL_HINTS_META = "torchtitan_chunk_symbol_hints"
_DEAD_CHUNK_SCALAR_TARGETS = {
    aten.sym_size.int,
    torch.sym_ite,
    operator.add,
    operator.and_,
    operator.floordiv,
    operator.mod,
    operator.mul,
    operator.or_,
    operator.sub,
    operator.eq,
    operator.ge,
    operator.gt,
    operator.le,
    operator.lt,
}
_ASSERT_TARGETS = {aten._assert_scalar.default}


def is_module_fqn_inside_root(fqn: str, root_fqn: str) -> bool:
    """Return whether ``fqn`` is exactly ``root_fqn`` or one of its children."""
    return fqn == root_fqn or fqn.startswith(root_fqn + ".")


def tensor_meta(node: fx.Node) -> torch.Tensor | None:
    """Return tensor fake value metadata, if this node has tensor metadata."""
    val = node.meta.get("val")
    return val if isinstance(val, torch.Tensor) else None


def is_c10d_functional_node(node: fx.Node) -> bool:
    """Return whether a node is a distributed functional op (AG/RS/A2A/wait)."""
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )


def is_view_target(target: object) -> bool:
    """Return whether a target is a true PyTorch view op overload."""
    return isinstance(target, torch._ops.OpOverload) and target.is_view


def same_tensor_domain(lhs: fx.Node, rhs: fx.Node) -> bool:
    """Return whether two tensor nodes share dtype/device/layout metadata."""
    lhs_val, rhs_val = tensor_meta(lhs), tensor_meta(rhs)
    if lhs_val is None or rhs_val is None:
        return False
    return (
        lhs_val.dtype == rhs_val.dtype
        and lhs_val.device == rhs_val.device
        and lhs_val.layout == rhs_val.layout
    )


def view_shape_arg_index(target: object) -> int | None:
    """Return the positional shape arg index for Tensor -> Tensor view ops."""
    if not isinstance(target, torch._ops.OpOverload):
        return None
    schema = target._schema
    if (
        not schema.arguments
        or str(schema.arguments[0].type) != "Tensor"
        or len(schema.returns) != 1
        or str(schema.returns[0].type) != "Tensor"
    ):
        return None
    schema_text = str(schema)
    for idx, arg in enumerate(schema.arguments[1:], start=1):
        if (
            not arg.kwarg_only
            and arg.name in {"shape", "size"}
            and f"SymInt[] {arg.name}" in schema_text
        ):
            return idx
    return None


def free_symbols(value: object) -> frozenset[object]:
    """Return symbolic-shape free symbols for values accepted by PyTorch."""
    from torch.fx.experimental.symbolic_shapes import free_symbols

    return frozenset(free_symbols(value))


def ordered_nodes(gm: fx.GraphModule) -> dict[fx.Node, int]:
    """Map each graph node to its current topological position."""
    return {node: idx for idx, node in enumerate(gm.graph.nodes)}


@dataclass(frozen=True)
class ChunkOwner:
    """Stable identity for one chunk body in one direction of one module."""

    root_fqn: str
    is_backward: bool
    chunk_id: int


@dataclass(frozen=True)
class ChunkBody:
    """A planned chunk body plus its external graph inputs."""

    owner: ChunkOwner
    nodes: tuple[fx.Node, ...]
    node_set: frozenset[fx.Node]
    live_ins: frozenset[fx.Node]
    producer: str


@dataclass(frozen=True)
class ChunkedRegion:
    """The chunk bodies for one module/direction pair."""

    root_fqn: str
    is_backward: bool
    bodies_by_chunk: dict[int, ChunkBody]


def _chunk_owner(node: fx.Node) -> ChunkOwner | None:
    if node.meta.get("chunked_region_role") != "body":
        return None
    chunk_id = node.meta.get("chunk_id")
    root = node.meta.get("chunked_region_fqn")
    if chunk_id not in (0, 1) or not isinstance(root, str):
        raise ValueError(f"Chunk body node {node.name} has incomplete chunk metadata.")
    return ChunkOwner(
        root_fqn=root,
        is_backward=bool(
            node.meta.get("chunked_region_is_backward", _is_backward_node(node))
        ),
        chunk_id=chunk_id,
    )


def collect_chunked_regions(
    gm: fx.GraphModule, *, module_pattern: str
) -> list[ChunkedRegion]:
    """Collect chunk bodies matching ``module_pattern`` in graph order."""
    nodes_by_owner: dict[ChunkOwner, list[fx.Node]] = defaultdict(list)
    for node in gm.graph.nodes:
        owner = _chunk_owner(node)
        if owner and fnmatch.fnmatchcase(owner.root_fqn, module_pattern):
            nodes_by_owner[owner].append(node)

    grouped: dict[tuple[str, bool], dict[int, ChunkBody]] = defaultdict(dict)
    for owner, nodes in nodes_by_owner.items():
        node_set = frozenset(nodes)
        grouped[(owner.root_fqn, owner.is_backward)][owner.chunk_id] = ChunkBody(
            owner=owner,
            nodes=tuple(nodes),
            node_set=node_set,
            live_ins=frozenset(
                inp
                for node in nodes
                for inp in node.all_input_nodes
                if inp not in node_set
            ),
            producer=str(nodes[0].meta.get("chunked_region_producer", "graph")),
        )

    return [
        ChunkedRegion(root, is_backward, by_chunk)
        for (root, is_backward), by_chunk in grouped.items()
    ]


def _hint(value: object) -> int | None:
    """Return a concrete/optimization hint when PyTorch exposes one."""
    if isinstance(value, int):
        return value
    if hasattr(value, "node"):
        if value.node.hint is not None:
            return int(value.node.hint)
        shape_env = getattr(value.node, "shape_env", None)
        if shape_env is not None:
            try:
                return int(shape_env.optimization_hint(value.node.expr))
            except Exception:
                return None
    return None


def _eval_hint(value: object, hints: dict[object, int]) -> int | bool | None:
    """Evaluate ``value`` by substituting known symbol hints."""
    symbols = free_symbols(value)
    if not symbols:
        if (hint := _hint(value)) is not None:
            return hint
        try:
            return int(value)  # Handles concrete SymPy integers.
        except (TypeError, ValueError, OverflowError):
            return None
    if not symbols <= hints.keys():
        return None
    try:
        expr = value.node.expr if hasattr(value, "node") else value
        evaluated = expr.subs({symbol: hints[symbol] for symbol in symbols})
        if free_symbols(evaluated):
            return None
        if isinstance(value, torch.SymBool):
            return bool(evaluated)
        return int(evaluated)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _derive_symbol_hint(expr: object, extent_hint: int, symbol: object) -> int | None:
    expr = expr.node.expr if isinstance(expr, torch.SymInt) else expr
    try:
        zero = expr.subs(symbol, 0)
        one = expr.subs(symbol, 1)
        if free_symbols(zero):
            return None
        offset = int(zero)
        coeff = int(one - zero)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if offset != 0 or coeff <= 0 or extent_hint % coeff:
        return None
    return extent_hint // coeff


def _merge_symbol_hint(
    hints: dict[object, int], symbol: object, hint: int, *, source: str
) -> None:
    existing = hints.get(symbol)
    if existing is not None and existing != hint:
        raise ValueError(
            f"Chunk symbol {symbol} has conflicting hints: {existing} and "
            f"{hint} from {source}."
        )
    hints[symbol] = hint


def _record_symbols_from_extent(
    hints: dict[object, int], extent: object, *, source: str
) -> None:
    symbols = free_symbols(extent)
    if not symbols:
        return
    extent_hint = _hint(extent)
    if extent_hint is None:
        raise ValueError(
            f"Chunk pass found selected dynamic extent {extent} from {source} "
            "without an optimization hint."
        )
    for symbol in symbols:
        hint = _derive_symbol_hint(extent, int(extent_hint), symbol)
        if hint is None:
            raise ValueError(
                f"Chunk pass could not derive a base hint for symbol {symbol} "
                f"from {source} extent {extent}."
            )
        _merge_symbol_hint(hints, symbol, hint, source=source)


def _selected_symbol_hints(node: fx.Node) -> dict[object, int]:
    hints = node.meta.get(CHUNK_SYMBOL_HINTS_META, {})
    return hints if isinstance(hints, dict) else {}


def chunk_symbol_hints_for_mode(
    gm: fx.GraphModule, mode: str | None = None
) -> dict[object, int]:
    """Collect the selected chunk symbols and their optimization hints."""
    hints: dict[object, int] = {}
    for node in gm.graph.nodes:
        for symbol, hint in _selected_symbol_hints(node).items():
            _merge_symbol_hint(hints, symbol, int(hint), source=node.name)
        if (
            mode is None
            or node.op != "placeholder"
            or (val := tensor_meta(node)) is None
        ):
            continue
        dim = {"batch": 0, "seq": 1}.get(mode)
        if dim is None:
            raise ValueError(f"Unknown chunk mode: {mode!r}")
        if dim < len(val.shape):
            _record_symbols_from_extent(
                hints, val.shape[dim], source=f"{node.name}.shape[{dim}]"
            )
    return hints


def _placeholder_symbol_hints(gm: fx.GraphModule) -> dict[object, int]:
    hints: dict[object, int] = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder" or (val := tensor_meta(node)) is None:
            continue
        for dim, extent in enumerate(val.shape):
            _record_symbols_from_extent(
                hints, extent, source=f"{node.name}.shape[{dim}]"
            )
    return hints


def same_shape_with_hints(
    lhs: object,
    rhs: object,
    node: fx.Node,
    symbol_hints: dict[object, int] | None = None,
) -> bool:
    """Compare tensor shapes after substituting available chunk hints."""
    if (
        tensor_meta(node) is None
        or not hasattr(lhs, "shape")
        or not hasattr(rhs, "shape")
    ):
        return False
    hints = symbol_hints or _selected_symbol_hints(node)
    return len(lhs.shape) == len(rhs.shape) and all(
        same_symbolic_expr(a, b)
        or (_hint(a) is not None and _hint(a) == _hint(b))
        or (
            (a_hint := _eval_hint(a, hints)) is not None
            and (b_hint := _eval_hint(b, hints)) is not None
            and a_hint == b_hint
        )
        for a, b in zip(lhs.shape, rhs.shape)
    )


def symbolic_expr(value: object) -> object:
    """Return the symbolic expression object for a PyTorch symbolic scalar."""
    return value.node.expr if hasattr(value, "node") else value


def same_symbolic_expr(lhs: object, rhs: object) -> bool:
    """Return whether two symbolic/concrete expressions are structurally equal."""
    lhs_expr, rhs_expr = symbolic_expr(lhs), symbolic_expr(rhs)
    try:
        equal = lhs_expr == rhs_expr
        if isinstance(equal, bool):
            return equal
        if bool(equal):
            return True
    except (RuntimeError, TypeError, ValueError):
        pass
    try:
        equals = getattr(lhs_expr - rhs_expr, "equals", None)
        return callable(equals) and equals(0) is True
    except (AttributeError, TypeError, ValueError):
        return False


def expr_key(value: object) -> object:
    """Return the canonical object used to match identical symbolic expressions."""
    return symbolic_expr(value)


def dim_hint(dim: object, symbol_hints: dict[object, int]) -> int | None:
    """Return the hinted integer value for a dimension-like object."""
    if isinstance(dim, fx.Node):
        val = dim.meta.get("val")
        return None if val is None else dim_hint(val, symbol_hints)
    return _eval_hint(dim, symbol_hints) if free_symbols(dim) else _hint(dim)


def numel_hint(shape: object, symbol_hints: dict[object, int]) -> int | None:
    """Return hinted numel for a shape, or ``None`` if it is not provable."""
    if not isinstance(shape, (list, tuple, torch.Size)):
        return None
    result = 1
    for dim in shape:
        if dim == -1:
            return None
        hint = dim_hint(dim, symbol_hints)
        if hint is None:
            return None
        result *= int(hint)
    return result


def same_dim_hint(lhs: object, rhs: object, symbol_hints: dict[object, int]) -> bool:
    """Return whether two dimensions are equal under chunk hints."""
    if same_symbolic_expr(lhs, rhs):
        return True
    lhs_hint = dim_hint(lhs, symbol_hints)
    rhs_hint = dim_hint(rhs, symbol_hints)
    return lhs_hint is not None and lhs_hint == rhs_hint


def statically_equals_hint(value: object, hint: int | bool) -> bool:
    """Return whether symbolic reasoning proves ``value == hint``."""
    if isinstance(value, (torch.SymInt, torch.SymBool)):
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        return statically_known_true(value == hint)
    expr = value.node.expr if isinstance(value, torch.SymInt) else value
    equals = getattr(expr, "equals", None)
    return callable(equals) and equals(hint) is True


def depends_on_chunk_symbol(value: object, symbol_hints: dict[object, int]) -> bool:
    """Return whether ``value`` still references any selected chunk symbol."""
    if isinstance(value, torch.Tensor):
        return any(
            free_symbols(size_or_stride) & symbol_hints.keys()
            for size_or_stride in (*value.shape, *value.stride())
        )
    if isinstance(value, fx.Node):
        return False
    if not isinstance(
        value,
        (
            torch.SymInt,
            torch.SymFloat,
            torch.SymBool,
            int,
            float,
            bool,
            str,
            type(None),
        ),
    ):
        return False
    return bool(free_symbols(value) & symbol_hints.keys())


def _concretize_value(value: object, symbol_hints: dict[object, int]) -> object:
    symbols = free_symbols(value)
    if symbols:
        concrete = _eval_hint(value, symbol_hints)
        if concrete is not None:
            return concrete
        chunk_symbols = symbols & symbol_hints.keys()
        if not chunk_symbols:
            hint = _hint(value)
            if hint is not None:
                return hint
        if chunk_symbols and hasattr(value, "node"):
            expr = value.node.expr.subs(
                {symbol: symbol_hints[symbol] for symbol in chunk_symbols}
            )
            shape_env = value.node.shape_env
            if isinstance(value, torch.SymInt):
                return shape_env.create_symintnode(expr, hint=None)
            if isinstance(value, torch.SymFloat):
                return shape_env.create_symfloatnode(expr, hint=None)
            if isinstance(value, torch.SymBool):
                return shape_env.create_symboolnode(expr)
        return value
    return value


def _concretize_arg(arg: object, symbol_hints: dict[object, int]) -> object:
    if isinstance(arg, tuple):
        return tuple(_concretize_arg(item, symbol_hints) for item in arg)
    if isinstance(arg, list):
        mapped = [_concretize_arg(item, symbol_hints) for item in arg]
        try:
            return type(arg)(mapped)
        except TypeError:
            return mapped
    if isinstance(arg, dict):
        return {key: _concretize_arg(value, symbol_hints) for key, value in arg.items()}
    if isinstance(arg, (torch.SymInt, torch.SymFloat, torch.SymBool, int)):
        return _concretize_value(arg, symbol_hints)
    return arg


def _concretize_tensor_meta(
    val: torch.Tensor, symbol_hints: dict[object, int]
) -> torch.Tensor:
    if not any(
        free_symbols(size_or_stride) & symbol_hints.keys()
        for size_or_stride in (*val.shape, *val.stride())
    ):
        return val
    shape = tuple(_concretize_value(size, symbol_hints) for size in val.shape)
    stride = tuple(_concretize_value(extent, symbol_hints) for extent in val.stride())
    return val.new_empty_strided(shape, stride)


def _concretize_meta_value(value: object, symbol_hints: dict[object, int]) -> object:
    if isinstance(value, torch.Tensor):
        return _concretize_tensor_meta(value, symbol_hints)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return _concretize_value(value, symbol_hints)
    return value


def _clear_chunk_tensor_marks(
    tensor: torch.Tensor, symbol_hints: dict[object, int]
) -> None:
    dims = {
        dim
        for dim, extent in enumerate(tensor.shape)
        if bool(free_symbols(extent) & symbol_hints.keys())
    }
    if not dims:
        return
    for attr in (
        "_dynamo_unbacked_indices",
        "_dynamo_strict_unbacked_indices",
        "_dynamo_dynamic_indices",
    ):
        values = getattr(tensor, attr, None)
        if isinstance(values, set):
            values.difference_update(dims)
            if not values:
                delattr(tensor, attr)
    for attr in (
        "_dynamo_hint_overrides",
        "_dynamo_unbacked_bounds",
        "_dynamo_shape_ids",
        "_specialize_on",
    ):
        values = getattr(tensor, attr, None)
        if isinstance(values, dict):
            for dim in dims:
                values.pop(dim, None)
            if not values:
                delattr(tensor, attr)


def _concretize_chunk_example_inputs(
    example_inputs: list[Any] | tuple[Any, ...] | None,
    symbol_hints: dict[object, int],
) -> None:
    if example_inputs is None:
        return
    for idx, inp in enumerate(example_inputs):
        if not isinstance(inp, torch.Tensor):
            continue
        concrete = _concretize_tensor_meta(inp, symbol_hints)
        if concrete is not inp and isinstance(example_inputs, list):
            example_inputs[idx] = concrete
            inp = concrete
        _clear_chunk_tensor_marks(inp, symbol_hints)


def _validate_no_chunk_symbols(
    gm: fx.GraphModule, symbol_hints: dict[object, int]
) -> None:
    chunk_symbols = symbol_hints.keys()
    for graph_module in _iter_graph_modules(gm):
        for node in graph_module.graph.nodes:
            for value in tree_leaves(
                (
                    node.args,
                    node.kwargs,
                    node.meta.get("val"),
                    node.meta.get("example_value"),
                )
            ):
                if depends_on_chunk_symbol(value, symbol_hints):
                    raise ValueError(
                        "concretize_ep_chunk_symbolic_shapes_pass left chunk "
                        f"symbol(s) {free_symbols(value) & chunk_symbols} in "
                        f"{node.name}."
                    )


def _iter_graph_modules(gm: fx.GraphModule) -> list[fx.GraphModule]:
    """Return ``gm`` and GraphModule subgraphs referenced by its nodes."""
    seen: set[int] = set()
    ordered: list[fx.GraphModule] = []
    work = [gm]
    while work:
        graph_module = work.pop()
        if id(graph_module) in seen:
            continue
        seen.add(id(graph_module))
        ordered.append(graph_module)
        for node in graph_module.graph.nodes:
            if node.op == "get_attr" and isinstance(node.target, str):
                attr = getattr(graph_module, node.target, None)
                if isinstance(attr, fx.GraphModule):
                    work.append(attr)
            for value in tree_leaves((node.args, node.kwargs)):
                if isinstance(value, fx.GraphModule):
                    work.append(value)
    return ordered


def _concretize_graph_module(
    gm: fx.GraphModule, symbol_hints: dict[object, int]
) -> dict[fx.Node, int | bool]:
    scalar_constants: dict[fx.Node, int | bool] = {}
    for node in list(gm.graph.nodes):
        if node.op == "output":
            continue
        node.args = _concretize_arg(node.args, symbol_hints)
        node.kwargs = _concretize_arg(node.kwargs, symbol_hints)
        for meta_key in ("val", "example_value"):
            if meta_key not in node.meta:
                continue
            val = node.meta[meta_key]
            new_val = tree_map(
                lambda item: _concretize_meta_value(item, symbol_hints), val
            )
            node.meta[meta_key] = new_val
            if (
                isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool))
                and (concrete := _eval_hint(val, symbol_hints)) is not None
            ):
                scalar_constants[node] = concrete
        node.meta.pop(CHUNK_SYMBOL_HINTS_META, None)
    return scalar_constants


def _is_dead_chunk_scalar_plumbing(node: fx.Node) -> bool:
    """Return whether ``node`` is dead scalar shape/assert plumbing."""
    return (
        node.op == "call_function"
        and not node.users
        and (
            node.target in _DEAD_CHUNK_SCALAR_TARGETS or _is_assert_target(node.target)
        )
    )


def _is_assert_target(target: object) -> bool:
    return target in _ASSERT_TARGETS


def _concretized_guard_value(
    value: object,
    symbol_hints: dict[object, int],
    scalar_constants: dict[fx.Node, int | bool],
) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, fx.Node):
        if value in scalar_constants:
            concrete = scalar_constants[value]
            return concrete if isinstance(concrete, bool) else None
        value = value.meta.get("val")
    if isinstance(value, torch.SymBool):
        concrete = _eval_hint(value, symbol_hints)
        return concrete if isinstance(concrete, bool) else None
    return None


def _validate_no_false_guards(
    gm: fx.GraphModule,
    symbol_hints: dict[object, int],
    scalar_constants: dict[fx.Node, int | bool],
) -> None:
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or not _is_assert_target(node.target)
            or not node.args
        ):
            continue
        concrete = _concretized_guard_value(
            node.args[0], symbol_hints, scalar_constants
        )
        if concrete is False:
            message = node.args[1] if len(node.args) > 1 else ""
            raise ValueError(
                "concretize_ep_chunk_symbolic_shapes_pass found a guard that "
                f"evaluates to False after chunk-symbol substitution: "
                f"{node.name} {message}"
            )


def concretize_ep_chunk_symbolic_shapes_pass(
    gm: fx.GraphModule,
    example_inputs: list[Any] | tuple[Any, ...] | None = None,
) -> fx.GraphModule:
    """Concretize EP chunk symbolic shapes after scheduling.

    Graph chunking temporarily introduces unbacked SymInts as the selected
    chunk dimension. Once EP scheduling has consumed that symbolic contract,
    downstream compiler passes should see concrete hinted sizes. This pass
    rewrites graph args, metadata, scalar SymInt nodes, and example-input
    dynamic marks, then removes dead scalar/assert plumbing.
    """
    symbol_hints = chunk_symbol_hints_for_mode(gm)
    if not symbol_hints:
        return gm
    # Final FX codegen cannot reference raw SymPy names from partially
    # concretized expressions such as ``batch * chunked_seq``. Use placeholder
    # hints for non-selected symbols while keeping ``symbol_hints`` as the set
    # of symbols introduced/owned by EP chunking.
    concretization_hints = dict(_placeholder_symbol_hints(gm))
    concretization_hints.update(symbol_hints)

    logger.debug(
        "Concretizing %d EP chunk symbol(s) after scheduling", len(symbol_hints)
    )

    graph_modules = _iter_graph_modules(gm)
    scalar_constants: dict[fx.GraphModule, dict[fx.Node, int | bool]] = {
        graph_module: _concretize_graph_module(graph_module, concretization_hints)
        for graph_module in graph_modules
    }

    if total_scalar_constants := sum(
        len(constants) for constants in scalar_constants.values()
    ):
        logger.debug(
            "Replacing %d EP chunk scalar node(s) with hinted constants",
            total_scalar_constants,
        )
        for graph_module, constants in scalar_constants.items():
            if not constants:
                continue

            for node in graph_module.graph.nodes:

                def replace_scalar(arg):
                    if arg in constants:
                        return constants[arg]
                    return arg

                node.args = fx.map_arg(node.args, replace_scalar)
                node.kwargs = fx.map_arg(node.kwargs, replace_scalar)

    for graph_module, constants in scalar_constants.items():
        _validate_no_false_guards(graph_module, concretization_hints, constants)

    for graph_module in graph_modules:
        for node in reversed(list(graph_module.graph.nodes)):
            if _is_dead_chunk_scalar_plumbing(node):
                graph_module.graph.erase_node(node)

        graph_module.graph.lint()
        graph_module.recompile()
    _concretize_chunk_example_inputs(example_inputs, symbol_hints)
    _validate_no_chunk_symbols(gm, symbol_hints)
    logger.info("Concretized EP chunk symbols after scheduling")
    return gm
