# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for reasoning about tensor storage and schema-declared FX mutations."""

from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from typing import Any, TypeVar

import torch
import torch.fx as fx
import torch.utils._pytree as pytree


_Identity = TypeVar("_Identity", bound=Hashable)


def tensor_storages(graph: fx.Graph) -> dict[fx.Node, frozenset[torch.UntypedStorage]]:
    """Map each node to the tensor storages in its existing value metadata."""
    return {
        node: frozenset(
            value.untyped_storage()
            for value in pytree.tree_leaves(node.meta.get("val"))
            if isinstance(value, torch.Tensor)
        )
        for node in graph.nodes
    }


def mutation_write_sets(
    graph: fx.Graph, aliases: Mapping[fx.Node, Set[_Identity]]
) -> dict[fx.Node, set[_Identity]]:
    """Map each node to the supplied identities written by mutable arguments."""
    return {
        node: set().union(*(aliases[target] for target in mutation_target_nodes(node)))
        for node in graph.nodes
    }


def can_swap_a_and_b_read_write_check(
    *,
    reads_a: Set[_Identity],
    writes_a: Set[_Identity],
    reads_b: Set[_Identity],
    writes_b: Set[_Identity],
) -> bool:
    """Return whether A and B can swap without read/write conflicts."""
    # weight = buffer.view(...)  # Shares storage with buffer.
    # copy_(buffer, source)      # A writes that storage.
    # linear(x, weight)          # B reads it, so A and B cannot swap.
    return not (writes_a & (reads_b | writes_b) or writes_b & reads_a)


def _try_move_nodes(
    nodes: Sequence[fx.Node],
    anchor: fx.Node,
    *,
    reads: Mapping[fx.Node, Set[_Identity]],
    writes: Mapping[fx.Node, Set[_Identity]],
    before: bool,
) -> bool:
    graph_nodes = tuple(anchor.graph.nodes)
    positions = {node: index for index, node in enumerate(graph_nodes)}
    moving = set(nodes)
    moving_reads = set().union(*(reads[node] for node in nodes))
    moving_writes = set().union(*(writes[node] for node in nodes))
    first = min(positions[anchor], *(positions[node] for node in nodes))
    last = max(positions[anchor], *(positions[node] for node in nodes))
    # FX value edges miss dependencies through aliases. Crossing a write
    # could change an input; crossing a read could expose an unfinished output.
    if not all(
        can_swap_a_and_b_read_write_check(
            reads_a=moving_reads,
            writes_a=moving_writes,
            reads_b=reads[node],
            writes_b=writes[node],
        )
        for node in graph_nodes[first : last + 1]
        if node not in moving
    ):
        return False
    if before:
        if any(
            positions[dependency] >= positions[anchor]
            for node in nodes
            for dependency in node.all_input_nodes
            if dependency not in moving
        ):
            return False
        for node in nodes:
            anchor.prepend(node)
    else:
        if any(
            positions[user] <= positions[anchor]
            for node in nodes
            for user in node.users
            if user not in moving
        ):
            return False
        for node in reversed(nodes):
            anchor.append(node)
    return True


def try_move_nodes_before(
    nodes: Sequence[fx.Node],
    anchor: fx.Node,
    *,
    reads: Mapping[fx.Node, Set[_Identity]],
    writes: Mapping[fx.Node, Set[_Identity]],
) -> bool:
    """Hoist a node group before anchor while preserving its internal order.

    Nodes must be nonempty, in graph order, and after anchor in the same graph.
    Return False without graph changes if FX or supplied mutation dependencies
    block the move. Read/write maps stay valid; node positions refresh per call.

    INPUT:
        wait0
        compute0
        prepare1
        launch1

    OUTPUT when movement is allowed:
        prepare1  # Move the full preparation group with its launch.
        launch1
        wait0
        compute0
    """
    return _try_move_nodes(nodes, anchor, reads=reads, writes=writes, before=True)


def try_move_nodes_after(
    nodes: Sequence[fx.Node],
    anchor: fx.Node,
    *,
    reads: Mapping[fx.Node, Set[_Identity]],
    writes: Mapping[fx.Node, Set[_Identity]],
) -> bool:
    """Sink a node group after anchor while preserving its internal order.

    Nodes must be nonempty, in graph order, and before anchor in the same graph.
    Return False without graph changes if FX or supplied mutation dependencies
    block the move. Read/write maps stay valid; node positions refresh per call.

    INPUT:
        launch0
        wait0
        unpack0
        backward1
        launch1

    OUTPUT when movement is allowed:
        launch0
        backward1
        launch1
        wait0    # Delay completion together with its output assembly.
        unpack0
    """
    return _try_move_nodes(nodes, anchor, reads=reads, writes=writes, before=False)


def _schema_and_bound_arguments(
    node: fx.Node,
) -> tuple[Any | None, dict[str, object]]:
    if node.op != "call_function":
        return None, {}
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return None, {}
    bound_arguments: dict[str, object] = {
        str(argument.name): value
        for argument, value in zip(schema.arguments, node.args)
    }
    bound_arguments.update(node.kwargs)
    return schema, bound_arguments


def is_mutation_node(node: fx.Node) -> bool:
    """Return whether ``node`` writes to a schema argument.

    Example::

        Input::
            node: add.out(x, y, out=buffer)

        ->

        Result::
            True
    """

    schema, _ = _schema_and_bound_arguments(node)
    if schema is None:
        return False
    return any(
        argument.alias_info is not None and argument.alias_info.is_write
        for argument in schema.arguments
    )


def mutation_arguments(node: fx.Node) -> dict[str, object]:
    """Return the schema argument names and values written by ``node``.

    Example::

        Input::
            node: add.out(x, y, out=buffer)

        ->

        Result::
            {"out": buffer}
    """

    schema, bound_arguments = _schema_and_bound_arguments(node)
    if schema is None:
        return {}
    return {
        str(argument.name): bound_arguments[str(argument.name)]
        for argument in schema.arguments
        if argument.alias_info is not None
        and argument.alias_info.is_write
        and str(argument.name) in bound_arguments
    }


def replace_mutation_argument(
    node: fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    argument_name: str,
    value: object,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Replace a writable argument while preserving its original call binding.

    Example::

        Input::
            node: add.out(x, y, out=buffer)
            args: (x, y)
            kwargs: {"out": buffer}
            argument_name: "out"
            value: previous_add_out

        ->

        Result::
            args: (x, y)
            kwargs: {"out": previous_add_out}
    """

    assert argument_name in mutation_arguments(node)
    if argument_name in node.kwargs:
        kwargs = dict(kwargs)
        kwargs[argument_name] = value
        return args, kwargs

    schema, _ = _schema_and_bound_arguments(node)
    assert schema is not None
    argument_index = next(
        index
        for index, argument in enumerate(schema.arguments)
        if str(argument.name) == argument_name
    )
    assert argument_index < len(args)
    args = tuple(
        value if index == argument_index else arg for index, arg in enumerate(args)
    )
    return args, kwargs


def mutation_target_nodes(node: fx.Node) -> list[fx.Node]:
    """Return FX nodes contained in the schema arguments written by ``node``.

    Example::

        Input::
            node: amp_foreach([x, y], found_inf, inv_scale)

        ->

        Result::
            [x, y, found_inf]
    """

    targets: list[fx.Node] = []
    for value in mutation_arguments(node).values():
        targets.extend(
            leaf for leaf in pytree.tree_leaves(value) if isinstance(leaf, fx.Node)
        )
    return list(dict.fromkeys(targets))


def base_tensor_for_mutation_target(node: fx.Node) -> fx.Node:
    """Return the alias base reached by following schema-declared view aliases.

    Example::

        Input::
            output_view = view(output, [-1])
            node: output_view

        ->

        Result::
            output
    """

    while (
        node.op == "call_function"
        and hasattr(node.target, "is_view")
        and node.target.is_view
    ):
        schema, bound_arguments = _schema_and_bound_arguments(node)
        if schema is None or len(schema.returns) != 1:
            break
        return_alias = schema.returns[0].alias_info
        if return_alias is None:
            break
        return_aliases = set(return_alias.before_set) | set(return_alias.after_set)
        alias_sources: list[fx.Node] = []
        for argument in schema.arguments:
            argument_alias = argument.alias_info
            if argument_alias is None:
                continue
            argument_aliases = set(argument_alias.before_set) | set(
                argument_alias.after_set
            )
            if return_aliases.isdisjoint(argument_aliases):
                continue
            alias_sources.extend(
                leaf
                for leaf in pytree.tree_leaves(bound_arguments.get(argument.name, ()))
                if isinstance(leaf, fx.Node)
            )
        alias_sources = list(dict.fromkeys(alias_sources))
        if len(alias_sources) != 1:
            break
        node = alias_sources[0]
    return node


def mutation_target_bases(node: fx.Node) -> list[fx.Node]:
    """Return alias bases of writable arguments in target order, including duplicates."""
    return [
        base_tensor_for_mutation_target(target)
        for target in mutation_target_nodes(node)
    ]


def mutation_deps(graph: fx.Graph) -> dict[fx.Node, list[fx.Node]]:
    """Map each mutated alias base to its mutation nodes in graph order.

    Example::

        Input::
            buffer = empty(...)
            add_out = add.out(x, y, out=buffer)
            result = cos(buffer)

        ->

        Result::
            {buffer: [add_out]}
    """

    mutation_deps: dict[fx.Node, list[fx.Node]] = {}
    for node in graph.nodes:
        for base in mutation_target_bases(node):
            mutation_deps.setdefault(base, []).append(node)
    return mutation_deps


def node_closure_with_mutations(
    values: Iterable[object], *, forward_only: bool = False
) -> set[fx.Node]:
    """Return roots, value dependencies, and writes through aliases before each read.

    INPUT: values=[result]
        source = placeholder("source")
        buffer = empty(...)
        alias = view(buffer, ...)
        write = _foreach_copy_([alias], [source])  # Returns None; no edge to result.
        result = sin(alias)                      # Requested root reads the buffer.
        later_write = zero_(buffer)              # After the read; excluded.
        unrelated = cos(source)                  # No dependency path; excluded.

    OUTPUT: closure members shown in graph order; returned as a set.
        source = placeholder("source")           # Needed by the included write.
        buffer = empty(...)
        alias = view(buffer, ...)
        write = _foreach_copy_([alias], [source])  # Included through buffer aliasing.
        result = sin(alias)

    Roots are read at graph end; dependencies are read at their users.
    With forward_only=True, skip writers tagged autograd_backward.
    """
    roots = [value for value in values if isinstance(value, fx.Node)]
    if not roots:
        return set()
    graph = roots[0].graph
    node_positions = {node: index for index, node in enumerate(graph.nodes)}
    writers_by_base = mutation_deps(graph)
    graph_end_position = len(node_positions)
    pending_reads = [(root, graph_end_position) for root in roots]
    visited_reads = set()
    closure = set()
    while pending_reads:
        node, read_position = pending_reads.pop()
        if (node, read_position) in visited_reads:
            continue
        visited_reads.add((node, read_position))
        for writer in writers_by_base.get(base_tensor_for_mutation_target(node), ()):
            if node_positions[writer] < read_position and not (
                forward_only and writer.meta.get("autograd_backward", False)
            ):
                pending_reads.append((writer, node_positions[writer]))
        if node not in closure:
            closure.add(node)
            pending_reads.extend(
                (dependency, node_positions[node])
                for dependency in node.all_input_nodes
            )
    return closure
