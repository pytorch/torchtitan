# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for reasoning about schema-declared tensor mutations in FX graphs."""

from typing import Any

import torch.fx as fx
import torch.utils._pytree as pytree


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
        for target in mutation_target_nodes(node):
            base = base_tensor_for_mutation_target(target)
            mutation_deps.setdefault(base, []).append(node)
    return mutation_deps
