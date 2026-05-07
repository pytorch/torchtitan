# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic mechanism for fixing the relative order of selected ops across a
graph pass that would otherwise be free to reorder them.

Pass-pipeline annotation that changes what a wrapped pass sees. The pattern is:

    1. Before the pass runs, ``_pin_ops_order(gm, fqns)`` walks the
       graph and pairs each call_function whose ``module_fqn`` metadata is
       in ``fqns`` with the previously-pinned node via
       ``preserve_node_ordering``. Each such node ends up wrapped in a
       ``control_deps`` HOP whose first arg points to the previous wrapped
       node. To downstream graph passes walking arg edges or doing
       topological scheduling, the listed ops form a single linear chain
       that cannot be reordered past one another.
    2. The wrapped pass runs against the annotated graph (it sees the
       chained HOPs and respects them).
    3. After the pass, ``_unpin_ops_order(gm)`` unwraps every
       ``control_deps`` HOP back to the original op so subsequent passes
       see the unmodified graph.

The ``@preserve_ops_order([...])`` decorator composes the three pieces.

The motivating use case was ``ChunkedCELoss`` whose chunk loop populates a
buffer via ``aten.copy_`` in-place writes. The ``module_fqn`` of those
writes (and of the downstream ``view + to`` that read the buffer) is
``"loss"``. Bucketing-style reordering passes don't track the implicit
aliasing dependency between ``copy_`` and views of the same storage, so
they freely move the readers past the writers, leaving the readers
seeing the unwritten zeros buffer. Pinning the loss region with
``control_deps`` HOPs forces the bucketing pass to keep the writes and
readers in their original order.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.fx as fx
from torch._inductor.fx_passes.control_dependencies import (
    control_deps,
    preserve_node_ordering,
)
from torch.utils._ordered_set import OrderedSet


_MODULE_FQN_KEY = "module_fqn"


def _node_module_fqn(node: fx.Node) -> str | None:
    """Return the ``module_fqn`` annotation on a node, if any."""
    custom = node.meta.get("custom")
    if custom is None:
        return None
    return custom.get(_MODULE_FQN_KEY)


def _pin_ops_order(gm: fx.GraphModule, fqns: Iterable[str]) -> None:
    """Pin the relative order of nodes whose ``module_fqn`` is in ``fqns``:
    each such node gets wrapped in a ``control_deps`` HOP that depends on
    the previous pinned node, so subsequent passes cannot reorder them.
    """
    fqns_set = frozenset(fqns)
    deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
    prev: fx.Node | None = None
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _node_module_fqn(node) not in fqns_set:
            continue
        # Wrap every pinned node — including the first — so the wrapping
        # pass cannot move ANY pinned node past a non-pinned node
        # either. The first node's deps tuple is empty.
        deps[node] = OrderedSet([prev]) if prev is not None else OrderedSet()
        prev = node
    preserve_node_ordering(gm.graph, deps)


def _unpin_ops_order(gm: fx.GraphModule) -> None:
    """Inline every ``control_deps(deps_tuple, get_attr, *operands)`` call
    in ``gm.graph`` back to the wrapped op, removing the HOP wrappers
    introduced by ``_pin_ops_order``.
    """
    control_deps_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is control_deps
    ]
    for node in control_deps_nodes:
        get_attr_node = node.args[1]
        operands = node.args[2:]
        subgraph: fx.GraphModule = getattr(gm, str(get_attr_node.target))

        env: dict[fx.Node, Any] = dict(
            zip(subgraph.graph.find_nodes(op="placeholder"), operands)
        )
        with gm.graph.inserting_before(node):
            inlined: fx.Node | None = None
            for sub_node in subgraph.graph.nodes:
                if sub_node.op == "placeholder":
                    continue
                if sub_node.op == "output":
                    output_value = sub_node.args[0]
                    inlined = env.get(output_value, output_value)
                    break
                new_node = gm.graph.node_copy(sub_node, lambda n: env[n])
                env[sub_node] = new_node

        assert inlined is not None
        node.replace_all_uses_with(inlined)
        gm.graph.erase_node(node)
        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)
            delattr(gm, str(get_attr_node.target))

    if control_deps_nodes:
        gm.graph.lint()
        gm.recompile()


def preserve_ops_order(fqns: list[str]) -> Callable[[Callable], Callable]:
    """Decorator that fixes the relative order of nodes whose
    ``module_fqn`` is in ``fqns`` across the wrapped graph pass.

    Pins each such node into a ``control_deps``-chained sequence before
    calling the pass (so the pass cannot reorder them past one another or
    past non-pinned ops), and inlines the HOPs back to plain ops
    afterward (so downstream passes see the original graph).
    """

    def wrap(pass_fn: Callable) -> Callable:
        @functools.wraps(pass_fn)
        def wrapped(
            gm: fx.GraphModule,
            example_inputs: tuple | None = None,
            **kwargs: Any,
        ) -> fx.GraphModule:
            _pin_ops_order(gm, fqns)
            result = gm
            try:
                returned = pass_fn(gm, example_inputs, **kwargs)
                if returned is not None:
                    result = returned
            finally:
                _unpin_ops_order(result)
            return result

        return wrapped

    return wrap
