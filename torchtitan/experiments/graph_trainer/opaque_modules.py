# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic mechanism for making selected modules look opaque to a graph pass.

Pass-pipeline annotation that changes what a wrapped pass sees. The pattern is:

    1. Before the pass runs, ``_make_modules_opaque(gm, fqns)`` walks the
       graph and pairs each call_function whose ``module_fqn`` metadata is
       in ``fqns`` with the previously-anchored node via
       ``preserve_node_ordering``. Each such node ends up wrapped in a
       ``control_deps`` HOP whose first arg points to the previous wrapped
       node. To downstream graph passes walking arg edges or doing
       topological scheduling, the listed module regions look like single
       linear chains of opaque ops that cannot be reordered past one
       another.
    2. The wrapped pass runs against the annotated graph (it sees opaque
       HOPs and respects them).
    3. After the pass, ``_restore_modules(gm)`` unwraps every
       ``control_deps`` HOP back to the original op so subsequent passes
       see the unmodified graph.

The ``@opaque_modules([...])`` decorator composes the three pieces.

The motivating use case was ``ChunkedCELoss`` whose chunk loop populates a
buffer via ``aten.copy_`` in-place writes. The ``module_fqn`` of those
writes (and of the downstream ``view + to`` that read the buffer) is
``"loss"``. Bucketing-style reordering passes don't track the implicit
aliasing dependency between ``copy_`` and views of the same storage, so
they freely move the readers past the writers, leaving the readers
seeing the unwritten zeros buffer. Wrapping the loss region in
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


def _make_modules_opaque(
    gm: fx.GraphModule, fqns: Iterable[str]
) -> None:
    """Make nodes whose ``module_fqn`` is in ``fqns`` opaque to subsequent
    passes by anchoring them to their relative order: each such node gets
    wrapped in a ``control_deps`` HOP that depends on the previous
    anchored node.
    """
    fqns_set = frozenset(fqns)
    deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
    prev: fx.Node | None = None
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _node_module_fqn(node) not in fqns_set:
            continue
        # Wrap every anchored node — including the first — so the wrapping
        # pass cannot move ANY anchored node past a non-anchored node
        # either. The first node's deps tuple is empty.
        deps[node] = OrderedSet([prev]) if prev is not None else OrderedSet()
        prev = node
    preserve_node_ordering(gm.graph, deps)


def _restore_modules(gm: fx.GraphModule) -> None:
    """Inline every ``control_deps(deps_tuple, get_attr, *operands)`` call
    in ``gm.graph`` back to the wrapped op, removing the HOP wrappers
    introduced by ``_make_modules_opaque``.
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


def opaque_modules(fqns: list[str]) -> Callable[[Callable], Callable]:
    """Decorator that makes nodes whose ``module_fqn`` is in ``fqns`` look
    opaque to the wrapped graph pass.

    Wraps each such node in a ``control_deps`` HOP before calling the
    pass (so the pass sees an opaque chain that respects original
    ordering), and inlines the HOPs back to plain ops afterward (so
    downstream passes see the original graph).
    """

    def wrap(pass_fn: Callable) -> Callable:
        @functools.wraps(pass_fn)
        def wrapped(
            gm: fx.GraphModule,
            example_inputs: tuple | None = None,
            **kwargs: Any,
        ) -> fx.GraphModule:
            _make_modules_opaque(gm, fqns)
            result = gm
            try:
                returned = pass_fn(gm, example_inputs, **kwargs)
                if returned is not None:
                    result = returned
            finally:
                _restore_modules(result)
            return result

        return wrapped

    return wrap
