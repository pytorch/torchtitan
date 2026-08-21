# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Dependency-based dI/dW classification of backward graph nodes.

Contract:
  Given a backward (or joint) FX graph and the flat output value nodes for
  input gradients and parameter gradients, classify every node in either
  ancestor closure:

    di_ancestors = ancestors(input_grad_outputs)
    dw_ancestors = ancestors(param_grad_outputs)
    di_nodes      = di_ancestors - dw_ancestors
    dw_only_nodes = dw_ancestors - di_ancestors
    shared_nodes  = di_ancestors & dw_ancestors

  Classification is pure dataflow: it never inspects op names or dtypes, so a
  BF16 backward and a quantized backward with the same dependency structure
  classify identically. Only ``dw_only_nodes`` may be deferred past the dI
  outputs; ``shared_nodes`` stay on the dgrad path. ``movable_nodes`` is the
  subset of ``dw_only_nodes`` a scheduler may DEFER — move later, never
  earlier — which is what the safety rules assume. Symbolic-shape nodes
  classify like any other node; a scheduler that emits deferred nodes in
  topological order keeps sym producers ahead of their consumers.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.ep_pass_utils import is_c10d_functional_node
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    base_tensor_for_mutation_target,
    is_mutation_node,
    node_closure,
)

_COLLECTIVE_NAMESPACES = ("_c10d_functional_autograd", "_dtensor")


@dataclass(frozen=True, slots=True)
class BackwardNodePartition:
    """dI/dW classification of backward nodes plus the deferrable subset.

    ``di_nodes``, ``dw_only_nodes``, and ``shared_nodes`` are disjoint and
    together cover the union of both ancestor closures. ``movable_nodes`` is
    the subset of ``dw_only_nodes`` that a deferral pass may move past the dI
    outputs; it is empty when the graph cannot be proven safe to reorder.
    """

    di_nodes: set[fx.Node]
    dw_only_nodes: set[fx.Node]
    shared_nodes: set[fx.Node]
    movable_nodes: set[fx.Node]


def _is_collective_node(node: fx.Node) -> bool:
    if is_c10d_functional_node(node):
        return True
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace in _COLLECTIVE_NAMESPACES
    )


def _device_type(value: object) -> str | None:
    device = getattr(value, "device", None)
    return None if device is None else device.type


def _is_cpu_sync_node(node: fx.Node) -> bool:
    """Return whether ``node`` reads device values back to the host."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops.aten._local_scalar_dense.default:
        return True
    device = node.kwargs.get("device")
    out_type = (
        torch.device(device).type
        if device is not None
        else _device_type(node.meta.get("val"))
    )
    if out_type != "cpu":
        return False
    return any(
        _device_type(inp.meta.get("val")) not in (None, "cpu")
        for inp in node.all_input_nodes
    )


def _is_inert(node: fx.Node, memo: dict[fx.Node, bool]) -> bool:
    """Return whether ``node`` is a pure value whose uses are all inert.

    An inert user cannot pin its producer: def-before-use survives any
    topological reorder, and no effect or classified consumer depends on the
    node's position. The common case is a dead ``getitem`` of a multi-output
    op whose other outputs are the ones actually consumed.
    """
    cached = memo.get(node)
    if cached is not None:
        return cached
    memo[node] = False
    result = (
        node.op == "call_function"
        and not _is_collective_node(node)
        and not is_mutation_node(node)
        and not _is_cpu_sync_node(node)
        and not node.is_impure()
        and all(_is_inert(user, memo) for user in node.users)
    )
    memo[node] = result
    return result


def _written_value_bases(node: fx.Node) -> list[fx.Node] | None:
    """Return view bases of the graph values ``node`` writes to.

    Returns ``None`` when a written argument cannot be resolved to graph
    nodes; callers must treat that as unsafe.
    """
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return None
    bases: list[fx.Node] = []
    for index, arg_spec in enumerate(schema.arguments):
        if arg_spec.alias_info is None or not arg_spec.alias_info.is_write:
            continue
        if arg_spec.name in node.kwargs:
            value = node.kwargs[arg_spec.name]
        elif index < len(node.args):
            value = node.args[index]
        else:
            # Defaulted mutable argument: no graph value is written.
            continue
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            if item is None:
                continue
            base = base_tensor_for_mutation_target(item)
            if base is None:
                return None
            bases.append(base)
    return bases


def _value_descendants(node: fx.Node) -> set[fx.Node]:
    """Return all nodes deriving a value from ``node`` (transitive users)."""
    seen: set[fx.Node] = set()
    stack = [node]
    while stack:
        for user in stack.pop().users:
            if user not in seen:
                seen.add(user)
                stack.append(user)
    return seen


def _mutation_deferral_pins(graph: fx.Graph) -> tuple[bool, set[fx.Node]]:
    """Return ``(unresolvable, pinned_readers)`` for graph mutations.

    Deferral only moves nodes LATER, and mutations are never movable, so a
    write is a hazard only for a node that reads the written buffer's alias
    family WITHOUT a data edge through the write and sits BEFORE the write:
    deferring that reader could push its read past the write. Readers
    downstream of the write (data edge keeps them after it) or originally
    after the fixed write keep their relative order. A write whose target
    cannot be resolved to graph nodes makes every deferral unprovable.
    """
    order = {node: index for index, node in enumerate(graph.nodes)}
    pinned: set[fx.Node] = set()
    for node in graph.nodes:
        if not is_mutation_node(node):
            continue
        bases = _written_value_bases(node)
        if bases is None:
            return True, set()
        post_write = _value_descendants(node)
        for base in bases:
            for reader in _value_descendants(base):
                if (
                    reader is not node
                    and reader not in post_write
                    and order[reader] < order[node]
                ):
                    pinned.add(reader)
    return False, pinned


def partition_backward_nodes(
    graph_or_gm: fx.Graph | fx.GraphModule,
    *,
    input_grad_outputs: Sequence[fx.Node],
    param_grad_outputs: Sequence[fx.Node],
) -> BackwardNodePartition:
    """Classify backward nodes into dI-only, dW-only, and shared sets.

    Slicing the flat backward outputs into input-gradient and
    parameter-gradient value nodes is the caller's job; ``None`` grad slots
    are ignored.

    ``movable_nodes`` keeps only ``dw_only_nodes`` that are provably safe to
    defer. A dw-only node is pinned (classified but not movable) when it is:

    * not a ``call_function`` node (placeholders and get_attr constants are
      live-ins, not deferrable computation);
    * a collective (``_c10d_functional``/``_dtensor`` namespaces);
    * a mutation (writes through an aliased argument);
    * a CPU synchronization (``aten._local_scalar_dense`` or a device-to-host
      read);
    * side-effectful per ``fx.Node.is_impure``;
    * used by an unclassified LIVE node other than the graph output, since
      deferring it would move a definition past that untracked use. An inert
      user (a pure dead subtree, e.g. the unused ``getitem`` outputs of a
      multi-output quantize op) does not pin its producer;
    * a pre-write reader of a mutated buffer's alias family (see
      ``_mutation_deferral_pins``) — deferring it could push the read past
      the write.

    ``movable_nodes`` is empty when a mutation's written target cannot be
    resolved to graph nodes.
    """
    graph = (
        graph_or_gm.graph if isinstance(graph_or_gm, fx.GraphModule) else graph_or_gm
    )
    di_ancestors = node_closure(input_grad_outputs)
    dw_ancestors = node_closure(param_grad_outputs)
    dw_only = dw_ancestors - di_ancestors
    inert_memo: dict[fx.Node, bool] = {}

    def is_movable(node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        if _is_collective_node(node):
            return False
        if is_mutation_node(node):
            return False
        if _is_cpu_sync_node(node):
            return False
        if node.is_impure():
            return False
        return all(
            user in dw_ancestors or user.op == "output" or _is_inert(user, inert_memo)
            for user in node.users
        )

    unresolvable, hazard_pins = _mutation_deferral_pins(graph)
    if unresolvable:
        movable: set[fx.Node] = set()
    else:
        movable = {
            node for node in dw_only if node not in hazard_pins and is_movable(node)
        }

    return BackwardNodePartition(
        di_nodes=di_ancestors - dw_ancestors,
        dw_only_nodes=dw_only,
        shared_nodes=di_ancestors & dw_ancestors,
        movable_nodes=movable,
    )
