# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule independent MoE work between a DeepEP dispatch launch and its wait.

The DeepEP token dispatcher with ``split_dispatch_launch_wait`` gives the traced
step a ``deepep.launch_dispatch`` node and a ``deepep.wait_dispatch`` node per MoE
layer. The launch returns once the communication is queued; the wait blocks the
host until the receive counts arrive and then materializes the received tokens.
Eager code runs the two back to back, so the GPU idles during the host wait.

This pass moves forward work that does not depend on the wait -- in the common
MoE layout the shared experts, which eager code computes after the combine -- in
front of the wait, directly after the launch. The GPU then runs that work while
the dispatch is in flight. It is a scheduler only: no node is added, removed, or
changed, and every data dependency is kept by a stable topological sort.

Contract
========
For each forward ``wait_dispatch`` node W with launch node L:

* the window is the graph order between L and the first consumer of the output
  of the ``deepep.combine`` node fed by W (the end of the MoE layer);
* filler is every window node that is not W, does not depend on W, has no side
  effect (no mutating schema, no DeepEP op, no effect token), and whose inputs
  are either before L or filler themselves, so it is computable before W;
* filler never reads a tensor that a window node mutates;
* the new order keeps L, then the filler, then W; all other nodes keep their
  relative order.

Backward is untouched: the dispatch backward is one ``dispatch_bwd`` op. The
two-chunk ``ep_overlap`` pass is unrelated; DeepEP's single communication buffer
cannot hold two in-flight dispatches, which is why this pass only sinks waits.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch.utils._ordered_set import OrderedSet

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.ep_pass_utils import ordered_nodes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepEPOverlapRecord:
    """What the pass did for one dispatch wait, kept in ``gm.meta``."""

    launch: str
    wait: str
    combine: str
    num_filler: int
    filler_targets: tuple[str, ...]


def _deepep_op(name: str):
    """Return the default overload of a ``deepep::`` op, or None when unregistered."""
    try:
        return getattr(torch.ops.deepep, name).default
    except (AttributeError, RuntimeError):
        return None


def _call_target(node: fx.Node):
    """Return the op a call node runs, looking through ``with_effects``."""
    if node.op != "call_function":
        return None
    if node.target is torch.ops.higher_order.with_effects:
        return node.args[1]
    return node.target


def _op_args(node: fx.Node) -> tuple:
    """Return the op arguments of a call node, looking through ``with_effects``."""
    if node.target is torch.ops.higher_order.with_effects:
        return tuple(node.args[2:])
    return tuple(node.args)


def _value_producer(node: fx.Node) -> fx.Node:
    """Return the call node behind a ``getitem`` of a multi-output op."""
    if node.op == "call_function" and node.target is operator.getitem:
        return node.args[0]
    return node


def _value_users(node: fx.Node) -> list[fx.Node]:
    """Return the consumers of a node's values, looking through tuple unpacking."""
    users: list[fx.Node] = []
    for user in node.users:
        if user.op == "call_function" and user.target is operator.getitem:
            if node.target is torch.ops.higher_order.with_effects and user.args[1] == 0:
                # The effect token is plumbing, not a value.
                continue
            users.extend(user.users)
        else:
            users.append(user)
    return users


def _descendants(node: fx.Node) -> set[fx.Node]:
    result: set[fx.Node] = set()
    stack = list(node.users)
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(current.users)
    return result


def _is_mutating(node: fx.Node) -> bool:
    target = node.target
    return isinstance(target, torch._ops.OpOverload) and target._schema.is_mutable


def _is_pure_forward_call(node: fx.Node) -> bool:
    """Whether a node may move: a forward call without side effects."""
    if node.op != "call_function" or _is_backward_node(node):
        return False
    if node.target is torch.ops.higher_order.with_effects:
        return False
    target = node.target
    if target is operator.getitem:
        return True
    if not isinstance(target, torch._ops.OpOverload):
        return False
    if target.namespace == "deepep" or target._schema.is_mutable:
        return False
    return True


def _collect_filler(
    window: list[fx.Node],
    *,
    launch: fx.Node,
    wait: fx.Node,
    wait_descendants: set[fx.Node],
    order: dict[fx.Node, int],
) -> list[fx.Node]:
    mutated: set[fx.Node] = set()
    for node in window:
        if _is_mutating(node):
            mutated.update(node.all_input_nodes)
    filler: list[fx.Node] = []
    filler_set: set[fx.Node] = set()
    launch_order = order[launch]
    for node in window:
        if node is wait or node in wait_descendants:
            continue
        if not _is_pure_forward_call(node):
            continue
        inputs = node.all_input_nodes
        if any(order[inp] > launch_order and inp not in filler_set for inp in inputs):
            continue
        if any(inp in mutated for inp in inputs):
            continue
        filler.append(node)
        filler_set.add(node)
    return filler


def deepep_dispatch_overlap_pass(
    gm: fx.GraphModule, example_inputs: tuple | None = None
) -> fx.GraphModule:
    """Move wait-independent MoE work between each DeepEP dispatch launch and wait."""
    del example_inputs
    launch_op = _deepep_op("launch_dispatch")
    wait_op = _deepep_op("wait_dispatch")
    combine_op = _deepep_op("combine")
    if launch_op is None or wait_op is None or combine_op is None:
        logger.info("deepep_dispatch_overlap: DeepEP ops are not registered, skipping")
        return gm

    order = ordered_nodes(gm)
    nodes = list(gm.graph.nodes)
    deps: dict[fx.Node, OrderedSet[fx.Node]] = {}
    records: list[DeepEPOverlapRecord] = []
    expected: list[tuple[fx.Node, list[fx.Node], fx.Node]] = []
    for wait in nodes:
        if _call_target(wait) is not wait_op or _is_backward_node(wait):
            continue
        launch = _value_producer(_op_args(wait)[3])
        if _call_target(launch) is not launch_op:
            raise ValueError(
                f"deepep_dispatch_overlap: {wait.name} does not consume a "
                f"deepep.launch_dispatch result (got {launch.name})"
            )
        wait_descendants = _descendants(wait)
        combines = [n for n in wait_descendants if _call_target(n) is combine_op]
        if not combines:
            raise ValueError(
                f"deepep_dispatch_overlap: no deepep.combine depends on {wait.name}"
            )
        combine = min(combines, key=order.__getitem__)
        consumers = _value_users(combine)
        if not consumers:
            raise ValueError(f"deepep_dispatch_overlap: {combine.name} has no consumer")
        stop = min(order[n] for n in consumers)
        window = [n for n in nodes if order[launch] < order[n] < stop]
        filler = _collect_filler(
            window,
            launch=launch,
            wait=wait,
            wait_descendants=wait_descendants,
            order=order,
        )
        records.append(
            DeepEPOverlapRecord(
                launch=launch.name,
                wait=wait.name,
                combine=combine.name,
                num_filler=len(filler),
                filler_targets=tuple(sorted({str(n.target) for n in filler})),
            )
        )
        if not filler:
            continue
        for node in filler:
            deps.setdefault(node, OrderedSet()).add(launch)
        deps.setdefault(wait, OrderedSet()).update(filler)
        expected.append((launch, filler, wait))

    gm.meta["deepep_dispatch_overlap"] = records
    if not expected:
        logger.info(
            "deepep_dispatch_overlap: %d dispatch wait(s), nothing to move",
            len(records),
        )
        return gm

    _stable_topological_sort(gm.graph, deps)
    gm.graph.lint()
    gm.recompile()
    new_order = ordered_nodes(gm)
    for launch, filler, wait in expected:
        first = min(new_order[n] for n in filler)
        last = max(new_order[n] for n in filler)
        if not new_order[launch] < first and last < new_order[wait]:
            raise ValueError(
                "deepep_dispatch_overlap failed to place filler between "
                f"{launch.name} and {wait.name}"
            )
    logger.info(
        "deepep_dispatch_overlap: moved %d node(s) before %d of %d dispatch wait(s)",
        sum(len(filler) for _, filler, _ in expected),
        len(expected),
        len(records),
    )
    return gm
