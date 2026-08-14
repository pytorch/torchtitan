# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize chunk-local parameter-gradient collective chains.

Eager chunking can produce one collective per chunk, followed by an add at the
parameter-gradient output. Graph chunking may produce the same form when a
dtype cast prevents early materialization. This pass normalizes both to one
add before one collective so later schedulers see the same gradient buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from torch.utils._pytree import tree_leaves

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_module_fqn,
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    is_view_target,
    ordered_nodes,
    same_tensor_domain,
    tensor_meta,
)
from torchtitan.tools.logging import logger

aten = torch.ops.aten
c10d = torch.ops._c10d_functional

_GRAD_COLLECTIVES = {
    c10d.all_reduce.default,
    c10d.reduce_scatter_tensor.default,
}
_CHUNK_META = {
    "chunk_id",
    "chunked_region_fqn",
    "chunked_region_is_backward",
    "chunked_region_producer",
    "chunked_region_role",
}


@dataclass(frozen=True)
class _GradChain:
    grad: fx.Node
    collective: fx.Node
    tail: tuple[fx.Node, ...]
    chunk_id: int
    root_fqn: str

    @property
    def nodes(self) -> tuple[fx.Node, ...]:
        return (self.collective, *self.tail)


def _grad_output_ancestors(gm: fx.GraphModule) -> set[fx.Node]:
    output = next(node for node in gm.graph.nodes if node.op == "output")
    leaves = [leaf for leaf in tree_leaves(output.args[0]) if isinstance(leaf, fx.Node)]
    stack = leaves[1:]  # GraphTrainer returns [loss, *parameter_grads].
    ancestors: set[fx.Node] = set()
    while stack:
        node = stack.pop()
        if node not in ancestors:
            ancestors.add(node)
            stack.extend(node.all_input_nodes)
    return ancestors


def _chunk_owner(node: fx.Node) -> tuple[int, str] | None:
    chunk_id = node.meta.get("chunk_id")
    root_fqn = node.meta.get("chunked_region_fqn")
    if (
        node.meta.get("chunked_region_role") != "body"
        or chunk_id not in (0, 1)
        or not isinstance(root_fqn, str)
        or not node.meta.get("chunked_region_is_backward", _is_backward_node(node))
    ):
        return None
    return int(chunk_id), root_fqn


def _is_replayable_tail(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and len(node.all_input_nodes) == 1
        and (node.target is aten._to_copy.default or is_view_target(node.target))
        and isinstance(node.args[0], fx.Node)
    )


def _parse_chain(node: fx.Node) -> _GradChain | None:
    suffix = []
    while _is_replayable_tail(node):
        suffix.append(node)
        node = node.args[0]
    suffix.reverse()

    wait = None
    if (
        node.op == "call_function"
        and node.target is c10d.wait_tensor.default
        and len(node.args) == 1
        and isinstance(node.args[0], fx.Node)
    ):
        wait, node = node, node.args[0]

    if (
        node.op != "call_function"
        or node.target not in _GRAD_COLLECTIVES
        or len(node.args) < 2
        or node.args[1] != "sum"
        or not isinstance(node.args[0], fx.Node)
        or (owner := _chunk_owner(node.args[0])) is None
    ):
        return None
    chunk_id, root_fqn = owner
    return _GradChain(
        node.args[0],
        node,
        ((wait,) if wait is not None else ()) + tuple(suffix),
        chunk_id,
        root_fqn,
    )


def _same_static_call(left: fx.Node, right: fx.Node) -> bool:
    left_static = (left.args[1:], left.kwargs)
    right_static = (right.args[1:], right.kwargs)
    if left.target is not right.target or any(
        isinstance(leaf, fx.Node) for leaf in tree_leaves((left_static, right_static))
    ):
        return False
    return left_static == right_static


def _compatible(left: _GradChain, right: _GradChain) -> bool:
    left_val, right_val = tensor_meta(left.grad), tensor_meta(right.grad)
    return (
        left.root_fqn == right.root_fqn
        and {left.chunk_id, right.chunk_id} == {0, 1}
        and len(left.nodes) == len(right.nodes)
        and all(
            _same_static_call(left_node, right_node)
            for left_node, right_node in zip(left.nodes, right.nodes, strict=True)
        )
        and same_tensor_domain(left.grad, right.grad)
        and left_val is not None
        and right_val is not None
        and tuple(map(str, left_val.shape)) == tuple(map(str, right_val.shape))
    )


def _normalized_meta(
    source: fx.Node,
    *,
    root_fqn: str | None = None,
    val: Any | None = None,
) -> dict[str, Any]:
    meta = dict(source.meta)
    custom = dict(meta.get("custom", {}))
    for key in _CHUNK_META:
        meta.pop(key, None)
        custom.pop(key, None)
    module_fqn = (
        root_fqn.rsplit(".", 1)[0]
        if root_fqn is not None and "." in root_fqn
        else root_fqn or _get_module_fqn(source)
    )
    if module_fqn:
        custom[_MODULE_FQN] = module_fqn
    meta["custom"] = custom
    meta["autograd_backward"] = True
    if root_fqn is not None:
        meta["chunked_region_fqn"] = root_fqn
        meta["chunked_region_role"] = "materialization"
    if val is not None:
        meta["val"] = val
    return meta


def _clone_with_input(
    gm: fx.GraphModule,
    template: fx.Node,
    input_node: fx.Node,
) -> fx.Node:
    node = gm.graph.call_function(
        template.target,
        args=(input_node, *template.args[1:]),
        kwargs=dict(template.kwargs),
    )
    node._rename(f"{template.name}_chunk_normalized")
    node.meta = _normalized_meta(template)
    return node


def _rewrite(
    gm: fx.GraphModule,
    add: fx.Node,
    left: _GradChain,
    right: _GradChain,
) -> bool:
    for chain in (left, right):
        for node, user in zip(chain.nodes, (*chain.nodes[1:], add), strict=True):
            if set(node.users) != {user}:
                return False

    order = ordered_nodes(gm)
    template = max((left, right), key=lambda chain: order[chain.collective])
    with gm.graph.inserting_before(template.collective):
        replacement = gm.graph.call_function(
            add.target,
            args=(left.grad, right.grad, *add.args[2:]),
            kwargs=dict(add.kwargs),
        )
        replacement._rename(f"{left.grad.name}_chunk_normalized")
        replacement.meta = _normalized_meta(
            left.grad,
            root_fqn=left.root_fqn,
            val=tensor_meta(left.grad),
        )
        for node in template.nodes:
            replacement = _clone_with_input(gm, node, replacement)

    add.replace_all_uses_with(replacement)
    stale = {add, *left.nodes, *right.nodes}
    for node in sorted(stale, key=order.__getitem__, reverse=True):
        if not node.users:
            gm.graph.erase_node(node)
    return True


def normalize_chunked_grad_collective_chains_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> fx.GraphModule:
    """Accumulate sibling chunk gradients before one equivalent collective."""
    del example_inputs
    grad_ancestors = _grad_output_ancestors(gm)
    rewrites = 0
    for node in tuple(gm.graph.nodes):
        if (
            node not in grad_ancestors
            or node.op != "call_function"
            or node.target is not aten.add.Tensor
            or len(node.args) < 2
            or not isinstance(node.args[0], fx.Node)
            or not isinstance(node.args[1], fx.Node)
        ):
            continue
        left, right = _parse_chain(node.args[0]), _parse_chain(node.args[1])
        if left is not None and right is not None and _compatible(left, right):
            rewrites += _rewrite(gm, node, left, right)

    if rewrites:
        logger.debug("Normalized %d chunked gradient collective chain(s)", rewrites)
        gm.graph.lint()
        gm.recompile()
    return gm
