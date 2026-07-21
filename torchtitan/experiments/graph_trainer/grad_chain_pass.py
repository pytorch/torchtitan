# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize chunked parameter-gradient collective chains.

EP chunking can expose one parameter-gradient chain per chunk:

    chunk0_grad -> reduce_scatter -> wait -> cast --+
                                                   add -> graph grad output
    chunk1_grad -> reduce_scatter -> wait -> cast --+

That is mathematically equivalent to reducing the chunk-local gradients before
the collective:

    add(chunk0_grad, chunk1_grad) -> reduce_scatter -> wait -> cast -> grad output

The normalization belongs after eager or graph chunking and before dense FSDP
or EP overlap scheduling so those schedulers see the same grad buckets for both
chunking implementations.
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


_GRAD_COLLECTIVE_TARGETS = {
    c10d.all_reduce.default,
    c10d.reduce_scatter_tensor.default,
}

_CHUNK_META_KEYS = (
    "chunk_id",
    "chunked_region_fqn",
    "chunked_region_is_backward",
    "chunked_region_producer",
    "chunked_region_role",
)


@dataclass(frozen=True)
class _GradCollectiveChain:
    input: fx.Node
    collective: fx.Node
    wait: fx.Node | None
    suffix: tuple[fx.Node, ...]
    chunk_id: int
    root_fqn: str


def _graph_output_nodes(gm: fx.GraphModule) -> list[fx.Node]:
    output = next(node for node in gm.graph.nodes if node.op == "output")
    return [leaf for leaf in tree_leaves(output.args[0]) if isinstance(leaf, fx.Node)]


def _grad_output_ancestors(gm: fx.GraphModule) -> set[fx.Node]:
    outputs = _graph_output_nodes(gm)
    stack = list(outputs[1:])  # GraphTrainer returns [loss, *param_grads].
    ancestors: set[fx.Node] = set()
    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        stack.extend(node.all_input_nodes)
    return ancestors


def _chunk_body(node: fx.Node) -> tuple[int, str] | None:
    if node.meta.get("chunked_region_role") != "body":
        return None
    chunk_id = node.meta.get("chunk_id")
    root_fqn = node.meta.get("chunked_region_fqn")
    if chunk_id not in (0, 1) or not isinstance(root_fqn, str):
        return None
    if not node.meta.get("chunked_region_is_backward", _is_backward_node(node)):
        return None
    return int(chunk_id), root_fqn


def _is_replayable_suffix(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and len(node.all_input_nodes) == 1
        and (node.target is aten._to_copy.default or is_view_target(node.target))
        and node.args
        and isinstance(node.args[0], fx.Node)
    )


def _chain_from_add_input(node: fx.Node) -> _GradCollectiveChain | None:
    suffix = []
    while _is_replayable_suffix(node):
        suffix.append(node)
        node = node.args[0]
    suffix.reverse()

    wait = None
    collective = node
    if (
        node.op == "call_function"
        and node.target is c10d.wait_tensor.default
        and len(node.args) == 1
        and isinstance(node.args[0], fx.Node)
    ):
        wait = node
        collective = node.args[0]

    if (
        collective.op != "call_function"
        or collective.target not in _GRAD_COLLECTIVE_TARGETS
        or not collective.args
        or not isinstance(collective.args[0], fx.Node)
    ):
        return None

    grad_input = collective.args[0]
    chunk_body = _chunk_body(grad_input)
    if chunk_body is None:
        return None
    chunk_id, root_fqn = chunk_body
    return _GradCollectiveChain(
        input=grad_input,
        collective=collective,
        wait=wait,
        suffix=tuple(suffix),
        chunk_id=chunk_id,
        root_fqn=root_fqn,
    )


def _same_static_call(lhs: fx.Node, rhs: fx.Node) -> bool:
    if lhs.target is not rhs.target:
        return False
    if len(lhs.args) != len(rhs.args):
        return False
    if set(lhs.kwargs) != set(rhs.kwargs):
        return False

    for left, right in zip(lhs.args[1:], rhs.args[1:]):
        if any(isinstance(leaf, fx.Node) for leaf in tree_leaves((left, right))):
            return False
        if left != right:
            return False
    for key in lhs.kwargs:
        left, right = lhs.kwargs[key], rhs.kwargs[key]
        if any(isinstance(leaf, fx.Node) for leaf in tree_leaves((left, right))):
            return False
        if left != right:
            return False
    return True


def _same_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if len(lhs.shape) != len(rhs.shape):
        return False
    return all(str(left) == str(right) for left, right in zip(lhs.shape, rhs.shape))


def _compatible_chains(left: _GradCollectiveChain, right: _GradCollectiveChain) -> bool:
    if left.root_fqn != right.root_fqn:
        return False
    if {left.chunk_id, right.chunk_id} != {0, 1}:
        return False
    if not _same_static_call(left.collective, right.collective):
        return False
    if (left.wait is None) != (right.wait is None):
        return False
    if len(left.suffix) != len(right.suffix) or any(
        not _same_static_call(left_node, right_node)
        for left_node, right_node in zip(left.suffix, right.suffix)
    ):
        return False
    if not same_tensor_domain(left.input, right.input):
        return False
    left_val, right_val = tensor_meta(left.input), tensor_meta(right.input)
    if left_val is None or right_val is None or not _same_shape(left_val, right_val):
        return False
    return True


def _materialization_meta(
    source: fx.Node,
    *,
    root_fqn: str,
    val: Any,
) -> dict[str, Any]:
    meta = dict(source.meta)
    _clear_chunk_body_meta_for_dict(meta)
    custom = dict(meta.get("custom", {}))
    custom[_MODULE_FQN] = _parent_module_fqn(root_fqn)
    meta["custom"] = custom
    meta["autograd_backward"] = True
    meta["chunked_region_fqn"] = root_fqn
    meta["chunked_region_role"] = "materialization"
    meta["val"] = val
    return meta


def _clear_chunk_body_meta_for_dict(meta: dict[str, Any]) -> None:
    for key in _CHUNK_META_KEYS:
        meta.pop(key, None)
    if isinstance(meta.get("custom"), dict):
        custom = dict(meta["custom"])
        for key in _CHUNK_META_KEYS:
            custom.pop(key, None)
        meta["custom"] = custom


def _parent_module_fqn(root_fqn: str) -> str:
    return root_fqn.rsplit(".", 1)[0] if "." in root_fqn else root_fqn


def _suffix_meta(source: fx.Node) -> dict[str, Any]:
    meta = dict(source.meta)
    _clear_chunk_body_meta_for_dict(meta)
    custom = dict(meta.get("custom", {}))
    if module_fqn := _get_module_fqn(source):
        custom[_MODULE_FQN] = module_fqn
    meta["custom"] = custom
    meta["autograd_backward"] = True
    return meta


def _erase_if_unused(gm: fx.GraphModule, nodes: set[fx.Node]) -> None:
    order = ordered_nodes(gm)
    for node in sorted(nodes, key=lambda n: order[n], reverse=True):
        if node.users:
            continue
        gm.graph.erase_node(node)


def _rewrite_add_of_collective_chains(
    gm: fx.GraphModule,
    add_node: fx.Node,
    left: _GradCollectiveChain,
    right: _GradCollectiveChain,
) -> bool:
    for chain in (left, right):
        linear_suffix = ((chain.wait,) if chain.wait is not None else ()) + chain.suffix
        users = (*linear_suffix, add_node)
        for current, user in zip((chain.collective, *linear_suffix), users):
            if set(current.users) != {user}:
                return False

    chains_by_chunk = {left.chunk_id: left, right.chunk_id: right}
    chunk0, chunk1 = chains_by_chunk[0], chains_by_chunk[1]

    order = ordered_nodes(gm)
    insertion_chain = max((left, right), key=lambda chain: order[chain.collective])
    with gm.graph.inserting_before(insertion_chain.collective):
        grad_sum = gm.graph.call_function(
            aten.add.Tensor, args=(chunk0.input, chunk1.input)
        )
        grad_sum._rename(f"{chunk0.input.name}_chunk_normalized")
        grad_sum.meta = _materialization_meta(
            chunk0.input,
            root_fqn=chunk0.root_fqn,
            val=tensor_meta(chunk0.input),
        )

        collective_args = (grad_sum, *insertion_chain.collective.args[1:])
        collective = gm.graph.call_function(
            insertion_chain.collective.target,
            args=collective_args,
            kwargs=dict(insertion_chain.collective.kwargs),
        )
        collective._rename(f"{insertion_chain.collective.name}_chunk_normalized")
        collective.meta = _suffix_meta(insertion_chain.collective)

        replacement = collective
        if insertion_chain.wait is not None:
            replacement = gm.graph.call_function(
                c10d.wait_tensor.default, args=(collective,)
            )
            replacement._rename(f"{insertion_chain.wait.name}_chunk_normalized")
            replacement.meta = _suffix_meta(insertion_chain.wait)

        for suffix_node in insertion_chain.suffix:
            replacement = gm.graph.call_function(
                suffix_node.target,
                args=(replacement, *suffix_node.args[1:]),
                kwargs=dict(suffix_node.kwargs),
            )
            replacement._rename(f"{suffix_node.name}_chunk_normalized")
            replacement.meta = _suffix_meta(suffix_node)

    add_node.replace_all_uses_with(replacement)
    old_nodes = {
        add_node,
        left.collective,
        right.collective,
        *left.suffix,
        *right.suffix,
    }
    if left.wait is not None:
        old_nodes.add(left.wait)
    if right.wait is not None:
        old_nodes.add(right.wait)
    _erase_if_unused(
        gm,
        old_nodes,
    )
    return True


def normalize_chunked_grad_collective_chains_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> fx.GraphModule:
    """Deduplicate chunk sibling grad collective chains before scheduling."""
    del example_inputs
    grad_ancestors = _grad_output_ancestors(gm)
    rewrites = 0
    for node in tuple(gm.graph.nodes):
        if node not in grad_ancestors:
            continue
        if (
            node.op != "call_function"
            or node.target is not aten.add.Tensor
            or len(node.args) < 2
            or not isinstance(node.args[0], fx.Node)
            or not isinstance(node.args[1], fx.Node)
        ):
            continue

        left = _chain_from_add_input(node.args[0])
        right = _chain_from_add_input(node.args[1])
        if left is None or right is None or not _compatible_chains(left, right):
            continue

        rewrites += int(_rewrite_add_of_collective_chains(gm, node, left, right))

    if rewrites:
        logger.debug(
            "Normalized %d chunked parameter-gradient collective chain(s)", rewrites
        )
        gm.graph.lint()
        gm.recompile()
    return gm
