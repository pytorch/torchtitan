# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for graph PP passes, extracted from autoparallel.

These are small FX graph helpers used by split_fsdp_collectives.py and
split_di_dw_graph.py.
"""

import operator
from typing import Optional

import torch
import torch.fx


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_reduce_scatter_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def find_last_all_gather_in_chain(
    start_node: torch.fx.Node,
) -> Optional[torch.fx.Node]:
    """Follow a linear chain from start_node and return the last all_gather node."""
    node = start_node
    last_ag_node = None
    while True:
        if len(node.users) != 1:
            break
        user = next(iter(node.users))
        if len(user.all_input_nodes) > 1:
            break
        node = user
        if is_all_gather_into_tensor(node):
            last_ag_node = node
    return last_ag_node


def find_last_user_in_wait_chain(wait_node: torch.fx.Node) -> torch.fx.Node:
    """Follow the linear chain from a wait_tensor node to find the last node
    in the FSDP prefetch pattern (handles split -> getitem -> cat)."""
    w = wait_node
    while True:
        if len(w.users) != 1:
            if w.op == "call_function" and w.target == torch.ops.aten.split.Tensor:
                if all(
                    split_user.op == "call_function"
                    and split_user.target == operator.getitem
                    and len(split_user.users) == 1
                    for split_user in w.users
                ):
                    getitem_users = [
                        next(iter(getitem_node.users)) for getitem_node in w.users
                    ]
                    potential_cat_op = getitem_users[0]
                    if all(
                        potential_cat_op == getitem_user
                        for getitem_user in getitem_users
                    ) and (
                        potential_cat_op.op == "call_function"
                        and potential_cat_op.target == torch.ops.aten.cat.default
                    ):
                        w = potential_cat_op
                        continue
            else:
                break
        user = next(iter(w.users))
        if len(user.all_input_nodes) > 1:
            break
        w = user
    return w


def find_last_non_view_node_in_chain(node: torch.fx.Node) -> torch.fx.Node:
    """Traverse backwards through view ops to find the last non-view node."""
    result = node
    while hasattr(result.target, "is_view") and result.target.is_view:
        assert (
            len(result.all_input_nodes) == 1
        ), "View op should have only one input node"
        result = result.all_input_nodes[0]
    return result


def rename_placeholder_node(
    fx_g: torch.fx.GraphModule, node: torch.fx.Node, new_name: str
) -> None:
    """Rename a placeholder node in an FX graph."""
    assert node.op == "placeholder", f"only placeholder node supported, got {node.op}"
    with fx_g.graph.inserting_before(node):
        new_node = fx_g.graph.placeholder(new_name)
        new_node.meta.update(node.meta)
        node.replace_all_uses_with(new_node)
        fx_g.graph.erase_node(node)
