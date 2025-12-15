# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.checkpoint import CheckpointPolicy


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


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


def is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    """
    Returns True if the node is a wait_tensor node that is the result of an all_gather
    that can be arbitrarily prefetched, i.e., if all its recursive inputs are
    single-input operators that leads to a graph input.
    """
    if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):
        n: torch.fx.Node = node.all_input_nodes[0]
        while len(n.all_input_nodes) == 1:
            if is_graph_input(n.all_input_nodes[0]):
                return True
            n = n.all_input_nodes[0]
    return False


def annotate_fsdp_all_gather(
    gm: torch.fx.GraphModule, reshard_after_forward: bool
) -> None:
    """
    Force recompute all_gather nodes from simple fsdp in the graph.
    This pass should be added in torch._inductor.config.joint_custom_post_pass
    """
    graph = gm.graph

    def force_recompute_node(node):
        if reshard_after_forward:
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        else:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        # ac_graph_id is used in the partitioner to decide
        # if two nodes which have AC applied come from a different
        # AC regions. This is needed because nodes in the boundary
        # of two AC regions are marked as MUST_SAVE. In our case
        # we just add a large value of ac_graph_id so that
        # all nodes we tag for recomputation do indeed get recomputed
        # and are not influenced by other nodes in the graph with
        # nearby ac_graph_id values
        node.meta["ac_graph_id"] = 100000

    # Make all-gather nodes (and related nodes) recomputable, to circumvent
    # https://github.com/pytorch/pytorch/issues/136433
    for node in graph.nodes:
        if is_wait_tensor_from_fsdp(node):
            ag_node = node.args[0]
            force_recompute_node(ag_node)  # all_gather
            force_recompute_node(node)  # wait_tensor
            # Force-recompute slice that comes after wait
            for user in node.users:
                if (
                    user.op == "call_function"
                    and user.target == torch.ops.aten.slice.Tensor
                ):
                    force_recompute_node(user)
            # Force-recompute potential dtype casts from all_gather
            if (
                ag_node.all_input_nodes[0].op == "call_function"
                and ag_node.args[0].target
                == torch.ops.prims.convert_element_type.default
            ):
                force_recompute_node(ag_node.all_input_nodes[0])

    return gm
