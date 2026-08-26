# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Min-cut activation-checkpoint policy for GraphTrainer FX graphs."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.fx as fx
from torch._functorch import config as functorch_config
from torch._functorch.partitioners import (
    force_save_bw_mutation_src,
    force_save_collectives,
    force_save_effectful_ops,
    get_default_op_list,
    MinCutOptions,
    NodeInfo,
    solve_min_cut,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node


_INF_DISTANCE = int(1e9)


def _is_backward_side(node: fx.Node, backward_side: set[fx.Node]) -> bool:
    return _is_backward_node(node) or any(
        inp in backward_side for inp in node.all_input_nodes
    )


def _backward_side_nodes(gm: fx.GraphModule) -> OrderedSet[fx.Node]:
    backward_side = OrderedSet()
    for node in gm.graph.nodes:
        if node.op != "output" and _is_backward_side(node, backward_side):
            backward_side.add(node)
    return backward_side


def _node_info_for_graph_trainer(
    gm: fx.GraphModule,
    backward_side: OrderedSet[fx.Node],
) -> NodeInfo | None:
    nodes = list(gm.graph.nodes)
    required_bw_nodes = OrderedSet(
        node for node in nodes if node in backward_side and node.op != "output"
    )
    if not required_bw_nodes:
        return None

    required_fw_nodes = OrderedSet(
        node for node in nodes if node not in required_bw_nodes and node.op != "output"
    )
    fw_order = {node: idx for idx, node in enumerate(required_fw_nodes)}
    static_lifetime_input_nodes = OrderedSet(
        node for node in required_fw_nodes if node.op in ("placeholder", "get_attr")
    )

    for node in reversed(nodes):
        if node.op == "output":
            node.dist_from_bw = _INF_DISTANCE
        elif node in required_bw_nodes:
            node.dist_from_bw = 0
        elif node in required_fw_nodes:
            user_distances = [
                getattr(user, "dist_from_bw", _INF_DISTANCE) + 1 for user in node.users
            ]
            node.dist_from_bw = min(user_distances, default=_INF_DISTANCE)
        else:
            node.dist_from_bw = _INF_DISTANCE

    return NodeInfo(
        list(static_lifetime_input_nodes),
        required_fw_nodes,
        required_bw_nodes.copy(),
        required_bw_nodes.copy(),
        OrderedSet(),
        fw_order,
        static_lifetime_input_nodes,
    )


def _min_cut_choose_saved_values_set(
    gm: fx.GraphModule,
    node_info: NodeInfo,
    *,
    ban_if_materialized_backward: bool,
) -> list[fx.Node]:
    min_cut_options = MinCutOptions(
        ban_if_used_far_apart=functorch_config.ban_recompute_used_far_apart,
        ban_if_long_fusible_chains=functorch_config.ban_recompute_long_fusible_chains,
        ban_if_materialized_backward=ban_if_materialized_backward,
        ban_if_not_in_allowlist=functorch_config.ban_recompute_not_in_allowlist,
        ban_if_reduction=functorch_config.ban_recompute_reductions,
    )
    if functorch_config.aggressive_recomputation:
        min_cut_options = replace(
            min_cut_options,
            ban_if_used_far_apart=False,
            ban_if_long_fusible_chains=False,
            ban_if_materialized_backward=False,
            ban_if_not_in_allowlist=False,
        )
    saved_values, _ = solve_min_cut(gm.graph, node_info, min_cut_options)
    return saved_values


def _apply_min_cut_policies(
    gm: fx.GraphModule,
    backward_side: OrderedSet[fx.Node],
    saved_values: set[fx.Node],
) -> tuple[list[fx.Node], list[fx.Node]]:
    required_fw_nodes = {
        node
        for node in gm.graph.nodes
        if node not in backward_side and node.op != "output"
    }
    saved_boundaries = set(saved_values)
    op_types = get_default_op_list()
    for node in list(saved_boundaries):
        if node not in required_fw_nodes or node.op != "call_function":
            continue
        if (
            node.target == torch.ops.aten.detach.default or op_types.is_view(node)
        ) and any(inp in required_fw_nodes for inp in node.all_input_nodes):
            saved_boundaries.remove(node)

    saved_boundaries.update(
        node
        for node in required_fw_nodes
        if node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
    )
    for node in saved_boundaries:
        if node in required_fw_nodes:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE

    marked = OrderedSet()
    seen = set()

    def visit(node: fx.Node) -> None:
        if node in seen or node in saved_boundaries:
            return
        seen.add(node)
        if node in backward_side:
            for inp in node.all_input_nodes:
                visit(inp)
            return
        if node not in required_fw_nodes or node.op in ("placeholder", "get_attr"):
            return
        if node.op == "call_function":
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
            marked.add(node)
            for inp in node.all_input_nodes:
                visit(inp)

    for node in backward_side:
        for inp in node.all_input_nodes:
            visit(inp)

    return list(marked), list(saved_boundaries)


def _apply_min_cut_to_module(
    gm: fx.GraphModule,
    *,
    ban_if_materialized_backward: bool,
) -> int:
    backward_side = _backward_side_nodes(gm)
    node_info = _node_info_for_graph_trainer(gm, backward_side)
    if node_info is None:
        return 0

    force_save_collectives(gm)
    force_save_effectful_ops(gm)
    force_save_bw_mutation_src(gm)
    saved_values = _min_cut_choose_saved_values_set(
        gm,
        node_info,
        ban_if_materialized_backward=ban_if_materialized_backward,
    )
    recompute_nodes, saved_values = _apply_min_cut_policies(
        gm, backward_side, set(saved_values)
    )
    if not recompute_nodes:
        return 0

    gm.graph.lint()
    gm.recompile()
    return len(recompute_nodes)


def apply_min_cut_policy(
    gm: fx.GraphModule,
    *,
    ban_if_materialized_backward: bool = False,
) -> fx.GraphModule:
    """Tag a graph with a min-cut AC policy.

    GraphTrainer uses the raw min-cut solution, then marks the saved cut as
    ``MUST_SAVE`` and every unsaved forward-to-backward crossing as
    ``MUST_RECOMPUTE``. Decomposition and rematerialization are separate passes.
    """
    _apply_min_cut_to_module(
        gm,
        ban_if_materialized_backward=ban_if_materialized_backward,
    )
    return gm
