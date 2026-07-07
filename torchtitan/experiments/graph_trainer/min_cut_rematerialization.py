# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Min-cut rematerialization for GraphTrainer FX graphs and subgraphs."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.fx as fx
import torch.fx.traceback as fx_traceback
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
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)


_INF_DISTANCE = int(1e9)


def _is_backward_side(node: fx.Node, backward_side: set[fx.Node]) -> bool:
    return _is_backward_node(node) or any(
        inp in backward_side for inp in node.all_input_nodes
    )


def _decomposition_inputs_from_meta(gm: fx.GraphModule):
    inputs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if "val" not in node.meta:
            return None
        inputs.append(node.meta["val"])
    return tuple(inputs)


def _fetch_attr(root, target):
    for atom in target.split("."):
        root = getattr(root, atom)
    return root


def _assign_attr(root, target, value) -> None:
    atoms = target.split(".")
    for atom in atoms[:-1]:
        if not hasattr(root, atom):
            setattr(root, atom, torch.nn.Module())
        root = getattr(root, atom)
    setattr(root, atoms[-1], value)


def _copy_getattrs(dst: fx.GraphModule, src: fx.GraphModule) -> None:
    for node in src.graph.nodes:
        if node.op == "get_attr":
            _assign_attr(dst, node.target, _fetch_attr(src, node.target))


def _apply_decomposition_table_for_min_cut(
    gm: fx.GraphModule,
    example_inputs,
    decomposition_table,
) -> None:
    if decomposition_table is None:
        return
    if callable(decomposition_table):
        decomposition_table = decomposition_table()
    if not decomposition_table:
        return

    from torch.fx.experimental.proxy_tensor import decompose, make_fx

    inputs = (
        tuple(example_inputs)
        if example_inputs is not None
        else _decomposition_inputs_from_meta(gm)
    )
    if inputs is None:
        return

    from torch._subclasses.fake_tensor import fake_tensor_tls

    class DecomposeInterpreter(fx.Interpreter):
        def run_node(self, node):
            with decompose(decomposition_table):
                return super().run_node(node)

    def run(*args):
        return DecomposeInterpreter(gm).run(*args)

    old_allow_non_fake = fake_tensor_tls.allow_non_fake_inputs_override
    fake_tensor_tls.allow_non_fake_inputs_override = True
    try:
        with fx_traceback.preserve_node_meta():
            decomposed = make_fx(
                run,
                decomposition_table={},
                _allow_non_fake_inputs=True,
            )(*inputs)
    finally:
        fake_tensor_tls.allow_non_fake_inputs_override = old_allow_non_fake

    _copy_getattrs(gm, decomposed)
    gm.graph = decomposed.graph
    gm.graph.lint()
    gm.recompile()


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
        node
        for node in nodes
        if node not in required_bw_nodes and node.op != "output"
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
                getattr(user, "dist_from_bw", _INF_DISTANCE) + 1
                for user in node.users
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
        node for node in gm.graph.nodes if node not in backward_side and node.op != "output"
    }
    saved_boundaries = set(saved_values)
    op_types = get_default_op_list()
    for node in list(saved_boundaries):
        if node not in required_fw_nodes or node.op != "call_function":
            continue
        if (
            node.target == torch.ops.aten.detach.default
            or op_types.is_view(node)
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
    example_inputs,
    decomposition_table,
    materialize: bool,
    ban_if_materialized_backward: bool,
) -> int:
    _apply_decomposition_table_for_min_cut(gm, example_inputs, decomposition_table)
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

    if materialize:
        selective_activation_remat_pass(gm)
    gm.graph.lint()
    gm.recompile()
    return len(recompute_nodes)


def min_cut_rematerialization_pass(
    gm: fx.GraphModule,
    example_inputs=None,
    *,
    recurse: bool = False,
    apply_to_root: bool = True,
    materialize: bool = True,
    decomposition_table=None,
    ban_if_materialized_backward: bool = False,
) -> fx.GraphModule:
    """Apply compile-style min-cut recompute to a graph and/or nested subgraphs.

    GraphTrainer uses the raw min-cut solution, then marks the saved cut as
    ``MUST_SAVE`` and every unsaved forward-to-backward crossing as
    ``MUST_RECOMPUTE``. ``decomposition_table`` applies AOT/proxy decompositions
    before min-cut.
    """
    modules: list[tuple[str, fx.GraphModule]] = []
    if apply_to_root:
        modules.append(("", gm))
    if recurse:
        modules.extend(
            (name, module)
            for name, module in gm.named_modules()
            if name and isinstance(module, fx.GraphModule)
        )

    for _, module in modules:
        _apply_min_cut_to_module(
            module,
            example_inputs=example_inputs if module is gm else None,
            decomposition_table=decomposition_table,
            materialize=materialize,
            ban_if_materialized_backward=ban_if_materialized_backward,
        )

    return gm
