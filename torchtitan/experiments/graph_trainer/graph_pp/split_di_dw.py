# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    _extract_fwd_bwd_modules,
    _extract_fwd_bwd_outputs,
    is_sym_node,
)
from torch.utils._ordered_set import OrderedSet

from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    _output_names,
    _placeholder_names,
    extract_graph_with_graph_pp_abi,
)


@dataclass(frozen=True, slots=True)
class GraphPPDiDwSplit:
    """Backward graph split into dI and dW GraphPP callables."""

    bw_di_module: fx.GraphModule
    bw_dw_module: fx.GraphModule
    num_input_grads: int
    bw_di_input_names: tuple[str, ...]
    bw_di_output_names: tuple[str, ...]
    bw_dw_input_names: tuple[str, ...]
    bw_dw_output_names: tuple[str, ...]


def _rename_placeholder_node(gm: fx.GraphModule, node: fx.Node, new_name: str) -> None:
    if node.op != "placeholder":
        raise ValueError(f"Can only rename placeholder nodes, got {node.op}")
    with gm.graph.inserting_before(node):
        new_node = gm.graph.placeholder(new_name)
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


def _remove_recompute_tags(gm: fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        node.meta.pop("recompute", None)


def _extract_di_dw_modules(
    bw_gm: fx.GraphModule,
    saved_values: list[fx.Node],
    saved_sym_nodes: list[fx.Node],
    *,
    num_input_grads: int,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """Extract dI/dW graphs after GraphPP has selected dW live-ins."""
    return _extract_fwd_bwd_modules(
        bw_gm,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_input_grads,
        ignore_must_be_in_fw_bw=True,
        omit_aot_autograd_runtime=True,
    )


def _reorder_backward_outputs_for_di(
    gm: fx.GraphModule, *, num_param_grads: int
) -> int:
    outputs = gm.graph.find_nodes(op="output")
    if len(outputs) != 1:
        raise ValueError(f"Expected exactly one output node, found {len(outputs)}")
    output = outputs[0]
    if not isinstance(output.args[0], tuple):
        raise ValueError("Backward graph output must be a tuple")

    output_values = output.args[0]
    if len(output_values) < num_param_grads:
        raise ValueError(
            f"Backward graph has {len(output_values)} outputs but "
            f"{num_param_grads} parameter grads were requested"
        )

    param_grads = output_values[:num_param_grads]
    input_grads = output_values[num_param_grads:]
    with gm.graph.inserting_after(output):
        new_output = gm.graph.output(tuple(input_grads + param_grads))
    new_output.meta.update(output.meta)
    gm.graph.erase_node(output)
    gm.graph.lint()
    gm.recompile()
    return len(input_grads)


def _is_fake_tensor_value(node: fx.Node) -> bool:
    return isinstance(node.meta.get("val"), torch._subclasses.FakeTensor)


def _collect_saved_values_for_dw(
    bw_gm: fx.GraphModule,
    di_graph: fx.Graph,
) -> tuple[list[fx.Node], list[fx.Node]]:
    di_node_names = OrderedSet(
        node.name for node in di_graph.nodes if node.op != "output"
    )
    saved_values: list[fx.Node] = []
    saved_sym_nodes: list[fx.Node] = []

    for node in bw_gm.graph.nodes:
        if node.name not in di_node_names:
            continue
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        elif (
            "tensor_meta" not in node.meta
            and node.op == "call_function"
            and not _is_fake_tensor_value(node)
        ):
            users = list(node.users)
            if all(user.target == operator.getitem for user in users):
                # Multi-output non-tensor node (e.g. a tuple-returning op): save
                # its getitem results across the di/dW boundary.
                saved_values.extend(users)
            # Otherwise this is a single-output non-tensor node, e.g. a
            # compile-on-one-rank device_mesh/coor op like
            # mesh_get_process_group whose users consume it directly. Don't save
            # it; the partitioner recomputes it in the dW graph, which is cheap,
            # deterministic, and correct.
        else:
            dw_users = [user for user in node.users if user.name not in di_node_names]
            if "tensor_meta" in node.meta and all(
                is_sym_node(user) for user in dw_users
            ):
                saved_sym_nodes.extend(dw_users)
            else:
                saved_values.append(node)

    return (
        list(dict.fromkeys(saved_values).keys()),
        list(dict.fromkeys(saved_sym_nodes).keys()),
    )


def split_di_dw_graph(
    bw_module: fx.GraphModule,
    *,
    num_param_grads: int,
) -> GraphPPDiDwSplit | None:
    """Split a backward graph into input-gradient and weight-gradient graphs.

    Contract:
      The input backward graph returns parameter grads first, then input grads.
      PP schedules with split backward actions need ``BACKWARD_INPUT`` to emit
      input grads early and ``BACKWARD_WEIGHT`` to finish parameter grads later.
      If the stage has no input grads, GraphPP keeps the full backward graph and
      skips the split.

    Pseudocode:
      rename tangent placeholders to avoid AOTAutograd reserved names
      reorder outputs so input grads are the "forward" outputs for extraction
      extract bw_di(original backward inputs) -> input grads + dW live-ins
      collect saved values needed only by dW computation
      extract bw_dw(dW live-ins) -> parameter grads
      return both graphs and their flat ABI names

    The backward graph is expected to return parameter gradients first,
    followed by input gradients.  If there are no input gradients, the caller
    should skip `BACKWARD_INPUT` and run the original full backward graph at the
    `BACKWARD_WEIGHT` action.
    """
    if num_param_grads < 0:
        raise ValueError(f"num_param_grads must be non-negative, got {num_param_grads}")

    bw_gm = copy.deepcopy(bw_module)
    for placeholder in list(bw_gm.graph.find_nodes(op="placeholder")):
        if placeholder.name.startswith("tangent"):
            _rename_placeholder_node(
                bw_gm,
                placeholder,
                f"graph_pp_runtime_grad{placeholder.name[len('tangent') :]}",
            )

    _remove_recompute_tags(bw_gm)
    num_input_grads = _reorder_backward_outputs_for_di(
        bw_gm, num_param_grads=num_param_grads
    )
    if num_input_grads == 0:
        return None

    placeholders = list(bw_gm.graph.find_nodes(op="placeholder"))
    di_outputs, _, di_output_descs, _ = _extract_fwd_bwd_outputs(
        bw_gm, num_fwd_outputs=num_input_grads
    )
    di_graph = extract_graph_with_graph_pp_abi(
        bw_gm.graph,
        placeholders,
        di_outputs,
        di_output_descs,
        "forward",
    )
    saved_values, saved_sym_nodes = _collect_saved_values_for_dw(bw_gm, di_graph)
    bw_di_module, bw_dw_module = _extract_di_dw_modules(
        bw_gm,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_input_grads=num_input_grads,
    )
    bw_di_module.graph.lint()
    bw_dw_module.graph.lint()
    bw_di_module.recompile()
    bw_dw_module.recompile()

    return GraphPPDiDwSplit(
        bw_di_module=bw_di_module,
        bw_dw_module=bw_dw_module,
        num_input_grads=num_input_grads,
        bw_di_input_names=_placeholder_names(bw_di_module),
        bw_di_output_names=_output_names(bw_di_module),
        bw_dw_input_names=_placeholder_names(bw_dw_module),
        bw_dw_output_names=_output_names(bw_dw_module),
    )
