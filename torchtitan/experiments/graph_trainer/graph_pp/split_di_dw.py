# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator
from dataclasses import dataclass

import torch.fx as fx
from torch._functorch.partitioners import (
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
    is_sym_node,
)
from torch.fx._lazy_graph_module import _make_graph_module

from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    is_fake_tensor_node,
    output_names,
    placeholder_names,
    rename_placeholder,
    trace_graph_pp_graph,
    unique_in_order,
)


@dataclass(frozen=True, slots=True)
class GraphPPDiDwSplit:
    """Backward graph split into dI and dW GraphPP callables.

    Attributes:
        bw_di_module (fx.GraphModule): Backward-input graph. It computes input
            gradients to send to the previous PP stage and live-ins needed by
            the backward-weight graph.
        bw_dw_module (fx.GraphModule): Backward-weight graph. It consumes the
            dW live-ins and computes parameter gradients.
        num_input_grads (int): Number of leading ``bw_di_module`` outputs that
            are input_grads_to_prev.
        bw_di_input_names (tuple[str, ...]): ``bw_di_module`` placeholder
            names.
        bw_di_output_names (tuple[str, ...]): ``bw_di_module`` output names.
        bw_dw_input_names (tuple[str, ...]): ``bw_dw_module`` placeholder
            names.
        bw_dw_output_names (tuple[str, ...]): ``bw_dw_module`` output names.
    """

    bw_di_module: fx.GraphModule
    bw_dw_module: fx.GraphModule
    num_input_grads: int
    bw_di_input_names: tuple[str, ...]
    bw_di_output_names: tuple[str, ...]
    bw_dw_input_names: tuple[str, ...]
    bw_dw_output_names: tuple[str, ...]


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

    # The upstream extractor treats the first outputs as the "forward" side.
    # For dI/dW splitting, those early outputs are the input grads that PP
    # schedules send to the previous stage before parameter grads are computed.
    param_grads = output_values[:num_param_grads]
    input_grads = output_values[num_param_grads:]
    with gm.graph.inserting_after(output):
        new_output = gm.graph.output(tuple(input_grads + param_grads))
    new_output.meta.update(output.meta)
    gm.graph.erase_node(output)
    gm.graph.lint()
    gm.recompile()
    return len(input_grads)


def _collect_saved_values_for_dw(
    bw_gm: fx.GraphModule,
    di_graph: fx.Graph,
    dw_output_nodes: list[fx.Node] | None = None,
) -> tuple[list[fx.Node], list[fx.Node]]:
    di_node_names = {node.name for node in di_graph.nodes if node.op != "output"}
    dw_output_set = set(dw_output_nodes or ())
    saved_values: list[fx.Node] = []
    saved_sym_nodes: list[fx.Node] = []

    for node in bw_gm.graph.nodes:
        if node.name not in di_node_names:
            continue
        dw_users = [
            user
            for user in node.users
            if user.name not in di_node_names and user.op != "output"
        ]
        is_dw_output = node in dw_output_set
        if not dw_users and not is_dw_output:
            continue
        if node.op == "get_attr":
            # get_attr nodes are graph constants, such as FlexAttention's
            # mask/score submodules. The dW graph should retain them as
            # get_attr references instead of receiving Python objects as
            # runtime live-ins from the dI graph.
            continue
        if is_sym_node(node):
            # Symbolic shape values used by dW must stay as SymInt live-ins,
            # not tensor placeholders.
            saved_sym_nodes.append(node)
        elif (
            "tensor_meta" not in node.meta
            and node.op == "call_function"
            and not is_fake_tensor_node(node)
        ):
            # Non-tensor tuple-like results flow through getitem leaves. Save
            # the leaves because the dW graph consumes those concrete values.
            users = list(node.users)
            if not all(user.target == operator.getitem for user in users):
                raise ValueError(
                    f"Non-tensor multi-output node {node.name} has unexpected users"
                )
            saved_values.extend(user for user in users if user in dw_users)
        else:
            if (
                not is_dw_output
                and "tensor_meta" in node.meta
                and all(is_sym_node(user) for user in dw_users)
            ):
                # If dW only needs shape reads from this tensor, pass the
                # symbolic reads instead of keeping the tensor live.
                saved_sym_nodes.extend(dw_users)
            else:
                saved_values.append(node)

    return (
        unique_in_order(saved_values),
        unique_in_order(saved_sym_nodes),
    )


def split_di_dw_graph(
    bw_module: fx.GraphModule,
    *,
    num_param_grads: int,
) -> GraphPPDiDwSplit | None:
    """Split a backward graph into input-gradient and weight-gradient graphs.

    Contract:
      Input:
        backward(saved_values_for_backward, output_grads_from_next?)
          -> param_grads, input_grads_to_prev

      Output:
        bw_di(saved_values_for_backward, output_grads_from_next?)
          -> input_grads_to_prev, dw_live_ins

        bw_dw(dw_live_ins)
          -> param_grads

    Terms:
      ``num_param_grads`` is the leading output count for parameter gradients.
      ``input_grads_to_prev`` are remaining backward outputs sent to the
      previous PP stage. ``dw_live_ins`` are values computed by bw_di that the
      dW graph still needs. ``saved_sym_nodes`` carries symbolic shape live-ins.

    If a stage has no input grads, GraphPP skips ``BACKWARD_INPUT`` and keeps
    the original full backward graph for ``BACKWARD_WEIGHT``. This pass runs
    after AC/remat has already materialized recomputed backward nodes, so those
    nodes are treated like ordinary backward graph nodes.

    Args:
        bw_module (fx.GraphModule): Backward graph whose outputs are ordered as
            parameter gradients followed by input gradients.
        num_param_grads (int): Number of leading backward outputs that are
            parameter-gradient slots.

    Returns:
        GraphPPDiDwSplit | None: Split dI/dW graph modules and metadata, or
        ``None`` when the stage has no input gradients and the full backward
        graph should be used unchanged.

    Raises:
        ValueError: If ``num_param_grads`` is invalid or the backward graph
            does not match the expected flat output convention.
    """
    if num_param_grads < 0:
        raise ValueError(f"num_param_grads must be non-negative, got {num_param_grads}")

    bw_gm = copy.deepcopy(bw_module)
    for placeholder in list(bw_gm.graph.find_nodes(op="placeholder")):
        if placeholder.name.startswith("tangent"):
            rename_placeholder(
                bw_gm,
                placeholder,
                f"graph_pp_runtime_grad{placeholder.name[len('tangent') :]}",
            )

    num_input_grads = _reorder_backward_outputs_for_di(
        bw_gm, num_param_grads=num_param_grads
    )
    trace_graph_pp_graph("graph_pp_split_di_dw_input", bw_gm)
    if num_input_grads == 0:
        return None

    placeholders = list(bw_gm.graph.find_nodes(op="placeholder"))
    di_outputs, dw_outputs, di_output_descs, dw_output_descs = _extract_fwd_bwd_outputs(
        bw_gm, num_fwd_outputs=num_input_grads
    )
    di_closure_graph = _extract_graph_with_inputs_outputs(
        bw_gm.graph,
        placeholders,
        di_outputs,
        di_output_descs,
        "forward",
        ignore_must_be_in_fw_bw=True,
    )
    saved_values, saved_sym_nodes = _collect_saved_values_for_dw(
        bw_gm,
        di_closure_graph,
        dw_outputs,
    )
    di_runtime_outputs = [*di_outputs, *saved_values, *saved_sym_nodes]
    di_runtime_output_descs = list(di_output_descs) + [None] * (
        len(saved_values) + len(saved_sym_nodes)
    )
    bw_di_graph = _extract_graph_with_inputs_outputs(
        bw_gm.graph,
        placeholders,
        di_runtime_outputs,
        di_runtime_output_descs,
        "bw_di",
        ignore_must_be_in_fw_bw=True,
    )
    bw_dw_graph = _extract_graph_with_inputs_outputs(
        bw_gm.graph,
        [*saved_values, *saved_sym_nodes],
        dw_outputs,
        dw_output_descs,
        "bw_dw",
        ignore_must_be_in_fw_bw=True,
    )
    bw_di_module = _make_graph_module(bw_gm, bw_di_graph)
    bw_dw_module = _make_graph_module(bw_gm, bw_dw_graph)
    bw_di_module.graph.lint()
    bw_dw_module.graph.lint()
    bw_di_module.recompile()
    bw_dw_module.recompile()
    trace_graph_pp_graph("graph_pp_bw_di", bw_di_module)
    trace_graph_pp_graph("graph_pp_bw_dw", bw_dw_module)

    return GraphPPDiDwSplit(
        bw_di_module=bw_di_module,
        bw_dw_module=bw_dw_module,
        num_input_grads=num_input_grads,
        bw_di_input_names=placeholder_names(bw_di_module),
        bw_di_output_names=output_names(bw_di_module),
        bw_dw_input_names=placeholder_names(bw_dw_module),
        bw_dw_output_names=output_names(bw_dw_module),
    )
