# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator

import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    _extract_fwd_bwd_modules,
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
    is_sym_node,
)
from torch.utils._ordered_set import OrderedSet

from .graph_pp_utils import rename_placeholder_node

# we are running the default partitioner on the bw graph, which requires AC tags being removed.
# At this stage we have already finished running AC anyway, since we have a bw graph


def remove_recompute_tags(bw_gm):
    for n in bw_gm.graph.nodes:
        if "recompute" in n.meta:
            del n.meta["recompute"]


# We are using the default partitioner to split our backward into dI and dW subgraphs.
# We want to generate the dI subgraph *first*, because:
# - in pipelining we generally want to schedule dI compute before dW
# - the dI compute will potentially compute more activations that we need to plumb into dW compute
# Today, the default partitioner requires that your split on the first K outputs of your combined graph.
# So here, we reorder the outputs of the backward so grad_inputs are first.


def reorder_output_grads(bw_gm, num_weight_gradients):
    outputs = bw_gm.graph.find_nodes(op="output")
    assert len(outputs) == 1
    output = outputs[0]
    assert isinstance(output.args[0], tuple)
    grad_weights, grad_inputs = (
        output.args[0][:num_weight_gradients],
        output.args[0][num_weight_gradients:],
    )
    new_out_tuple = grad_inputs + grad_weights
    with bw_gm.graph.inserting_after(output):
        # TODO: also set the new node's meta properly
        new_out = bw_gm.graph.output(new_out_tuple)
    output.replace_all_uses_with(new_out)
    bw_gm.graph.erase_node(output)
    return len(grad_inputs)


# TODO: in theory we can infer num_weight_gradients from the graph metadata directly
def split_di_dw_graph(
    bw_gm_old: fx.GraphModule, *, num_weight_gradients: int
) -> tuple[fx.GraphModule, fx.GraphModule, int]:
    # we could consider doing this is a non-mutating way
    bw_gm = copy.deepcopy(bw_gm_old)
    placeholders = bw_gm.graph.find_nodes(op="placeholder")
    for p in placeholders:
        if p.name.startswith("tangent"):
            name_suffix = p.name[8:]
            rename_placeholder_node(bw_gm, p, f"not_tngnt{name_suffix}")

    remove_recompute_tags(bw_gm)
    num_input_gradients = reorder_output_grads(bw_gm, num_weight_gradients)
    bw_gm.recompile()

    args = list(bw_gm.graph.find_nodes(op="placeholder"))

    #    bw_inputs, bw_weights = default_partition(bw_gm, args, num_fwd_outputs=num_input_gradients)
    #    return bw_inputs, bw_weights, num_input_gradients

    (
        grad_inps,
        grad_weights,
        grad_inp_descs,
        grad_weight_descs,
    ) = _extract_fwd_bwd_outputs(bw_gm, num_fwd_outputs=num_input_gradients)
    bw_inputs_gm = _extract_graph_with_inputs_outputs(
        bw_gm.graph,
        args,
        grad_inps,
        grad_inp_descs,
        "forward",
        ignore_must_be_in_fw_bw=True,
    )
    bw_inputs_gm_node_names = OrderedSet(
        node.name for node in bw_inputs_gm.nodes if node.op != "output"
    )
    saved_values = []
    saved_sym_nodes = []

    # TODO: this classification loop is a simplified version of default_partition's
    # node classification. It does not handle: get_attr nodes, _assert_scalar/profiler
    # ops, MUST_SAVE tags, impure/effectful ops, force_save_collectives,
    # force_save_bw_mutation_src, must_recompute skipping, or post-split DCE.
    # Ideally we would call default_partition directly instead of reimplementing.
    for node in bw_gm.graph.nodes:
        if node.name not in bw_inputs_gm_node_names:
            # Not handling mutations for now,
            # we can try to re-use more of and/or consolidate with default partitioner
            continue
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        elif (
            "tensor_meta" not in node.meta
            and node.op == "call_function"
            and not isinstance(node.meta.get("val"), torch._subclasses.FakeTensor)
        ):
            users = node.users
            assert all(user.target == operator.getitem for user in users)
            saved_values.extend(users)
        else:
            backward_usages = [
                n for n in node.users if n.name not in bw_inputs_gm_node_names
            ]
            if "tensor_meta" in node.meta and all(
                is_sym_node(n) for n in backward_usages
            ):
                saved_sym_nodes.extend(backward_usages)
            else:
                saved_values.append(node)
    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())
    bw_inputs, bw_weights = _extract_fwd_bwd_modules(
        bw_gm,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_input_gradients,
        ignore_must_be_in_fw_bw=True,
        omit_aot_autograd_runtime=True,
    )
    return bw_inputs, bw_weights, num_input_gradients
