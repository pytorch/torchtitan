# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from itertools import dropwhile

import torch.fx as fx
from torch.fx.graph_module import _assign_attr, _get_attr

from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    _find_fake_mode,
    _MetaShapeEnvTransfer,
    trace_graph_pp_graph,
)


def _copy_prefixed_get_attrs(
    dst: fx.GraphModule,
    src: fx.GraphModule,
    *,
    prefix: str,
) -> dict[str, str]:
    remap: dict[str, str] = {}
    for node in src.graph.find_nodes(op="get_attr"):
        attr_name = str(node.target)
        if attr_name in remap:
            continue
        base_name = f"{prefix}{attr_name.replace('.', '_')}"
        new_attr_name = base_name
        suffix = 0
        # GraphModule attributes share one namespace; probe it directly to
        # avoid colliding with existing get_attr targets.
        while hasattr(dst, new_attr_name):
            suffix += 1
            new_attr_name = f"{base_name}_{suffix}"
        _assign_attr(copy.deepcopy(_get_attr(src, attr_name)), dst, new_attr_name)
        remap[attr_name] = new_attr_name
    return remap


def multiplex_fw_bw_graph(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
    *,
    overlap: bool = False,
) -> fx.GraphModule:
    """Concatenate backward and forward graphs into one boxed GraphPP callable.

    Contract:
      ``OVERLAP_F_B`` schedule actions need one callable that performs a
      backward action for one stage and a forward action for another stage. The
      returned graph preserves both flat calling conventions by ordering
      placeholders as ``bw_inputs + fw_inputs`` and outputs as
      ``bw_outputs + fw_outputs``.

    Pseudocode:
      deep-copy the forward graph as the destination
      transfer backward metadata into the forward FakeTensorMode/ShapeEnv
      insert copied backward placeholders before the forward placeholders
      copy backward get_attr targets with a prefix to avoid attr collisions
      copy backward compute nodes before the forward compute nodes
      replace the output tuple with backward outputs followed by forward outputs

    ``overlap`` is reserved for future DualPipeV scheduling work; the current
    placeholder does not change graph construction.

    The forward graph remains the destination module because its ShapeEnv owns
    the dynamic collective-size constraints needed by full Inductor for MoE
    all-to-all outputs.  The backward graph is inserted in topological order
    before the existing forward compute, with disjoint placeholders and
    prefixed attributes, so this helper does not need dependency analysis or a
    scheduling policy.  EP-overlap annotations are intentionally not applied
    inside this graph; pass ordering keeps EP-overlap on the standalone no-FSDP
    forward/backward graphs.

    Args:
        fw_gm (fx.GraphModule): Forward graph whose ShapeEnv and module
            namespace become the destination for the multiplexed graph.
        bw_gm (fx.GraphModule): Backward graph copied before the forward graph
            inside the multiplexed callable.
        overlap (bool): Reserved placeholder for future DualPipeV scheduling
            work. Defaults to ``False`` and currently has no behavior.

    Returns:
        fx.GraphModule: One graph module with placeholders ordered as
        ``bw_inputs + fw_inputs`` and outputs ordered as
        ``bw_outputs + fw_outputs``.

    Raises:
        ValueError: If the forward graph does not contain the placeholders or
            output node needed to build the multiplexed callable.
    """
    old_to_new: dict[fx.Node, fx.Node] = {}
    # Preserve the forward ShapeEnv exactly as traced.  Reconstructing it from
    # copied metadata can lose collective-size hints that full Inductor needs.
    multiplexed_gm = copy.deepcopy(fw_gm)
    dst_fake_mode = _find_fake_mode(multiplexed_gm)
    meta_transfer = _MetaShapeEnvTransfer(dst_fake_mode)
    for node in bw_gm.graph.nodes:
        meta_transfer.collect(node.meta)
    meta_transfer.seed()
    bw_get_attr_remap = _copy_prefixed_get_attrs(
        multiplexed_gm,
        bw_gm,
        prefix="bw",
    )

    fw_placeholders = multiplexed_gm.graph.find_nodes(op="placeholder")
    if not fw_placeholders:
        raise ValueError("GraphPP forward graph has no placeholders to multiplex")
    first_fw_placeholder = fw_placeholders[0]
    insert_point: fx.Node | None = None
    for node in bw_gm.graph.find_nodes(op="placeholder"):
        if insert_point is None:
            with multiplexed_gm.graph.inserting_before(first_fw_placeholder):
                new_placeholder = multiplexed_gm.graph.placeholder(f"bw_{node.name}")
        else:
            with multiplexed_gm.graph.inserting_after(insert_point):
                new_placeholder = multiplexed_gm.graph.placeholder(f"bw_{node.name}")
        new_placeholder.meta = meta_transfer.copy_meta(node.meta)
        old_to_new[node] = new_placeholder
        insert_point = new_placeholder

    first_fw_compute = next(
        (node for node in multiplexed_gm.graph.nodes if node.op != "placeholder"),
        None,
    )
    if first_fw_compute is None:
        raise ValueError("GraphPP forward graph has no output node to multiplex")

    bw_nodes = iter(bw_gm.graph.nodes)
    bw_nodes = dropwhile(lambda node: node.op == "placeholder", bw_nodes)
    insert_point = None
    for node in bw_nodes:
        if node.op == "output":
            break
        if insert_point is None:
            with multiplexed_gm.graph.inserting_before(first_fw_compute):
                new_node = multiplexed_gm.graph.node_copy(
                    node, lambda arg: old_to_new[arg]
                )
        else:
            with multiplexed_gm.graph.inserting_after(insert_point):
                new_node = multiplexed_gm.graph.node_copy(
                    node, lambda arg: old_to_new[arg]
                )
        new_node.meta = meta_transfer.copy_meta(node.meta)
        if new_node.op == "get_attr":
            attr_name = str(new_node.target)
            if attr_name not in bw_get_attr_remap:
                raise ValueError(
                    "GraphPP multiplexed graph missing copied get_attr target "
                    f"for backward attribute {attr_name!r}."
                )
            new_node.target = bw_get_attr_remap[attr_name]
        old_to_new[node] = new_node
        insert_point = new_node

    multiplexed_output = multiplexed_gm.graph.find_nodes(op="output")[0]
    bw_output_node = bw_gm.graph.find_nodes(op="output")[0]
    bw_outputs = [
        old_to_new[value] if isinstance(value, fx.Node) else value
        for value in bw_output_node.args[0]
    ]
    fw_outputs = list(multiplexed_output.args[0])
    multiplexed_output.args = (tuple(bw_outputs + fw_outputs),)

    multiplexed_gm.graph.eliminate_dead_code()
    multiplexed_gm.graph.lint()
    multiplexed_gm.recompile()
    trace_graph_pp_graph("graph_pp_multiplexed_graph", multiplexed_gm)
    return multiplexed_gm
