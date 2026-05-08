# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
from itertools import dropwhile

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch._logging import trace_structured


def _add_compute_annotations(gm: fx.GraphModule, tag: str) -> bool:
    """Add compute_region annotations to nodes without custom metadata."""
    has_comm_region = False
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            continue
        if n.meta.get("custom", None) is None:
            n.meta["custom"] = {"compute_region": tag}
        else:
            if "comm_region" in n.meta["custom"]:
                has_comm_region = True
                val = n.meta["custom"]["comm_region"]
                n.meta["custom"]["comm_region"] = tag + " " + val
            elif "compute_region" in n.meta["custom"]:
                val = n.meta["custom"]["compute_region"]
                n.meta["custom"]["compute_region"] = tag + " " + val
            else:
                n.meta["custom"]["compute_region"] = tag
    return has_comm_region


def _move_wait_tensors_to_compute_region(gm: fx.GraphModule, tag: str):
    """Move the last wait_tensor node from each contiguous comm_region to the compute_region of its first user."""
    # First pass: identify the last wait_tensor in each contiguous comm region
    last_waits: list[fx.Node] = []
    last_wait: fx.Node | None = None
    in_comm_region = False

    for n in gm.graph.nodes:
        if n.op == "placeholder":
            continue
        if "comm_region" in n.meta["custom"]:
            in_comm_region = True
            if is_wait_tensor(n):
                last_wait = n
        else:
            # Transitioning out of a comm region — flush
            if in_comm_region and last_wait is not None:
                last_waits.append(last_wait)
                last_wait = None
            in_comm_region = False

    # Handle graph ending inside a comm region
    if in_comm_region and last_wait is not None:
        last_waits.append(last_wait)

    # Second pass: re-tag and move only the collected last-wait nodes
    for n in last_waits:
        assert len(n.users) >= 1, "wait tensor must have at least one user"
        user: fx.Node = next(iter(n.users))
        if "compute_region" in user.meta["custom"]:
            val = n.meta["custom"].pop("comm_region")
            if tag not in val:
                val = tag + " " + val
            n.meta["custom"].update({"compute_region": val + " " + "wait"})
            if n.next is not user:
                user.prepend(n)


def multiplex_fw_bw_graph(
    fw_gm: fx.GraphModule, bw_gm: fx.GraphModule, overlap_with_annotations: bool = True
) -> fx.GraphModule:
    """
    Multiplexes forward and backward graphs into a single unified graph module.

    This function combines a forward graph and a backward graph into one multiplexed
    graph by merging their nodes and outputs. The resulting graph has:
    - All placeholders from both forward and backward graphs (backward followed by forward)
    - All computation nodes from both graphs (backward followed by forward)
    - Combined outputs (backward outputs followed by forward outputs)

    Args:
        fw_gm: The forward graph module containing the forward computation
        bw_gm: The backward graph module containing the backward computation

    Returns:
        A multiplexed fx.GraphModule containing both forward and backward computations
        with backward outputs appearing before forward outputs

    Note:
        The function preserves node metadata during the merging process.
    """
    if overlap_with_annotations:
        fw_has_comm = _add_compute_annotations(fw_gm, "forward")
        bw_has_comm = _add_compute_annotations(bw_gm, "backward")
        assert fw_has_comm and bw_has_comm, "No comm region found in either graph"
        _move_wait_tensors_to_compute_region(fw_gm, "forward")
        _move_wait_tensors_to_compute_region(bw_gm, "backward")

    # Mapping to track correspondence between forward graph nodes and new nodes
    old_node_to_new_node: dict[torch.fx.Node, torch.fx.Node] = {}

    # Start with a deep copy of the backward graph as the base
    multiplexed_gm = copy.deepcopy(bw_gm)

    # Copy tensor constant attributes from fw_gm to multiplexed_gm with "fw_" prefix
    # to avoid collision with bw's tensor constants
    fw_tensor_constant_remap: dict[str, str] = {}
    for attr_name in dir(fw_gm):
        if attr_name.startswith("_tensor_constant"):
            fw_attr = getattr(fw_gm, attr_name)
            new_attr_name = (
                f"fw{attr_name}"  # e.g., _tensor_constant0 -> fw_tensor_constant0
            )
            setattr(multiplexed_gm, new_attr_name, fw_attr)
            fw_tensor_constant_remap[attr_name] = new_attr_name

    # Collect all placeholder nodes from all the graphs
    bw_placeholders = bw_gm.graph.find_nodes(op="placeholder")
    fw_placeholders = fw_gm.graph.find_nodes(op="placeholder")
    insert_point = multiplexed_gm.graph.find_nodes(op="placeholder")[-1]

    # Insert forward placeholders after the backward placeholders of the multiplexed graph
    for n in fw_placeholders:
        with multiplexed_gm.graph.inserting_after(insert_point):
            new_placeholder = multiplexed_gm.graph.placeholder(n.name)
            new_placeholder.meta = copy.copy(n.meta)
            new_placeholder.target = new_placeholder.name
            old_node_to_new_node[n] = new_placeholder
            insert_point = new_placeholder

    multiplexed_gm_placeholders = multiplexed_gm.graph.find_nodes(op="placeholder")
    assert len(multiplexed_gm_placeholders) == len(fw_placeholders) + len(
        bw_placeholders
    )
    fw_nodes_iter = iter(fw_gm.graph.nodes)
    fw_nodes_iter = dropwhile(lambda n: n.op == "placeholder", fw_nodes_iter)
    # Initialize the forward node to be the first non-placeholder node
    fn = next(fw_nodes_iter)
    if overlap_with_annotations:
        # Interleave forward and backward nodes to create overlap pattern:
        # bw_compute (if any) -> bw_comm -> fw_compute (if any) -> fw_comm -> [repeat]
        # This allows bw_comm to overlap with fw_compute, and fw_comm to overlap with bw_compute
        bw_in_comm = False
        for bn in multiplexed_gm.graph.nodes:
            if bn.op == "placeholder" or bn.op == "output":
                continue
            # Track when we enter a backward comm region
            if "comm_region" in bn.meta["custom"] and not bw_in_comm:
                bw_in_comm = True
            # When we transition from bw_comm to bw_compute, insert forward nodes
            elif "compute_region" in bn.meta["custom"] and bw_in_comm:
                bw_in_comm = False
                fw_in_comm = False
                insert_point = bn
                # Insert forward nodes before this bw_compute node
                # Note: We cannot reorder nodes within a graph, only their relative order between graphs
                while fn.op != "output":
                    if "comm_region" in fn.meta["custom"] and not fw_in_comm:
                        fw_in_comm = True
                    elif "compute_region" in fn.meta["custom"] and fw_in_comm:
                        # Stop when we reach the next fw_compute after fw_comm
                        # This ensures we insert one fw_compute + fw_comm cycle per bw_comm -> bw_compute transition
                        # If fw starts with comm (no compute before it), we still insert it to overlap with future bw_compute
                        fw_in_comm = False
                        break
                    with multiplexed_gm.graph.inserting_before(insert_point):
                        # Copy node and remap its arguments using the node mapping
                        new_node = multiplexed_gm.graph.node_copy(
                            fn, lambda x: old_node_to_new_node[x]
                        )
                        new_node.meta = copy.copy(fn.meta)
                        # Remap get_attr targets for tensor constants to avoid collision
                        if (
                            new_node.op == "get_attr"
                            and new_node.target in fw_tensor_constant_remap
                        ):
                            new_node.target = fw_tensor_constant_remap[
                                str(new_node.target)
                            ]
                        old_node_to_new_node[fn] = new_node
                    fn = next(fw_nodes_iter)
    # Insert any remaining forward nodes at the end
    # If overlap_with_annotations is False, this concatenates all fw nodes after bw nodes
    insert_point = multiplexed_gm.graph.find_nodes(op="output")[-1]
    while fn.op != "output":
        with multiplexed_gm.graph.inserting_before(insert_point):
            # Copy node and remap its arguments using the node mapping
            new_node = multiplexed_gm.graph.node_copy(
                fn, lambda x: old_node_to_new_node[x]
            )
            new_node.meta = copy.copy(fn.meta)
            # Remap get_attr targets for tensor constants to avoid collision
            if (
                new_node.op == "get_attr"
                and new_node.target in fw_tensor_constant_remap
            ):
                new_node.target = fw_tensor_constant_remap[str(new_node.target)]
            old_node_to_new_node[fn] = new_node
        fn = next(fw_nodes_iter)

    # Collect output arguments from forward graph, remapping to new nodes
    fw_outputs = fw_gm.graph.find_nodes(op="output")
    multiplexed_graph_outputs = multiplexed_gm.graph.find_nodes(op="output")
    assert len(multiplexed_graph_outputs) == 1 and len(fw_outputs) == 1
    fw_graph_op_node = fw_outputs[0]
    fw_op_node_args = [
        old_node_to_new_node[n] if n is not None else None
        for n in fw_graph_op_node.args[0]
    ]

    # Collect output arguments from multiplexed graph (will contain only bwd_outs)
    multiplexed_graph_op_node = multiplexed_graph_outputs[0]
    bw_op_node_args = list(multiplexed_graph_op_node.args[0])

    # Update output node args to prepend backward outputs before forward outputs
    multiplexed_graph_op_node.args = (tuple(bw_op_node_args + fw_op_node_args),)

    multiplexed_gm.graph.eliminate_dead_code()
    multiplexed_gm.graph.lint()
    multiplexed_gm.recompile()
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "autoparallel_multiplexed_graph",
            "encoding": "string",
        },
        payload_fn=lambda: multiplexed_gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )
    return multiplexed_gm
