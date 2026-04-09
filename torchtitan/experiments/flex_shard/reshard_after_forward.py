# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph pass for FlexShard reshard-after-forward.

Generalizes graph_trainer's ``annotate_fsdp_all_gather`` to handle all
FlexShard unshard patterns (Shard(0), Shard(dim!=0), FlatShard) in addition
to SimpleFSDP's pattern. Annotates unshard sequence nodes with
``CheckpointPolicy.MUST_RECOMPUTE`` or ``MUST_SAVE`` so the compiler can
free unsharded parameters between forward and backward.

Unshard sequences by placement type (brackets = optional):

    Shard(0):       placeholder → [_to_copy] → all_gather → wait_tensor
                    → [convert_element_type]
    Shard(dim!=0):  placeholder → [_to_copy] → all_gather → wait_tensor
                    → chunk → getitem(0..N) → cat → [convert_element_type]
    FlatShard:      placeholder → [_to_copy] → all_gather → wait_tensor
                    → view → [convert_element_type]
    SimpleFSDP:     [convert_element_type] → all_gather → wait_tensor
                    → [slice]
"""

import operator

import torch
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_wait_tensor,
)
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.reshard_after_forward import (
    is_wait_tensor_from_fsdp,
)


# ---------------------------------------------------------------------------
# Helper predicates for FX node targets
# ---------------------------------------------------------------------------


def _is_chunk(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.chunk.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
    )


def _is_getitem(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def _is_cat(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (torch.ops.aten.cat.default,)


def _is_view(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    )


def _is_slice(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (torch.ops.aten.slice.Tensor,)


def _is_convert_element_type(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.prims.convert_element_type.default,
    )


def _is_to_copy(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        torch.ops.aten._to_copy.default,
        torch.ops.aten.to.dtype_layout,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.device,
    )


# ---------------------------------------------------------------------------
# Core annotation logic
# ---------------------------------------------------------------------------


def _force_recompute_node(node: torch.fx.Node, reshard_after_forward: bool) -> None:
    """Mark a node for recomputation or forced save."""
    if reshard_after_forward:
        node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    else:
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    # ac_graph_id prevents the partitioner from treating these nodes as part
    # of a user-level activation checkpoint region. A large value ensures our
    # annotations aren't influenced by nearby ac_graph_id values.
    node.meta["ac_graph_id"] = 100000


def _annotate_unshard_sequence(
    wait_node: torch.fx.Node,
    reshard_after_forward: bool,
) -> None:
    """Annotate all nodes in an unshard sequence starting from wait_tensor.

    Walks backward from all_gather to annotate pre-processing ops (.to(),
    convert_element_type) and forward from wait_tensor to annotate
    post-processing ops (chunk+cat, view, slice, convert_element_type).
    """

    def force(node: torch.fx.Node) -> None:
        _force_recompute_node(node, reshard_after_forward)

    ag_node = wait_node.args[0]
    assert is_all_gather_into_tensor(ag_node)

    # 1. Annotate all_gather + wait_tensor
    force(ag_node)
    force(wait_node)

    # 2. Walk backward from all_gather: annotate .to() (offload H2D) and
    #    convert_element_type (SimpleFSDP mixed precision pre-cast)
    pre = ag_node.all_input_nodes[0]
    while pre.op == "call_function":
        if _is_to_copy(pre) or _is_convert_element_type(pre):
            force(pre)
        pre_inputs = pre.all_input_nodes
        if len(pre_inputs) == 1:
            pre = pre_inputs[0]
        else:
            break

    # 3. Walk forward from wait_tensor: identify post-processing pattern
    terminal = wait_node
    for user in wait_node.users:
        if _is_chunk(user):
            # Shard(dim!=0): chunk → getitem(0..N) → cat
            force(user)
            for gi_user in user.users:
                if _is_getitem(gi_user):
                    force(gi_user)
                    for cat_user in gi_user.users:
                        if _is_cat(cat_user):
                            force(cat_user)
                            terminal = cat_user
            break
        elif _is_view(user):
            # FlatShard: wait_tensor → view
            force(user)
            terminal = user
            break
        elif _is_slice(user):
            # SimpleFSDP: wait_tensor → slice
            force(user)
            terminal = user
            break

    # 4. Walk forward from terminal: convert_element_type (mixed precision)
    for user in terminal.users:
        if _is_convert_element_type(user):
            force(user)
            terminal = user
            break

    # 5. Tag the final output node with metadata for downstream passes
    terminal.meta["flex_shard_placement"] = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_flex_shard_all_gather(
    gm: torch.fx.GraphModule, reshard_after_forward: bool
) -> torch.fx.GraphModule:
    """Annotate all FlexShard and SimpleFSDP unshard sequences in the graph.

    This is a superset of ``annotate_fsdp_all_gather`` — it handles all
    unshard patterns (Shard(0), Shard(dim!=0), FlatShard, SimpleFSDP).
    """
    for node in gm.graph.nodes:
        if is_wait_tensor_from_fsdp(node):
            _annotate_unshard_sequence(node, reshard_after_forward)
    return gm


def flex_shard_reshard_after_fwd_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    reshard_after_forward: bool,
) -> torch.fx.GraphModule:
    """Graph pass: annotate FlexShard/SimpleFSDP all-gathers for reshard.

    Drop-in replacement for ``fsdp_reshard_after_fwd_pass`` that handles
    FlexShard's richer unshard patterns in addition to SimpleFSDP.

    When ``reshard_after_forward=True``, all-gather nodes are marked
    ``MUST_RECOMPUTE`` so the compiler frees unsharded params after forward
    and recomputes them in backward.

    When ``reshard_after_forward=False``, nodes are marked ``MUST_SAVE``
    to prevent the compiler from accidentally recomputing them.
    """
    gm = annotate_flex_shard_all_gather(gm, reshard_after_forward)
    gm.recompile()
    return gm
