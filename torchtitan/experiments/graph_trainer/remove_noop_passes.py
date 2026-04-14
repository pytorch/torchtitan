# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph passes that remove semantically no-op nodes.

These passes simplify traced forward-backward graphs by eliminating nodes
that are identity operations in the context of a fully traced graph (no
autograd, no symbolic shape changes).  Removing them reduces graph noise
and improves downstream pass effectiveness (bucketing, scheduling,
cudagraph compatibility).
"""

import sys

import torch

from torchtitan.tools.logging import logger


def remove_detach_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove ``aten.detach.default`` nodes from the graph.

    In a traced fwd+bwd graph there is no autograd context, so detach is
    semantically a no-op.  Removing these nodes simplifies the graph for
    downstream passes.

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with all detach nodes removed.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target is torch.ops.aten.detach.default:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} aten.detach.default node(s) from the graph")

    return gm


_IDENTITY_VIEW_TARGETS = {
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
}


def remove_identity_view_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove identity ``view``, ``reshape``, and ``_unsafe_view`` nodes.

    In a traced graph these ops are no-ops when the output shape equals
    the input shape.  Removing them simplifies the graph for downstream
    passes (bucketing, scheduling, cudagraph).

    Args:
        gm: The traced graph module.
        example_inputs: Unused, accepted for pass interface compatibility.

    Returns:
        The graph module with identity view nodes removed.
    """
    count = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _IDENTITY_VIEW_TARGETS:
            continue

        # Skip nodes without fake tensor metadata (e.g. symbolic or opaque).
        inp = node.args[0]
        inp_val = inp.meta.get("val") if isinstance(inp, torch.fx.Node) else None
        out_val = node.meta.get("val")
        if inp_val is None or out_val is None:
            continue
        if not isinstance(inp_val, torch.Tensor) or not isinstance(
            out_val, torch.Tensor
        ):
            continue

        if inp_val.shape == out_val.shape:
            node.replace_all_uses_with(inp)
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.info(f"Removed {count} identity view/reshape node(s) from the graph")

    return gm


def remove_identity_slice_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """Remove identity aten.slice.Tensor ops that select the full dimension.

    An ``aten.slice.Tensor(input, dim, start, end, step)`` is a no-op when
    ``start == 0``, ``end >= dim_size``, and ``step == 1``.  This pass
    replaces such nodes with their input tensor, reducing graph noise from
    decompositions and simplifying downstream passes.

    Default args for ``aten.slice.Tensor``: dim=0, start=0, end=sys.maxsize,
    step=1.
    """
    count = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is not torch.ops.aten.slice.Tensor:
            continue

        args = node.args
        input_node = args[0]

        # Parse args with defaults matching aten.slice.Tensor signature
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else sys.maxsize
        step = args[4] if len(args) > 4 else 1

        if start != 0 or step != 1:
            continue

        # Use fake tensor metadata to determine the actual dimension size.
        # Skip nodes without metadata (e.g. from hand-built test graphs).
        val = input_node.meta.get("val")
        if val is None:
            continue

        shape = val.shape
        dim_size = shape[dim]

        if end >= dim_size:
            node.replace_all_uses_with(input_node)
            gm.graph.erase_node(node)
            count += 1

    if count > 0:
        logger.info(f"Removed {count} identity slice node(s)")

    gm.graph.lint()
    gm.recompile()
    return gm
