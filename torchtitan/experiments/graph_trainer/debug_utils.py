# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict

import torch
from torch._logging import trace_structured

from torchtitan.tools.logging import logger


def timing_log(msg: str, *args) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    text = msg % args if args else msg
    if logger.handlers and logger.isEnabledFor(logging.INFO):
        logger.info(text)
    else:
        print(f"[graph_trainer_timing] {text}", flush=True)


def _get_node_target_name(node: torch.fx.Node) -> str:
    """Return a human-readable name for a node's target."""
    if node.op == "call_function":
        target = node.target
        if hasattr(target, "__name__"):
            return target.__name__
        return str(target).split(".")[-1]
    return f"{node.op}:{node.target}"


def snapshot_graph(gm: torch.fx.GraphModule) -> dict:
    """Capture a structural snapshot of a graph for comparison."""
    op_counts: dict[str, int] = defaultdict(int)
    num_nodes = 0
    num_placeholders = 0
    num_outputs = 0

    for node in gm.graph.nodes:
        num_nodes += 1
        if node.op == "placeholder":
            num_placeholders += 1
        elif node.op == "output":
            num_outputs += 1
        elif node.op == "call_function":
            op_counts[_get_node_target_name(node)] += 1
        elif node.op == "get_attr":
            op_counts["get_attr"] += 1

    return {
        "op_counts": dict(op_counts),
        "num_nodes": num_nodes,
        "num_placeholders": num_placeholders,
        "num_outputs": num_outputs,
    }


def log_graph_diff(
    before: dict,
    after: dict,
    pass_name: str,
) -> None:
    """Log a structured summary of what a graph pass changed."""
    lines = []

    node_delta = after["num_nodes"] - before["num_nodes"]
    if node_delta:
        lines.append(
            f"  nodes: {before['num_nodes']} -> {after['num_nodes']} ({node_delta:+d})"
        )

    ph_delta = after["num_placeholders"] - before["num_placeholders"]
    if ph_delta:
        lines.append(
            f"  placeholders: {before['num_placeholders']} -> {after['num_placeholders']} ({ph_delta:+d})"
        )

    out_delta = after["num_outputs"] - before["num_outputs"]
    if out_delta:
        lines.append(
            f"  outputs: {before['num_outputs']} -> {after['num_outputs']} ({out_delta:+d})"
        )

    all_ops = sorted(set(before["op_counts"]) | set(after["op_counts"]))
    changed_ops = []
    for op in all_ops:
        b = before["op_counts"].get(op, 0)
        a = after["op_counts"].get(op, 0)
        if b != a:
            changed_ops.append(f"  {op}: {b} -> {a} ({a - b:+d})")

    if changed_ops:
        lines.append("  op changes:")
        lines.extend(f"    {line}" for line in changed_ops)

    if not lines:
        logger.info(f"[{pass_name}] graph unchanged")
    else:
        logger.info(f"[{pass_name}] graph diff:\n" + "\n".join(lines))


def tlparse_log_graph_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    graph_name: str,
    debug: bool = False,
) -> torch.fx.GraphModule:
    """Log the transformed graph to tlparse via trace_structured.

    This pass should be added as the last transform in fwd/bwd_transforms
    so that the logged graph reflects all prior transformations.

    Args:
        gm: The graph module to log.
        example_inputs: The example inputs (unused, required by protocol).
        graph_name: The name for this graph artifact
            (e.g. "aot_forward_graph_transformed").
        debug: When True, include additional metadata in the printed nodes.

    Returns:
        The graph module unchanged.
    """
    additional_meta = ["autograd_backward"]
    if debug:
        additional_meta.append("seq_nr")

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": graph_name,
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
            additional_meta=additional_meta,
        ),
        expect_trace_id=False,
    )

    return gm
