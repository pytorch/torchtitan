# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch

from torchtitan.tools.logging import logger


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
