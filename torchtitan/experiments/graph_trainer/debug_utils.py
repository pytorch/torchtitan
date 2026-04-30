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


class FQNInterpreter(torch.fx.Interpreter):
    """Interpreter that sets activation tracer context vars from node metadata.

    For each node, reads:
    - ``node.meta["custom"]["module_fqn"]`` → sets _current_module_name
    - ``node.meta["stack_trace"]`` → parsed and set as _current_stack_frames
    - ``node.meta["autograd_backward"]`` → sets _current_is_backward

    This is needed because traced graph replay via ``gm(*inputs)`` bypasses
    module forwards entirely, so the monkeypatched forwards from _patch_model
    never fire.  FQNInterpreter walks the graph node-by-node, restoring the
    context that _patch_model would have set during eager execution.
    """

    def run_node(self, n: torch.fx.Node):
        from contextvars import Token

        from torchtitan.tools.activation_tracer import (
            _current_module_name,
            _current_phase_override,
            _current_stack_frames,
            _parse_stack_trace,
        )

        fqn = (n.meta.get("custom") or {}).get("module_fqn")
        stack_trace = n.meta.get("stack_trace")
        is_backward = n.meta.get("autograd_backward", False)

        # Each ContextVar.set() returns a Token that remembers the
        # previous value.  We collect them so the finally block can
        # restore all vars in reverse order (LIFO, like nested withs).
        tokens: list[Token] = []
        if fqn:
            tokens.append(_current_module_name.set(fqn))
        if stack_trace:
            tokens.append(
                _current_stack_frames.set(_parse_stack_trace(stack_trace))
            )
        tokens.append(
            _current_phase_override.set("backward" if is_backward else "forward")
        )
        try:
            return super().run_node(n)
        finally:
            # Reset each context var to its value before this node,
            # so the next node starts with a clean slate.
            for token in reversed(tokens):
                token.var.reset(token)
