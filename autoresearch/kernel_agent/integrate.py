"""Integrate KernelAgent-generated Triton kernels into FX graph passes.

Provides utilities to:
1. Load a generated Triton kernel from disk
2. Register it as a torch.library custom op (FX-compatible)
3. Replace matched FX graph patterns with the custom op
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import torch

_lib = torch.library.Library("kernel_agent", "DEF")
_registered_ops: dict[str, Any] = {}


def load_kernel(kernel_path: str | Path) -> Callable:
    """Load a Triton kernel module and return its kernel_function.

    The kernel file must define a top-level ``kernel_function`` callable
    (the standard KernelAgent output format).
    """
    path = Path(kernel_path)
    if not path.exists():
        raise FileNotFoundError(f"Kernel file not found: {path}")

    spec = importlib.util.spec_from_file_location(f"_kernel_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "kernel_function"):
        raise AttributeError(
            f"Kernel module {path} does not define 'kernel_function'. "
            f"Available: {[a for a in dir(module) if not a.startswith('_')]}"
        )
    return module.kernel_function


def register_triton_op(
    name: str,
    kernel_fn: Callable,
    *,
    schema: str,
    fake_fn: Callable,
) -> Any:
    """Register a Triton kernel as a torch.library custom op.

    After registration the op is callable as ``torch.ops.kernel_agent.<name>``.

    Args:
        name: Op name (e.g., "fused_swiglu").
        kernel_fn: The kernel_function callable from KernelAgent output.
        schema: Torch schema string, e.g. "(Tensor x, Tensor gate) -> Tensor".
        fake_fn: FakeTensor implementation for shape/dtype propagation during
            tracing. Must accept the same signature as kernel_fn and return
            tensors with correct shapes/dtypes (use torch.empty_like, etc.).

    Returns:
        The registered op (torch.ops.kernel_agent.<name>).

    Example::

        kernel_fn = load_kernel("generated/swiglu/kernel.py")
        op = register_triton_op(
            "fused_swiglu",
            kernel_fn,
            schema="(Tensor x, Tensor gate) -> Tensor",
            fake_fn=lambda x, gate: torch.empty_like(x),
        )
        # Now callable as torch.ops.kernel_agent.fused_swiglu(x, gate)
    """
    if name in _registered_ops:
        return _registered_ops[name]

    _lib.define(f"{name}{schema}")
    _lib.impl(name, kernel_fn, "CUDA")
    _lib.impl(name, fake_fn, "Meta")

    op = getattr(torch.ops.kernel_agent, name)
    _registered_ops[name] = op
    return op


def replace_pattern(
    gm: torch.fx.GraphModule,
    match_fn: Callable[[torch.fx.Node], list[torch.fx.Node] | None],
    replacement_fn: Callable[[list[torch.fx.Node], torch.fx.Graph], torch.fx.Node],
    *,
    anchor_target: Any,
) -> int:
    """Replace all instances of a pattern in the FX graph.

    Walks the graph looking for ``anchor_target`` nodes. For each, calls
    ``match_fn`` to check if a full pattern match exists. If so, calls
    ``replacement_fn`` to insert the replacement node(s), then cleans up
    the old nodes.

    Args:
        gm: The GraphModule to transform (modified in place).
        match_fn: Given an anchor node, returns the full list of nodes
            in the pattern (in topological order) or None.
        replacement_fn: Given (matched_nodes, graph), inserts replacement
            node(s) and returns the node that replaces the pattern's output.
        anchor_target: The aten op to scan for (e.g., aten.silu.default).

    Returns:
        Number of replacements made.

    Example — replacing silu+mul with a fused custom op::

        def match_swiglu(silu_node):
            for user in silu_node.users:
                if user.target == torch.ops.aten.mul.Tensor:
                    return [silu_node, user]
            return None

        def replace_swiglu(nodes, graph):
            silu, mul = nodes
            x = silu.args[0]
            gate = mul.args[1] if mul.args[0] is silu else mul.args[0]
            with graph.inserting_after(mul):
                new = graph.call_function(
                    torch.ops.kernel_agent.fused_swiglu, args=(x, gate)
                )
                new.meta = mul.meta.copy()
            return new

        count = replace_pattern(
            gm, match_swiglu, replace_swiglu,
            anchor_target=torch.ops.aten.silu.default,
        )
    """
    graph = gm.graph
    replaced = 0
    matched_nodes: set[torch.fx.Node] = set()
    # Collect all replacements first, then apply (avoid modifying during iteration)
    replacements: list[tuple[list[torch.fx.Node], torch.fx.Node]] = []

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != anchor_target:
            continue
        if node in matched_nodes:
            continue

        pattern_nodes = match_fn(node)
        if pattern_nodes is None:
            continue

        output_node = pattern_nodes[-1]
        new_node = replacement_fn(pattern_nodes, graph)
        replacements.append((pattern_nodes, new_node))
        matched_nodes.update(pattern_nodes)

    for pattern_nodes, new_node in replacements:
        output_node = pattern_nodes[-1]
        output_node.replace_all_uses_with(new_node)
        # Erase in reverse topological order
        for old in reversed(pattern_nodes):
            if not old.users:
                graph.erase_node(old)
        replaced += 1

    if replaced > 0:
        graph.lint()
        gm.recompile()

    return replaced
