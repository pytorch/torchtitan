"""Extract FX subgraphs as KernelAgent-compatible problem descriptions.

Given an FX GraphModule and a set of nodes forming a fusable subgraph,
produces a problem description string (Model class + get_inputs + get_init_inputs)
that KernelAgent can consume to generate a custom Triton kernel.
"""

from __future__ import annotations

import operator
from collections import Counter
from typing import Callable, Sequence

import torch

aten = torch.ops.aten


# ---------------------------------------------------------------------------
# Aten op → readable PyTorch code formatters
# ---------------------------------------------------------------------------

def _fmt_to_copy(a: list[str], kw: dict[str, str]) -> str:
    if "dtype" in kw:
        return f"{a[0]}.to(dtype={kw['dtype']})"
    return f"{a[0]}.clone()"


def _fmt_reduction(method: str):
    def _fmt(a: list[str], kw: dict[str, str]) -> str:
        parts = [a[0], f".{method}("]
        if len(a) > 1:
            parts.append(f"dim={a[1]}")
            if len(a) > 2:
                parts.append(f", keepdim={a[2]}")
        parts.append(")")
        return "".join(parts)
    return _fmt


_OP_FORMATTERS: dict[object, Callable[[list[str], dict[str, str]], str]] = {
    # Unary element-wise
    aten.silu.default: lambda a, kw: f"F.silu({a[0]})",
    aten.relu.default: lambda a, kw: f"F.relu({a[0]})",
    aten.gelu.default: lambda a, kw: f"F.gelu({a[0]})",
    aten.sigmoid.default: lambda a, kw: f"torch.sigmoid({a[0]})",
    aten.tanh.default: lambda a, kw: f"torch.tanh({a[0]})",
    aten.exp.default: lambda a, kw: f"torch.exp({a[0]})",
    aten.log.default: lambda a, kw: f"torch.log({a[0]})",
    aten.neg.default: lambda a, kw: f"(-{a[0]})",
    aten.abs.default: lambda a, kw: f"torch.abs({a[0]})",
    aten.rsqrt.default: lambda a, kw: f"torch.rsqrt({a[0]})",
    aten.sqrt.default: lambda a, kw: f"torch.sqrt({a[0]})",
    aten.clone.default: lambda a, kw: f"{a[0]}.clone()",
    aten.contiguous.default: lambda a, kw: f"{a[0]}.contiguous()",
    # Binary element-wise
    aten.mul.Tensor: lambda a, kw: f"({a[0]} * {a[1]})",
    aten.add.Tensor: lambda a, kw: f"({a[0]} + {a[1]})",
    aten.sub.Tensor: lambda a, kw: f"({a[0]} - {a[1]})",
    aten.div.Tensor: lambda a, kw: f"({a[0]} / {a[1]})",
    aten.pow.Tensor_Scalar: lambda a, kw: f"({a[0]} ** {a[1]})",
    aten.maximum.default: lambda a, kw: f"torch.maximum({a[0]}, {a[1]})",
    aten.minimum.default: lambda a, kw: f"torch.minimum({a[0]}, {a[1]})",
    # Type conversion
    aten._to_copy.default: _fmt_to_copy,
    # Shape ops
    aten.reshape.default: lambda a, kw: f"{a[0]}.reshape({a[1]})",
    aten.view.default: lambda a, kw: f"{a[0]}.view({a[1]})",
    aten.unsqueeze.default: lambda a, kw: f"{a[0]}.unsqueeze({a[1]})",
    aten.squeeze.dim: lambda a, kw: f"{a[0]}.squeeze({a[1]})",
    aten.permute.default: lambda a, kw: f"{a[0]}.permute(*{a[1]})",
    aten.transpose.int: lambda a, kw: f"{a[0]}.transpose({a[1]}, {a[2]})",
    aten.t.default: lambda a, kw: f"{a[0]}.t()",
    aten.expand.default: lambda a, kw: f"{a[0]}.expand({a[1]})",
    aten.slice.Tensor: (
        lambda a, kw: f"{a[0]}[{a[1] if len(a) > 1 else 0}"
        f":{a[2] if len(a) > 2 else ''}:{a[3] if len(a) > 3 else ''}]"
    ),
    # Complex ops
    aten.view_as_complex.default: lambda a, kw: f"torch.view_as_complex({a[0]})",
    aten.view_as_real.default: lambda a, kw: f"torch.view_as_real({a[0]})",
    # Reductions
    aten.sum.dim_IntList: _fmt_reduction("sum"),
    aten.mean.dim: _fmt_reduction("mean"),
    aten.amax.default: _fmt_reduction("amax"),
    # Matmul
    aten.mm.default: lambda a, kw: f"torch.mm({a[0]}, {a[1]})",
    aten.bmm.default: lambda a, kw: f"torch.bmm({a[0]}, {a[1]})",
    aten.addmm.default: lambda a, kw: f"torch.addmm({a[0]}, {a[1]}, {a[2]})",
    # Cat/stack
    aten.cat.default: lambda a, kw: f"torch.cat({a[0]}, dim={a[1] if len(a) > 1 else 0})",
    aten.stack.default: lambda a, kw: f"torch.stack({a[0]}, dim={a[1] if len(a) > 1 else 0})",
    # Getitem (tuple unpacking)
    operator.getitem: lambda a, kw: f"{a[0]}[{a[1]}]",
}


# ---------------------------------------------------------------------------
# Argument formatting
# ---------------------------------------------------------------------------

def _format_arg(arg: object, var_map: dict[torch.fx.Node, str]) -> str:
    """Format an FX node argument as Python source code."""
    if isinstance(arg, torch.fx.Node):
        return var_map.get(arg, f"<unresolved:{arg.name}>")
    if isinstance(arg, torch.dtype):
        return str(arg)
    if isinstance(arg, torch.device):
        return repr(str(arg))
    if isinstance(arg, torch.memory_format):
        return str(arg)
    if isinstance(arg, torch.layout):
        return str(arg)
    if isinstance(arg, (int, float, bool)):
        return repr(arg)
    if arg is None:
        return "None"
    if isinstance(arg, (list, tuple)):
        elems = [_format_arg(a, var_map) for a in arg]
        if isinstance(arg, tuple):
            inner = ", ".join(elems) + ("," if len(elems) == 1 else "")
            return f"({inner})"
        return f"[{', '.join(elems)}]"
    return repr(arg)


def _format_op(
    target: object,
    formatted_args: list[str],
    formatted_kwargs: dict[str, str],
) -> str:
    """Format an FX call_function node as readable Python code."""
    if target in _OP_FORMATTERS:
        return _OP_FORMATTERS[target](formatted_args, formatted_kwargs)
    # Fallback: torch.ops.<namespace>.<op>.<overload>(...)
    all_parts = list(formatted_args)
    all_parts.extend(f"{k}={v}" for k, v in formatted_kwargs.items())
    return f"torch.ops.{target}({', '.join(all_parts)})"


# ---------------------------------------------------------------------------
# Shape / dtype helpers
# ---------------------------------------------------------------------------

def _resolve_shape(shape: torch.Size) -> tuple[int, ...]:
    """Resolve a shape that may contain SymInts to concrete ints."""
    resolved = []
    for s in shape:
        if isinstance(s, int):
            resolved.append(s)
        elif hasattr(s, "node") and hasattr(s.node, "hint"):
            resolved.append(s.node.hint)
        else:
            try:
                resolved.append(int(s))
            except (TypeError, RuntimeError):
                resolved.append(1)  # last resort
    return tuple(resolved)


def _get_tensor_spec(node: torch.fx.Node) -> dict | None:
    """Extract shape/dtype from a node's FakeTensor metadata."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return {
            "shape": _resolve_shape(val.shape),
            "dtype": val.dtype,
        }
    if isinstance(val, (tuple, list)) and len(val) > 0:
        first = val[0]
        if isinstance(first, torch.Tensor):
            return {
                "shape": _resolve_shape(first.shape),
                "dtype": first.dtype,
            }
    return None


def _dtype_to_randn(dtype: torch.dtype) -> str:
    """Return the torch function call to generate random data for a dtype."""
    if dtype.is_complex:
        return "torch.randn"
    if dtype.is_floating_point:
        return "torch.randn"
    if dtype == torch.bool:
        return "torch.randint(0, 2, "
    # Integer types
    return "torch.randint(0, 10, "


def _format_input_creation(name: str, spec: dict) -> str:
    """Generate code to create a random input tensor."""
    shape = spec["shape"]
    dtype = spec["dtype"]
    shape_str = repr(shape)

    if dtype.is_floating_point or dtype.is_complex:
        return f"    {name} = torch.randn({shape_str}, dtype={dtype}, device='cuda')"
    if dtype == torch.bool:
        return f"    {name} = torch.randint(0, 2, {shape_str}, dtype={dtype}, device='cuda')"
    return f"    {name} = torch.randint(0, 10, {shape_str}, dtype={dtype}, device='cuda')"


# ---------------------------------------------------------------------------
# Core: extract subgraph as KernelAgent problem description
# ---------------------------------------------------------------------------

def extract_subgraph_as_problem(
    gm: torch.fx.GraphModule,
    nodes: Sequence[torch.fx.Node | str],
    *,
    description: str = "",
) -> str:
    """Convert a set of FX graph nodes into a KernelAgent problem description.

    Args:
        gm: The FX GraphModule containing the full graph.
        nodes: Node objects or node name strings identifying the subgraph.
        description: Human-readable description of the fusion target.

    Returns:
        A string containing a Model class with forward(), get_inputs(),
        and get_init_inputs() suitable for TritonKernelAgent.generate_kernel().
    """
    # Resolve names to Node objects
    if nodes and isinstance(nodes[0], str):
        name_to_node = {n.name: n for n in gm.graph.nodes}
        resolved = []
        for name in nodes:
            if name not in name_to_node:
                raise ValueError(
                    f"Node '{name}' not found in graph. "
                    f"Available: {list(name_to_node.keys())[:20]}..."
                )
            resolved.append(name_to_node[name])
        nodes = resolved

    node_set = set(nodes)

    # Topological ordering (preserve graph order)
    sorted_nodes = [n for n in gm.graph.nodes if n in node_set]

    # Identify external inputs: args from outside the subgraph that are Nodes
    external_inputs: list[torch.fx.Node] = []
    seen_inputs: set[torch.fx.Node] = set()
    for node in sorted_nodes:
        for arg in _iter_node_args(node):
            if isinstance(arg, torch.fx.Node) and arg not in node_set and arg not in seen_inputs:
                seen_inputs.add(arg)
                external_inputs.append(arg)

    # Identify outputs: subgraph nodes used by nodes outside the subgraph
    outputs: list[torch.fx.Node] = []
    for node in sorted_nodes:
        for user in node.users:
            if user not in node_set:
                if node not in outputs:
                    outputs.append(node)
                break

    if not outputs:
        outputs = [sorted_nodes[-1]]

    # Build variable name map
    var_map: dict[torch.fx.Node, str] = {}
    input_names: list[str] = []
    input_specs: list[dict] = []

    for i, inp_node in enumerate(external_inputs):
        name = f"input_{i}"
        var_map[inp_node] = name
        input_names.append(name)
        spec = _get_tensor_spec(inp_node)
        if spec is None:
            spec = {"shape": (1,), "dtype": torch.float32}
        input_specs.append(spec)

    # Generate forward() body
    body_lines: list[str] = []
    for i, node in enumerate(sorted_nodes):
        var_name = f"v{i}"
        var_map[node] = var_name

        if node.op == "call_function":
            fargs = [_format_arg(a, var_map) for a in node.args]
            fkwargs = {k: _format_arg(v, var_map) for k, v in node.kwargs.items()}
            code = _format_op(node.target, fargs, fkwargs)
            body_lines.append(f"        {var_name} = {code}")
        elif node.op == "get_attr":
            body_lines.append(f"        {var_name} = self.{node.target}")
        else:
            body_lines.append(f"        {var_name} = {_format_arg(node, var_map)}  # {node.op}")

    # Return statement
    if len(outputs) == 1:
        body_lines.append(f"        return {var_map[outputs[0]]}")
    else:
        out_vars = ", ".join(var_map[o] for o in outputs)
        body_lines.append(f"        return ({out_vars})")

    forward_body = "\n".join(body_lines)

    # Parameter signature
    param_parts = []
    for name, spec in zip(input_names, input_specs):
        param_parts.append(f"{name}: torch.Tensor")
    param_str = ", ".join(param_parts)

    # get_inputs()
    input_lines = [_format_input_creation(name, spec) for name, spec in zip(input_names, input_specs)]
    inputs_body = "\n".join(input_lines)
    return_list = ", ".join(input_names)

    # Return type annotation
    if len(outputs) == 1:
        ret_annotation = "torch.Tensor"
    else:
        ret_annotation = f"tuple[{', '.join(['torch.Tensor'] * len(outputs))}]"

    # Assemble the full problem description
    desc_block = f"{description}\n\n" if description else ""
    problem = f"""\
{desc_block}import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def forward(self, {param_str}) -> {ret_annotation}:
{forward_body}

def get_inputs():
{inputs_body}
    return [{return_list}]

def get_init_inputs():
    return []
"""
    return problem


def _iter_node_args(node: torch.fx.Node):
    """Yield all Node-typed arguments from a node's args and kwargs."""
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            yield arg
        elif isinstance(arg, (list, tuple)):
            for a in arg:
                if isinstance(a, torch.fx.Node):
                    yield a
    for v in node.kwargs.values():
        if isinstance(v, torch.fx.Node):
            yield v
        elif isinstance(v, (list, tuple)):
            for a in v:
                if isinstance(a, torch.fx.Node):
                    yield a


# ---------------------------------------------------------------------------
# Graph analysis for LLM-driven fusion candidate identification
# ---------------------------------------------------------------------------

def dump_graph_for_analysis(
    gm: torch.fx.GraphModule,
    *,
    include_shapes: bool = True,
    include_backward: bool = False,
    max_nodes: int = 500,
) -> str:
    """Dump FX graph in a format suitable for LLM analysis of fusion candidates.

    Returns a structured summary with per-node info and op frequency counts,
    designed to help an LLM identify profitable fusion patterns.
    """
    lines: list[str] = []
    op_counts: Counter = Counter()
    total = 0
    fwd_count = 0
    bwd_count = 0

    for node in gm.graph.nodes:
        if node.op == "call_function":
            is_bwd = node.meta.get("autograd_backward", False)
            if is_bwd:
                bwd_count += 1
            else:
                fwd_count += 1
            op_counts[str(node.target)] += 1
        total += 1

    lines.append(f"# FX Graph Summary: {total} total nodes ({fwd_count} fwd, {bwd_count} bwd)")
    lines.append("")
    lines.append("## Op frequency (top 30):")
    for op_name, count in op_counts.most_common(30):
        lines.append(f"  {op_name}: {count}")

    lines.append("")
    lines.append("## Node details (forward only):")
    lines.append(f"{'name':<40} {'target':<45} {'output_shape':<30} {'dtype':<12} {'module_fqn'}")
    lines.append("-" * 140)

    shown = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        is_bwd = node.meta.get("autograd_backward", False)
        if is_bwd and not include_backward:
            continue
        if shown >= max_nodes:
            lines.append(f"  ... ({total - shown} more nodes)")
            break

        target_str = str(node.target)
        shape_str = ""
        dtype_str = ""
        if include_shapes:
            spec = _get_tensor_spec(node)
            if spec:
                shape_str = str(spec["shape"])
                dtype_str = str(spec["dtype"]).replace("torch.", "")

        fqn = node.meta.get("custom", {}).get("module_fqn", "")

        bwd_marker = " [bwd]" if is_bwd else ""
        lines.append(
            f"{node.name:<40} {target_str:<45} {shape_str:<30} {dtype_str:<12} {fqn}{bwd_marker}"
        )
        shown += 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern finding
# ---------------------------------------------------------------------------

def find_pattern_instances(
    gm: torch.fx.GraphModule,
    anchor_target: object,
    match_fn: Callable[[torch.fx.Node], list[torch.fx.Node] | None],
) -> list[list[torch.fx.Node]]:
    """Find all instances of a pattern in the graph.

    Args:
        anchor_target: The aten op to search for (e.g., aten.silu.default).
        match_fn: Given an anchor node, return the full list of nodes in
            the pattern instance, or None if the anchor doesn't match.

    Returns:
        List of node groups, one per pattern instance found.

    Example:
        def match_swiglu(silu_node):
            for user in silu_node.users:
                if user.target == aten.mul.Tensor:
                    return [silu_node, user]
            return None

        instances = find_pattern_instances(gm, aten.silu.default, match_swiglu)
    """
    instances: list[list[torch.fx.Node]] = []
    matched_nodes: set[torch.fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != anchor_target:
            continue
        if node in matched_nodes:
            continue
        result = match_fn(node)
        if result is not None:
            instances.append(result)
            matched_nodes.update(result)
    return instances
