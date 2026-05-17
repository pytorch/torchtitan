# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused kernel registry and replacement pass.

Provides a general framework for replacing aten subgraphs in the FX graph
with optimized kernel implementations. The kernels can come from any backend:
Triton, torch.compile, handwritten CUDA, etc.

The registry is populated by loading a directory of generated kernels
(see ``load_from_dir``). Each kernel directory contains:
  - ``problem.py``: reference PyTorch implementation (``Model.forward``)
  - ``kernel.py`` or ``optimized_kernel.py``: Triton implementation (optional)
  - ``benchmark.json``: timing data for backend selection (optional)

The replacement pass matches subgraphs by their op-target signature
(the sequence of ``torch.ops.aten.*`` targets) and replaces them with
a ``torch.ops.fused_kernel.*`` custom op that dispatches to the best
available backend at call time.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import operator
import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from torchtitan.tools.logging import logger

_lib = torch.library.Library("fused_kernel", "DEF")
_registered_schemas: set[str] = set()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Implementation:
    """A single backend implementation of a fused op."""

    backend: str  # "triton", "compile", "eager"
    fn: Callable
    time_ms: float = float("inf")


@dataclass
class FusedOp:
    """A fused op with multiple backend implementations and dispatch."""

    name: str
    signature: list[OpSig]  # op targets + shapes identifying the subgraph
    num_inputs: int
    num_outputs: int = 1    # >1 means the op returns a tuple of tensors
    reference_fn: Callable | None = None
    implementations: dict[str, Implementation] = field(default_factory=dict)

    @property
    def best_backend(self) -> str | None:
        """Backend with lowest benchmark time (from benchmark.json)."""
        if not self.implementations:
            return None
        return min(
            self.implementations, key=lambda k: self.implementations[k].time_ms
        )

    def dispatch(self, *args: Any) -> Any:
        """Call the best backend based on microbenchmark results."""
        backend = self.best_backend
        if backend is None:
            raise RuntimeError(f"No implementations for {self.name}")
        return self.implementations[backend].fn(*args)


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def _load_module_from_file(path: Path, name: str) -> types.ModuleType:
    content = path.read_text()
    idx = content.find("import torch")
    if idx > 0:
        content = content[idx:]
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(content, str(path), "exec"), mod.__dict__)
    return mod


def _load_kernel_fn(path: Path) -> Callable | None:
    if not path.exists():
        return None
    try:
        mod = _load_module_from_file(path, f"_kernel_{path.parent.name}")
        return getattr(mod, "kernel_function", None)
    except Exception as e:
        logger.debug(f"Failed to load {path}: {e}")
        return None


@dataclass
class OpSig:
    """Signature element: op target + output shape for shape-constrained matching."""
    target: str           # e.g. "aten.reshape.default"
    output_shape: tuple[int, ...]  # e.g. (1, 8192, 4096)
    output_dtype: str     # e.g. "bf16"

    def matches_node(self, node: torch.fx.Node) -> bool:
        """Check if an FX node matches this signature element."""
        if str(node.target) != self.target:
            return False
        if not self.output_shape:
            return True
        val = node.meta.get("val")
        if val is None:
            return True
        if isinstance(val, torch.Tensor):
            return tuple(val.shape) == self.output_shape
        return True


def _parse_shape_from_comment(comment: str) -> tuple[str, tuple[int, ...]]:
    """Parse dtype and shape from a type comment like 'bf16[1, 8192, 4096][...]cuda:0'."""
    m = re.match(r"(\w+)\[([^\]]*)\]", comment)
    if not m:
        return "", ()
    dtype = m.group(1)
    shape_str = m.group(2)
    if not shape_str:
        return dtype, ()
    try:
        shape = tuple(int(x.strip()) for x in shape_str.split(","))
        return dtype, shape
    except ValueError:
        return dtype, ()


def _extract_signature(problem_path: Path) -> list[OpSig]:
    """Extract op signature with shape constraints from problem.py.

    Parses the forward body for op targets and their output type comments,
    producing a list of OpSig elements that match by both target and shape.
    """
    content = problem_path.read_text()
    fwd_start = content.find("def forward(")
    if fwd_start < 0:
        return []
    body = content[fwd_start:]

    sig: list[OpSig] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("def ") or line.startswith("return"):
            continue

        # Extract type comment if present: ... # bf16[1, 8192][...]cuda:0
        dtype, shape = "", ()
        comment_m = re.search(r"#\s*(.+)$", line)
        if comment_m:
            dtype, shape = _parse_shape_from_comment(comment_m.group(1).strip())

        # torch.ops.X.Y.Z(...)
        m = re.search(r"torch\.ops\.(\S+)\(", line)
        if m:
            sig.append(OpSig(target=m.group(1), output_shape=shape, output_dtype=dtype))
            continue

        # v_N = v_M[idx]  (getitem)
        if re.match(r"\w+\s*=\s*\w+\[\d+\]", line):
            sig.append(OpSig(target="<built-in function getitem>", output_shape=shape, output_dtype=dtype))
            continue

    return sig


def _sanitize_op_name(name: str) -> str:
    clean = re.sub(r"^[\d_]+", "", name)
    if not clean:
        clean = "op"
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", clean)
    return f"f_{clean}"


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------


def _register_custom_op(fused_op: FusedOp) -> Callable:
    op_name = _sanitize_op_name(fused_op.name)
    if op_name in _registered_schemas:
        return getattr(torch.ops.fused_kernel, op_name)

    args = ", ".join(f"Tensor x{i}" for i in range(fused_op.num_inputs))
    n_out = fused_op.num_outputs
    if n_out == 1:
        ret_type = "Tensor"
    else:
        ret_type = f"({', '.join(['Tensor'] * n_out)})"
    schema = f"{op_name}({args}) -> {ret_type}"
    _lib.define(schema)

    def cuda_impl(*args: torch.Tensor):
        return fused_op.dispatch(*args)

    def meta_impl(*args: torch.Tensor):
        if fused_op.reference_fn is not None:
            try:
                return fused_op.reference_fn(*args)
            except Exception:
                pass
        if n_out == 1:
            return torch.empty_like(args[0])
        return tuple(torch.empty_like(args[0]) for _ in range(n_out))

    _lib.impl(op_name, cuda_impl, "CUDA")
    _lib.impl(op_name, meta_impl, "Meta")

    _registered_schemas.add(op_name)
    return getattr(torch.ops.fused_kernel, op_name)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FusedKernelRegistry:
    """Global registry of fused kernel replacements."""

    def __init__(self) -> None:
        self.ops: dict[str, FusedOp] = {}

    def register(self, op: FusedOp) -> None:
        self.ops[op.name] = op

    def load_from_dir(self, generated_dir: str | Path) -> None:
        """Load fused ops from a directory of generated kernels.

        Expected layout per pattern:
          generated/<name>/
            problem.py              # reference Model + get_inputs
            kernel.py               # Triton kernel (optional)
            optimized_kernel.py     # NCU-optimized Triton (optional)
            benchmark.json          # timing results (optional)
        """
        generated = Path(generated_dir)
        if not generated.exists():
            logger.warning(f"Fused kernel dir not found: {generated}")
            return

        for d in sorted(generated.iterdir()):
            if not d.is_dir():
                continue
            problem_path = d / "problem.py"
            if not problem_path.exists():
                continue

            name = d.name
            signature = _extract_signature(problem_path)
            if not signature:
                continue

            # Load reference and detect num_inputs / num_outputs
            ref_fn = None
            num_inputs = 0
            num_outputs = 1
            try:
                mod = _load_module_from_file(problem_path, f"_prob_{name}")
                model_cls = getattr(mod, "Model", None)
                get_inputs_fn = getattr(mod, "get_inputs", None)
                if model_cls and get_inputs_fn:
                    ref_fn = model_cls().forward
                    inputs = get_inputs_fn()
                    num_inputs = len(inputs)
                    # Detect multi-output by running the reference
                    try:
                        with torch.no_grad():
                            ref_out = ref_fn(*inputs)
                        if isinstance(ref_out, tuple):
                            num_outputs = len(ref_out)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Failed to load reference from {problem_path}: {e}")
                continue

            op = FusedOp(
                name=name,
                signature=signature,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                reference_fn=ref_fn,
            )

            # Load benchmark data
            bench = {}
            bench_path = d / "benchmark.json"
            if bench_path.exists():
                try:
                    bench = json.loads(bench_path.read_text())
                except Exception:
                    pass

            # Register eager backend
            if ref_fn is not None:
                op.implementations["eager"] = Implementation(
                    backend="eager",
                    fn=ref_fn,
                    time_ms=bench.get("eager_ms", float("inf")),
                )

            # Register torch.compile backend
            if ref_fn is not None:
                try:
                    compiled = torch.compile(ref_fn, fullgraph=True)
                    op.implementations["compile"] = Implementation(
                        backend="compile",
                        fn=compiled,
                        time_ms=bench.get("compile_ms", float("inf")),
                    )
                except Exception:
                    pass

            # Register Triton backend
            for kernel_name in ("optimized_kernel.py", "kernel.py"):
                kernel_fn = _load_kernel_fn(d / kernel_name)
                if kernel_fn is not None:
                    op.implementations["triton"] = Implementation(
                        backend="triton",
                        fn=kernel_fn,
                        time_ms=bench.get("triton_ms", float("inf")),
                    )
                    break

            if op.implementations:
                self.register(op)

        logger.info(
            f"Loaded {len(self.ops)} fused ops from {generated} "
            f"(backends per op: "
            f"{', '.join(f'{n}={list(o.implementations)}' for n, o in list(self.ops.items())[:3])}"
            f"{'...' if len(self.ops) > 3 else ''})"
        )


# ---------------------------------------------------------------------------
# Graph pass
# ---------------------------------------------------------------------------


def _iter_node_args(node: torch.fx.Node):
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


def fused_kernel_replacement_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    registry: FusedKernelRegistry | None = None,
) -> torch.fx.GraphModule:
    """Replace matched aten subgraphs with fused custom ops.

    No-op when the registry is None or empty.
    """
    if registry is None or not registry.ops:
        return gm

    # Register all ops as torch.library custom ops
    # Group by first op target for fast lookup
    first_target_to_ops: dict[str, list[tuple[FusedOp, Any]]] = {}
    for name, fused_op in registry.ops.items():
        if not fused_op.signature:
            continue
        custom_op = _register_custom_op(fused_op)
        first_target = fused_op.signature[0].target
        first_target_to_ops.setdefault(first_target, []).append((fused_op, custom_op))

    graph = gm.graph
    nodes = list(graph.nodes)
    replaced = 0

    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node.op != "call_function":
            i += 1
            continue

        node_target = str(node.target)
        candidates = first_target_to_ops.get(node_target, [])
        matched = False

        for fused_op, custom_op in candidates:
            sig = fused_op.signature
            if i + len(sig) > len(nodes):
                continue

            # Shape-aware matching: check each node against its OpSig
            window = nodes[i : i + len(sig)]
            match = True
            for fx_node, op_sig in zip(window, sig):
                if not op_sig.matches_node(fx_node):
                    match = False
                    break
            if not match:
                continue

            matched_names = {n.name for n in window}
            external_inputs = []
            for n in window:
                for arg in _iter_node_args(n):
                    if arg.name not in matched_names and arg not in external_inputs:
                        external_inputs.append(arg)

            # Find output nodes: nodes in the window that have users outside it
            output_nodes = []
            for n in window:
                for user in n.users:
                    if user.name not in matched_names:
                        if n not in output_nodes:
                            output_nodes.append(n)
                        break
            if not output_nodes:
                output_nodes = [window[-1]]

            # Check that num_outputs matches the registered op
            if len(output_nodes) != fused_op.num_outputs:
                i += 1
                continue

            last_node = window[-1]
            with graph.inserting_after(last_node):
                new_node = graph.call_function(
                    custom_op, args=tuple(external_inputs)
                )

                if len(output_nodes) == 1:
                    new_node.meta = output_nodes[0].meta.copy()
                    output_nodes[0].replace_all_uses_with(new_node)
                else:
                    # Multi-output: the fused op returns a tuple.
                    # Insert getitem nodes to fan out to each output's users.
                    new_node.meta = {}
                    for out_idx, out_node in enumerate(output_nodes):
                        getitem_node = graph.call_function(
                            operator.getitem, args=(new_node, out_idx)
                        )
                        getitem_node.meta = out_node.meta.copy()
                        out_node.replace_all_uses_with(getitem_node)

            # Fix self-references (new_node/getitem using themselves)
            for user_node in list(new_node.users):
                if user_node.target is operator.getitem:
                    continue
                user_node.replace_input_with(new_node, last_node)

            for old in reversed(window):
                if not old.users:
                    graph.erase_node(old)

            nodes = list(graph.nodes)
            replaced += 1
            matched = True
            break

        if not matched:
            i += 1

    if replaced > 0:
        graph.lint()
        gm.recompile()
        logger.info(f"Fused kernel replacement pass: replaced {replaced} subgraphs")

    return gm


# ---------------------------------------------------------------------------
# Extraction pass: discover fusible patterns from the live GraphModule
# ---------------------------------------------------------------------------

# Ops that must not appear in a fused region.
_UNFUSABLE_OPS = {
    "torch.ops.bucketing._pre_bucket_all_gather.default",
    "torch.ops.bucketing._pre_bucket_reduce_scatter.default",
    "torch.ops.ao.offload.default",
    "torch.ops.ao.reload.default",
    "torch.ops.ao.wait_tensor.default",
    "torch.ops.aten._scaled_dot_product_flash_attention.default",
    "torch.ops.aten._scaled_dot_product_flash_attention_backward.default",
    "torch.ops.aten.embedding.default",
    "torch.ops.aten.embedding_dense_backward.default",
}

_ALREADY_OPTIMIZED_OPS = {
    "torch.ops.aten.mm.default",
    "torch.ops.aten.bmm.default",
    "torch.ops.aten.addmm.default",
}

_METADATA_OPS = {
    "torch.ops.aten.reshape.default",
    "torch.ops.aten.view.default",
    "torch.ops.aten._unsafe_view.default",
    "torch.ops.aten.t.default",
    "torch.ops.aten.transpose.int",
    "torch.ops.aten.unsqueeze.default",
    "torch.ops.aten.squeeze.dim",
    "torch.ops.aten.squeeze.dims",
    "torch.ops.aten.slice.Tensor",
    "torch.ops.aten.expand.default",
    "torch.ops.aten.permute.default",
    "operator.getitem",
}

_DTYPE_MAP = {
    torch.bfloat16: "torch.bfloat16",
    torch.float16: "torch.float16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.complex64: "torch.complex64",
    torch.complex128: "torch.complex128",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.bool: "torch.bool",
}

_RANDN_DTYPES = {
    torch.bfloat16, torch.float16, torch.float32, torch.float64,
    torch.complex64, torch.complex128,
}


def _get_node_fqn(node: torch.fx.Node) -> str:
    """Extract module_fqn from node metadata."""
    custom = node.meta.get("custom", {})
    if isinstance(custom, dict):
        return custom.get("module_fqn", "")
    return ""


def _normalize_fqn(fqn: str) -> str:
    return re.sub(r"layers\.\d+", "layers.*", fqn)


def _get_tensor_info(node: torch.fx.Node) -> tuple[tuple[int, ...], torch.dtype | None]:
    """Extract shape and dtype from a node's FakeTensor metadata."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape), val.dtype
    if isinstance(val, (tuple, list)) and val and isinstance(val[0], torch.Tensor):
        return tuple(val[0].shape), val[0].dtype
    return (), None


def _is_unfusable(node: torch.fx.Node) -> bool:
    target_str = str(node.target)
    if target_str in _UNFUSABLE_OPS:
        return True
    # Block all _c10d_functional ops
    if "_c10d_functional" in target_str:
        return True
    return False


@dataclass
class _Region:
    nodes: list[torch.fx.Node]
    norm_fqn: str

    @property
    def op_sig(self) -> tuple[str, ...]:
        return tuple(str(n.target) for n in self.nodes)

    @property
    def shape_sig(self) -> tuple[tuple[tuple[int, ...], str], ...]:
        result = []
        for n in self.nodes:
            shape, dtype = _get_tensor_info(n)
            result.append((shape, str(dtype) if dtype else ""))
        return tuple(result)

    @property
    def key(self) -> tuple:
        return (self.norm_fqn, self.op_sig, self.shape_sig)

    @property
    def num_compute_ops(self) -> int:
        return sum(1 for n in self.nodes if str(n.target) not in _METADATA_OPS)

    @property
    def estimated_bytes(self) -> int:
        max_bytes = 0
        for n in self.nodes:
            shape, dtype = _get_tensor_info(n)
            if shape and dtype:
                elem = dtype.itemsize if hasattr(dtype, "itemsize") else 4
                size = 1
                for d in shape:
                    size *= d
                max_bytes = max(max_bytes, size * elem)
        return max_bytes


def _split_connected_fx(nodes: list[torch.fx.Node]) -> list[list[torch.fx.Node]]:
    """Split FX nodes into connected components via union-find."""
    if not nodes:
        return []

    node_set = {n.name for n in nodes}
    name_to_idx = {n.name: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, n in enumerate(nodes):
        for arg in _iter_node_args(n):
            if arg.name in name_to_idx:
                union(i, name_to_idx[arg.name])

    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)

    return [[nodes[i] for i in sorted(idxs)] for idxs in groups.values()]


def _segment_regions(gm: torch.fx.GraphModule) -> list[_Region]:
    """Segment FX graph into fusible regions at fqn boundaries."""
    regions: list[_Region] = []
    current: list[torch.fx.Node] = []
    current_fqn = ""

    def flush() -> None:
        if not current:
            return
        for comp in _split_connected_fx(current):
            # Skip if any node is unfusable
            if any(_is_unfusable(n) for n in comp):
                return
            # Skip if only compute is already-optimized (bare matmul)
            compute = {str(n.target) for n in comp if str(n.target) not in _METADATA_OPS}
            if compute and compute <= {str(t) for t in _ALREADY_OPTIMIZED_OPS}:
                return
            regions.append(_Region(nodes=list(comp), norm_fqn=current_fqn))

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        fqn = _normalize_fqn(_get_node_fqn(node))
        if fqn != current_fqn:
            flush()
            current = []
            current_fqn = fqn
        current.append(node)

    flush()
    return regions


def _format_arg_for_problem(arg: object, input_map: dict[str, str], var_map: dict[str, str]) -> str:
    """Format an FX node argument as Python source for problem.py."""
    if isinstance(arg, torch.fx.Node):
        return input_map.get(arg.name, var_map.get(arg.name, arg.name))
    if isinstance(arg, torch.dtype):
        return str(arg)
    if isinstance(arg, torch.device):
        return f"torch.device('{arg}')"
    if isinstance(arg, torch.memory_format):
        return str(arg)
    if isinstance(arg, torch.layout):
        return str(arg)
    if isinstance(arg, (int, float, bool)):
        return repr(arg)
    if arg is None:
        return "None"
    if isinstance(arg, (list, tuple)):
        elems = [_format_arg_for_problem(a, input_map, var_map) for a in arg]
        if isinstance(arg, tuple):
            return f"({', '.join(elems)}{',' if len(elems) == 1 else ''})"
        return f"[{', '.join(elems)}]"
    return repr(arg)


def _generate_problem_from_region(
    region: _Region,
    count: int,
) -> str:
    """Generate a KernelAgent problem.py from a live FX region."""
    nodes = region.nodes
    node_names = {n.name for n in nodes}

    # External inputs
    external_inputs: list[torch.fx.Node] = []
    seen: set[str] = set()
    for n in nodes:
        for arg in _iter_node_args(n):
            if arg.name not in node_names and arg.name not in seen:
                seen.add(arg.name)
                external_inputs.append(arg)

    input_map = {inp.name: f"input_{i}" for i, inp in enumerate(external_inputs)}
    var_map = {n.name: f"v_{i}" for i, n in enumerate(nodes)}

    # Forward body
    body_lines = []
    for n in nodes:
        var = var_map[n.name]
        target_str = str(n.target)

        if n.target is operator.getitem:
            src = _format_arg_for_problem(n.args[0], input_map, var_map)
            idx = n.args[1]
            shape, dtype = _get_tensor_info(n)
            comment = f"  # {dtype} {list(shape)}" if shape else ""
            body_lines.append(f"        {var} = {src}[{idx}]{comment}")
        else:
            formatted_args = [_format_arg_for_problem(a, input_map, var_map) for a in n.args]
            formatted_kwargs = {
                k: _format_arg_for_problem(v, input_map, var_map)
                for k, v in n.kwargs.items()
            }
            all_args = list(formatted_args)
            all_args.extend(f"{k}={v}" for k, v in formatted_kwargs.items())

            shape, dtype = _get_tensor_info(n)
            comment = f"  # {dtype} {list(shape)}" if shape else ""
            body_lines.append(
                f"        {var} = torch.ops.{target_str}({', '.join(all_args)}){comment}"
            )

    # Outputs: nodes consumed outside the region
    node_names = {n.name for n in nodes}
    output_nodes = []
    for n in nodes:
        for user in n.users:
            if user.name not in node_names:
                if n not in output_nodes:
                    output_nodes.append(n)
                break
    if not output_nodes:
        output_nodes = [nodes[-1]]

    num_outputs = len(output_nodes)
    if num_outputs == 1:
        body_lines.append(f"        return {var_map[output_nodes[0].name]}")
        ret_type = "torch.Tensor"
    else:
        out_vars = ", ".join(var_map[n.name] for n in output_nodes)
        body_lines.append(f"        return ({out_vars})")
        ret_type = f"tuple[{', '.join(['torch.Tensor'] * num_outputs)}]"

    # Parameters and get_inputs
    param_parts = []
    input_lines = []
    return_parts = []
    for inp in external_inputs:
        pname = input_map[inp.name]
        param_parts.append(f"{pname}: torch.Tensor")
        return_parts.append(pname)
        shape, dtype = _get_tensor_info(inp)
        if shape and dtype:
            torch_dtype = _DTYPE_MAP.get(dtype, "torch.float32")
            shape_str = repr(shape)
            if dtype in _RANDN_DTYPES:
                input_lines.append(f"    {pname} = torch.randn({shape_str}, dtype={torch_dtype}, device='cuda')")
            elif dtype == torch.bool:
                input_lines.append(f"    {pname} = torch.randint(0, 2, {shape_str}, dtype={torch_dtype}, device='cuda')")
            else:
                input_lines.append(f"    {pname} = torch.randint(0, 10, {shape_str}, dtype={torch_dtype}, device='cuda')")
        else:
            input_lines.append(f"    {pname} = torch.randn((1,), dtype=torch.float32, device='cuda')")

    compute_ops = [
        str(n.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")
        for n in nodes if str(n.target) not in _METADATA_OPS
    ]
    slug = "_".join(compute_ops[:4]) if compute_ops else "reshape_chain"
    desc = f"Fused region ({region.norm_fqn}): {' -> '.join(compute_ops) if compute_ops else 'reshape chain'}\n"
    desc += f"Instances: {count}. Ops: {len(nodes)}, compute: {len(compute_ops)}, outputs: {num_outputs}.\n"

    return desc + f"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, {', '.join(param_parts)}) -> {ret_type}:
{chr(10).join(body_lines)}

def get_inputs():
{chr(10).join(input_lines) if input_lines else '    pass'}
    return [{', '.join(return_parts)}]

def get_init_inputs():
    return []
""", slug


class ExtractFusedKernelsExit(Exception):
    """Raised after extraction to stop training without error."""
    pass


def extract_fused_kernels_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    output_dir: str = "",
    min_ops: int = 2,
    min_compute_ops: int = 1,
    min_count: int = 2,
    top_n: int = 20,
) -> torch.fx.GraphModule:
    """Extract fusible patterns from the live FX graph and write problem.py files.

    This pass runs on the real GraphModule with FakeTensor metadata, avoiding
    the fragile text-dump round-trip. After writing problems, raises
    ``ExtractFusedKernelsExit`` to stop training.
    """
    if not output_dir:
        return gm

    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting fusible patterns from live FX graph...")

    regions = _segment_regions(gm)
    logger.info(f"  Total regions (after fqn split + connectivity): {len(regions)}")

    # Group by key
    by_key: dict[tuple, list[_Region]] = {}
    for r in regions:
        by_key.setdefault(r.key, []).append(r)

    logger.info(f"  Unique (fqn, ops, shapes) signatures: {len(by_key)}")

    # Filter and rank
    candidates: list[tuple[int, int, int, _Region]] = []
    for key, instances in by_key.items():
        rep = instances[0]
        count = len(instances)
        if len(rep.nodes) < min_ops:
            continue
        if rep.num_compute_ops < min_compute_ops:
            continue
        if count < min_count:
            continue
        candidates.append((count, rep.num_compute_ops, rep.estimated_bytes, rep))

    candidates.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
    top = candidates[:top_n]

    logger.info(f"  Top {len(top)} fusible regions:")
    written = 0
    for i, (count, num_compute, est_bytes, region) in enumerate(top):
        problem_text, slug = _generate_problem_from_region(region, count)
        dir_name = f"{i:02d}_{slug}"
        problem_dir = out / dir_name
        problem_dir.mkdir(parents=True, exist_ok=True)
        (problem_dir / "problem.py").write_text(problem_text)
        written += 1
        logger.info(
            f"    {i:>2}  {count:>4}×  {len(region.nodes):>3} ops  "
            f"{num_compute:>2} compute  {est_bytes:>12,} bytes  {dir_name}"
        )

    logger.info(f"  Wrote {written} problem files to {out}/")
    raise ExtractFusedKernelsExit(f"Extracted {written} patterns to {out}")
