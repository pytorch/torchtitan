# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused kernel registry and replacement pass.

Provides a general framework for replacing aten subgraphs in the FX graph
with optimized kernel implementations. The kernels can come from any backend:
Triton, torch.compile, Helion, handwritten CUDA, etc.

The registry is populated by loading a directory of generated kernels
(see ``load_from_dir``). Each kernel directory contains:
  - ``problem.py``: reference PyTorch implementation (``Model.forward``)
  - ``kernel.py`` or ``optimized_kernel.py``: Triton implementation (optional)
  - ``helion_kernel.py``: Helion implementation (optional)
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

    backend: str  # "triton", "compile", "eager", "helion"
    fn: Callable
    time_ms: float = float("inf")


@dataclass
class FusedOp:
    """A fused op with multiple backend implementations and dispatch."""

    name: str
    signature: list[OpSig]  # op targets + shapes identifying the subgraph
    num_inputs: int
    reference_fn: Callable | None = None
    implementations: dict[str, Implementation] = field(default_factory=dict)
    _active_backend: str | None = None

    @property
    def active_backend(self) -> str | None:
        return self._active_backend

    @active_backend.setter
    def active_backend(self, backend: str) -> None:
        if backend not in self.implementations:
            available = list(self.implementations.keys())
            raise ValueError(
                f"Backend '{backend}' not available for {self.name}. "
                f"Available: {available}"
            )
        self._active_backend = backend

    @property
    def best_backend(self) -> str | None:
        if not self.implementations:
            return None
        return min(
            self.implementations, key=lambda k: self.implementations[k].time_ms
        )

    def dispatch(self, *args: Any) -> Any:
        backend = self._active_backend or self.best_backend
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
    schema = f"{op_name}({args}) -> Tensor"
    _lib.define(schema)

    def cuda_impl(*args: torch.Tensor) -> torch.Tensor:
        return fused_op.dispatch(*args)

    def meta_impl(*args: torch.Tensor) -> torch.Tensor:
        if fused_op.reference_fn is not None:
            try:
                return fused_op.reference_fn(*args)
            except Exception:
                pass
        return torch.empty_like(args[0])

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
        self._default_backend: str | None = None

    def register(self, op: FusedOp) -> None:
        self.ops[op.name] = op
        if self._default_backend and self._default_backend in op.implementations:
            op.active_backend = self._default_backend

    def set_default_backend(self, backend: str) -> None:
        self._default_backend = backend
        for op in self.ops.values():
            if backend in op.implementations:
                op.active_backend = backend

    def set_backend(self, op_name: str, backend: str) -> None:
        self.ops[op_name].active_backend = backend

    def load_from_dir(self, generated_dir: str | Path) -> None:
        """Load fused ops from a directory of generated kernels.

        Expected layout per pattern:
          generated/<name>/
            problem.py              # reference Model + get_inputs
            kernel.py               # Triton kernel (optional)
            optimized_kernel.py     # NCU-optimized Triton (optional)
            helion_kernel.py        # Helion kernel (optional)
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

            # Load reference
            ref_fn = None
            num_inputs = 0
            try:
                mod = _load_module_from_file(problem_path, f"_prob_{name}")
                model_cls = getattr(mod, "Model", None)
                get_inputs_fn = getattr(mod, "get_inputs", None)
                if model_cls and get_inputs_fn:
                    ref_fn = model_cls().forward
                    num_inputs = len(get_inputs_fn())
            except Exception as e:
                logger.debug(f"Failed to load reference from {problem_path}: {e}")
                continue

            op = FusedOp(
                name=name,
                signature=signature,
                num_inputs=num_inputs,
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

            # Register Helion backend
            helion_fn = _load_kernel_fn(d / "helion_kernel.py")
            if helion_fn is not None:
                op.implementations["helion"] = Implementation(
                    backend="helion",
                    fn=helion_fn,
                    time_ms=bench.get("helion_ms", float("inf")),
                )

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

            last_node = window[-1]
            with graph.inserting_after(last_node):
                new_node = graph.call_function(
                    custom_op, args=tuple(external_inputs)
                )
                new_node.meta = last_node.meta.copy()

            last_node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, last_node)

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
