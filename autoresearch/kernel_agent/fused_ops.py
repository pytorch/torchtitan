"""Fused op registry: backend-agnostic custom ops with lightweight dispatch.

Offline step (kernel gen):
  1. extract_from_dump.py discovers fusible regions → problem.py
  2. Various backends produce implementations (Triton, torch.compile, ...)
  3. benchmark_all.py measures each → benchmark.json
  4. Results stored in generated/<pattern>/

Runtime step (graph pass):
  1. Registry loads patterns + implementations from generated/
  2. A graph pass replaces matched aten subgraphs with custom ops
  3. Each custom op dispatches to the selected backend at call time

Usage:
  registry = FusedOpRegistry.from_generated("autoresearch/kernel_agent/generated")
  # Select backend globally or per-op
  registry.set_default_backend("triton")
  registry.set_backend("00_rope_bwd", "compile")
  # Create and apply the graph pass
  pass_fn = registry.make_graph_pass()
  gm = pass_fn(gm, example_inputs)
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

logger = logging.getLogger(__name__)

_lib = torch.library.Library("fused", "DEF")
_registered_schemas: set[str] = set()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Implementation:
    """A single backend implementation of a fused op."""
    backend: str          # "triton", "compile", "eager"
    fn: Callable          # the callable: kernel_function or compiled fn
    time_ms: float = float("inf")  # benchmark result (lower is better)


@dataclass
class FusedOp:
    """A fused op with multiple backend implementations."""
    name: str
    signature: tuple[str, ...]      # aten op targets identifying the subgraph
    input_specs: list[dict]         # shape/dtype for each input
    reference_fn: Callable | None = None  # eager PyTorch reference (Model.forward)
    implementations: dict[str, Implementation] = field(default_factory=dict)
    _active_backend: str | None = None

    @property
    def active_backend(self) -> str | None:
        return self._active_backend

    @active_backend.setter
    def active_backend(self, backend: str) -> None:
        if backend not in self.implementations:
            available = list(self.implementations.keys())
            raise ValueError(f"Backend '{backend}' not available for {self.name}. Available: {available}")
        self._active_backend = backend

    @property
    def best_backend(self) -> str | None:
        """Backend with lowest benchmark time."""
        if not self.implementations:
            return None
        return min(self.implementations, key=lambda k: self.implementations[k].time_ms)

    def dispatch(self, *args: Any) -> Any:
        """Call the active backend (or best if none set)."""
        backend = self._active_backend or self.best_backend
        if backend is None:
            raise RuntimeError(f"No implementations registered for {self.name}")
        return self.implementations[backend].fn(*args)

    @property
    def num_inputs(self) -> int:
        return len(self.input_specs)


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

def _load_module_from_file(path: Path, module_name: str) -> types.ModuleType:
    content = path.read_text()
    idx = content.find("import torch")
    if idx > 0:
        content = content[idx:]
    mod = types.ModuleType(module_name)
    mod.__file__ = str(path)
    exec(compile(content, str(path), "exec"), mod.__dict__)
    return mod


def _load_kernel_fn(kernel_path: Path) -> Callable | None:
    """Load kernel_function from a kernel.py file."""
    if not kernel_path.exists():
        return None
    try:
        mod = _load_module_from_file(kernel_path, f"_kernel_{kernel_path.parent.name}")
        return getattr(mod, "kernel_function", None)
    except Exception as e:
        logger.warning(f"Failed to load {kernel_path}: {e}")
        return None


def _load_reference(problem_path: Path) -> tuple[Callable | None, list[dict]]:
    """Load Model.forward and get_inputs from problem.py."""
    if not problem_path.exists():
        return None, []
    try:
        mod = _load_module_from_file(problem_path, f"_problem_{problem_path.parent.name}")
        model_cls = getattr(mod, "Model", None)
        get_inputs = getattr(mod, "get_inputs", None)
        if model_cls is None or get_inputs is None:
            return None, []
        model = model_cls()
        ref_fn = model.forward
        # Extract input specs from get_inputs
        inputs = get_inputs()
        specs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                specs.append({"shape": tuple(inp.shape), "dtype": str(inp.dtype)})
            else:
                specs.append({})
        return ref_fn, specs
    except Exception as e:
        logger.warning(f"Failed to load {problem_path}: {e}")
        return None, []


def _make_compiled_fn(ref_fn: Callable) -> Callable | None:
    """Create a torch.compile'd version of the reference function."""
    try:
        return torch.compile(ref_fn, fullgraph=True)
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Op signature extraction from problem.py
# ---------------------------------------------------------------------------

def _extract_signature(problem_path: Path) -> tuple[str, ...]:
    """Extract the op target sequence from a problem.py forward body.

    Strips ``torch.ops.`` prefix so signatures match ``str(node.target)``
    on FX graph nodes.
    """
    content = problem_path.read_text()
    raw = re.findall(r'torch\.ops\.(\S+)\(', content)
    return tuple(raw)


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

def _sanitize_op_name(name: str) -> str:
    """Make a name safe for torch.library schema (must be a valid identifier)."""
    # Strip leading digits/underscores
    clean = re.sub(r'^[\d_]+', '', name)
    if not clean:
        clean = "op"
    # Replace any remaining non-alphanumeric chars
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', clean)
    return f"f_{clean}"


def _register_custom_op(fused_op: FusedOp) -> Callable:
    """Register a fused op as torch.ops.fused.<name>.

    The op dispatches to fused_op.dispatch() at call time, which picks
    the active backend.
    """
    name = _sanitize_op_name(fused_op.name)
    if name in _registered_schemas:
        return getattr(torch.ops.fused, name)

    # Build schema: (Tensor x0, Tensor x1, ...) -> Tensor
    n = fused_op.num_inputs
    args = ", ".join(f"Tensor x{i}" for i in range(n))
    schema = f"{name}({args}) -> Tensor"
    _lib.define(schema)

    # CUDA impl: dispatch to active backend
    def cuda_impl(*args: torch.Tensor) -> torch.Tensor:
        return fused_op.dispatch(*args)

    # Meta impl: return empty tensor with correct shape/dtype (inferred from first call)
    def meta_impl(*args: torch.Tensor) -> torch.Tensor:
        # Run reference to get output shape/dtype
        if fused_op.reference_fn is not None:
            return fused_op.reference_fn(*args)
        return torch.empty_like(args[0])

    _lib.impl(name, cuda_impl, "CUDA")
    _lib.impl(name, meta_impl, "Meta")

    _registered_schemas.add(name)
    return getattr(torch.ops.fused, name)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class FusedOpRegistry:
    """Registry of fused ops with backend dispatch."""

    def __init__(self) -> None:
        self.ops: dict[str, FusedOp] = {}
        self._default_backend: str | None = None

    def register(self, op: FusedOp) -> None:
        self.ops[op.name] = op
        if self._default_backend and self._default_backend in op.implementations:
            op.active_backend = self._default_backend

    def set_default_backend(self, backend: str) -> None:
        """Set the default backend for all ops that support it."""
        self._default_backend = backend
        for op in self.ops.values():
            if backend in op.implementations:
                op.active_backend = backend

    def set_backend(self, op_name: str, backend: str) -> None:
        """Override the backend for a specific op."""
        self.ops[op_name].active_backend = backend

    @classmethod
    def from_generated(cls, generated_dir: str | Path) -> FusedOpRegistry:
        """Load all fused ops from the generated/ directory structure.

        Expected layout per pattern:
          generated/<name>/
            problem.py              # reference Model + get_inputs
            kernel.py               # Triton kernel (optional)
            optimized_kernel.py     # NCU-optimized Triton (optional)
            benchmark.json          # timing results (optional)
        """
        registry = cls()
        generated = Path(generated_dir)
        if not generated.exists():
            return registry

        for d in sorted(generated.iterdir()):
            if not d.is_dir():
                continue
            problem_path = d / "problem.py"
            if not problem_path.exists():
                continue

            name = d.name
            signature = _extract_signature(problem_path)
            ref_fn, input_specs = _load_reference(problem_path)

            op = FusedOp(
                name=name,
                signature=signature,
                input_specs=input_specs,
                reference_fn=ref_fn,
            )

            # Load benchmark results if available
            bench_path = d / "benchmark.json"
            bench_data = {}
            if bench_path.exists():
                try:
                    bench_data = json.loads(bench_path.read_text())
                except Exception:
                    pass

            # Register eager backend (the reference itself)
            if ref_fn is not None:
                op.implementations["eager"] = Implementation(
                    backend="eager",
                    fn=ref_fn,
                    time_ms=bench_data.get("eager_ms", float("inf")),
                )

            # Register torch.compile backend
            if ref_fn is not None:
                compiled = _make_compiled_fn(ref_fn)
                if compiled is not None:
                    op.implementations["compile"] = Implementation(
                        backend="compile",
                        fn=compiled,
                        time_ms=bench_data.get("compile_ms", float("inf")),
                    )

            # Register Triton backend (prefer optimized over initial)
            for kernel_name in ("optimized_kernel.py", "kernel.py"):
                kernel_fn = _load_kernel_fn(d / kernel_name)
                if kernel_fn is not None:
                    op.implementations["triton"] = Implementation(
                        backend="triton",
                        fn=kernel_fn,
                        time_ms=bench_data.get("triton_ms", float("inf")),
                    )
                    break

            if op.implementations:
                registry.register(op)
                logger.info(
                    f"Registered {name}: {list(op.implementations.keys())}, "
                    f"best={op.best_backend}"
                )

        return registry

    # -------------------------------------------------------------------
    # Graph pass generation
    # -------------------------------------------------------------------

    def make_graph_pass(
        self,
    ) -> Callable[[torch.fx.GraphModule, tuple], torch.fx.GraphModule]:
        """Create a graph pass that replaces matched subgraphs with fused ops.

        Returns a pass with the standard signature:
            def pass_fn(gm, example_inputs) -> gm
        """
        # Pre-register all ops as torch.library custom ops
        op_map: dict[str, tuple[FusedOp, Any]] = {}
        for name, fused_op in self.ops.items():
            custom_op = _register_custom_op(fused_op)
            op_map[name] = (fused_op, custom_op)

        # Build a lookup: signature → (fused_op, custom_op)
        sig_to_op: dict[tuple[str, ...], tuple[FusedOp, Any]] = {}
        for name, (fused_op, custom_op) in op_map.items():
            sig_to_op[fused_op.signature] = (fused_op, custom_op)

        def fused_kernel_pass(
            gm: torch.fx.GraphModule,
            example_inputs: tuple | None = None,
        ) -> torch.fx.GraphModule:
            """Replace matched aten subgraphs with fused custom ops."""
            graph = gm.graph
            nodes = list(graph.nodes)
            replaced = 0

            # Sliding window: try to match signatures starting at each node
            i = 0
            while i < len(nodes):
                node = nodes[i]
                if node.op != "call_function":
                    i += 1
                    continue

                # Try each registered signature
                for sig, (fused_op, custom_op) in sig_to_op.items():
                    if i + len(sig) > len(nodes):
                        continue

                    # Check if the window matches the signature
                    window = nodes[i:i + len(sig)]
                    window_targets = tuple(
                        str(n.target) if n.op == "call_function" else ""
                        for n in window
                    )
                    if window_targets != sig:
                        continue

                    # Collect external inputs to the matched subgraph
                    matched_names = {n.name for n in window}
                    external_inputs = []
                    for n in window:
                        for arg in _iter_node_args(n):
                            if isinstance(arg, torch.fx.Node) and arg.name not in matched_names:
                                if arg not in external_inputs:
                                    external_inputs.append(arg)

                    # Insert the fused op call after the last matched node
                    last_node = window[-1]
                    with graph.inserting_after(last_node):
                        new_node = graph.call_function(
                            custom_op, args=tuple(external_inputs)
                        )
                        new_node.meta = last_node.meta.copy()

                    # Replace uses of the last matched node
                    last_node.replace_all_uses_with(new_node)
                    # Fix self-reference
                    new_node.replace_input_with(new_node, last_node)

                    # Remove matched nodes in reverse order
                    for old in reversed(window):
                        if not old.users:
                            graph.erase_node(old)

                    # Refresh node list after modification
                    nodes = list(graph.nodes)
                    replaced += 1
                    break  # restart scan from current position
                else:
                    i += 1

            if replaced > 0:
                graph.lint()
                gm.recompile()
                logger.info(f"Fused kernel pass: replaced {replaced} subgraphs")

            return gm

        return fused_kernel_pass


def _iter_node_args(node: torch.fx.Node):
    """Yield all Node-typed arguments."""
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
