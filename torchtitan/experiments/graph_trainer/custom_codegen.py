# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom FX codegen pass: save generated code to disk with dual-path profiling."""

from __future__ import annotations

import ast
import os
import re
import sys
from typing import Any

import torch
import torch.distributed as dist
from torch._logging import trace_structured

from torchtitan.tools.logging import logger


class _CodegenGraphModule(torch.fx.GraphModule):
    """GraphModule that saves generated code to disk with two-path profiling support.

    Extends torch.fx.GraphModule to save generated code to disk and automatically
    reload it when modified. Enables inspection, breakpoints, and code modifications
    for experimentation.

    Always generates two paths:
    1. profiler-enabled with _RecordFunctionFast for stack trace
    2. profiler-disabled runs directly
    At runtime, the file is loaded and selects path with
    torch.autograd.profiler._is_profiler_enabled.

    Subgraph modules are included as module-level functions in a single file,
    loaded and executed at runtime via compile() + exec().
    """

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute SHA256 hash of content for change detection."""
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _write_code_atomically(path: str, content: str) -> None:
        """Atomically write to avoid race conditions in distributed setups."""
        import tempfile as _tempfile

        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        with _tempfile.NamedTemporaryFile(
            mode="w", dir=dir_path, delete=False, suffix=".py.tmp"
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _get_rank_suffix() -> str:
        """Returns rank suffix for distributed setups, empty string otherwise."""
        if dist.is_initialized():
            return f"_rank{dist.get_rank()}"
        return ""

    @staticmethod
    def _get_graph_module_code(gm: torch.fx.GraphModule) -> str:
        """Extract source code from a GraphModule.

        Always generates fresh code from the graph IR via python_code()
        WITHOUT record_func, to get clean code free of profiler instrumentation.

        This is necessary because GraphModule.recompile() may inject
        _RecordFunctionFast into ._code when enrich_profiler_metadata=True.
        Our custom codegen generates its own per-op profiling in
        _forward_profiled, so the base code must be clean.
        """
        code = gm.graph.python_code("self").src
        assert "_RecordFunctionFast" not in code, (
            "Expected clean graph IR code without _RecordFunctionFast. "
            "graph.python_code('self') should not inject profiler instrumentation. "
            "If this fires, check if python_code() default for record_func changed."
        )
        return code

    def __init__(
        self,
        root: torch.nn.Module,
        graph: torch.fx.Graph,
        codegen_dir: str,
        class_name: str = "_CodegenGraphModule",
        subgraph_modules: dict[str, torch.fx.GraphModule] | None = None,
    ):
        assert codegen_dir, "codegen dir path must be provided."
        self._codegen_dir = codegen_dir
        self._code_path: str | None = None
        self._code_hash: str | None = None
        self._python_code_globals: dict[str, Any] | None = None
        self._custom_codegen_initialized = False
        self._subgraph_modules = subgraph_modules or {}

        super().__init__(root, graph, class_name)

        # Get python_code with globals before any modifications
        python_code = graph.python_code("self")
        self._python_code_globals = python_code.globals

        # Also collect globals from subgraph modules
        for subgraph in self._subgraph_modules.values():
            subgraph_python_code = subgraph.graph.python_code("self")
            if subgraph_python_code.globals:
                self._python_code_globals.update(subgraph_python_code.globals)

        self._setup_custom_codegen()
        self._custom_codegen_initialized = True
        self._root = root

    def _extract_imports(self, code: str) -> str:
        """Extract import statements from code."""
        return "\n".join(
            line
            for line in code.split("\n")
            if line.strip().startswith(("import ", "from "))
        )

    def _find_forward_boundaries(self, code: str) -> tuple[int, int]:
        """Use AST to find forward function's start and end line numbers.

        Returns:
            Tuple of (start_line, end_line) using 1-based line numbers.
            Returns (0, 0) if forward not found.
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "forward":
                    start_line = node.lineno
                    end_line = node.end_lineno if node.end_lineno else start_line
                    return (start_line, end_line)
        except SyntaxError:
            pass
        return (0, 0)

    def _generate_wrapper_code(self) -> str:
        """Generate wrapper code with dual-path profiling.

        Strategy: Keep the ENTIRE original code as-is, just rename 'forward'
        to '_forward_impl', then add a new 'forward' that dispatches based on
        profiler state.
        """
        original_code = self._get_graph_module_code(self)

        original_imports = self._extract_imports(original_code)

        # Collect imports from subgraph modules
        subgraph_imports = []
        for subgraph in self._subgraph_modules.values():
            subgraph_code = self._get_graph_module_code(subgraph)
            subgraph_imports.append(self._extract_imports(subgraph_code))

        all_imports = "\n".join([original_imports] + subgraph_imports)
        # Deduplicate import lines
        import_lines = list(dict.fromkeys(all_imports.split("\n")))
        all_imports = "\n".join(line for line in import_lines if line.strip())

        header = f"""# Auto-generated by _CodegenGraphModule
# Runtime-conditional profiling: uses _RecordFunctionFast when profiler is enabled

import operator
import torch
import torch.fx
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx.node import map_aggregate
from torch import device, dtype
import torch.autograd.profiler as _autograd_profiler

{all_imports}

"""

        code_lines = [
            line
            for line in original_code.split("\n")
            if not line.strip().startswith(("import ", "from "))
        ]
        code_without_imports = "\n".join(code_lines)

        modified_code = code_without_imports.replace(
            "def forward(", "def _forward_impl(", 1
        )

        profiled_forward = self._generate_profiled_forward()

        footer = """

def forward(self, *args, **kwargs):
    '''Dispatch to profiled or non-profiled path based on profiler state.'''
    if _autograd_profiler._is_profiler_enabled:
        return _forward_profiled(self, *args, **kwargs)
    else:
        return _forward_impl(self, *args, **kwargs)

def call(self, *args, **kwargs):
    '''Entry point for FX graph execution. Receives self from _CodegenGraphModule.'''
    return forward(self, *args, **kwargs)
"""

        main_module_code = header + modified_code + "\n" + profiled_forward + footer

        # Generate module-level functions for each subgraph (loaded at runtime)
        subgraph_code_sections = []
        for subgraph_name, subgraph in self._subgraph_modules.items():
            subgraph_section = self._generate_subgraph_functions(
                subgraph_name, subgraph
            )
            subgraph_code_sections.append(subgraph_section)

        if subgraph_code_sections:
            subgraph_code = "\n".join(subgraph_code_sections)
            return main_module_code + subgraph_code
        else:
            return main_module_code

    def _generate_subgraph_functions(
        self, subgraph_name: str, subgraph: torch.fx.GraphModule
    ) -> str:
        """Generate module-level functions for a subgraph module.

        These functions are loaded at runtime and bound to the subgraph's
        forward method, enabling dual-path profiling for subgraphs just like
        the main module.
        """
        subgraph_code = self._get_graph_module_code(subgraph)

        # Remove imports (they're already in the header)
        code_lines = [
            line
            for line in subgraph_code.split("\n")
            if not line.strip().startswith(("import ", "from "))
        ]
        code_without_imports = "\n".join(code_lines)

        # Rename forward to _{name}_forward_impl
        modified_code = code_without_imports.replace(
            "def forward(", f"def _{subgraph_name}_forward_impl(", 1
        )

        # Generate profiled forward for the subgraph
        profiled_forward = self._generate_profiled_forward(
            code=subgraph_code, graph=subgraph.graph
        )
        # Rename _forward_profiled to _{name}_forward_profiled
        profiled_forward = profiled_forward.replace(
            "def _forward_profiled(", f"def _{subgraph_name}_forward_profiled(", 1
        )

        # Build the dispatcher function
        dispatcher = "\n".join(
            [
                f"def _{subgraph_name}_forward(self, *args, **kwargs):",
                '    """Dispatch to profiled or non-profiled path based on profiler state."""',
                "    if _autograd_profiler._is_profiler_enabled:",
                f"        return _{subgraph_name}_forward_profiled(self, *args, **kwargs)",
                f"    return _{subgraph_name}_forward_impl(self, *args, **kwargs)",
            ]
        )

        header = f"\n# ===== Subgraph: {subgraph_name} ====="
        return f"{header}\n{modified_code.strip()}\n\n{profiled_forward.strip()}\n\n{dispatcher}\n"

    def _generate_profiled_forward(
        self,
        code: str | None = None,
        graph: torch.fx.Graph | None = None,
    ) -> str:
        """Generate _forward_profiled with _RecordFunctionFast wrapping each op.

        Uses AST to reliably find forward function boundaries instead of
        fragile string matching.

        Args:
            code: Source code to generate profiled forward from. Defaults to
                fresh code from graph IR.
            graph: FX graph to get node info from. Defaults to self.graph.
        """
        original_code = (code or self._get_graph_module_code(self)).strip()
        target_graph = graph or self.graph
        original_lines = original_code.split("\n")

        # Use AST to find forward function boundaries
        start_line, end_line = self._find_forward_boundaries(original_code)
        if start_line == 0:
            return ""

        # Extract forward function lines (1-based to 0-based index)
        forward_lines = original_lines[start_line - 1 : end_line]

        # Get signature line and body
        signature_line = forward_lines[0].replace(
            "def forward(", "def _forward_profiled("
        )
        body_lines = forward_lines[1:]

        node_info = {
            node.name: node
            for node in target_graph.nodes
            if node.op in ("call_function", "call_method", "call_module")
        }

        # Pre-compile a single regex to extract the leading identifier from
        # assignment lines (``name = ...``) or tuple-unpacking lines
        # (``name, ...``).  This replaces the previous O(lines × nodes)
        # per-node regex scan with O(lines) extraction + O(1) dict lookup.
        _leading_ident_re = re.compile(r"^([A-Za-z_]\w*)(?:\s*=\s|\s*,)")

        profiled_lines = [signature_line]
        for line in body_lines:
            stripped = line.strip()
            if not stripped:
                profiled_lines.append(line)
                continue

            indent = line[: len(line) - len(line.lstrip())]

            matched_node = None
            m = _leading_ident_re.match(stripped)
            if m:
                matched_node = node_info.get(m.group(1))

            if matched_node:
                label = self._get_profiler_label(matched_node).replace('"', '\\"')
                args_tuple = self._format_args_tuple(matched_node.args)
                profiled_lines.append(
                    f'{indent}with torch._C._profiler._RecordFunctionFast("{label}", {args_tuple}):'
                )
                profiled_lines.append(f"{indent}    {stripped}")
            else:
                profiled_lines.append(line)

        return "\n".join(profiled_lines)

    def _get_op_name(self, node: torch.fx.Node) -> str:
        """Extract readable operation name from node."""
        if node.op == "call_function":
            if hasattr(node.target, "__name__"):
                return node.target.__name__
            elif hasattr(node.target, "_name"):
                return node.target._name
            return str(node.target).split(".")[-1]
        return str(node.target)

    def _get_profiler_label(self, node: torch.fx.Node) -> str:
        """Generate profiler label: 'node_name: op_name (file:line)'."""
        op_name = self._get_op_name(node)
        stack_info = self._get_stack_trace_summary(node)
        if stack_info:
            return f"{node.name}: {op_name} ({stack_info})"
        return f"{node.name}: {op_name}"

    def _format_args_tuple(self, args: tuple) -> str:
        """Prepare args as a tuple expression for _RecordFunctionFast."""
        parts = []

        def collect_nodes(item):
            """Recursively collect fx.Node names from nested structures."""
            if isinstance(item, torch.fx.Node):
                parts.append(item.name)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    collect_nodes(sub_item)

        for arg in args:
            collect_nodes(arg)

        if not parts:
            return "()"
        return f"({', '.join(parts)},)"

    def _get_stack_trace_summary(self, node: torch.fx.Node) -> str:
        """Extract filename:line from node stack trace for profiling labels."""
        stack_trace = node.meta.get("stack_trace", None)
        if not stack_trace:
            return ""

        pattern = re.compile(r'^File "(.+)", line (\d+), in (.+)$')
        lines = stack_trace.strip().split("\n")

        for idx in range(len(lines) - 2, -1, -1):
            line = lines[idx].strip()
            matches = pattern.match(line)
            if matches:
                filepath = matches.group(1)
                lineno = matches.group(2)

                if "/torch/" in filepath or "/site-packages/" in filepath:
                    continue

                filename = filepath.split("/")[-1]
                return f"{filename}:{lineno}"

        return ""

    def _setup_custom_codegen(self) -> None:
        """Generate code, save to disk, and load forward method."""
        wrapper_code = self._generate_wrapper_code()
        code_hash = self._compute_hash(wrapper_code)
        self._code_hash = code_hash

        rank_suffix = self._get_rank_suffix()
        key = f"fx_{code_hash}{rank_suffix}"
        self._code_path = os.path.join(self._codegen_dir, f"{key}.py")

        if os.path.exists(self._code_path):
            with open(self._code_path, "r") as f:
                existing_code = f.read()
            existing_hash = self._compute_hash(existing_code)
            self._code_hash = existing_hash
            logger.info("[CUSTOM_CODEGEN] Loading from disk: %s", self._code_path)
        else:
            self._write_code_atomically(self._code_path, wrapper_code)
            logger.info("[CUSTOM_CODEGEN] Dumped new file: %s", self._code_path)

        # Send generated code to tlparse via trace_structured
        self._trace_structured_codegen(wrapper_code)

        self._reload_forward_from_disk()

    def _trace_structured_codegen(self, code: str) -> None:
        """Send generated code to tlparse via trace_structured."""
        code_path = self._code_path
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_codegen",
                "encoding": "string",
                "filename": os.path.basename(code_path) if code_path else "unknown",
                "file_path": os.path.abspath(code_path) if code_path else "",
            },
            payload_fn=lambda: code,
            expect_trace_id=False,
        )

    def _reload_forward_from_disk(self) -> None:
        """Load or reload the forward method from disk file."""
        import importlib

        assert self._code_path is not None

        with open(self._code_path, "r") as f:
            current_code = f.read()
        new_hash = self._compute_hash(current_code)

        importlib.invalidate_caches()

        # Use compile() + exec() to avoid bytecode caching issues
        module = type(sys.modules[__name__])("fx_dynamic_module")
        module.__file__ = self._code_path

        for name, obj in self.__dict__.items():
            if not name.startswith("_") and not callable(obj):
                setattr(module, name, obj)

        # Inject globals from python_code() for proper variable resolution
        if self._python_code_globals is not None:
            for name, obj in self._python_code_globals.items():
                if name.startswith("__") or name in module.__dict__:
                    continue
                setattr(module, name, obj)

        code_obj = compile(current_code, self._code_path, "exec")
        exec(code_obj, module.__dict__)  # noqa: S102

        if not hasattr(module, "call"):
            raise RuntimeError(
                f"Generated code at {self._code_path} must define a 'call' function"
            )

        loaded_call_fn = module.call
        self._code_hash = new_hash
        self._code_mtime = os.path.getmtime(self._code_path)

        logger.info(
            "[CUSTOM_CODEGEN] Loaded module from %s, hash: %s",
            self._code_path,
            new_hash[:8],
        )

        def forward_wrapper(self, *args, **kwargs):
            if self._check_file_modified():
                logger.warning(
                    "[CUSTOM_CODEGEN] Detected modification, reloading: %s",
                    self._code_path,
                )
                self._reload_forward_from_disk()
                return self.forward(*args, **kwargs)
            return loaded_call_fn(self, *args, **kwargs)

        self.forward = forward_wrapper.__get__(self, type(self))

        # Bind subgraph forward functions to their respective modules
        for subgraph_name in self._subgraph_modules:
            forward_fn_name = f"_{subgraph_name}_forward"
            if hasattr(module, forward_fn_name):
                loaded_forward = getattr(module, forward_fn_name)
                subgraph_module = getattr(self, subgraph_name, None)
                if subgraph_module is not None:
                    subgraph_module.forward = loaded_forward.__get__(
                        subgraph_module, type(subgraph_module)
                    )
                    logger.info(
                        "[CUSTOM_CODEGEN] Bound forward for subgraph: %s",
                        subgraph_name,
                    )

    def _check_file_modified(self) -> bool:
        """Check if the code file has been modified since last load.

        Uses file modification time as a cheap first check before reading
        and hashing file contents for better performance.
        """
        if self._code_path is None or not os.path.exists(self._code_path):
            return False

        current_mtime = os.path.getmtime(self._code_path)
        if hasattr(self, "_code_mtime") and current_mtime == self._code_mtime:
            return False

        # mtime changed, check if content actually changed
        with open(self._code_path, "r") as f:
            current_code = f.read()

        if self._compute_hash(current_code) != self._code_hash:
            return True

        # Content unchanged, update mtime cache
        self._code_mtime = current_mtime
        return False

    def recompile(self):
        """Recompile graph from IR."""
        return super().recompile()

    def __getstate__(self):
        """When pickling we just pickle the underlying root module since it's
        unsafe to depend on file system references."""
        return self._root.__dict__.copy()


def custom_codegen_pass(
    gm: torch.fx.GraphModule,
    example_inputs: list[Any] | None = None,
    *,
    codegen_dir: str | None = None,
) -> torch.fx.GraphModule:
    """Save generated code to disk with dual-path profiling.

    Always generates both profiler and non-profiler paths:
    - If torch.autograd.profiler enabled: uses _RecordFunctionFast for
      minimal overhead profiling
    - If torch.autograd.profiler disabled: direct execution with zero overhead

    Args:
        gm: Input graph module.
        example_inputs: Placeholder arg for compiler signature compatibility.
        codegen_dir: Directory for generated code files. Defaults to
            ``<tempdir>/torchtitan_fx_codegen``.
    """
    import tempfile

    if codegen_dir is None:
        codegen_dir = os.path.join(tempfile.gettempdir(), "torchtitan_fx_codegen")

    logger.info("[CUSTOM_CODEGEN] Saving code to %s", codegen_dir)

    if isinstance(gm, _CodegenGraphModule):
        return gm

    # Collect all subgraph modules (single file with module-level functions)
    subgraph_modules: dict[str, torch.fx.GraphModule] = {}
    subgraph_names = set()
    for name, child in gm.named_children():
        if isinstance(child, torch.fx.GraphModule) and hasattr(child, "graph"):
            subgraph_modules[name] = child
            subgraph_names.add(name)
            logger.info("[CUSTOM_CODEGEN] Found subgraph: %s", name)

    # Create single _CodegenGraphModule with all subgraphs included
    custom_gm = _CodegenGraphModule(
        root=gm,
        graph=gm.graph,
        codegen_dir=codegen_dir,
        class_name=gm.__class__.__name__,
        subgraph_modules=subgraph_modules,
    )

    # Copy subgraph modules as attributes
    for name, subgraph in subgraph_modules.items():
        setattr(custom_gm, name, subgraph)

    for attr_name in dir(gm):
        if attr_name.startswith("_") or attr_name in ["forward", "graph", "code"]:
            continue
        if attr_name in subgraph_names:
            continue
        try:
            attr = getattr(gm, attr_name)
            if not callable(attr):
                setattr(custom_gm, attr_name, attr)
        except AttributeError:
            pass

    return custom_gm
