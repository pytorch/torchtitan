# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Substitute outlined graph regions with tuned kernels from a two-level store.

``substitute_with_kernels_pass`` looks at every ``call_module`` node whose
target is an ``fx.GraphModule`` and replaces it with a cached kernel if one
exists::

    <cache_dir>/
      pat_<l1>/                   # level 1: the submodule's generated code
        reference.py              # standalone eager implementation
        specialization_<l2>.json  # level 2: spec of one specialization
        <l2>.py                   # optimized kernel for that specialization

The level-2 key is the AOTAutograd cache key of the submodule on its inputs.
Inputs with symbolic sizes (e.g. the routed-row count of a token dispatcher)
are keyed by their description instead, with the symbols renamed ``N0``,
``N1``, ... in order of first appearance: the AOTAutograd key would include the
enclosing graph's symbol names, which differ at every call site. Neither key
depends on the device index, so every rank finds the same kernels.
``specialization_<l2>.json`` is written when a specialization is first seen::

    {
      "kernel_name": "<l2>.py",
      "reference": "reference.py",
      "signature": "def run(in0, in1, in2)",
      "inputs": [{"shape": [8, 16], "stride": [16, 1],
                  "dtype": "float32", "device": "cpu"}, ...],
      "outputs": {"shape": [8, 16], "stride": [16, 1], ...}
    }

To provide a kernel, write ``kernel_name`` next to it, defining ``run`` with
that signature: it receives inputs as described by ``inputs`` and must return
outputs as described by ``outputs``, computing what ``reference.py``
computes. Without that file the ``call_module`` is left untouched, i.e. it
runs exactly ``reference.py``.

Submodules must not read attributes: tensor constants have to be passed as
``call_module`` arguments, so that patterns do not depend on their values and
the cache stores no tensors. Higher-order ops with subgraphs (``get_attr`` of
a GraphModule) are not supported yet.

Custom ops must be defined with ``torch.library.custom_op`` or ``triton_op``
in an importable module. ``reference.py`` imports those modules so it runs in
a fresh process, and their source is part of the level-1 key. Submodules
calling any other non-torch op are not cacheable.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sympy
import torch
import torch._guards
from torch._functorch._aot_autograd.autograd_cache import (
    autograd_cache_key,
    sanitize_gm_for_cache,
)
from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._inductor.codecache import FxGraphCachePickler
from torch._inductor.runtime.cache_dir_utils import cache_dir as inductor_cache_dir
from torch._library import simple_registry
from torch._library.custom_ops import OPDEFS
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph_module import _format_import_block
from torch.package import sys_importer


logger = logging.getLogger(__name__)

_AOT_CONFIG = AOTConfig(
    fw_compiler=None,
    bw_compiler=None,
    partition_fn=None,
    decompositions=None,
    num_params_buffers=0,
    aot_id=0,
    keep_inference_input_mutations=False,
)


# Op namespaces registered by torch itself; anything else is a custom op.
_TORCH_NAMESPACES = frozenset(
    {
        "aten",
        "prims",
        "c10d",
        "_c10d_functional",
        "_c10d_functional_autograd",
        "_dtensor",
        "profiler",
    }
)

# FX prefills every generated module's globals with these builtins. The first
# set can never be used by a normalized graph (flat placeholders, no type
# annotations); the second is only needed when the generated code uses it.
_UNUSED_FX_BUILTINS = frozenset({"NoneType", "fx_pytree", "pytree"})
_OPTIONAL_FX_BUILTINS = frozenset({"inf", "nan", "device"})

# ((module, sha256 of its source, ops it registers), ...), sorted by module.
OpDeps = tuple[tuple[str, str, tuple[str, ...]], ...]


def default_cache_dir() -> Path:
    return Path(
        os.environ.get("TORCHTITAN_SUBSTITUTE_KERNELS_DIR")
        or os.path.join(inductor_cache_dir(), "substitute_kernels")
    )


def normalize_names(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Return a copy of ``gm`` with nodes renamed by position and device indices dropped.

    Node names end up in the generated code that both cache keys hash, so two
    copies of the same region whose nodes were named differently by the
    enclosing graph (``mm`` vs. ``mm_17``) would otherwise get different keys.
    Likewise for device literals: every rank traces the same region on its own
    device (``device=cuda:3``), and all ranks must find the same kernels. Without
    the index (``device=cuda``), ``reference.py`` runs on the current device.
    """
    graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    indices: set[tuple[str, int]] = set()

    def key_arg(a: Any) -> Any:
        if isinstance(a, torch.fx.Node):
            return env[a]
        if isinstance(a, torch.device) and a.index is not None:
            indices.add((a.type, a.index))
            return torch.device(a.type)
        return a

    n_in = 0
    for i, node in enumerate(gm.graph.nodes):
        args = torch.fx.node.map_aggregate(node.args, key_arg)
        kwargs = torch.fx.node.map_aggregate(node.kwargs, key_arg)
        if node.op == "placeholder":
            new = graph.placeholder(f"in{n_in}")
            n_in += 1
        elif node.op == "get_attr":
            if isinstance(getattr(gm, node.target), torch.fx.GraphModule):
                raise ValueError(f"subgraph {node.target!r} is not supported")
            raise ValueError(f"attribute {node.target!r} must be passed as an argument")
        elif node.op == "output":
            new = graph.output(args[0])
        else:
            new = graph.create_node(node.op, node.target, args, kwargs, name=f"v{i}")
        new.meta = dict(node.meta)
        env[node] = new
    if len(indices) != len({t for t, _ in indices}):
        raise ValueError(
            f"submodule uses several devices of one type: {sorted(indices)}"
        )
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _module_of_file(path: str) -> str | None:
    for name, module in list(sys.modules.items()):
        if getattr(module, "__file__", None) == path:
            return name
    return None


def _op_modules(op: torch._ops.OpOverload) -> set[str]:
    """Modules to import so that ``op`` and its fake kernel are registered."""
    opdef = OPDEFS.get(op._schema.name)
    if opdef is not None:
        fns = [opdef._init_fn, opdef._abstract_fn, *opdef._backend_fns.values()]
        modules = {inspect.unwrap(fn).__module__ for fn in fns if fn is not None}
    else:
        # An op defined with torch.library.Library (e.g. a C++ kernel plus a Python
        # fake kernel): the module that registers the fake kernel defines the op.
        kernel = simple_registry.singleton.find(op._schema.name).fake_impl.kernel
        module = kernel and _module_of_file(kernel.source.rsplit(":", 1)[0])
        modules = {module} if module else set()
    # custom_op / triton_op wrap the user function in torch.library helpers.
    modules = {
        m for m in modules if not m.startswith(("torch.library", "torch._library"))
    }
    # Ops defined inside torch (e.g. Inductor's bucketing ops) keep their torch module.
    modules = {
        m for m in modules if m != "torch" and not m.startswith("torch.")
    } or modules
    if not modules or "__main__" in modules:
        raise ValueError(f"{op} is not defined in an importable module")
    return modules


def custom_op_deps(gm: torch.fx.GraphModule) -> OpDeps:
    """Modules that must be imported to register the custom ops ``gm`` calls."""
    ops_by_module: dict[str, set[str]] = {}
    for node in gm.graph.nodes:
        op = node.target
        if (
            node.op == "call_function"
            and isinstance(op, torch._ops.OpOverload)
            and op.namespace not in _TORCH_NAMESPACES
        ):
            for module in _op_modules(op):
                ops_by_module.setdefault(module, set()).add(op.name())
    deps = []
    for module in sorted(ops_by_module):
        path = inspect.getsourcefile(sys.modules[module])
        if path is None:
            raise ValueError(f"no source file for {module}")
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        deps.append((module, sha, tuple(sorted(ops_by_module[module]))))
    return tuple(deps)


def pattern_key(gm: torch.fx.GraphModule, op_deps: OpDeps = ()) -> str:
    """Level-1 key: generated code and custom op sources."""
    with sanitize_gm_for_cache(gm):
        return "pat_" + FxGraphCachePickler(gm).get_hash((gm, op_deps))


def _symbol_names(values: list[Any]) -> dict[sympy.Symbol, str]:
    """``N0``, ``N1``, ... for the free symbols of ``values``' symbolic sizes,
    strides and ints, in order of first appearance."""
    names: dict[sympy.Symbol, str] = {}

    def visit(x: Any) -> None:
        if isinstance(x, torch.Tensor):
            for d in (*x.shape, *x.stride()):
                visit(d)
        elif isinstance(x, (tuple, list)):
            for v in x:
                visit(v)
        elif isinstance(x, torch.SymInt):
            # Symbols are numbered in creation order, the same at every call site.
            for sym in sorted(x.node.expr.free_symbols, key=_symbol_order):
                names.setdefault(sym, f"N{len(names)}")

    visit(values)
    return names


def _symbol_order(sym: sympy.Symbol) -> tuple[str, int]:
    name = str(sym)
    digits = name.lstrip("abcdefghijklmnopqrstuvwxyz")
    return name[: len(name) - len(digits)], int(digits) if digits.isdigit() else -1


def _render(x: Any, names: dict[sympy.Symbol, str]) -> Any:
    if isinstance(x, torch.SymInt):
        expr = x.node.expr.xreplace({s: sympy.Symbol(n) for s, n in names.items()})
        return int(expr) if expr.is_number else str(expr)
    return x


def spec_key(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> str:
    """Level-2 key: the AOTAutograd cache key for ``gm`` on ``example_inputs``.

    Inputs are keyed on device index 0, so every rank computes the same key
    (``normalize_names`` drops the index of device literals). Inputs with
    symbolic sizes are keyed by their description (see the module docstring)
    and the deterministic-algorithms flag, which the AOTAutograd key also
    includes.
    """
    names = _symbol_names(example_inputs)
    if names:
        described = [_describe(v, names) for v in example_inputs]
        for d in described:
            if isinstance(d, dict) and "device" in d:
                d["device"] = torch.device(d["device"]).type
        payload = json.dumps(
            [described, torch.are_deterministic_algorithms_enabled()], sort_keys=True
        )
        digest = base64.b32encode(hashlib.sha256(payload.encode()).digest())
        return "d" + digest.decode().lower().rstrip("=")[:51]
    # The FX graph cache validator requires an active ShapeEnv.
    fake_mode = FakeTensorMode(shape_env=ShapeEnv())
    inputs = []
    for v in example_inputs:
        if isinstance(v, torch.Tensor) and v.device.index:
            with fake_mode:
                v = torch.empty_strided(
                    v.shape,
                    v.stride(),
                    dtype=v.dtype,
                    device=torch.device(v.device.type, 0),
                    requires_grad=v.requires_grad,
                )
        inputs.append(v)
    with torch._guards.tracing(torch._guards.TracingContext(fake_mode)):
        key, _ = autograd_cache_key(gm, inputs, _AOT_CONFIG)
    return key


def reference_source(gm: torch.fx.GraphModule, op_deps: OpDeps = ()) -> str:
    python_code = gm.recompile()
    code = python_code.src
    used_globals = {
        name: obj
        for name, obj in python_code.globals.items()
        if name not in _UNUSED_FX_BUILTINS
        and (
            name not in _OPTIONAL_FX_BUILTINS
            or re.search(rf"\b{name}\b", code) is not None
        )
    }
    imports = _format_import_block(used_globals, sys_importer)
    op_imports = "".join(
        f"import {module}  # {', '.join(ops)}\n" for module, _, ops in op_deps
    )
    body = "\n".join("    " + line for line in gm.code.strip().splitlines())
    return (
        f"{imports}\n"
        f"{op_imports}\n\n"
        "class Reference(torch.nn.Module):\n"
        f"{body}\n\n\n"
        "def run(*args):\n"
        "    return Reference()(*args)\n"
    )


def _describe(val: Any, names: dict[sympy.Symbol, str] | None = None) -> Any:
    """JSON description of an input or output ``meta["val"]``; symbolic sizes are
    written with ``names``."""
    names = names or {}
    if isinstance(val, torch.Tensor):
        return {
            "shape": [_render(d, names) for d in val.shape],
            "stride": [_render(d, names) for d in val.stride()],
            "dtype": str(val.dtype).removeprefix("torch."),
            "device": str(val.device),
        }
    if isinstance(val, (tuple, list)):
        return [_describe(v, names) for v in val]
    if isinstance(val, torch.SymInt):
        return {"value": _render(val, names)}
    if val is None or isinstance(val, (bool, int, float, str)):
        return {"value": val}
    return {"value": repr(val)}


def _atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".tmp"
    ) as f:
        f.write(text)
    os.replace(f.name, path)


def _spec_json(spec: dict[str, Any]) -> str:
    """``json.dumps(spec, indent=2)`` with lists of numbers on one line."""
    text = json.dumps(spec, indent=2)
    return re.sub(r"\[[\d\s,.-]*\]", lambda m: json.dumps(json.loads(m.group())), text)


def _ensure_spec(
    pattern_dir: Path,
    l2: str,
    gm: torch.fx.GraphModule,
    inputs: list[Any],
    output: Any,
) -> dict[str, Any]:
    """Return the spec of specialization ``l2``, writing it if it is new."""
    path = pattern_dir / f"specialization_{l2}.json"
    if path.exists():
        return json.loads(path.read_text())
    params = [n.name for n in gm.graph.find_nodes(op="placeholder")]
    # Output-only symbols (sizes computed inside the region) continue the numbering.
    names = _symbol_names([*inputs, output])
    spec = {
        "kernel_name": f"{l2}.py",
        "reference": "reference.py",
        "signature": f"def run({', '.join(params)})",
        "inputs": [_describe(v, names) for v in inputs],
        "outputs": _describe(output, names),
    }
    _atomic_write(path, _spec_json(spec) + "\n")
    return spec


def _write_reference(
    pattern_dir: Path, gm: torch.fx.GraphModule, op_deps: OpDeps
) -> None:
    pattern_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(pattern_dir / "reference.py", reference_source(gm, op_deps))


def _load_run(path: Path) -> Callable[..., Any]:
    # One module per file content: every call site of a pattern shares it, and a
    # replaced file is loaded anew.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    name = f"_substitute_kernels_{path.parent.name}_{path.stem}_{digest}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name].run


def cache_keys(
    submod: torch.fx.GraphModule, example_inputs: list[Any]
) -> tuple[str, str, torch.fx.GraphModule, OpDeps]:
    """Return ``(l1, l2, normalized submod, custom op deps)`` for a submodule."""
    gm = normalize_names(submod)
    op_deps = custom_op_deps(gm)
    return pattern_key(gm, op_deps), spec_key(gm, example_inputs), gm, op_deps


def lookup(
    submod: torch.fx.GraphModule,
    example_inputs: list[Any],
    cache_dir: Path,
    example_output: Any = None,
) -> Callable[..., Any] | None:
    """Return the cached kernel for ``submod``, or None to keep the reference.

    Writes ``reference.py`` the first time a pattern is seen and
    ``specialization_<l2>.json`` the first time a specialization is seen.
    """
    try:
        l1, l2, gm, op_deps = cache_keys(submod, example_inputs)
    except Exception:
        logger.warning("substitute_kernels: submodule is not cacheable", exc_info=True)
        return None

    pattern_dir = cache_dir / l1
    if not (pattern_dir / "reference.py").exists():
        _write_reference(pattern_dir, gm, op_deps)

    spec = _ensure_spec(pattern_dir, l2, gm, example_inputs, example_output)
    impl = pattern_dir / spec["kernel_name"]
    if not impl.exists():
        logger.info("substitute_kernels miss: %s", impl)
        return None
    logger.info("substitute_kernels hit: %s", impl)
    return _load_run(impl)


def _example_inputs(node: torch.fx.Node) -> list[Any] | None:
    if node.kwargs:
        return None
    inputs = []
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            if "val" not in arg.meta:
                return None
            inputs.append(arg.meta["val"])
        else:
            inputs.append(arg)
    return inputs


def substitute_with_kernels_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    cache_dir: str | os.PathLike | None = None,
) -> torch.fx.GraphModule:
    """Replace ``call_module`` nodes of GraphModules with cached kernels.

    Only ``gm``'s own graph is scanned; nested GraphModules are left alone.
    Arguments must be positional and carry ``meta["val"]`` (fake or real
    tensors); other ``call_module`` nodes are skipped.
    """
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    replaced: set[str] = set()
    for node in list(gm.graph.find_nodes(op="call_module")):
        submod = gm.get_submodule(node.target)
        if not isinstance(submod, torch.fx.GraphModule):
            continue
        inputs = _example_inputs(node)
        if inputs is None:
            logger.info("substitute_kernels: skipping %s (no example inputs)", node)
            continue
        impl = lookup(submod, inputs, root, node.meta.get("val"))
        if impl is None:
            continue
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(impl, node.args)
        new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)
        replaced.add(node.target)

    still_called = {n.target for n in gm.graph.find_nodes(op="call_module")}
    for target in replaced - still_called:
        gm.delete_submodule(target)
    gm.recompile()
    return gm
