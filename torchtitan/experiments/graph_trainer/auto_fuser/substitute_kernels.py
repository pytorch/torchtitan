# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Substitute outlined graph regions with tuned kernels from a two-level cache.

``substitute_with_kernels_pass`` looks at every ``call_module`` node whose
target is an ``fx.GraphModule`` and replaces it with a cached kernel if one
exists::

    <cache_dir>/
      pat_<l1>/    # level 1: the submodule's structure (see _pattern_key)
        <l2>.py    # level 2: one specialization (spec, reference, kernel)

The level-1 key ignores literal values and view ops, so submodules that only
differ in values baked into op arguments (shapes, scalars, dtypes) or in the
position of views share a pattern. The level-2 key is the AOTAutograd cache
key of the exact submodule on its inputs, so a kernel only runs on the code
and inputs it was written for. Inputs with symbolic sizes (e.g. the routed-row
count of a token dispatcher) are keyed by the submodule's code and their
description instead (plus the torch build and system, which the AOTAutograd key
also covers), with the symbols renamed ``N0``, ``N1``, ... in order of first
appearance: the AOTAutograd key would include the enclosing graph's symbol
names, which differ at every call site. Either form of the level-2 key changes
with the torch build, so kernels are re-checked after every torch change; the
level-1 key does not, so the new specialization is written next to the old
kernel. Neither level depends on the device index, so every rank finds the
same kernels. ``<l2>.py`` is written
when a specialization is first seen::

    SPEC = json.loads(r\"\"\"
    {
      "signature": "def run(in0, in1, in2)",
      "inputs": [{"shape": [8, 16], "stride": [16, 1],
                  "dtype": "float32", "device": "cpu"}, ...],
      "outputs": {"shape": [8, 16], "stride": [16, 1], ...}
    }
    \"\"\")

    def reference(in0, in1, in2):  # the exact submodule code
        ...

    def impl(in0, in1, in2):
        raise NotImplementedError

    run = reference

To provide a kernel, write it in ``impl`` and set ``run = impl``: it receives
inputs as described by ``inputs`` and must return outputs as described by
``outputs``, computing what ``reference`` computes. While ``run`` is
``reference``, the ``call_module`` is left untouched, i.e. it runs exactly the
reference. Otherwise the ``call_module`` is pointed at a ``_KernelModule`` that
holds the file's absolute path and calls its ``run``; the graph keeps no
reference to generated code, so it can be serialized (e.g. by precompile) and
loaded wherever that path exists. ``SPEC`` and ``reference`` are generated and must not be edited.

Submodules must not read attributes: tensor constants have to be passed as
``call_module`` arguments, so that patterns do not depend on their values and
the cache stores no tensors. Higher-order ops with subgraphs (``get_attr`` of
a GraphModule) are not supported yet.

Custom ops must be defined in an importable module: with
``torch.library.custom_op`` or ``triton_op``, or with ``torch.library.Library``
and a Python fake kernel (e.g. torchao's ops), whose module is the one that
registers the fake kernel. ``<l2>.py`` imports those modules so it runs in a
fresh process, and their source is part of the level-1 key. Submodules
calling any other non-torch op are not cacheable.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import logging
import operator
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sympy
import torch
import torch._guards

# TODO: the imports below are private PyTorch APIs (AOTAutograd cache key,
# PyCodeCache, torch_key, OPDEFS, simple_registry, FX import formatting) without public
# equivalents; they can change in any nightly. Replace them once upstream
# exposes public ones.
from torch._functorch._aot_autograd.autograd_cache import (
    autograd_cache_key,
    BypassAOTAutogradCache,
)
from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._inductor.codecache import CacheBase, PyCodeCache, torch_key, write_atomic
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


# Op namespaces registered by ``import torch`` (the namespaces of
# ``torch._C._dispatch_get_all_op_names()``); anything else is a custom op.
_TORCH_NAMESPACES = frozenset(
    {
        "_c10d_functional",
        "_c10d_functional_autograd",
        "_dtensor",
        "_native",
        "_quantized",
        "_test",
        "aten",
        "c10d",
        "coor",
        "debug_mode_ops",
        "debugprims",
        "export",
        "flex_gemm",
        "fsdp",
        "inductor",
        "inductor_prims",
        "mkldnn",
        "mkldnn_prepacked",
        "onednn",
        "prims",
        "profiler",
        "quantization",
        "quantized",
        "rngprims",
        "sparse",
        "static_runtime",
        "symm_mem",
        "torch_nn",
    }
)

# FX prefills every generated module's globals with these builtins. The first
# set can never be used by a normalized graph (flat placeholders, no type
# annotations); the second is only needed when the generated code uses it.
_UNUSED_FX_BUILTINS = frozenset({"NoneType", "fx_pytree", "pytree"})
_OPTIONAL_FX_BUILTINS = frozenset({"inf", "nan", "device"})

# ((module, sha256 of its source, ops it registers), ...), sorted by module.
OpDeps = tuple[tuple[str, str, tuple[str, ...]], ...]


def normalize_names(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Return a copy of ``gm`` with nodes renamed by position and device indices dropped.

    Node names end up in the generated code that both cache keys hash, so two
    copies of the same region whose nodes were named differently by the
    enclosing graph (``mm`` vs. ``mm_17``) would otherwise get different keys.
    Likewise for device literals: every rank traces the same region on its own
    device (``device=cuda:3``), and all ranks must find the same kernels. Without
    the index (``device=cuda``), ``reference`` runs on the current device.
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


def _op_modules(op: torch._ops.OpOverload) -> set[str]:
    """Non-torch modules defining ``op``'s implementation and fake kernel."""
    opdef = OPDEFS.get(op._schema.name)
    if opdef is not None:
        fns = [opdef._init_fn, opdef._abstract_fn, *opdef._backend_fns.values()]
        modules = {inspect.unwrap(fn).__module__ for fn in fns if fn is not None}
    else:
        # An op defined with torch.library.Library, e.g. torchao's C++ kernels with
        # Python fake kernels: the module that registers its fake kernel defines it.
        kernel = simple_registry.singleton.find(op._schema.name).fake_impl.kernel
        if kernel is None:
            raise ValueError(f"{op} has no Python fake kernel")
        path = kernel.source.rsplit(":", 1)[0]
        modules = {
            name
            for name, module in list(sys.modules.items())
            if getattr(module, "__file__", None) == path
        }
    # triton_op wraps the user function in a torch-internal backend function.
    modules = {m for m in modules if m != "torch" and not m.startswith("torch.")}
    if not modules or "__main__" in modules:
        raise ValueError(f"{op} is not defined in an importable module")
    return modules


def _custom_op_deps(gm: torch.fx.GraphModule) -> OpDeps:
    """Modules that must be imported to register the custom ops ``gm`` calls."""
    ops_by_module: dict[str, set[str]] = {}
    for node in gm.graph.nodes:
        op = node.target
        if isinstance(op, torch._ops.HigherOrderOperator) and any(
            isinstance(a, torch._ops.OpOverload)
            for a in torch.utils._pytree.tree_leaves((node.args, node.kwargs))
        ):
            # E.g. auto_functionalized(op, ...): the wrapped op would be neither
            # imported by <l2>.py nor part of the keys.
            raise ValueError(f"{op.name()} wrapping an op is not supported")
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


def _pattern_key(gm: torch.fx.GraphModule, op_deps: OpDeps = ()) -> str:
    """Level-1 key: the submodule's structure, ignoring literal values and views.

    Calls to view ops (``OpOverload.is_view``: view, reshape, permute, expand,
    ...) are skipped, and their users read the viewed tensor instead. Every
    other node contributes its op and its arguments, with nodes referenced by
    position, inputs numbered by first use and non-tensor literals replaced by
    their type. Custom op sources are part of the key.
    """
    names: dict[torch.fx.Node, str] = {}
    viewed: dict[torch.fx.Node, torch.fx.Node] = {}
    num_inputs = 0
    structure = []

    def key_arg(a: Any) -> Any:
        nonlocal num_inputs
        if isinstance(a, torch.fx.Node):
            a = viewed.get(a, a)
            if a not in names:  # a placeholder, numbered by first use
                names[a] = f"in{num_inputs}"
                num_inputs += 1
            return names[a]
        if isinstance(a, (list, tuple)):
            return [key_arg(x) for x in a]
        if isinstance(a, dict):
            return {k: key_arg(v) for k, v in a.items()}
        return None if a is None else type(a).__name__

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        source = node.args[0] if node.args else None
        if (
            node.op == "call_function"
            and isinstance(source, torch.fx.Node)
            and (
                (isinstance(node.target, torch._ops.OpOverload) and node.target.is_view)
                # An element of a multi-output view such as split.
                or (node.target is operator.getitem and source in viewed)
            )
        ):
            viewed[node] = viewed.get(source, source)
            continue
        args = key_arg((node.args, node.kwargs))
        names[node] = f"v{len(structure)}"
        structure.append((node.op, str(node.target), args))
    return "pat_" + hashlib.sha256(repr((structure, op_deps)).encode()).hexdigest()


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


def _spec_key(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> str:
    """Level-2 key: the AOTAutograd cache key for ``gm`` on ``example_inputs``.

    Inputs are keyed on device index 0, so every rank computes the same key
    (``normalize_names`` drops the index of device literals). With symbolic
    input sizes the key is a hash of ``gm``'s code (the level-1 key ignores
    literals), the inputs' description (see the module docstring), and the
    deterministic-algorithms flag, torch build and system (GPU, CUDA and Triton
    versions), which the AOTAutograd key also includes.
    """
    names = _symbol_names(example_inputs)
    if names:
        described = [_describe(v, names) for v in example_inputs]
        payload = json.dumps(
            [
                gm.code,
                described,
                torch.are_deterministic_algorithms_enabled(),
                torch_key().hex(),
                CacheBase.get_system()["hash"],
            ],
            sort_keys=True,
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


def _kernel_source(
    gm: torch.fx.GraphModule, op_deps: OpDeps, spec: dict[str, Any]
) -> str:
    """Source of ``<l2>.py``: ``spec``, the reference and an empty ``impl``."""
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
    reference = re.sub(
        r"def forward\(self(, )?", "def reference(", gm.code.strip(), count=1
    )
    names = ", ".join(n.name for n in gm.graph.find_nodes(op="placeholder"))
    return (
        '"""One specialization of a substitute_kernels pattern.\n\n'
        "SPEC and reference are generated: do not edit them. Write the kernel in\n"
        "impl, then set ``run = impl``.\n"
        '"""\n\n'
        "import json\n"
        f"{imports}\n"
        f"{op_imports}\n"
        f'SPEC = json.loads(r"""\n{_spec_json(spec)}\n""")\n\n\n'
        f"{reference}\n\n\n"
        f"def impl({names}):\n"
        "    raise NotImplementedError\n\n\n"
        "run = reference\n"
    )


def _describe(val: Any, names: dict[sympy.Symbol, str] | None = None) -> Any:
    """JSON description of an input or output ``meta["val"]``; symbolic sizes are
    written with ``names``. Devices are written without their index, so the
    spec is the same whichever rank writes it."""
    names = names or {}
    if isinstance(val, torch.Tensor):
        return {
            "shape": [_render(d, names) for d in val.shape],
            "stride": [_render(d, names) for d in val.stride()],
            "dtype": str(val.dtype).removeprefix("torch."),
            "device": val.device.type,
        }
    if isinstance(val, (tuple, list)):
        return [_describe(v, names) for v in val]
    if isinstance(val, torch.SymInt):
        return {"value": _render(val, names)}
    if val is None or isinstance(val, (bool, int, float, str)):
        return {"value": val}
    return {"value": repr(val)}


def _spec_json(spec: dict[str, Any]) -> str:
    """``json.dumps(spec, indent=2)`` with lists of numbers on one line."""
    text = json.dumps(spec, indent=2)
    return re.sub(r"\[[\d\s,.-]*\]", lambda m: json.dumps(json.loads(m.group())), text)


def cache_keys(
    submod: torch.fx.GraphModule, example_inputs: list[Any]
) -> tuple[str, str, torch.fx.GraphModule, OpDeps]:
    """Return ``(l1, l2, normalized submod, custom op deps)`` for a submodule."""
    gm = normalize_names(submod)
    op_deps = _custom_op_deps(gm)
    return _pattern_key(gm, op_deps), _spec_key(gm, example_inputs), gm, op_deps


def _lookup(
    submod: torch.fx.GraphModule,
    example_inputs: list[Any],
    cache_dir: Path,
    example_output: Any = None,
) -> Path | None:
    """Return the path of ``submod``'s kernel file, or None to keep the reference.

    Writes ``<l2>.py`` the first time a specialization is seen.
    """
    try:
        l1, l2, gm, op_deps = cache_keys(submod, example_inputs)
    except (ValueError, BypassAOTAutogradCache):
        logger.warning("substitute_kernels: submodule is not cacheable", exc_info=True)
        return None

    path = cache_dir / l1 / f"{l2}.py"
    if path.exists():
        source = path.read_text(encoding="utf-8")
    else:
        params = [n.name for n in gm.graph.find_nodes(op="placeholder")]
        # Output-only symbols (sizes computed inside the region) continue the numbering.
        names = _symbol_names([*example_inputs, example_output])
        spec = {
            "signature": f"def run({', '.join(params)})",
            "inputs": [_describe(v, names) for v in example_inputs],
            "outputs": _describe(example_output, names),
        }
        source = _kernel_source(gm, op_deps, spec)
        write_atomic(str(path), source, make_dirs=True, encode_utf_8=True)
    module = PyCodeCache.load(source)
    if module.run is getattr(module, "reference", None):
        logger.info("substitute_kernels miss: %s", path)
        return None
    logger.info("substitute_kernels hit: %s", path)
    return path


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


class _KernelModule(torch.nn.Module):
    """Calls ``run`` of the specialization file ``path``, loaded on first use.

    Holding the path rather than the loaded function keeps a graph that calls
    this module picklable (``GraphPickler``).
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self._run: Callable[..., Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_run": None}

    def forward(self, *args: Any) -> Any:
        if self._run is None:
            source = Path(self.path).read_text(encoding="utf-8")
            self._run = PyCodeCache.load(source).run
        return self._run(*args)


def substitute_with_kernels_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    cache_dir: str | os.PathLike,
) -> torch.fx.GraphModule:
    """Point ``call_module`` nodes of GraphModules that have a kernel at it.

    Each distinct kernel becomes one ``_KernelModule`` submodule, shared by all
    nodes that hit it; submodules no node calls anymore are deleted. Only
    ``gm``'s own graph is scanned; nested GraphModules are left alone.
    Arguments must be positional and carry ``meta["val"]`` (fake or real
    tensors); other ``call_module`` nodes are skipped.
    """
    cache_dir = Path(cache_dir)
    kernels: dict[Path, str] = {}  # kernel file -> submodule name
    replaced: set[str] = set()
    for node in list(gm.graph.find_nodes(op="call_module")):
        submod = gm.get_submodule(node.target)
        if not isinstance(submod, torch.fx.GraphModule):
            continue
        inputs = _example_inputs(node)
        if inputs is None:
            logger.info("substitute_kernels: skipping %s (no example inputs)", node)
            continue
        path = _lookup(submod, inputs, cache_dir, node.meta.get("val"))
        if path is None:
            continue
        if path not in kernels:
            name = f"kernel{len(kernels)}"
            while hasattr(gm, name):
                name += "_"
            gm.add_submodule(name, _KernelModule(str(path.resolve())))
            kernels[path] = name
        replaced.add(node.target)
        node.target = kernels[path]

    still_called = {n.target for n in gm.graph.find_nodes(op="call_module")}
    for target in replaced - still_called:
        gm.delete_submodule(target)
    gm.recompile()
    return gm
