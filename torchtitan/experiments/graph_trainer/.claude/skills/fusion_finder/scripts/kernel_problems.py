# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write the lifted regions as kernel problems, for kernel-writing agents.

A recipe runs the lift passes, then this pass::

    from kernel_problems import register_kernel_problems
    from lift_regions import register_lift_passes

    config.compile.extra_passes = [
        *register_lift_passes("<capture dir>"),
        register_kernel_problems("<capture dir>", "<problems dir>"),
    ]

A problem is one specialization of a lifted submodule: its exact code on inputs
of its exact shapes, strides and dtypes. It is keyed as ``substitute_kernels``
keys its kernel cache (``cache_keys``): level 1 (``<l1>``) is the submodule's
structure, ignoring literal values and views, so the problems of one region at
other sizes share it; level 2 (``<l2>``) is the exact code and inputs::

    <problems dir>/
      problems.json           # the problems of this run, by gap
      check_kernel.py         # the bitwise gate, shared by all problems
      pattern_<l1>/<l2>/
        problem.py            # reference(in0, ...) and the KernelBench interface
        spec.json             # contract, input cases, the calls it comes from
        inputs.pt             # the integer, bool and complex inputs of each case
        kernel.py             # written by a kernel agent: kernel_function(in0, ...)

``problem.py`` defines ``reference``, the submodule's code, and ``Model``,
``get_inputs`` and ``get_init_inputs`` for agents that take KernelBench problems.
``spec.json`` holds the contract (the inputs and outputs, the inputs written in
place, the inputs each output aliases), the saved cases, and the calls of the
step with this specialization, with their measured and ideal time when the call
is an instance of the capture. The first ``_NUM_CASES`` calls of each problem in
the step are its input cases: ``spec.json`` holds the layout of their inputs (shapes, strides,
alignment, shared storage) and ``inputs.pt`` the values of their integer, bool
and complex inputs (indices, masks, counts: values a generator cannot make up).
Floating-point inputs are generated from a seed (``check_kernel.build_args``).
Each case checks that ``reference`` on the call's inputs reproduces its outputs
and in-place writes bit for bit. A kernel must match ``reference`` bit for
bit on every case: ``python check_kernel.py <problem dir> kernel.py``.

Only rank 0 writes problems. The pass must run after the lift passes.
"""

import gzip
import json
import operator
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import sympy
import torch
import torch.fx as fx

from check_kernel import compare, copy_args, describe_args, load_module
from fusion_finder_capture import encode, match_key, node_phase
from lift_regions import submodule_storage
from regions import find_regions, key_skeleton
from torch._subclasses.fake_tensor import is_fake

# TODO: _format_import_block is a private FX API without a public equivalent;
# substitute_kernels uses it the same way to make generated code standalone.
from torch.fx.graph_module import _format_import_block
from torch.package import sys_importer

from torchtitan.experiments.graph_trainer.auto_fuser.substitute_kernels import (
    cache_keys,
)
from torchtitan.experiments.graph_trainer.registry import register_graph_pass

# Calls per problem that save their inputs: values from several layers, and for
# symbolic sizes several sizes, at a bounded size of inputs.pt.
_NUM_CASES = 3
_CALL_RE = re.compile(r"fusion_(\w+)_i\d+")
# Globals that FX adds to every generated module, needed only when the code uses them.
_UNUSED_FX_GLOBALS = {"NoneType", "fx_pytree", "pytree"}
_OPTIONAL_FX_GLOBALS = {"inf", "nan", "device"}


def _symbol_order(sym: sympy.Symbol) -> tuple[str, int]:
    name = str(sym)
    digits = name.lstrip("abcdefghijklmnopqrstuvwxyz")
    return name[: len(name) - len(digits)], int(digits) if digits.isdigit() else -1


def _symbol_names(values: list[Any]) -> dict[sympy.Symbol, sympy.Symbol]:
    """``N0``, ``N1``, ... for the symbols in the sizes and strides of ``values``,
    in order of first appearance (as ``substitute_kernels`` names them)."""
    names: dict[sympy.Symbol, sympy.Symbol] = {}
    for v in values:
        dims = [*v.shape, *v.stride()] if isinstance(v, torch.Tensor) else [v]
        for d in dims:
            if isinstance(d, torch.SymInt):
                for sym in sorted(d.node.expr.free_symbols, key=_symbol_order):
                    names.setdefault(sym, sympy.Symbol(f"N{len(names)}"))
    return names


def _describe(val: Any, names: dict[sympy.Symbol, sympy.Symbol]) -> dict[str, Any]:
    def size(d: Any) -> int | str:
        return str(d.node.expr.xreplace(names)) if isinstance(d, torch.SymInt) else d

    if isinstance(val, torch.Tensor):
        return {
            "shape": [size(d) for d in val.shape],
            "stride": [size(d) for d in val.stride()],
            "dtype": str(val.dtype).removeprefix("torch."),
            "device": val.device.type,
        }
    if isinstance(val, torch.SymInt):
        return {"value": size(val)}
    if val is None or isinstance(val, (bool, int, float, str)):
        return {"value": val}
    return {"value": repr(val)}


def _reference_source(gm: fx.GraphModule, op_modules: list[str]) -> str:
    """Imports and ``def reference(...)``: the code of ``gm`` (``cache_keys``'
    copy with nodes named by position), standalone."""
    python_code = gm.recompile()
    used = {
        name: obj
        for name, obj in python_code.globals.items()
        if name not in _UNUSED_FX_GLOBALS
        and (
            name not in _OPTIONAL_FX_GLOBALS
            or re.search(rf"\b{name}\b", python_code.src) is not None
        )
    }
    imports = _format_import_block(used, sys_importer).strip()
    # Importing a custom op's module registers the op.
    imports += "".join(f"\nimport {m}" for m in op_modules)
    reference = re.sub(r"def forward\(self,? ?", "def reference(", gm.code.strip())
    return f"{imports}\n\n\n{reference}\n"


def _problem_source(docstring: str, gm: fx.GraphModule, op_modules: list[str]) -> str:
    names = ", ".join(n.name for n in gm.graph.find_nodes(op="placeholder"))
    return f'''"""{docstring}"""

import importlib.util
from pathlib import Path

{_reference_source(gm, op_modules)}

class Model(torch.nn.Module):
    """``reference`` as a KernelBench model."""

    def forward(self, {names}):
        return reference({names})


def get_input_cases(seed=0):
    """The inputs of each case (calls of the training run, see spec.json): the
    integer, bool and complex inputs as saved, floating-point inputs generated from
    ``seed``."""
    path = Path(__file__).parents[2] / "check_kernel.py"
    spec = importlib.util.spec_from_file_location("check_kernel", path)
    check_kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_kernel)
    return check_kernel.load_cases(Path(__file__).parent, seed)


def get_inputs():
    return get_input_cases()[0]


def get_init_inputs():
    return []
'''


def _docstring(
    problem_id: str, region: str, capture_dir: str, calls: list[dict]
) -> str:
    modules = list(dict.fromkeys(c["module"] for c in calls))
    timed = [c for c in calls if c["measured_us"] is not None]
    lines = [
        f"Kernel problem {problem_id}: region {region} of {capture_dir}.",
        "",
        f"{len(calls)} call{'s' if len(calls) > 1 else ''} per step, "
        f"{'/'.join(sorted({c['phase'] for c in calls}))}, "
        f"in {', '.join(modules[:3])}{' ...' if len(modules) > 3 else ''}.",
    ]
    if timed:
        measured = sum(c["measured_us"] for c in timed) / len(timed)
        ideal = sum(c["ideal_us"] for c in timed) / len(timed)
        lines.append(
            f"Measured {measured:.1f} us per call in the capture; ideal (HBM roofline) "
            f"{ideal:.1f} us."
        )
    lines += [
        "",
        "``reference`` is the computation a kernel replaces. Write ``kernel_function``,",
        "with the signature of ``reference``, in ``kernel.py`` next to this file. It must",
        "match ``reference`` bit for bit, including in-place writes to its inputs, on",
        "every case in ``inputs.pt``: ``python ../../check_kernel.py . kernel.py``.",
        "",
    ]
    return "\n".join(lines)


class _Capture:
    """The capture's nodes, and each instance's measured and ideal time."""

    def __init__(self, capture_dir: str) -> None:
        with gzip.open(os.path.join(capture_dir, "graph.jsonl.gz"), "rt") as f:
            self.records = {r["name"]: r for r in map(json.loads, f)}
        _, regions = find_regions(capture_dir)
        self.instance_of: dict[str, tuple[frozenset[str], dict]] = {}
        for r in regions:
            for members, stats in zip(r["members"], r["instance_stats"]):
                for name in members:
                    self.instance_of[name] = (frozenset(members), stats)

    def stats_of(self, call: fx.Node, members: list[fx.Node]) -> dict | None:
        """The stats of the capture instance that ``call`` lifted: one with the
        same node names (``members``, and the ``getitem``s of outside tuples that
        stayed outside as inputs of the call), and per member the same match key
        up to input names (an input lifted by an earlier pass is a new ``getitem``)
        and the same value."""
        for n in members:
            record = self.records.get(n.name)
            if record is None or json.loads(
                json.dumps([key_skeleton(match_key(n))[0], encode(n.meta.get("val"))])
            ) != [key_skeleton(record["match_key"])[0], record["val"]]:
                return None
        instance, stats = self.instance_of.get(members[0].name, (frozenset(), None))
        names = {n.name for n in members}
        getitems = {
            a.name for a in call.all_input_nodes if a.target is operator.getitem
        }
        return stats if names <= instance and instance - names <= getitems else None


class _Problem:
    """The files of one problem, and the input cases its calls save."""

    def __init__(self, directory: Path, spec: dict) -> None:
        self.directory = directory
        self.spec = spec
        self.saved: list[list[torch.Tensor | None]] = []
        self._reference = None

    def wants_case(self) -> bool:
        return len(self.saved) < _NUM_CASES

    def add_case(self, call: str, run: Any, args: list[Any]) -> Any:
        """Run the call (``run(*args)``) and add its inputs as a case, after checking
        that ``reference`` reproduces its outputs and in-place writes."""
        layout, data = describe_args(args, generate_floating=True, device="cpu")
        # reference reads the inputs after the call, or a copy from before it if the
        # call writes into them.
        inputs = copy_args(args) if self.spec["mutated_inputs"] else args
        outputs = run(*args)
        if self._reference is None:
            self._reference = load_module(
                self.directory / "problem.py", "problem"
            ).reference
        difference = compare(outputs, self._reference(*inputs), args, inputs)
        if difference is not None:
            raise RuntimeError(
                f"{self.spec['id']}: reference on the inputs of {call} differs from "
                f"the training run: {difference}"
            )
        self.saved.append(data)
        self.spec["cases"].append({"call": call, **layout})
        tmp = self.directory / "inputs.pt.tmp"
        torch.save(self.saved, tmp)
        os.replace(tmp, self.directory / "inputs.pt")
        _write_json(self.directory / "spec.json", self.spec)
        return outputs


class _SaveCases(torch.nn.Module):
    """A lifted submodule whose first call is a case of its problem."""

    def __init__(self, module: fx.GraphModule, problem: _Problem, call: str) -> None:
        super().__init__()
        self.module = module
        self.problem = problem
        self.call = call
        self.saved = False
        # Dead code elimination keeps a call of an impure module (see lift_regions).
        self._is_impure = module._is_impure

    def forward(self, *args: Any) -> Any:
        if (
            self.saved
            or not self.problem.wants_case()
            or any(is_fake(a) for a in args if isinstance(a, torch.Tensor))
        ):
            return self.module(*args)
        self.saved = True
        return self.problem.add_case(self.call, self.module, list(args))


def _write_json(path: Path, value: Any) -> None:
    # Lists of numbers and short strings (shapes, strides) on one line.
    text = re.sub(
        r"\[[^\[\]{}]*\]",
        lambda m: json.dumps(json.loads(m.group())),
        json.dumps(value, indent=1),
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n")
    os.replace(tmp, path)


def _write_problems(
    gm: fx.GraphModule, capture_dir: str, problems_dir: str
) -> fx.GraphModule:
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return gm
    capture = _Capture(capture_dir)
    root = Path(problems_dir)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(__file__).with_name("check_kernel.py"), root / "check_kernel.py"
    )
    env = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }

    problems: dict[str, dict] = {}  # id -> spec
    gms: dict[str, tuple[fx.GraphModule, list[str]]] = {}
    calls_of: dict[str, list[fx.Node]] = {}
    uncacheable = []
    for node in gm.graph.find_nodes(op="call_module"):
        m = _CALL_RE.fullmatch(node.target)
        sub = gm.get_submodule(node.target)
        if m is None or not isinstance(sub, fx.GraphModule):
            continue
        inputs = [a.meta["val"] if isinstance(a, fx.Node) else a for a in node.args]
        # TODO: substitute_kernels' level-2 key (the AOTAutograd cache key) includes
        # node.meta["custom"], where GraphTrainer notes each node's module_fqn, so
        # the instances of a region in different layers would never share a key.
        # Drop it in substitute_kernels.normalize_names.
        custom = {
            n: n.meta.pop("custom") for n in sub.graph.nodes if "custom" in n.meta
        }
        try:
            l1, l2, normalized, op_deps = cache_keys(sub, inputs)
        except ValueError as e:
            uncacheable.append({"call": node.target, "reason": str(e)})
            continue
        finally:
            for n, value in custom.items():
                n.meta["custom"] = value
        # TODO: investigate making regions and patterns one to one. A pattern skips
        # views, so regions that differ only in views share one (e.g. t vs
        # transpose); the instances of a region whose submodules return different
        # outputs (a member also used outside in some of them) get several.
        problem_id = f"pattern_{l1.removeprefix('pat_')}/{l2}"
        members = [n for n in sub.graph.nodes if n.op not in ("placeholder", "output")]
        stats = capture.stats_of(node, members)
        modules = Counter(
            n.meta.get("custom", {}).get("module_fqn", "") for n in members
        )
        call = {
            "name": node.target,
            "module": modules.most_common(1)[0][0],
            "phase": "/".join(sorted({node_phase(n) for n in members})),
            "measured_us": stats and round(stats["measured_us"], 1),
            "ideal_us": stats and round(stats["ideal_us"], 1),
        }
        if problem_id not in problems:
            written, output_storage = submodule_storage(sub)
            outputs = list(node.meta["val"])
            names = _symbol_names([*inputs, *outputs])
            placeholders = [
                n.name for n in normalized.graph.find_nodes(op="placeholder")
            ]
            problems[problem_id] = {
                "id": problem_id,
                "signature": f"def kernel_function({', '.join(placeholders)})",
                "inputs": [_describe(v, names) for v in inputs],
                "outputs": [_describe(v, names) for v in outputs],
                "mutated_inputs": sorted(written),
                "output_aliases": [
                    f"in{b}" if isinstance(b, int) else None for b in output_storage
                ],
                "gate": "bitwise",
                "cases": [],
                "source": {
                    "capture": capture_dir,
                    "region": m.group(1),
                    "calls": [],
                },
                "env": env,
            }
            gms[problem_id] = (normalized, [module for module, _, _ in op_deps])
            calls_of[problem_id] = []
        problems[problem_id]["source"]["calls"].append(call)
        calls_of[problem_id].append(node)

    for problem_id, spec in problems.items():
        directory = root / problem_id
        directory.mkdir(parents=True, exist_ok=True)
        normalized, op_modules = gms[problem_id]
        docstring = _docstring(
            problem_id, spec["source"]["region"], capture_dir, spec["source"]["calls"]
        )
        (directory / "problem.py").write_text(
            _problem_source(docstring, normalized, op_modules)
        )
        _write_json(directory / "spec.json", spec)
        (directory / "inputs.pt").unlink(missing_ok=True)
        problem = _Problem(directory, spec)
        for node in calls_of[problem_id]:
            gm.add_submodule(
                node.target,
                _SaveCases(gm.get_submodule(node.target), problem, node.target),
            )

    index = []
    for spec in problems.values():
        calls = spec["source"]["calls"]
        entry = {
            "id": spec["id"],
            "region": spec["source"]["region"],
            "calls": len(calls),
            "measured_us": None,
            "ideal_us": None,
            "gap_us": None,
        }
        # Per step; only when every call is an instance of the capture.
        if all(c["measured_us"] is not None for c in calls):
            entry["measured_us"] = round(sum(c["measured_us"] for c in calls), 1)
            entry["ideal_us"] = round(sum(c["ideal_us"] for c in calls), 1)
            entry["gap_us"] = round(
                sum(max(c["measured_us"] - c["ideal_us"], 0.0) for c in calls), 1
            )
        index.append(entry)
    index.sort(key=lambda p: -(p["gap_us"] if p["gap_us"] is not None else -1.0))
    _write_json(
        root / "problems.json",
        {"capture": capture_dir, "problems": index, "uncacheable": uncacheable},
    )
    return gm


def register_kernel_problems(capture_dir: str, problems_dir: str) -> str:
    """Register the pass that writes the lifted regions of the graph as problems in
    ``problems_dir``, with times from the capture in ``capture_dir``; return its
    name, for ``compile.extra_passes`` after the lift passes."""

    def write_kernel_problems(
        gm: fx.GraphModule, example_inputs: Any = None
    ) -> fx.GraphModule:
        return _write_problems(gm, capture_dir, problems_dir)

    register_graph_pass(write_kernel_problems)
    return write_kernel_problems.__name__
