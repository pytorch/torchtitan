# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.substitute_kernels import (
    cache_keys,
    load_module,
    register_kernel_store,
    substitute_with_kernels_pass,
)
from torchtitan.experiments.graph_trainer.tests import _substitute_kernels_ops

_REPO_ROOT = Path(__file__).resolve().parents[4]

# An op that is not a torch.library.custom_op, hence not cacheable.
_lib = torch.library.Library("sk_test_lib", "DEF")
_lib.define("neg(Tensor x) -> Tensor")
_lib.impl("neg", lambda x: -x, "CompositeExplicitAutograd")


def _load_module(path):
    return load_module(path.read_text())


class _MemoryKernelStore:
    """In-memory ``memtest://<name>`` store; ``stores[name]`` maps path -> bytes."""

    stores: dict[str, dict[str, bytes]] = {}

    def __init__(self, location):
        self.files = self.stores.setdefault(location, {})

    def get(self, path):
        return self.files.get(path)

    def put(self, path, data):
        self.files[path] = data


register_kernel_store("memtest", _MemoryKernelStore)


def _swiglu(x, w1, w2):
    return torch.nn.functional.silu(x @ w1) * (x @ w2)


def _scaled_swiglu_op(x, w1, w2):
    return torch.ops.sk_test.scale(_swiglu(x, w1, w2), 2.0)


def _swiglu_triton_add(x, w1, w2):
    return torch.ops.sk_test.add(torch.nn.functional.silu(x @ w1), x @ w2)


def _inputs(m=8, k=16, dtype=torch.float32, device="cpu"):
    return (
        torch.randn(m, k, dtype=dtype, device=device),
        torch.randn(k, k, dtype=dtype, device=device),
        torch.randn(k, k, dtype=dtype, device=device),
    )


def _layer(fn=_swiglu, name_suffix="", device="cpu", extra_args=()):
    """Trace ``fn`` into a submodule, optionally renaming its nodes the way an
    enclosing graph would (``mm`` -> ``mm_17``)."""
    gm = make_fx(fn)(*_inputs(device=device), *extra_args)
    if name_suffix:
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                node._rename(node.name + name_suffix)
        gm.recompile()
    return gm


def _scaled_swiglu(scale):
    def fn(x, w1, w2):
        return _swiglu(x, w1, w2) * scale

    return fn


def _swiglu_times(x, w1, w2, c):
    return _swiglu(x, w1, w2) * c


def _parent(layers, args, fake=False):
    """``h = x; h = layer_i(h + 1 if i else h, *args[1:])`` as ``call_module``s.

    Every node carries ``meta["val"]``, fake if ``fake`` else real tensors.
    """
    mode = FakeTensorMode() if fake else contextlib.nullcontext()
    vals = tuple(mode.from_tensor(a) for a in args) if fake else args
    root = torch.nn.Module()
    graph = fx.Graph()
    inputs = [graph.placeholder(f"arg{i}") for i in range(len(args))]
    for node, val in zip(inputs, vals):
        node.meta["val"] = val
    h, h_val = inputs[0], vals[0]
    for i, layer in enumerate(layers):
        with mode:
            if i:
                h = graph.call_function(torch.ops.aten.add.Tensor, (h, 1))
                h_val = h.meta["val"] = h_val + 1
            root.add_module(f"layer{i}", layer)
            h = graph.call_module(f"layer{i}", (h, *inputs[1:]))
            h_val = h.meta["val"] = layer(h_val, *vals[1:])
    graph.output(h)
    return fx.GraphModule(root, graph)


def _keys(gm):
    """(l1, l2) for each ``call_module`` in ``gm``, as computed by the pass."""
    return [
        cache_keys(gm.get_submodule(n.target), [a.meta["val"] for a in n.args])[:2]
        for n in gm.graph.find_nodes(op="call_module")
    ]


class TestSubstituteWithKernelsPass(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _patterns(self):
        return sorted(p for p in self.cache.iterdir() if p.is_dir())

    def _spec_text(self, l1, l2):
        """The ``SPEC`` header of ``<l1>/<l2>.py``."""
        text = (self.cache / l1 / f"{l2}.py").read_text()
        return text.split('SPEC = json.loads(r"""\n', 1)[1].split('\n""")', 1)[0]

    def _reference_text(self, path):
        """The imports and ``reference`` function of the specialization file ``path``."""
        text = path.read_text()
        imports = text[text.index("import json") : text.index("SPEC =")]
        return f"{imports.strip()}\n\n{text[text.index('def reference') : text.index('def impl')].strip()}"

    def _run_in_fresh_process(self, reference, args):
        """Run the specialization file ``reference`` on ``args`` in a new interpreter."""
        inputs, output = self.cache / "inputs.pt", self.cache / "output.pt"
        torch.save(args, inputs)
        script = (
            "import importlib.util, torch\n"
            f"spec = importlib.util.spec_from_file_location('ref', {str(reference)!r})\n"
            "ref = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(ref)\n"
            f"torch.save(ref.run(*torch.load({str(inputs)!r})), {str(output)!r})\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_REPO_ROOT), env.get("PYTHONPATH", "")]
        )
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return torch.load(output, weights_only=True)

    def test_renamed_identical_submodules_share_pattern_and_specialization(self):
        args = _inputs()
        gm = _parent([_layer(), _layer(name_suffix="_17")], args)
        keys = _keys(gm)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])

    def test_fake_and_real_meta_vals_give_same_keys(self):
        args = _inputs()
        real = _keys(_parent([_layer()], args))
        fake = _keys(_parent([_layer()], args, fake=True))
        self.assertEqual(real, fake)

    def test_device_index_does_not_change_keys(self):
        # Every rank traces its regions on its own device; all ranks must find
        # the same kernels.
        def fn(x):
            return x + torch.ops.aten.scalar_tensor.default(1.0, device=x.device)

        keys = []
        for device in ("cuda:0", "cuda:1"):
            with FakeTensorMode():
                x = torch.empty(8, 16, device=device)
            gm = make_fx(fn, tracing_mode="fake")(x)
            keys.append(cache_keys(gm, [x])[:2])
        self.assertEqual(keys[0], keys[1])

    def test_symbolic_sizes_are_keyed_by_description(self):
        # A token dispatcher's routed-row count is a new unbacked symbol in every
        # layer; identical regions must still share one specialization.
        layer = _layer(lambda x, w1, w2: torch.nn.functional.silu(x) * 2)
        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            vals = [
                torch.empty(shape_env.create_unbacked_symint(), 16) for _ in range(2)
            ]
            w = torch.empty(16, 16)
            out = torch.empty(vals[1].shape[0], 16)
        keys = [cache_keys(layer, [x, w, w])[:2] for x in vals]
        self.assertEqual(keys[0], keys[1])

        root = torch.nn.Module()
        root.add_module("layer0", layer)
        graph = fx.Graph()
        inputs = [graph.placeholder(f"arg{i}") for i in range(3)]
        for node, val in zip(inputs, (vals[1], w, w)):
            node.meta["val"] = val
        call = graph.call_module("layer0", tuple(inputs))
        call.meta["val"] = out
        graph.output(call)
        substitute_with_kernels_pass(fx.GraphModule(root, graph), cache_dir=self.cache)
        spec = json.loads(self._spec_text(*keys[0]))
        self.assertEqual(spec["inputs"][0]["shape"], ["N0", 16])
        self.assertEqual(spec["outputs"]["shape"], ["N0", 16])

    def test_symbolic_sizes_keep_literals_in_specialization(self):
        # The pattern ignores literals, so with symbolic sizes the specialization
        # must still tell `* 2` from `* 3`.
        def silu_times(c):
            def fn(x, w1, w2):
                return torch.nn.functional.silu(x) * c

            return fn

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            x = torch.empty(shape_env.create_unbacked_symint(), 16)
            w = torch.empty(16, 16)
        keys = [cache_keys(_layer(silu_times(c)), [x, w, w])[:2] for c in (2, 3)]
        self.assertEqual(keys[0][0], keys[1][0])
        self.assertNotEqual(keys[0][1], keys[1][1])

    def test_miss_keeps_call_module_and_writes_specialization_file(self):
        args = _inputs()
        gm = _parent([_layer(), _layer(name_suffix="_17")], args)
        ((_, l2), _) = _keys(gm)
        expected = gm(*args)
        gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)

        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 2)
        self.assertEqual(gm(*args), expected)
        (pattern,) = self._patterns()
        self.assertEqual(sorted(p.name for p in pattern.iterdir()), [f"{l2}.py"])
        module = _load_module(pattern / f"{l2}.py")
        self.assertIs(module.run, module.reference)
        self.assertEqual(module.reference(*args), _swiglu(*args))
        self.assertExpectedInline(
            (pattern / f"{l2}.py").read_text().strip(),
            """\
\"\"\"One specialization of a substitute_kernels pattern.

SPEC and reference are generated: do not edit them. Write the kernel in
impl, then set ``run = impl``.
\"\"\"

import json
import torch

SPEC = json.loads(r\"\"\"
{
  "signature": "def run(in0, in1, in2)",
  "inputs": [
    {
      "shape": [8, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    }
  ],
  "outputs": {
    "shape": [8, 16],
    "stride": [16, 1],
    "dtype": "float32",
    "device": "cpu"
  }
}
\"\"\")


def reference(in0, in1, in2):
    v3 = torch.ops.aten.mm.default(in0, in1);  in1 = None
    v4 = torch.ops.aten.silu.default(v3);  v3 = None
    v5 = torch.ops.aten.mm.default(in0, in2);  in0 = in2 = None
    v6 = torch.ops.aten.mul.Tensor(v4, v5);  v4 = v5 = None
    return v6


def impl(in0, in1, in2):
    raise NotImplementedError


run = reference""",
        )

    def test_reference_imports_only_fx_builtins_it_uses(self):
        def fn(x, w1, w2):
            masked = x.masked_fill(x > 0, float("-inf"))
            return masked + torch.ones(x.shape, device=x.device)

        args = _inputs()
        gm = _parent([_layer(fn)], args)
        expected = gm(*args)
        substitute_with_kernels_pass(gm, cache_dir=self.cache)

        (pattern,) = self._patterns()
        (reference,) = pattern.iterdir()
        self.assertExpectedInline(
            self._reference_text(reference),
            """\
import json
from math import inf
from torch import device
import torch

def reference(in0, in1, in2):
    v3 = torch.ops.aten.gt.Scalar(in0, 0)
    v4 = torch.ops.aten.masked_fill.Scalar(in0, v3, -inf);  in0 = v3 = None
    v5 = torch.ops.aten.ones.default([8, 16], device = device(type='cpu'), pin_memory = False)
    v6 = torch.ops.aten.add.Tensor(v4, v5);  v4 = v5 = None
    return v6""",
        )
        self.assertEqual(self._run_in_fresh_process(reference, args), expected)

    def _write_kernel(self, path, value):
        path.write_text(
            "import torch\n"
            "def run(x, w1, w2):\n"
            f"    return torch.full((x.shape[0], w1.shape[1]), {value})\n"
        )

    def test_hit_replaces_call_module_with_kernel_run(self):
        args = _inputs()
        gm = _parent([_layer(), _layer(name_suffix="_17")], args)
        ((l1, l2), _) = _keys(gm)
        (self.cache / l1).mkdir()
        self._write_kernel(self.cache / l1 / f"{l2}.py", 42.0)
        gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)

        self.assertEqual(gm.graph.find_nodes(op="call_module"), [])
        self.assertFalse(hasattr(gm, "layer0") or hasattr(gm, "layer1"))
        self.assertEqual(gm(*args), torch.full((8, 16), 42.0))
        # The kernel file is imported once and shared by both layers.
        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        targets.discard(torch.ops.aten.add.Tensor)
        self.assertEqual(len(targets), 1)

    def test_shape_and_dtype_change_specialization_not_pattern(self):
        keys = []
        for args in (_inputs(m=8), _inputs(m=32), _inputs(dtype=torch.float64)):
            gm = _parent([_layer()], args)
            keys += _keys(gm)
            substitute_with_kernels_pass(gm, cache_dir=self.cache)
        self.assertEqual(len({l1 for l1, _ in keys}), 1)
        self.assertEqual(len({l2 for _, l2 in keys}), 3)
        self.assertExpectedInline(
            self._spec_text(*keys[2]),
            """\
{
  "signature": "def run(in0, in1, in2)",
  "inputs": [
    {
      "shape": [8, 16],
      "stride": [16, 1],
      "dtype": "float64",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float64",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float64",
      "device": "cpu"
    }
  ],
  "outputs": {
    "shape": [8, 16],
    "stride": [16, 1],
    "dtype": "float64",
    "device": "cpu"
  }
}""",
        )

    def test_same_submodule_different_shapes_writes_one_specialization_each(self):
        layer = _layer()
        specs = []
        for m in (8, 32):
            args = _inputs(m=m)
            gm = _parent([layer], args)
            expected = gm(*args)
            ((l1, l2),) = _keys(gm)
            gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)
            self.assertEqual(gm(*args), expected)
            specs.append((l1, l2))

        # One pattern folder, one specialization per input shape.
        ((l1_a, l2_a), (l1_b, l2_b)) = specs
        self.assertEqual(l1_a, l1_b)
        self.assertNotEqual(l2_a, l2_b)
        (pattern,) = self._patterns()
        self.assertEqual(
            sorted(p.name for p in pattern.iterdir()),
            sorted([f"{l2_a}.py", f"{l2_b}.py"]),
        )
        self.assertExpectedInline(
            self._spec_text(l1_b, l2_b),
            """\
{
  "signature": "def run(in0, in1, in2)",
  "inputs": [
    {
      "shape": [32, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    }
  ],
  "outputs": {
    "shape": [32, 16],
    "stride": [16, 1],
    "dtype": "float32",
    "device": "cpu"
  }
}""",
        )

    def test_scalar_constant_changes_specialization_not_pattern(self):
        args = _inputs()
        keys = [
            _keys(_parent([_layer(_scaled_swiglu(s))], args))[0] for s in (2.0, 3.0)
        ]
        self.assertEqual(keys[0][0], keys[1][0])
        self.assertNotEqual(keys[0][1], keys[1][1])

    def test_view_position_changes_specialization_not_pattern(self):
        # Moving an independent view also reorders the inputs by first use; the
        # pattern ignores both, the specialization does not.
        def view_first(w, x):
            v = w.view(16, 16)
            return torch.ops.aten.mm.default(x + 1, v)

        def view_last(x, w):
            s = x + 1
            return torch.ops.aten.mm.default(s, w.view(16, 16))

        def other_op(x, w):
            return torch.ops.aten.mm.default(x - 1, w.view(16, 16))

        x, w = torch.randn(8, 16), torch.randn(256)
        first = cache_keys(make_fx(view_first)(w, x), [w, x])[:2]
        last = cache_keys(make_fx(view_last)(x, w), [x, w])[:2]
        other = cache_keys(make_fx(other_op)(x, w), [x, w])[:2]
        self.assertEqual(first[0], last[0])
        self.assertNotEqual(first[1], last[1])
        self.assertNotEqual(first[0], other[0])

    def test_submodule_with_tensor_attribute_is_skipped(self):
        args = _inputs()
        layer = _layer(_scaled_swiglu(torch.full((16,), 2.0)))
        self.assertTrue(layer.graph.find_nodes(op="get_attr"))
        gm = substitute_with_kernels_pass(_parent([layer], args), cache_dir=self.cache)
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 1)
        self.assertEqual(self._patterns(), [])

    def test_tensor_constant_passed_as_argument_shares_pattern(self):
        args = _inputs()
        for value in (2.0, 3.0):
            c = torch.full((16,), value)
            layer = _layer(_swiglu_times, extra_args=(c,))
            gm = _parent([layer], (*args, c))
            gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)
            self.assertEqual(gm(*args, c), _swiglu(*args) * value)

        ((l1, l2),) = _keys(gm)
        self.assertEqual(
            sorted(p.name for p in (self.cache / l1).iterdir()), [f"{l2}.py"]
        )
        self.assertExpectedInline(
            self._spec_text(l1, l2),
            """\
{
  "signature": "def run(in0, in1, in2, in3)",
  "inputs": [
    {
      "shape": [8, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16],
      "stride": [1],
      "dtype": "float32",
      "device": "cpu"
    }
  ],
  "outputs": {
    "shape": [8, 16],
    "stride": [16, 1],
    "dtype": "float32",
    "device": "cpu"
  }
}""",
        )
        run = _load_module(self.cache / l1 / f"{l2}.py").reference
        for value in (2.0, 3.0):
            c = torch.full((16,), value)
            self.assertEqual(run(*args, c), _swiglu(*args) * value)

    def test_specialization_describes_tuple_outputs_and_strides(self):
        def fn(x, w1, w2):
            return torch.nn.functional.silu(x @ w1), (x @ w2).t()

        args = _inputs()
        gm = _parent([_layer(fn)], args)
        substitute_with_kernels_pass(gm, cache_dir=self.cache)
        ((l1, l2),) = _keys(gm)
        self.assertExpectedInline(
            self._spec_text(l1, l2),
            """\
{
  "signature": "def run(in0, in1, in2)",
  "inputs": [
    {
      "shape": [8, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    }
  ],
  "outputs": [
    {
      "shape": [8, 16],
      "stride": [16, 1],
      "dtype": "float32",
      "device": "cpu"
    },
    {
      "shape": [16, 8],
      "stride": [1, 16],
      "dtype": "float32",
      "device": "cpu"
    }
  ]
}""",
        )

    def test_submodule_with_higher_order_op_subgraph_is_skipped(self):
        def fn(x, w1, w2):
            return torch.cond(x.sum() > 0, torch.sin, torch.cos, (x @ w1,))

        args = _inputs()
        layer = _layer(fn)
        self.assertTrue(layer.graph.find_nodes(op="get_attr"))
        gm = substitute_with_kernels_pass(_parent([layer], args), cache_dir=self.cache)
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 1)
        self.assertEqual(self._patterns(), [])

    def test_custom_op_reference_imports_op_module_and_runs_in_fresh_process(self):
        args = _inputs()
        gm = _parent([_layer(_scaled_swiglu_op)], args)
        expected = gm(*args)
        substitute_with_kernels_pass(gm, cache_dir=self.cache)

        (pattern,) = self._patterns()
        (reference,) = pattern.iterdir()
        self.assertExpectedInline(
            self._reference_text(reference),
            """\
import json
import torch
import torchtitan.experiments.graph_trainer.tests._substitute_kernels_ops  # sk_test::scale

def reference(in0, in1, in2):
    v3 = torch.ops.aten.mm.default(in0, in1);  in1 = None
    v4 = torch.ops.aten.silu.default(v3);  v3 = None
    v5 = torch.ops.aten.mm.default(in0, in2);  in0 = in2 = None
    v6 = torch.ops.aten.mul.Tensor(v4, v5);  v4 = v5 = None
    v7 = torch.ops.sk_test.scale.default(v6, 2.0);  v6 = None
    return v7""",
        )
        self.assertEqual(self._run_in_fresh_process(reference, args), expected)

    @unittest.skipIf(
        _substitute_kernels_ops.triton is None or not torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_triton_op_reference_imports_op_module_and_runs_in_fresh_process(self):
        args = _inputs(device="cuda")
        gm = _parent([_layer(_swiglu_triton_add, device="cuda")], args)
        expected = gm(*args)
        substitute_with_kernels_pass(gm, cache_dir=self.cache)

        (pattern,) = self._patterns()
        (reference,) = pattern.iterdir()
        self.assertIn(
            "import torchtitan.experiments.graph_trainer.tests._substitute_kernels_ops"
            "  # sk_test::add",
            reference.read_text(),
        )
        self.assertEqual(self._run_in_fresh_process(reference, args), expected)

    def test_custom_op_source_edit_changes_pattern_not_specialization(self):
        module_dir = Path(self._tmp.name) / "ops_src"
        module_dir.mkdir()
        source = module_dir / "sk_tmp_ops.py"
        source.write_text(
            "import torch\n"
            "@torch.library.custom_op('sk_tmp::double', mutates_args=())\n"
            "def double(x: torch.Tensor) -> torch.Tensor:\n"
            "    return x * 2\n"
        )
        sys.path.insert(0, str(module_dir))
        try:
            import sk_tmp_ops  # noqa: F401

            args = _inputs()
            gm = _parent([_layer(lambda x, w1, w2: torch.ops.sk_tmp.double(x))], args)
            before = _keys(gm)
            source.write_text(source.read_text() + "# edited\n")
            after = _keys(gm)
        finally:
            sys.path.remove(str(module_dir))
            sys.modules.pop("sk_tmp_ops", None)
        self.assertNotEqual(before[0][0], after[0][0])
        self.assertEqual(before[0][1], after[0][1])

    def test_submodule_with_library_defined_op_is_skipped(self):
        args = _inputs()
        gm = _parent([_layer(lambda x, w1, w2: torch.ops.sk_test_lib.neg(x))], args)
        gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 1)
        self.assertEqual(self._patterns(), [])

    def test_registered_store_scheme_serves_cache_dir(self):
        args = _inputs()
        make_gm = lambda: _parent([_layer()], args)  # noqa: E731
        ((l1, l2),) = _keys(make_gm())
        files = _MemoryKernelStore.stores.setdefault("bucket/prefix", {})
        files.clear()

        gm = substitute_with_kernels_pass(
            make_gm(), cache_dir="memtest://bucket/prefix"
        )
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 1)
        self.assertEqual(sorted(files), [f"{l1}/{l2}.py"])
        self.assertEqual(self._patterns(), [])  # nothing written locally

        files[f"{l1}/{l2}.py"] = (
            b"import torch\n"
            b"def run(x, w1, w2):\n"
            b"    return torch.full((x.shape[0], w1.shape[1]), 5.0)\n"
        )
        gm = substitute_with_kernels_pass(
            make_gm(), cache_dir="memtest://bucket/prefix"
        )
        self.assertEqual(gm(*args), torch.full((8, 16), 5.0))

    def test_unknown_store_scheme_raises(self):
        gm = _parent([_layer()], _inputs())
        with self.assertRaisesRegex(ValueError, "no kernel store registered"):
            substitute_with_kernels_pass(gm, cache_dir="nosuchscheme://x")

    def test_manifold_cache_dir_requires_internal_pytorch(self):
        try:
            import torch._inductor.fb.remote_cache  # noqa: F401
        except ImportError:
            pass
        else:
            self.skipTest("internal PyTorch build: Manifold is available")
        gm = _parent([_layer()], _inputs())
        with self.assertRaisesRegex(ImportError, "requires an internal build"):
            substitute_with_kernels_pass(
                gm, cache_dir="manifold://torchtitan_kernel_cache/test"
            )

    def test_non_graph_module_and_missing_meta_val_call_modules_are_skipped(self):
        args = _inputs()
        gm = _parent([_layer()], args)
        gm.layer0_linear = torch.nn.Linear(16, 16)
        x = next(iter(gm.graph.nodes))
        with gm.graph.inserting_after(x):
            gm.graph.call_module("layer0_linear", (x,))  # not a GraphModule
        del gm.graph.find_nodes(op="placeholder")[0].meta["val"]  # no example input
        gm.recompile()

        gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 2)
        self.assertEqual(self._patterns(), [])


if __name__ == "__main__":
    run_tests()
