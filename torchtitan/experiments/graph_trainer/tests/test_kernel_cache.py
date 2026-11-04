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
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.kernel_cache import (
    _load_run,
    cache_keys,
    substitute_with_kernels_pass,
)
from torchtitan.experiments.graph_trainer.tests import _kernel_cache_ops

_REPO_ROOT = Path(__file__).resolve().parents[4]

# An op that is not a torch.library.custom_op, hence not cacheable.
_lib = torch.library.Library("kc_test_lib", "DEF")
_lib.define("neg(Tensor x) -> Tensor")
_lib.impl("neg", lambda x: -x, "CompositeExplicitAutograd")


def _swiglu(x, w1, w2):
    return torch.nn.functional.silu(x @ w1) * (x @ w2)


def _scaled_swiglu_op(x, w1, w2):
    return torch.ops.kc_test.scale(_swiglu(x, w1, w2), 2.0)


def _swiglu_triton_add(x, w1, w2):
    return torch.ops.kc_test.add(torch.nn.functional.silu(x @ w1), x @ w2)


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


class TestSubstituteWithKernels(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _patterns(self):
        return sorted(p for p in self.cache.iterdir() if p.is_dir())

    def _spec_text(self, l1, l2):
        """The spec of ``(l1, l2)``, with the torch-dependent hash redacted."""
        return (
            (self.cache / l1 / f"specialization_{l2}.json")
            .read_text()
            .strip()
            .replace(l2, "<l2>")
        )

    def _run_in_fresh_process(self, reference, args):
        """Run ``reference.py`` on ``args`` in a new interpreter."""
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

    def test_identical_layers_share_keys(self):
        args = _inputs()
        gm = _parent([_layer(), _layer(name_suffix="_17")], args)
        keys = _keys(gm)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])

    def test_fake_and_real_inputs_share_keys(self):
        args = _inputs()
        real = _keys(_parent([_layer()], args))
        fake = _keys(_parent([_layer()], args, fake=True))
        self.assertEqual(real, fake)

    def test_miss_keeps_reference(self):
        args = _inputs()
        gm = _parent([_layer(), _layer(name_suffix="_17")], args)
        ((_, l2), _) = _keys(gm)
        expected = gm(*args)
        gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)

        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 2)
        self.assertEqual(gm(*args), expected)
        (pattern,) = self._patterns()
        self.assertEqual(
            sorted(p.name for p in pattern.iterdir()),
            ["reference.py", f"specialization_{l2}.json"],
        )
        self.assertExpectedInline(
            (pattern / f"specialization_{l2}.json")
            .read_text()
            .strip()
            .replace(l2, "<l2>"),
            """\
{
  "kernel_name": "<l2>.py",
  "reference": "reference.py",
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
}""",
        )
        self.assertEqual(_load_run(pattern / "reference.py")(*args), _swiglu(*args))
        self.assertExpectedInline(
            (pattern / "reference.py").read_text().strip(),
            """\
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.mm.default(in0, in1);  in1 = None
        v4 = torch.ops.aten.silu.default(v3);  v3 = None
        v5 = torch.ops.aten.mm.default(in0, in2);  in0 = in2 = None
        v6 = torch.ops.aten.mul.Tensor(v4, v5);  v4 = v5 = None
        return v6


def run(*args):
    return Reference()(*args)""",
        )

    def test_reference_imports_only_used_builtins(self):
        def fn(x, w1, w2):
            masked = x.masked_fill(x > 0, float("-inf"))
            return masked + torch.ones(x.shape, device=x.device)

        args = _inputs()
        gm = _parent([_layer(fn)], args)
        expected = gm(*args)
        substitute_with_kernels_pass(gm, cache_dir=self.cache)

        (pattern,) = self._patterns()
        reference = pattern / "reference.py"
        self.assertExpectedInline(
            reference.read_text().strip(),
            """\
from math import inf
from torch import device
import torch


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.gt.Scalar(in0, 0)
        v4 = torch.ops.aten.masked_fill.Scalar(in0, v3, -inf);  in0 = v3 = None
        v5 = torch.ops.aten.ones.default([8, 16], device = device(type='cpu'), pin_memory = False)
        v6 = torch.ops.aten.add.Tensor(v4, v5);  v4 = v5 = None
        return v6


def run(*args):
    return Reference()(*args)""",
        )
        self.assertEqual(self._run_in_fresh_process(reference, args), expected)

    def _write_kernel(self, path, value):
        path.write_text(
            "import torch\n"
            "def run(x, w1, w2):\n"
            f"    return torch.full((x.shape[0], w1.shape[1]), {value})\n"
        )

    def test_hit_substitutes_kernel(self):
        args = _inputs()
        gm = _parent([_layer(), _layer(name_suffix="_17")], args)
        ((l1, l2), _) = _keys(gm)
        (self.cache / l1).mkdir()
        self._write_kernel(self.cache / l1 / f"{l2}.py", 42.0)
        gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)

        self.assertEqual(gm.graph.find_nodes(op="call_module"), [])
        self.assertFalse(hasattr(gm, "layer0") or hasattr(gm, "layer1"))
        self.assertEqual(gm(*args), torch.full((8, 16), 42.0))

    def test_kernel_name_selects_kernel_file(self):
        args = _inputs()
        make_gm = lambda: _parent([_layer()], args)  # noqa: E731
        substitute_with_kernels_pass(make_gm(), cache_dir=self.cache)  # writes spec
        ((l1, l2),) = _keys(make_gm())
        spec_path = self.cache / l1 / f"specialization_{l2}.json"
        spec = json.loads(spec_path.read_text())
        spec["kernel_name"] = "swiglu_fast.py"
        spec_path.write_text(json.dumps(spec))
        self._write_kernel(self.cache / l1 / f"{l2}.py", 1.0)  # ignored now
        self._write_kernel(self.cache / l1 / "swiglu_fast.py", 7.0)

        gm = substitute_with_kernels_pass(make_gm(), cache_dir=self.cache)
        self.assertEqual(gm(*args), torch.full((8, 16), 7.0))

    def test_shape_and_dtype_change_spec_not_pattern(self):
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
  "kernel_name": "<l2>.py",
  "reference": "reference.py",
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

    def test_same_submodule_different_shapes(self):
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
            sorted(
                [
                    "reference.py",
                    f"specialization_{l2_a}.json",
                    f"specialization_{l2_b}.json",
                ]
            ),
        )
        self.assertExpectedInline(
            self._spec_text(l1_b, l2_b),
            """\
{
  "kernel_name": "<l2>.py",
  "reference": "reference.py",
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

    def test_scalar_constant_changes_pattern(self):
        args = _inputs()
        keys = [_keys(_parent([_layer(_scaled_swiglu(s))], args)) for s in (2.0, 3.0)]
        self.assertNotEqual(keys[0][0][0], keys[1][0][0])

    def test_tensor_attribute_is_not_cacheable(self):
        args = _inputs()
        layer = _layer(_scaled_swiglu(torch.full((16,), 2.0)))
        self.assertTrue(layer.graph.find_nodes(op="get_attr"))
        gm = substitute_with_kernels_pass(_parent([layer], args), cache_dir=self.cache)
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 1)
        self.assertEqual(self._patterns(), [])

    def test_tensor_constant_argument_shares_pattern(self):
        args = _inputs()
        for value in (2.0, 3.0):
            c = torch.full((16,), value)
            layer = _layer(_swiglu_times, extra_args=(c,))
            gm = _parent([layer], (*args, c))
            gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)
            self.assertEqual(gm(*args, c), _swiglu(*args) * value)

        ((l1, l2),) = _keys(gm)
        self.assertEqual(
            sorted(p.name for p in (self.cache / l1).iterdir()),
            ["reference.py", f"specialization_{l2}.json"],
        )
        self.assertExpectedInline(
            self._spec_text(l1, l2),
            """\
{
  "kernel_name": "<l2>.py",
  "reference": "reference.py",
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
        run = _load_run(self.cache / l1 / "reference.py")
        for value in (2.0, 3.0):
            c = torch.full((16,), value)
            self.assertEqual(run(*args, c), _swiglu(*args) * value)

    def test_spec_describes_tuple_outputs(self):
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
  "kernel_name": "<l2>.py",
  "reference": "reference.py",
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

    def test_subgraph_is_not_cacheable(self):
        def fn(x, w1, w2):
            return torch.cond(x.sum() > 0, torch.sin, torch.cos, (x @ w1,))

        args = _inputs()
        layer = _layer(fn)
        self.assertTrue(layer.graph.find_nodes(op="get_attr"))
        gm = substitute_with_kernels_pass(_parent([layer], args), cache_dir=self.cache)
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 1)
        self.assertEqual(self._patterns(), [])

    def test_custom_op_reference_runs_in_fresh_process(self):
        args = _inputs()
        gm = _parent([_layer(_scaled_swiglu_op)], args)
        expected = gm(*args)
        substitute_with_kernels_pass(gm, cache_dir=self.cache)

        (pattern,) = self._patterns()
        reference = pattern / "reference.py"
        self.assertExpectedInline(
            reference.read_text().strip(),
            """\
import torch
import torchtitan.experiments.graph_trainer.tests._kernel_cache_ops  # kc_test::scale


class Reference(torch.nn.Module):
    def forward(self, in0, in1, in2):
        v3 = torch.ops.aten.mm.default(in0, in1);  in1 = None
        v4 = torch.ops.aten.silu.default(v3);  v3 = None
        v5 = torch.ops.aten.mm.default(in0, in2);  in0 = in2 = None
        v6 = torch.ops.aten.mul.Tensor(v4, v5);  v4 = v5 = None
        v7 = torch.ops.kc_test.scale.default(v6, 2.0);  v6 = None
        return v7


def run(*args):
    return Reference()(*args)""",
        )
        self.assertEqual(self._run_in_fresh_process(reference, args), expected)

    @unittest.skipIf(
        _kernel_cache_ops.triton is None or not torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_triton_op_reference_runs_in_fresh_process(self):
        args = _inputs(device="cuda")
        gm = _parent([_layer(_swiglu_triton_add, device="cuda")], args)
        expected = gm(*args)
        substitute_with_kernels_pass(gm, cache_dir=self.cache)

        (pattern,) = self._patterns()
        reference = pattern / "reference.py"
        self.assertIn(
            "import torchtitan.experiments.graph_trainer.tests._kernel_cache_ops"
            "  # kc_test::add",
            reference.read_text(),
        )
        self.assertEqual(self._run_in_fresh_process(reference, args), expected)

    def test_custom_op_source_change_changes_pattern(self):
        module_dir = Path(self._tmp.name) / "ops_src"
        module_dir.mkdir()
        source = module_dir / "kc_tmp_ops.py"
        source.write_text(
            "import torch\n"
            "@torch.library.custom_op('kc_tmp::double', mutates_args=())\n"
            "def double(x: torch.Tensor) -> torch.Tensor:\n"
            "    return x * 2\n"
        )
        sys.path.insert(0, str(module_dir))
        try:
            import kc_tmp_ops  # noqa: F401

            args = _inputs()
            gm = _parent([_layer(lambda x, w1, w2: torch.ops.kc_tmp.double(x))], args)
            before = _keys(gm)
            source.write_text(source.read_text() + "# edited\n")
            after = _keys(gm)
        finally:
            sys.path.remove(str(module_dir))
            sys.modules.pop("kc_tmp_ops", None)
        self.assertNotEqual(before[0][0], after[0][0])
        self.assertEqual(before[0][1], after[0][1])

    def test_non_custom_op_is_not_cacheable(self):
        args = _inputs()
        gm = _parent([_layer(lambda x, w1, w2: torch.ops.kc_test_lib.neg(x))], args)
        gm = substitute_with_kernels_pass(gm, cache_dir=self.cache)
        self.assertEqual(len(gm.graph.find_nodes(op="call_module")), 1)
        self.assertEqual(self._patterns(), [])

    def test_skips_unsupported_call_modules(self):
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
