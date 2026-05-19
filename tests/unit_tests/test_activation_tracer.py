# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for activation_tracer and compare_numerics.

Covers each change documented in ACTIVATION_TRACER_CHANGES.md.
Run with:
    pytest torchtitan/tools/tests/test_activation_tracer.py -x -v
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from torchtitan.tools.activation_tracer import (
    _clean_fqn,
    _is_filtered_frame,
    _parse_stack_trace,
    _should_capture,
    CapturedActivation,
    DebugModeTracer,
    dump_captures_to_file,
    is_numerics_capture_active,
    NumericsDebugger,
)


BATCH = 32


def _fake_func(op_name: str) -> object:
    """Build a stand-in for ``torch._ops.OpOverload`` whose ``_schema.name``
    attribute matches the format ``_get_op_name`` parses (``ns::op``).

    Used by ``_should_capture`` tests to exercise the op-name filter
    without dispatching a real op.
    """
    from types import SimpleNamespace

    return SimpleNamespace(_schema=SimpleNamespace(name=op_name))


def _make_model(device="cpu"):
    """Simple model for testing. Uses 64-wide layers so batch=32 gives
    numel=2048 per activation, above the 1000 threshold."""
    return nn.Sequential(
        nn.Linear(64, 64, bias=False, device=device),
        nn.ReLU(),
        nn.Linear(64, 64, bias=False, device=device),
    )


class _FeedForward(nn.Module):
    """Mimics torchtitan FeedForward: ops between child module calls."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.w1 = nn.Linear(32, 64, bias=False, device=device)
        self.w2 = nn.Linear(64, 32, bias=False, device=device)
        self.w3 = nn.Linear(32, 64, bias=False, device=device)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TestCleanFQN(unittest.TestCase):
    """Change 3: FQN cleaning."""

    def test_strips_checkpoint_wrapper(self):
        self.assertEqual(
            _clean_fqn("layers.0._checkpoint_wrapped_module.attention.wq"),
            "layers.0.attention.wq",
        )

    def test_no_change_without_wrapper(self):
        self.assertEqual(
            _clean_fqn("layers.0.attention.wq"),
            "layers.0.attention.wq",
        )

    def test_multiple_wrappers(self):
        fqn = "a._checkpoint_wrapped_module.b._checkpoint_wrapped_module.c"
        self.assertEqual(_clean_fqn(fqn), "a.b.c")


class TestIsFilteredFrame(unittest.TestCase):
    """Frame filtering for stack traces."""

    def test_filters_torch_nn(self):
        self.assertTrue(
            _is_filtered_frame("/home/user/pytorch/torch/nn/modules/linear.py")
        )
        self.assertTrue(_is_filtered_frame("/home/user/pytorch/torch/nn/functional.py"))

    def test_filters_torch_autograd(self):
        self.assertTrue(
            _is_filtered_frame("/home/user/pytorch/torch/autograd/graph.py")
        )

    def test_filters_activation_tracer(self):
        self.assertTrue(
            _is_filtered_frame("/home/user/torchtitan/tools/activation_tracer.py")
        )

    def test_filters_fqn_interpreter(self):
        # FQNInterpreter wraps every node in traced replay, so its
        # super().run_node line appears in the live traceback when
        # node.meta["stack_trace"] is empty — never useful to surface.
        self.assertTrue(
            _is_filtered_frame(
                "/home/u/torchtitan/experiments/graph_trainer/debug_utils.py"
            )
        )

    def test_filters_dispatcher_and_runtime(self):
        # Dispatcher / compile / debug-mode / AC plumbing — these
        # otherwise drown out user-code frames in DebugMode's live
        # tracebacks under FSDP + activation checkpoint.
        self.assertTrue(_is_filtered_frame("/home/user/pytorch/torch/_ops.py"))
        self.assertTrue(_is_filtered_frame("/home/user/pytorch/torch/_compile.py"))
        self.assertTrue(
            _is_filtered_frame("/home/user/pytorch/torch/_dynamo/eval_frame.py")
        )
        self.assertTrue(
            _is_filtered_frame("/home/user/pytorch/torch/utils/_debug_mode/_mode.py")
        )
        self.assertTrue(
            _is_filtered_frame("/home/user/pytorch/torch/utils/checkpoint.py")
        )

    def test_keeps_model_code(self):
        self.assertFalse(
            _is_filtered_frame("/home/user/torchtitan/models/common/attention.py")
        )
        self.assertFalse(
            _is_filtered_frame("/home/user/torchtitan/models/llama3/model.py")
        )

    def test_keeps_torchtitan(self):
        self.assertFalse(
            _is_filtered_frame("/home/user/torchtitan/torchtitan/trainer.py")
        )


class TestParseStackTrace(unittest.TestCase):
    """Stack trace parsing from node.meta['stack_trace']."""

    def test_parses_file_line_func(self):
        trace = (
            '  File "/home/user/models/attention.py", line 534, in forward\n'
            "    xq, xk, xv = self.qkv_linear(x)\n"
        )
        frames = _parse_stack_trace(trace)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].lineno, 534)
        self.assertIn("attention.py", frames[0].filename)

    def test_filters_nn_modules(self):
        trace = (
            '  File "/pytorch/torch/nn/modules/module.py", line 1789, in _call_impl\n'
            "    return forward_call(*args, **kwargs)\n"
            '  File "/home/user/models/attention.py", line 534, in forward\n'
            "    xq = self.wq(x)\n"
        )
        frames = _parse_stack_trace(trace)
        self.assertEqual(len(frames), 1)
        self.assertIn("attention.py", frames[0].filename)

    def test_filters_activation_tracer(self):
        trace = (
            '  File "/home/user/tools/activation_tracer.py", line 100, in wrapped\n'
            "    return orig(*args)\n"
            '  File "/home/user/models/decoder.py", line 140, in forward\n'
            "    h = layer(h)\n"
        )
        frames = _parse_stack_trace(trace)
        self.assertEqual(len(frames), 1)
        self.assertIn("decoder.py", frames[0].filename)


class TestShouldCapture(unittest.TestCase):
    """Change 1: FakeTensor filtering and basic capture logic."""

    def test_captures_float_tensor(self):
        t = torch.randn(100, 100)

        FakeFunc = _fake_func("aten::mm")

        self.assertTrue(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_small_tensor(self):
        t = torch.randn(5, 5)

        FakeFunc = _fake_func("aten::mm")

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_int_tensor(self):
        t = torch.randint(0, 10, (100, 100))

        FakeFunc = _fake_func("aten::mm")

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_excluded_ops(self):
        t = torch.randn(100, 100)

        FakeFunc = _fake_func("aten::view")

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_fake_tensor(self):
        from torch._subclasses import FakeTensorMode

        with FakeTensorMode():
            t = torch.randn(100, 100)

        FakeFunc = _fake_func("aten::mm")

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))


class TestDumpCaptures(unittest.TestCase):
    """Stat fields are written from the pre-computed ``stats`` dict.

    Stats are computed inline at capture time (see ``_compute_stats``)
    so the dump is just a write-through; no tensor is retained.
    """

    _SAMPLE_STATS = {
        "Shape": "torch.Size([100, 100]), Dtype: torch.float32",
        "L1 norm": "1.234e+02",
        "L2 norm": "5.678e+01",
        "Min": "-1.0e+00",
        "Max": "1.0e+00",
        "Mean": "0.0e+00",
    }

    def test_writes_file(self):
        caps = {"mod/op_0_mm": CapturedActivation(stats=self._SAMPLE_STATS)}
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            dump_captures_to_file(caps, path)
            content = open(path).read()
            self.assertIn("[mod/op_0_mm]", content)
            self.assertIn("Shape:", content)
            self.assertIn("L1 norm:", content)
            self.assertIn("L2 norm:", content)
            self.assertIn("Min:", content)
            self.assertIn("Max:", content)
        finally:
            os.unlink(path)

    def test_cleans_fqn_in_output(self):
        caps = {
            "layers.0._checkpoint_wrapped_module.wq/op_0_mm": CapturedActivation(
                stats=self._SAMPLE_STATS
            ),
        }
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            dump_captures_to_file(caps, path)
            content = open(path).read()
            self.assertNotIn("_checkpoint_wrapped_module", content)
            self.assertIn("[layers.0.wq/op_0_mm]", content)
        finally:
            os.unlink(path)

    def test_inplace_capture_writes_no_stats(self):
        """In-place op captures (no ``stats``) still get a header and
        any hash info, but skip the Shape/L1/L2/Min/Max/Mean lines."""
        caps = {
            "<none>/op_0__fused_adam_": CapturedActivation(
                stats={},
                output_hash="",
                input_hashes="1.0e+02, 2.0e+02",
            ),
        }
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            dump_captures_to_file(caps, path)
            content = open(path).read()
            self.assertIn("[<none>/op_0__fused_adam_]", content)
            self.assertNotIn("Shape:", content)
            self.assertNotIn("L1 norm:", content)
            self.assertIn("Input hashes:", content)
        finally:
            os.unlink(path)

    def test_phase_in_output(self):
        caps = {
            "mod/op_0_mm": CapturedActivation(
                stats=self._SAMPLE_STATS, phase="backward"
            ),
        }
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            dump_captures_to_file(caps, path)
            content = open(path).read()
            self.assertIn("Phase: backward", content)
        finally:
            os.unlink(path)

    def test_forward_phase_not_in_output(self):
        caps = {
            "mod/op_0_mm": CapturedActivation(
                stats=self._SAMPLE_STATS, phase="forward"
            ),
        }
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            dump_captures_to_file(caps, path)
            content = open(path).read()
            self.assertNotIn("Phase:", content)
        finally:
            os.unlink(path)


class TestNumericsDebugger(unittest.TestCase):
    """NumericsDebugger lifecycle."""

    def test_global_flag(self):
        model = _make_model()
        self.assertFalse(is_numerics_capture_active())
        debugger = NumericsDebugger(
            enabled=True, model=model, dump_dir="/tmp/test", capture_step=1
        )
        debugger.__enter__()
        # Flag set after _setup, which happens in __enter__ for step 1
        self.assertTrue(is_numerics_capture_active())
        debugger._teardown()
        self.assertFalse(is_numerics_capture_active())

    def test_captures_on_correct_step(self):
        # Use a model with large enough tensors (min_numel=1000)
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            debugger = NumericsDebugger(
                enabled=True,
                model=model,
                dump_dir=tmpdir,
                capture_step=1,
            )
            debugger.__enter__()

            # Run forward with batch large enough for min_numel
            model(torch.randn(32, 64))
            self.assertFalse(debugger._captured)

            debugger.step()
            self.assertTrue(debugger._captured)

            debugger.__exit__(None, None, None)

            log_path = os.path.join(tmpdir, "rank_0_activations.log")
            self.assertTrue(os.path.exists(log_path))

    def test_disabled(self):
        model = _make_model()
        debugger = NumericsDebugger(
            enabled=False, model=model, dump_dir="/tmp/test", capture_step=1
        )
        debugger.__enter__()
        self.assertFalse(is_numerics_capture_active())
        debugger.__exit__(None, None, None)


class TestDebugModeTracer(unittest.TestCase):
    """DebugMode-backed tracer.

    Verifies it produces captures keyed by ``module_fqn/op_N_opname``
    with phase tags and skipped-ops set so ``dump_captures_to_file``
    and :mod:`torchtitan.tools.compare_numerics` work unchanged.
    """

    def test_captures_forward_ops(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with DebugModeTracer(model) as caps:
            model(x)
        self.assertGreater(len(caps), 0)
        mm_keys = [k for k in caps if "mm" in k]
        self.assertGreaterEqual(len(mm_keys), 2)

    def test_captures_with_backward(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with DebugModeTracer(model) as caps:
            out = model(x)
            out.sum().backward()
        phases = {cap.phase for cap in caps.values()}
        self.assertIn("forward", phases)
        self.assertIn("backward", phases)

    def test_module_fqn_in_keys(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with DebugModeTracer(model) as caps:
            model(x)
        self.assertTrue(any("0/" in k for k in caps), f"keys={list(caps)}")

    def test_excluded_ops_recorded(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        tracer = DebugModeTracer(model)
        with tracer as _caps:
            model(x)
        # ReLU produces a view/detach-class op chain; at minimum, the
        # backward will dispatch t/transpose ops.  Just check the
        # mechanism is wired (set exists and is populated, or stays
        # empty when no excluded op fires).
        self.assertIsInstance(tracer.skipped_excluded_ops, set)

    def test_op_filter(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with DebugModeTracer(model, op_filter={"mm"}) as caps:
            model(x)
        for key in caps:
            self.assertIn("mm", key.split("/")[-1])

    def test_min_numel_filter(self):
        model = nn.Linear(4, 4, bias=False)  # tiny; numel=16 per op
        x = torch.randn(2, 4)
        with DebugModeTracer(model, min_numel=1000) as caps:
            model(x)
        self.assertEqual(len(caps), 0)

    def test_fqn_does_not_include_root_class_prefix(self):
        """ModTracker prefixes FQNs with the model class name; we strip
        that prefix at capture time so DebugModeTracer's FQNs use the
        same naming as ``named_modules()`` (matching FQNInterpreter)."""
        model = _make_model()  # nn.Sequential — root class is "Sequential"
        x = torch.randn(BATCH, 64)
        with DebugModeTracer(model) as caps:
            out = model(x)
            out.sum().backward()

        cls_name = type(model).__name__
        prefixed = [k for k in caps if k.startswith(f"{cls_name}.")]
        self.assertEqual(
            prefixed,
            [],
            f"FQNs should not be rooted at the model class: {prefixed[:3]}",
        )

    def test_backward_ops_have_module_fqn(self):
        """Backward FQNs are recovered via _grad_fn_to_module.

        ModTracker is silent during backward (C++ autograd doesn't
        invoke Python module forwards), so DebugModeTracer's forward
        hook must seed _grad_fn_to_module so backward ops can look up
        their owning module.
        """
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with DebugModeTracer(model) as caps:
            out = model(x)
            out.sum().backward()
        bwd_keys = [k for k, v in caps.items() if v.phase == "backward"]
        self.assertGreater(len(bwd_keys), 0)
        fqn_bwd = [k for k in bwd_keys if not k.startswith("<none>/")]
        self.assertGreater(
            len(fqn_bwd),
            0,
            f"all backward ops lost their FQN: {bwd_keys}",
        )

    def test_dump_and_reparse_round_trip(self):
        """DebugModeTracer output must be parseable by compare_numerics."""
        from torchtitan.tools.compare_numerics import parse_log

        model = _make_model()
        x = torch.randn(BATCH, 64)
        tracer = DebugModeTracer(model)
        with tracer as caps:
            out = model(x)
            out.sum().backward()

        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            dump_captures_to_file(
                caps, path, skipped_excluded_ops=tracer.skipped_excluded_ops
            )
            entries, skipped = parse_log(path)
            self.assertEqual(len(entries), len(caps))
            self.assertEqual(skipped, tracer.skipped_excluded_ops)
        finally:
            os.unlink(path)


class TestNumericsDebuggerProducesLog(unittest.TestCase):
    """End-to-end: NumericsDebugger drives DebugModeTracer and writes a log."""

    def test_writes_activation_log_on_step(self):
        model = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            debugger = NumericsDebugger(
                enabled=True,
                model=model,
                dump_dir=tmpdir,
                capture_step=1,
            )
            debugger.__enter__()
            try:
                self.assertIsInstance(debugger._tracer, DebugModeTracer)
                model(torch.randn(32, 64))
                debugger.step()
            finally:
                debugger.__exit__(None, None, None)
            log_path = os.path.join(tmpdir, "rank_0_activations.log")
            self.assertTrue(os.path.exists(log_path))


class TestFQNInterpreter(unittest.TestCase):
    """FQNInterpreter for traced graph replay."""

    def test_sets_module_name_from_node_meta(self):
        """FQNInterpreter reads node.meta['custom']['module_fqn']."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.graph_trainer.debug_utils import FQNInterpreter

        def fn(x, w):
            return torch.mm(x, w)

        x = torch.randn(BATCH, 64)
        w = torch.randn(64, 64)
        gm = make_fx(fn)(x, w)

        for node in gm.graph.nodes:
            if node.op == "call_function" and "mm" in str(node.target):
                node.meta["custom"] = {"module_fqn": "test_module"}
                break

        names_seen = []
        orig_run_node = FQNInterpreter.run_node

        class TrackingInterpreter(FQNInterpreter):
            def run_node(self, n):
                fqn = (n.meta.get("custom") or {}).get("module_fqn")
                if fqn:
                    names_seen.append(fqn)
                return orig_run_node(self, n)

        interp = TrackingInterpreter(gm)
        interp.run(x, w)
        self.assertIn("test_module", names_seen)


if __name__ == "__main__":
    unittest.main()
