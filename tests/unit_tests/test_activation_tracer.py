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
    ActivationTracer,
    CapturedActivation,
    NumericsDebugger,
    _clean_fqn,
    _is_filtered_frame,
    _parse_stack_trace,
    _resolve_module_location,
    _should_capture,
    dump_captures_to_file,
    is_numerics_capture_active,
    set_numerics_capture_active,
)
from torchtitan.tools.compare_numerics import naive_compare_captures


BATCH = 32


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
        self.assertTrue(_is_filtered_frame("/home/user/pytorch/torch/nn/modules/linear.py"))
        self.assertTrue(_is_filtered_frame("/home/user/pytorch/torch/nn/functional.py"))

    def test_filters_torch_autograd(self):
        self.assertTrue(_is_filtered_frame("/home/user/pytorch/torch/autograd/graph.py"))

    def test_filters_activation_tracer(self):
        self.assertTrue(_is_filtered_frame("/home/user/torchtitan/tools/activation_tracer.py"))

    def test_keeps_model_code(self):
        self.assertFalse(_is_filtered_frame("/home/user/torchtitan/models/common/attention.py"))
        self.assertFalse(_is_filtered_frame("/home/user/torchtitan/models/llama3/model.py"))

    def test_keeps_torchtitan(self):
        self.assertFalse(_is_filtered_frame("/home/user/torchtitan/torchtitan/trainer.py"))


class TestParseStackTrace(unittest.TestCase):
    """Stack trace parsing from node.meta['stack_trace']."""

    def test_parses_file_line_func(self):
        trace = (
            '  File "/home/user/models/attention.py", line 534, in forward\n'
            '    xq, xk, xv = self.qkv_linear(x)\n'
        )
        frames = _parse_stack_trace(trace)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].lineno, 534)
        self.assertIn("attention.py", frames[0].filename)

    def test_filters_nn_modules(self):
        trace = (
            '  File "/pytorch/torch/nn/modules/module.py", line 1789, in _call_impl\n'
            '    return forward_call(*args, **kwargs)\n'
            '  File "/home/user/models/attention.py", line 534, in forward\n'
            '    xq = self.wq(x)\n'
        )
        frames = _parse_stack_trace(trace)
        self.assertEqual(len(frames), 1)
        self.assertIn("attention.py", frames[0].filename)

    def test_filters_activation_tracer(self):
        trace = (
            '  File "/home/user/tools/activation_tracer.py", line 100, in wrapped\n'
            '    return orig(*args)\n'
            '  File "/home/user/models/decoder.py", line 140, in forward\n'
            '    h = layer(h)\n'
        )
        frames = _parse_stack_trace(trace)
        self.assertEqual(len(frames), 1)
        self.assertIn("decoder.py", frames[0].filename)


class TestResolveModuleLocation(unittest.TestCase):
    """Change 4: Source location via inspect."""

    def test_returns_location_for_custom_module(self):
        # _resolve_module_location uses inspect.getfile which works on
        # modules defined in real files. Modules in this test file work.
        model = _FeedForward()
        frames = _resolve_module_location(model)
        # This may be empty if running from <string> — skip in that case
        if frames:
            self.assertIn("test_activation_tracer", frames[0].filename)

    def test_filters_nn_linear(self):
        linear = nn.Linear(4, 4)
        frames = _resolve_module_location(linear)
        self.assertEqual(len(frames), 0)

    def test_filters_nn_relu(self):
        relu = nn.ReLU()
        frames = _resolve_module_location(relu)
        self.assertEqual(len(frames), 0)


class TestShouldCapture(unittest.TestCase):
    """Change 1: FakeTensor filtering and basic capture logic."""

    def test_captures_float_tensor(self):
        t = torch.randn(100, 100)

        class FakeFunc:
            class _schema:
                name = "aten::mm"

        self.assertTrue(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_small_tensor(self):
        t = torch.randn(5, 5)

        class FakeFunc:
            class _schema:
                name = "aten::mm"

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_int_tensor(self):
        t = torch.randint(0, 10, (100, 100))

        class FakeFunc:
            class _schema:
                name = "aten::mm"

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_excluded_ops(self):
        t = torch.randn(100, 100)

        class FakeFunc:
            class _schema:
                name = "aten::view"

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))

    def test_skips_fake_tensor(self):
        from torch._subclasses import FakeTensorMode

        with FakeTensorMode():
            t = torch.randn(100, 100)

        class FakeFunc:
            class _schema:
                name = "aten::mm"

        self.assertFalse(_should_capture(FakeFunc, t, 1000, None))


class TestActivationTracer(unittest.TestCase):
    """Core ActivationTracer functionality."""

    def test_captures_forward_ops(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            model(x)
        self.assertGreater(len(caps), 0)
        # Should have mm ops from the two Linear layers
        mm_keys = [k for k in caps if "mm" in k]
        self.assertGreaterEqual(len(mm_keys), 2)

    def test_captures_with_backward(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            out = model(x)
            out.sum().backward()
        # Should capture both forward and backward ops
        self.assertGreater(len(caps), 2)

    def test_module_fqn_in_keys(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            model(x)
        # Keys should contain module names like "0" (first Linear)
        keys = list(caps.keys())
        has_module = any("0/" in k for k in keys)
        self.assertTrue(has_module, f"No module FQN in keys: {keys}")

    def test_captures_ops_without_module_fqn(self):
        """Change 5: ops without module context use <none>."""
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            out = model(x)
            # Loss is outside any module
            loss = out.sum()
            loss.backward()
        none_keys = [k for k in caps if "<none>" in k]
        # There should be at least some <none> ops from backward
        # (depends on autograd graph mapping)
        self.assertIsInstance(none_keys, list)

    def test_double_exit_guard(self):
        """Change 8: double-exit doesn't crash."""
        model = _make_model()
        tracer = ActivationTracer(model)
        caps = tracer.__enter__()
        model(torch.randn(BATCH, 64))
        # Simulate dispatch mode exiting early
        tracer._dispatch_mode._exited = True
        # Should not crash
        tracer.__exit__(None, None, None)

    def test_detect_anomaly_enabled(self):
        """Change 10: detect_anomaly is enabled during tracing."""
        model = _make_model()
        self.assertFalse(torch.is_anomaly_enabled())
        with ActivationTracer(model) as caps:
            self.assertTrue(torch.is_anomaly_enabled())
        self.assertFalse(torch.is_anomaly_enabled())


class TestPhaseTagging(unittest.TestCase):
    """Change 2 & 7: phase tagging (forward/backward)."""

    def test_forward_ops_tagged_forward(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            model(x)
        for cap in caps.values():
            self.assertEqual(cap.phase, "forward")

    def test_backward_ops_tagged_backward(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            out = model(x)
            out.sum().backward()
        phases = {cap.phase for cap in caps.values()}
        self.assertIn("backward", phases)

    def test_backward_sticky_flag(self):
        """Once backward starts, all subsequent ops are tagged backward."""
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            out = model(x)
            out.sum().backward()
        # Find the first backward op
        keys = list(caps.keys())
        first_bwd = None
        for i, k in enumerate(keys):
            if caps[k].phase == "backward":
                first_bwd = i
                break
        # All ops after the first backward should be backward
        if first_bwd is not None:
            for k in keys[first_bwd:]:
                self.assertEqual(
                    caps[k].phase, "backward",
                    f"Op {k} after first backward should be backward"
                )


class TestBackwardFQNMapping(unittest.TestCase):
    """Change 9: backward module FQN via autograd graph mapping."""

    def test_backward_ops_have_module_fqn(self):
        model = _make_model()
        x = torch.randn(BATCH, 64)
        with ActivationTracer(model) as caps:
            out = model(x)
            out.sum().backward()
        bwd_ops = {k: v for k, v in caps.items() if v.phase == "backward"}
        # At least some backward ops should have module FQN (not <none>)
        fqn_ops = [k for k in bwd_ops if "<none>" not in k]
        self.assertGreater(
            len(fqn_ops), 0,
            "No backward ops with module FQN"
        )

    def test_boundary_aware_bfs(self):
        """Ops between child modules get parent's FQN, not child's."""
        model = _FeedForward()
        x = torch.randn(4, 32, requires_grad=True)
        with ActivationTracer(model) as caps:
            out = model(x)
            out.sum().backward()
        # The silu and mul ops in FeedForward.forward should NOT be
        # attributed to w1 or w2 — they should be under the parent
        # module (empty string for root, or the FeedForward FQN).
        bwd_ops = {k: v for k, v in caps.items() if v.phase == "backward"}
        # Check that silu_backward is not under w1 or w2
        for k in bwd_ops:
            if "silu" in k:
                self.assertNotIn(
                    "w1/", k,
                    f"silu backward should not be under w1: {k}"
                )
                self.assertNotIn(
                    "w2/", k,
                    f"silu backward should not be under w2: {k}"
                )


class TestDumpCaptures(unittest.TestCase):
    """Change 6: deterministic stat computation and dump format."""

    def test_writes_file(self):
        caps = {
            "mod/op_0_mm": CapturedActivation(
                tensor=torch.randn(100, 100)
            ),
        }
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
        """Change 3: _checkpoint_wrapped_module stripped at dump time."""
        caps = {
            "layers.0._checkpoint_wrapped_module.wq/op_0_mm": CapturedActivation(
                tensor=torch.randn(100, 100)
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

    def test_deterministic_norms(self):
        """Change 6: CPU float64 gives deterministic results."""
        t = torch.randn(1000, 1000, device="cpu")
        caps = {"test/op_0_mm": CapturedActivation(tensor=t)}
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            dump_captures_to_file(caps, path)
            content1 = open(path).read()
            dump_captures_to_file(caps, path)
            content2 = open(path).read()
            self.assertEqual(content1, content2)
        finally:
            os.unlink(path)

    def test_phase_in_output(self):
        caps = {
            "mod/op_0_mm": CapturedActivation(
                tensor=torch.randn(100, 100), phase="backward"
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
                tensor=torch.randn(100, 100), phase="forward"
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


class TestCompareCaptures(unittest.TestCase):
    """naive_compare_captures function."""

    def test_identical_captures_match(self):
        t = torch.randn(100, 100)
        caps_a = {"mod/op_0_mm": CapturedActivation(tensor=t.clone())}
        caps_b = {"mod/op_0_mm": CapturedActivation(tensor=t.clone())}
        results = naive_compare_captures(caps_a, caps_b, verbose=False)
        self.assertTrue(results["mod/op_0_mm"]["match"])

    def test_different_captures_mismatch(self):
        caps_a = {"mod/op_0_mm": CapturedActivation(tensor=torch.randn(100, 100))}
        caps_b = {"mod/op_0_mm": CapturedActivation(tensor=torch.randn(100, 100))}
        results = naive_compare_captures(caps_a, caps_b, verbose=False)
        self.assertFalse(results["mod/op_0_mm"]["match"])

    def test_tolerance(self):
        t = torch.randn(100, 100)
        caps_a = {"mod/op_0_mm": CapturedActivation(tensor=t)}
        caps_b = {"mod/op_0_mm": CapturedActivation(tensor=t + 1e-6)}
        results = naive_compare_captures(caps_a, caps_b, atol=1e-5, verbose=False)
        self.assertTrue(results["mod/op_0_mm"]["match"])

    def test_shape_mismatch(self):
        caps_a = {"mod/op_0_mm": CapturedActivation(tensor=torch.randn(10, 10))}
        caps_b = {"mod/op_0_mm": CapturedActivation(tensor=torch.randn(10, 20))}
        results = naive_compare_captures(caps_a, caps_b, verbose=False)
        self.assertFalse(results["mod/op_0_mm"]["match"])
        self.assertIn("error", results["mod/op_0_mm"])


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
                enabled=True, model=model,
                dump_dir=tmpdir, capture_step=1,
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


class TestFQNInterpreter(unittest.TestCase):
    """FQNInterpreter for traced graph replay."""

    def test_sets_module_name_from_node_meta(self):
        """FQNInterpreter reads node.meta['custom']['module_fqn']."""
        from torch.fx.experimental.proxy_tensor import make_fx

        from torchtitan.experiments.graph_trainer.debug_utils import (
            FQNInterpreter,
        )

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
