# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from torchtitan.observability.cuda_memory import capture as module, MemoryCapture


class CallbackBackend:
    """Only the shared callback contract; no activity or drop-accounting methods."""

    def __init__(self):
        self._cupti = SimpleNamespace(get_version=lambda: 130500)
        self.handlers = {}
        self.arms = {}
        self.enable = MagicMock()

    def register_callback_handler(self, domain, ident, callback):
        token = (domain, ident, callback)
        self.handlers[token] = callback
        return token

    def unregister_callback_handler(self, token):
        self.handlers.pop(token)

    def arm_callback(self, domain, ident):
        key = domain, ident
        count = self.arms.get(key, 0)
        self.arms[key] = count + 1
        if count == 0:
            self.enable(domain, ident)

    def disarm_callback(self, domain, ident):
        key = domain, ident
        self.arms[key] -= 1
        if not self.arms[key]:
            self.arms.pop(key)


class TestCapture(unittest.TestCase):
    def setUp(self):
        self.core = CallbackBackend()
        # Synthetic IDs: the consumer gets these from NVIDIA binding enums.
        self.info = {
            "domain": 1,
            "binding_versions": {"cupti-python": "13.5.0", "cuda-bindings": "13.5.0"},
        }
        self.info["apis"] = {
            name: ident
            for ident, name in enumerate(
                (
                    "cuMemAlloc_v2",
                    "cuMemAllocPitch_v2",
                    "cuMemAllocManaged",
                    "cuMemCreate",
                    "cuMemAllocAsync",
                    "cuMemAllocAsync_ptsz",
                    "cuMemAllocFromPoolAsync",
                    "cuMemAllocFromPoolAsync_ptsz",
                    "cuMemAlloc",
                    "cuMemAllocPitch",
                ),
                start=100,
            )
        }
        self.decoder = SimpleNamespace(
            info=self.info, decode=MagicMock(),
        )
        self.torch = SimpleNamespace(
            version=SimpleNamespace(cuda="13.5"),
            utils=SimpleNamespace(
                get_cpp_backtrace=MagicMock(
                    return_value=(
                        "frame #0: get_backtrace() (0x123 in /opt/libtorch_cpu.so)\n"
                        "frame #1: <unknown> (0x456 in /opt/libnccl.so.2)\n"
                    )
                )
            ),
        )
        self.enterContext(
            patch.object(
                module, "_load_backend",
                return_value=(self.torch, self.core, self.decoder),
            )
        )

    def record(self, **overrides):
        return {
            "api": "cuMemAlloc_v2",
            "operation": "alloc",
            "bytes": 4096,
            "address": 0xFEDCBA9876543210,
            "device": -1,
            "location_type": -1,
            "result": 0,
            **overrides,
        }

    def emit(self, capture, **overrides):
        with patch.object(
            self.decoder, "decode",
            return_value=self.record(**overrides),
        ):
            capture._callback(1, self.info["apis"]["cuMemAlloc_v2"], 0)

    def assert_idle(self):
        self.assertEqual(self.core.handlers, {})
        self.assertEqual(self.core.arms, {})

    def test_binding_callback_list_and_no_activity_dependency(self):
        with MemoryCapture() as capture:
            self.assertEqual(len(self.core.handlers), 10)
            self.emit(capture)
        self.assert_idle()
        self.assertEqual(capture.ids, list(self.info["apis"].values()))
        request = capture.report()["requests"][0]
        self.assertEqual(request["attribution"]["requester"], "NCCL")
        self.assertEqual(request["address"], str(0xFEDCBA9876543210))
        self.assertNotIn("activity_drops", capture.coverage)
        self.assertNotIn("live_bytes", capture.report())

    def test_bytes_only_never_unwinds(self):
        with MemoryCapture(stacks=False) as capture:
            self.emit(capture)
        self.torch.utils.get_cpp_backtrace.assert_not_called()
        self.assertEqual(
            capture.report()["requests"][0]["attribution"]["requester"], "unknown"
        )

    def test_capacity_failure_cleans_up(self):
        with self.assertRaisesRegex(RuntimeError, "1 dropped callbacks"):
            with MemoryCapture(stacks=False, max_records=1) as capture:
                self.emit(capture)
                self.emit(capture)
        self.assert_idle()
        with self.assertRaises(RuntimeError):
            capture.report()

    def test_entry_and_failed_calls_do_not_consume_capacity_or_unwind(self):
        with MemoryCapture(max_records=1) as capture:
            with patch.object(
                self.decoder, "decode", return_value=None
            ):
                capture._callback(1, 1, 0)
            self.emit(capture)
        self.assertEqual(len(capture.report()["requests"]), 1)
        self.assertEqual(
            self.torch.utils.get_cpp_backtrace.call_count, 2
        )  # warmup + allocation

    def test_decode_failure_is_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "1 collector errors"):
            with MemoryCapture(stacks=False) as capture:
                with patch.object(
                    self.decoder, "decode",
                    side_effect=ValueError("bad payload"),
                ):
                    capture._callback(1, 1, 0)
        self.assert_idle()

    def test_stack_truncation_is_reported(self):
        self.torch.utils.get_cpp_backtrace.return_value = "x" * 70000
        with MemoryCapture() as capture:
            self.emit(capture)
        request = capture.report()["requests"][0]
        self.assertEqual(len(request["native_stack_text"]), 65536)
        self.assertEqual(request["stack_status"], "truncated")
        self.assertEqual(request["attribution"]["requester"], "unknown")

    def test_missing_callback_capability_rejected_before_registration(self):
        for name in module._CALLBACK_METHODS:
            with self.subTest(name=name), patch.object(self.core, name, None):
                with self.assertRaisesRegex(RuntimeError, "callback API"):
                    MemoryCapture(stacks=False).start()
                self.assert_idle()

    def test_unknown_versions_rejected_before_registration(self):
        for cuda, cupti in (("13.4", 130500), ("13.5", 130400), ("13.6", 130600)):
            self.torch.version.cuda = cuda
            self.core._cupti.get_version = lambda: cupti
            with self.subTest(cuda=cuda, cupti=cupti), self.assertRaisesRegex(
                RuntimeError, "13.5 only"
            ):
                MemoryCapture(stacks=False).start()
            self.assert_idle()

    def test_missing_version_probe_rejected(self):
        self.core._cupti = None
        with self.assertRaisesRegex(RuntimeError, "report its CUPTI version"):
            MemoryCapture(stacks=False).start()
        self.assert_idle()

    def test_partial_registration_and_arm_failures_rollback(self):
        for name in ("register_callback_handler", "arm_callback"):
            original = getattr(self.core, name)
            calls = 0

            def fail_third(*args):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("injected failure")
                return original(*args)

            with self.subTest(name=name), patch.object(
                self.core, name, side_effect=fail_third
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    MemoryCapture(stacks=False).start()
            self.assert_idle()
        with MemoryCapture(stacks=False):
            pass
        self.assert_idle()

    def test_cleanup_failure_does_not_skip_other_handlers(self):
        original = self.core.unregister_callback_handler
        calls = 0

        def fail_after_unregister(token):
            nonlocal calls
            original(token)
            calls += 1
            if calls == 1:
                raise RuntimeError("unsubscribe error")

        with patch.object(
            self.core, "unregister_callback_handler", side_effect=fail_after_unregister
        ):
            with self.assertRaisesRegex(RuntimeError, "unsubscribe error"):
                with MemoryCapture(stacks=False):
                    pass
        self.assertEqual(calls, 10)
        self.assert_idle()

    def test_other_consumers_remain_registered_and_armed(self):
        ident = self.info["apis"]["cuMemAlloc_v2"]
        token = self.core.register_callback_handler(1, ident, lambda *_: None)
        self.core.arm_callback(1, ident)
        with MemoryCapture(stacks=False):
            pass
        self.assertEqual(list(self.core.handlers), [token])
        self.assertEqual(self.core.arms, {(1, ident): 1})
        self.core.disarm_callback(1, ident)
        self.core.unregister_callback_handler(token)

    @unittest.expectedFailure
    def test_failed_native_enable_preserves_registered_unarmed_consumer(self):
        # Known upstream Cuspy bug; its rollback fix is outside this design diff.
        ident = self.info["apis"]["cuMemAlloc_v2"]
        token = self.core.register_callback_handler(1, ident, lambda *_: None)

        def fail_selected(domain, selected):
            if selected == ident:
                raise RuntimeError("native enable failed")

        self.core.enable.side_effect = fail_selected
        with self.assertRaisesRegex(RuntimeError, "native enable failed"):
            MemoryCapture(stacks=False).start()
        self.assertEqual(list(self.core.handlers), [token])
        self.assertEqual(self.core.arms, {})
        self.core.enable.reset_mock(side_effect=True)
        self.core.arm_callback(1, ident)
        self.core.enable.assert_called_once_with(1, ident)
        self.assertEqual(self.core.arms, {(1, ident): 1})
        self.core.disarm_callback(1, ident)
        self.core.unregister_callback_handler(token)
        self.assert_idle()

    def test_stop_waits_for_admitted_callback_and_rejects_late_callbacks(self):
        capture = MemoryCapture().start()
        entered, release, stopped = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        errors = []

        def unwind(**kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test callback timed out")
            return ""

        def stop():
            try:
                capture.stop()
            except BaseException as error:
                errors.append(error)
            finally:
                stopped.set()

        self.torch.utils.get_cpp_backtrace.side_effect = unwind
        with patch.object(
            self.decoder, "decode",
            side_effect=lambda *_: self.record(),
        ):
            worker = threading.Thread(target=capture._callback, args=(1, 1, 0))
            worker.start()
            self.assertTrue(entered.wait(5))
            closer = threading.Thread(target=stop)
            closer.start()
            try:
                self.assertFalse(stopped.wait(0.05))
                self.assertEqual(len(self.core.handlers), 10)
            finally:
                release.set()
                worker.join(5)
                closer.join(5)
            capture._callback(1, 1, 0)
        self.assertFalse(worker.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(capture.report()["requests"]), 1)
        self.assert_idle()

    def test_concurrent_callbacks_reserve_capacity_before_unwinding(self):
        capture = MemoryCapture(max_records=1).start()
        entered, release = threading.Event(), threading.Event()

        def unwind(**kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test callback timed out")
            return ""

        self.torch.utils.get_cpp_backtrace.side_effect = unwind
        with patch.object(
            self.decoder, "decode",
            side_effect=lambda *_: self.record(),
        ):
            worker = threading.Thread(target=capture._callback, args=(1, 1, 0))
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                capture._callback(1, 1, 0)
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        with self.assertRaisesRegex(RuntimeError, "1 dropped callbacks"):
            capture.stop()
        self.assertEqual(len(capture._records), 1)
        self.assert_idle()

    def test_recursive_callback_fails_capture_without_recursing(self):
        capture = MemoryCapture().start()

        def unwind(**kwargs):
            capture._callback(1, 1, 0)
            return ""

        self.torch.utils.get_cpp_backtrace.side_effect = unwind
        self.emit(capture)
        with self.assertRaisesRegex(RuntimeError, "reentrant"):
            capture.stop()
        self.assert_idle()

    def test_exports_requests_and_coverage_without_lifetime_totals(self):
        with MemoryCapture(stacks=False) as capture:
            self.emit(capture)
        with tempfile.TemporaryDirectory() as directory:
            report_path, trace_path = (
                Path(directory) / "report.gz",
                Path(directory) / "trace.gz",
            )
            capture.export(report_path)
            capture.export_chrome_trace(trace_path)
            with gzip.open(report_path, "rt") as stream:
                report = json.load(stream)
            with gzip.open(trace_path, "rt") as stream:
                trace = json.load(stream)
        self.assertEqual(set(report), {"coverage", "requests", "summary"})
        self.assertEqual(trace["traceEvents"][0]["args"], report["requests"][0])
        self.assertEqual(trace["cuda_allocation_coverage"], report["coverage"])

    def test_single_use_and_invalid_capacity(self):
        for kwargs in (
            {"max_records": 0},
            {"maximum_frames": 129},
            {"max_records": True},
        ):
            with self.assertRaises(ValueError):
                MemoryCapture(**kwargs)
        with MemoryCapture(stacks=False) as capture:
            pass
        with self.assertRaisesRegex(RuntimeError, "new MemoryCapture"):
            capture.start()
