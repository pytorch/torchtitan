# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded successful allocation requests from synchronous Cuspy callbacks."""

import gzip
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .attribution import attribute_request, summarize_requests
from .stack import parse_stack

_LIFECYCLE = threading.RLock()
_CALLBACK_METHODS = (
    "register_callback_handler",
    "unregister_callback_handler",
    "arm_callback",
    "disarm_callback",
)


def _load_backend():
    import torch

    try:
        from torch.profiler._cuspy.core import Cuspy
    except ImportError as error:
        raise RuntimeError(
            "CUDA allocation capture requires PyTorch Cuspy and cupti-python"
        ) from error
    from .callbacks import AllocationCallbacks

    return torch, Cuspy(), AllocationCallbacks()


class MemoryCapture:
    """One window of successful allocation API exits; no lifetime accounting."""

    def __init__(self, *, stacks=True, max_records=262144, maximum_frames=96):
        if type(max_records) is not int or not 1 <= max_records <= 1048576:
            raise ValueError("record capacity must be between 1 and 1048576")
        if type(maximum_frames) is not int or not 1 <= maximum_frames <= 128:
            raise ValueError("frame capacity must be between 1 and 128")
        self.stacks = stacks
        self.max_records, self.maximum_frames = max_records, maximum_frames
        self.ids = []
        self._condition = threading.Condition(threading.RLock())
        self._tls = threading.local()
        self._state = "new"
        self._accepting = False
        self._inflight = self._reserved = self._drops = self._error_count = 0
        self._handlers, self._armed = [], []
        self._records: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self.events: list[dict[str, Any]] | None = None
        self.coverage: dict[str, Any] = {}

    def _error(self, error):
        with self._condition:
            self._error_count += 1
            if len(self._errors) < 8:
                self._errors.append(repr(error))

    def _callback(self, domain, ident, pointer):
        if getattr(self._tls, "active", False):
            self._error(RuntimeError("reentrant allocation callback discarded"))
            return
        with self._condition:
            if not self._accepting:
                return
            self._inflight += 1
        self._tls.active = True
        try:
            stamp = time.time_ns()
            record = self._decoder.decode(domain, ident, pointer)
            if record is None:
                return
            with self._condition:
                if self._reserved >= self.max_records:
                    self._drops += 1
                    return
                record["sequence"] = self._reserved
                self._reserved += 1
            record.update(native_stack_text="", stack_status="disabled")
            if self.stacks:
                stack = self._torch.utils.get_cpp_backtrace(
                    maximum_number_of_frames=self.maximum_frames
                )
                record["native_stack_text"] = stack[:65536]
                record["stack_text_truncated"] = len(stack) > 65536
            # JSON viewers use doubles: preserve full pointer/handle identity.
            for name in ("address", "handle", "stream"):
                if name in record:
                    record[name] = str(record[name])
            event = {
                "name": "[cuda_allocation_request]",
                "ph": "i",
                "s": "t",
                "cat": "cuda_memory",
                "ts": stamp / 1000,
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "args": record,
            }
            with self._condition:
                self._records.append(event)
        except BaseException as error:
            self._error(error)
        finally:
            self._tls.active = False
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def start(self):
        if getattr(self._tls, "active", False):
            raise RuntimeError("capture lifecycle cannot run inside a CUDA callback")
        with _LIFECYCLE:
            if self._state != "new":
                raise RuntimeError("use a new MemoryCapture for each window")
            self._state = "starting"
            try:
                self._torch, self._cuspy, self._decoder = _load_backend()
                missing = [
                    name
                    for name in _CALLBACK_METHODS
                    if not callable(getattr(self._cuspy, name, None))
                ]
                if missing:
                    raise RuntimeError(
                        f"Cuspy callback API is unavailable: {', '.join(missing)}"
                    )
                self._decoder_info = self._decoder.info
                version = getattr(
                    getattr(self._cuspy, "_cupti", None), "get_version", None
                )
                if not callable(version):
                    raise RuntimeError("Cuspy cannot report its CUPTI version")
                self._cupti_version = version()
                if self._torch.version.cuda != "13.5" or self._cupti_version != 130500:
                    raise RuntimeError("allocation decoder supports CUDA/CUPTI 13.5 only")
                self.ids = list(self._decoder_info["apis"].values())
                self._domain = self._decoder_info["domain"]
                if self.stacks:
                    self._torch.utils.get_cpp_backtrace(
                        maximum_number_of_frames=self.maximum_frames
                    )
                for ident in self.ids:
                    self._handlers.append(
                        self._cuspy.register_callback_handler(
                            self._domain, ident, self._callback
                        )
                    )
                for ident in self.ids:
                    self._cuspy.arm_callback(self._domain, ident)
                    self._armed.append(ident)
                with self._condition:
                    self.start_ns = time.time_ns()
                    self._accepting = True
                self._state = "running"
            except BaseException as error:
                self._state = "failed"
                self._cleanup(error)
                raise
        return self

    def _cleanup(self, primary=None):
        with self._condition:
            self._accepting = False
        failures = []
        for ident in reversed(self._armed):
            try:
                self._cuspy.disarm_callback(self._domain, ident)
            except BaseException as error:
                failures.append(error)
        self._armed.clear()
        # Retain our handlers/subscription until admitted Python callbacks finish.
        with self._condition:
            while self._inflight:
                self._condition.wait()
        for handler in reversed(self._handlers):
            try:
                self._cuspy.unregister_callback_handler(handler)
            except BaseException as error:
                failures.append(error)
        self._handlers.clear()
        if failures:
            error = primary if primary is not None else failures.pop(0)
            for failure in failures:
                error.add_note(f"capture cleanup: {failure!r}")
            if primary is None:
                raise error

    def stop(self):
        if getattr(self._tls, "active", False):
            raise RuntimeError("capture lifecycle cannot run inside a CUDA callback")
        with _LIFECYCLE:
            if self._state != "running":
                raise RuntimeError("memory capture is not running")
            with self._condition:
                self._accepting = False
                self.end_ns = time.time_ns()
            self._state = "failed"
            # These callbacks describe API exits, including async requests. There
            # is no activity buffer to flush and no GPU completion to wait for.
            self._cleanup()
            if self._drops or self._error_count:
                raise RuntimeError(
                    f"incomplete allocation capture: {self._drops} dropped callbacks, "
                    f"{self._error_count} collector errors: {self._errors}"
                )
            for event in self._records:
                record = event["args"]
                if self.stacks:
                    record.update(
                        parse_stack(record["native_stack_text"], self.maximum_frames)
                    )
                    if record["stack_text_truncated"]:
                        record["stack_status"] = "truncated"
                record["attribution"] = attribute_request(record)
            self.events = sorted(
                self._records, key=lambda event: event["args"]["sequence"]
            )
            self.coverage = {
                "schema_version": 1,
                "backend": "cuspy_callbacks",
                "capture_start_ns": str(self.start_ns),
                "capture_end_ns": str(self.end_ns),
                "enabled_apis": list(self._decoder_info["apis"]),
                "callback_capacity": self.max_records,
                "callback_drops": self._drops,
                "collector_errors": self._error_count,
                "native_stacks": self.stacks,
                "cuda_version": self._torch.version.cuda,
                "cupti_version": self._cupti_version,
                "binding_versions": self._decoder_info["binding_versions"],
                "scope": "successful API exits admitted during this window",
                "byte_semantics": "per-API request bytes; not additive across APIs or a live-memory total",
            }
            self._state = "complete"
        return self

    def report(self):
        if self.events is None:
            raise RuntimeError("only a completed capture can be exported")
        requests = [event["args"] for event in self.events]
        return {
            "coverage": self.coverage,
            "requests": requests,
            "summary": summarize_requests(requests),
        }

    def export(self, path):
        with gzip.open(Path(path), "wt") as stream:
            json.dump(self.report(), stream)

    def export_chrome_trace(self, path):
        if self.events is None:
            raise RuntimeError("only a completed capture can be exported")
        with gzip.open(Path(path), "wt") as stream:
            json.dump(
                {"traceEvents": self.events, "cuda_allocation_coverage": self.coverage},
                stream,
            )

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.stop()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"capture stop failed: {cleanup!r}")
        return False
