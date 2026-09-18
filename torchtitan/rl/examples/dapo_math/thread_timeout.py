# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vendored monarch.experimental.thread_timeout

monarch.experimental.thread_timeout is not shipped in a stable monarch
release yet, so we vendored from:
https://github.com/meta-pytorch/monarch/blob/main/python/monarch/experimental/thread_timeout.py

TODO: Delete this module, and use monarch.experimental.thread_timeout instead
once it ships in a stable torchmonarch release.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import sys
import threading
from collections.abc import Callable
from types import TracebackType


__all__ = [
    "ThreadTimeout",
    "ThreadTimeoutError",
]

logger: logging.Logger = logging.getLogger(__name__)


class ThreadTimeoutError(BaseException):
    """The exception injected when a thread exceeds its deadline.

    This inherits from ``BaseException`` so broad ``except Exception`` handlers
    inside the protected code do not suppress the timeout. Catch
    ``ThreadTimeoutError`` explicitly outside the ``ThreadTimeout`` context.
    """


class _ThreadTimeoutState(threading.local):
    """Track an active timeout so ``__enter__`` can reject nested contexts.

    CPython provides one pending asynchronous-exception slot per thread, so
    nested contexts could overwrite or clear each other's exception.
    """

    active: bool = False


@functools.cache
def _get_thread_timeout_state() -> _ThreadTimeoutState:
    return _ThreadTimeoutState()


@functools.cache
def _get_set_async_exception() -> Callable[..., int]:
    """Configure the shared ctypes binding once, on first use.

    ``ctypes.pythonapi`` reuses function objects, so setting the prototype on
    every injection would repeatedly mutate shared state. Caching also keeps
    this setup out of module import and makes later calls cheaper.
    """
    set_async_exception = ctypes.pythonapi.PyThreadState_SetAsyncExc
    set_async_exception.argtypes = (ctypes.c_ulong, ctypes.py_object)
    set_async_exception.restype = ctypes.c_int
    return set_async_exception


def _set_async_exception(thread_id: int) -> bool:
    modified = _get_set_async_exception()(
        ctypes.c_ulong(thread_id), ctypes.py_object(ThreadTimeoutError)
    )
    if modified == 0:
        return False
    if modified > 1:
        _clear_async_exception(thread_id)
        logger.warning(
            "async exception injection modified multiple threads",
            extra={"thread_id": thread_id, "modified": modified},
        )
        return False
    return True


def _clear_async_exception(thread_id: int) -> None:
    _get_set_async_exception()(ctypes.c_ulong(thread_id), ctypes.py_object())


class ThreadTimeout:
    """Bound the wall-clock time of a synchronous block on a CPython thread.

    The timeout targets the thread that enters the context. Instances are
    single-use. Once the watchdog fires, ``ThreadTimeoutError`` propagates from
    the context even if the protected code catches the injected exception.
    """

    def __init__(
        self,
        seconds: float,
    ) -> None:
        if sys.implementation.name != "cpython":
            raise RuntimeError("ThreadTimeout requires CPython")
        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds!r}")
        self._seconds = seconds
        self._fired = False
        self._target_thread_id: int | None = None
        self._finished = threading.Event()
        self._lock = threading.Lock()
        self._watchdog: threading.Thread | None = None

    def __enter__(self) -> ThreadTimeout:
        if self._target_thread_id is not None:
            raise RuntimeError("ThreadTimeout instances are single-use")

        thread_state = _get_thread_timeout_state()
        if thread_state.active:
            raise RuntimeError("ThreadTimeout contexts cannot be nested")
        thread_state.active = True

        target_thread_id = threading.get_ident()
        self._target_thread_id = target_thread_id
        self._watchdog = threading.Thread(
            target=self._run_watchdog,
            args=(target_thread_id,),
            name=f"thread-timeout-{target_thread_id}",
            daemon=True,
        )
        try:
            self._watchdog.start()
            return self
        except BaseException:
            thread_state.active = False
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception, traceback
        try:
            fired = self._finish()
        finally:
            _get_thread_timeout_state().active = False
        if fired and exception_type is None:
            raise ThreadTimeoutError

    def _finish(self) -> bool:
        with self._lock:
            self._finished.set()
            fired = self._fired
        if fired and self._target_thread_id is not None:
            _clear_async_exception(self._target_thread_id)
        return fired

    def _run_watchdog(self, target_thread_id: int) -> None:
        if self._finished.wait(self._seconds):
            return

        with self._lock:
            if self._finished.is_set():
                return
            if not _set_async_exception(target_thread_id):
                return
            self._fired = True
