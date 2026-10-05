# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Default JSONL backend: formatter, file handler, and factory.

``TraceJsonlFormatter`` is also the base class for backend-specific
formatters (e.g. the Scuba formatter under ``fb/``).
"""

import datetime as dt
import functools
import itertools
import json
import logging
import os
import queue
import random
import socket
import string
import threading
from logging.handlers import QueueHandler, QueueListener
from timeit import default_timer as timer
from typing import Any

from torchtitan.observability.structured_logger.step_state import (
    get_relative_step,
    get_step,
    get_step_tags,
)
from torchtitan.observability.structured_logger.structured_logging import (
    ExtraFields,
    LogType,
    TraceEventsOnlyFilter,
)

console_logger: logging.Logger = logging.getLogger(__name__)

MAX_MESSAGE_SIZE: int = 1000


# pathname is the source file of the logging call site, so the key cardinality
# should be low. But in case that assumption ever breaks, set maxsize to cap
# memory.
@functools.lru_cache(maxsize=4096)
def _relpath(pathname: str) -> str:
    return os.path.relpath(pathname)


class TraceJsonlFormatter(logging.Formatter):
    """Format trace records as one JSON line per record.

    Per-process fields (rank, source, hostname, local_rank) are captured
    in ``__init__``; per-step fields (step, relative_step, step_tags)
    are pulled from :mod:`.step_state` at emit time.

    Subclass to enrich records with backend-specific fields.

    Example output (wrapped for readability)::

        {"rank": 0, "source": "training", "step": 5,
         "log_type_name": "fwd_bwd_end", "value": 12.5, "task_name": "Task-1",
         "step_tags": ["gc"],
         "time_us": 1709500000123456, "caller": "trainer.py:796:train_step",
         "seq_id": 42}
    """

    def __init__(self, rank: int, source: str):
        super().__init__()
        self.rank = rank
        self.source = source
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._host_name = socket.gethostname()
        self._seq_counter = itertools.count()
        # Per-instance threadlocal: prevents delta_ms leakage when multiple
        # formatters (e.g. default JSONL + custom backend) are attached to
        # the same logger in the same process.
        self._thread_local = threading.local()

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self._log_dict(record))

    def _log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build the flat dict emitted as one JSONL line."""
        log_dict: dict[str, Any] = {}

        log_dict["delta_ms"] = self._refresh_event_delta()
        log_dict["tid"] = threading.get_native_id()

        # Rank/source from self (constants, set once in init_structured_logger)
        log_dict["rank"] = self.rank
        log_dict["global_rank"] = self.rank
        log_dict["local_rank"] = self._local_rank
        log_dict["source"] = self.source
        log_dict["host_name"] = self._host_name
        log_dict["pid"] = os.getpid()

        # Step/step_tags/relative_step from hybrid ContextVar/globals
        step = get_step()
        if step is not None:
            log_dict["step"] = step
        relative_step = get_relative_step()
        if relative_step is not None:
            log_dict["relative_step"] = relative_step
        step_tags = get_step_tags()
        if step_tags:
            log_dict["step_tags"] = list(step_tags)

        log_dict["time"] = int(record.created)
        log_dict["time_ms"] = int(record.created * 1000)
        log_dict["time_us"] = int(record.created * 1_000_000)

        log_dict["log_type"] = getattr(
            record, str(ExtraFields.LOG_TYPE), str(LogType.TEXT)
        )
        log_type_name = getattr(record, str(ExtraFields.LOG_TYPE_NAME), None)
        log_dict["log_type_name"] = log_type_name

        # Per-record step/relative_step override (from event_extra)
        record_step = getattr(record, str(ExtraFields.STEP), None)
        if record_step is not None:
            log_dict["step"] = record_step
        record_relative_step = getattr(record, str(ExtraFields.RELATIVE_STEP), None)
        if record_relative_step is not None:
            log_dict["relative_step"] = record_relative_step

        log_dict["event_name"] = getattr(record, str(ExtraFields.EVENT_NAME), None)

        value = getattr(record, str(ExtraFields.VALUE), None)
        if isinstance(value, (float, int)):
            log_dict["value"] = float(value)

        context = getattr(record, str(ExtraFields.CONTEXT), None)
        if context is not None:
            log_dict["context"] = context

        # task_name pairs start/end records
        task_name = getattr(record, str(ExtraFields.TASK_NAME), None)
        if task_name is not None:
            log_dict["task_name"] = task_name

        # Caller field for source traceability (file:line:function)
        log_dict[
            "caller"
        ] = f"{_relpath(record.pathname)}:{record.lineno}:{record.funcName}"
        log_dict["log_file"] = record.filename
        log_dict["log_function"] = record.funcName
        log_dict["log_level"] = record.levelname
        log_dict["logger_name"] = record.name
        if record.stack_info:
            log_dict["stack_info"] = record.stack_info

        log_dict["seq_id"] = next(self._seq_counter)

        # Truncate long free-text messages. Event records
        # (log_trace_span/scalar/instant) pass msg="", so this branch really
        # only catches plain logger.info() calls routed to the structured
        # logger -- usually a user mistake.
        message = record.getMessage()
        if message is not None:
            if len(message) <= MAX_MESSAGE_SIZE:
                log_dict["message"] = message
            else:
                half = MAX_MESSAGE_SIZE // 2
                log_dict["message"] = message[:half] + "..." + message[-half:]

        return log_dict

    def _refresh_event_delta(self) -> float:
        if not hasattr(self._thread_local, "last_event_time"):
            self._thread_local.last_event_time = timer()
        event_delta = (timer() - self._thread_local.last_event_time) * 1000
        self._thread_local.last_event_time = timer()
        return event_delta


class TraceJsonlHandler(logging.FileHandler):
    """Per-rank JSONL file handler.

    File path::

        {output_dir}/structured_logs/{source}.global_rank_{rank}.{timestamp}-{random}.jsonl
    """

    def __init__(self, rank: int, source: str, output_dir: str):
        timestamp_str = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        random_str = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
        )
        filename = f"{source}.global_rank_{rank}.{timestamp_str}-{random_str}.jsonl"
        filepath = os.path.join(output_dir, "structured_logs", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.setFormatter(TraceJsonlFormatter(rank=rank, source=source))
        self.addFilter(TraceEventsOnlyFilter())


class _JsonlQueueHandler(QueueHandler):
    """Format records on the emitting thread; write them on a listener thread.

    A stalled file write (e.g. on NFS) then does not block training. At
    interpreter exit, ``logging.shutdown`` calls ``close()``, which writes
    every queued record. Records emitted after ``close()`` (e.g. from an
    ``atexit`` hook registered before ``logging`` was imported, or a daemon
    thread) are written synchronously. Records still queued when the process
    is killed (SIGKILL, SIGTERM, a native crash, ``os._exit``) are lost:
    normally the listener lags by well under a millisecond, but longer while
    the emitting thread keeps the GIL busy with pure-Python work.
    """

    def __init__(self, file_handler: logging.Handler) -> None:
        # SimpleQueue.put is reentrant, so a record emitted from a signal
        # handler or __del__ cannot deadlock against an interrupted put().
        super().__init__(queue.SimpleQueue())
        # Only the write moves to the listener thread; prepare() formats on
        # the emitting thread. Formatting reads that thread's step state and
        # holds the GIL wherever it runs, so offloading it only helps an idle
        # caller and slowed a busy one in benchmarks.
        self.setFormatter(file_handler.formatter)
        self.addFilter(TraceEventsOnlyFilter())  # drop plain text before formatting
        file_handler.setFormatter(logging.Formatter())  # write the prepared line as-is
        self._listener = QueueListener(self.queue, file_handler)
        self._listener.start()
        self._listening = True

    def enqueue(self, record: logging.LogRecord) -> None:
        # Called from emit() under the handler lock, which close() also holds.
        if self._listening:
            self.queue.put_nowait(record)
        else:
            # After close(): write synchronously, like the old FileHandler did
            # (it reopens the file in append mode).
            self._listener.handle(record)

    def flush(self) -> None:
        """Block until every record queued so far has been written.

        This does not speed up writing: the listener writes and flushes each
        record as soon as it dequeues it, whether or not anyone calls flush().
        It only makes the caller wait for the listener to catch up, for
        callers that need everything in the file now (tests, and
        ``logging.shutdown`` at exit, where ``close()`` also drains).

        ``QueueListener.stop()`` is the wait: it enqueues a sentinel and joins
        the thread, which exits after writing every record ahead of the
        sentinel (the queue is FIFO). stop() ends the thread, so a new one is
        started. This takes ~0.1 ms plus the time to drain the backlog, and
        holds the handler lock, so other threads' log calls wait meanwhile.
        Do not call it per record.
        """
        self.acquire()
        try:
            if self._listening:
                self._listener.stop()
                self._listener.start()
        finally:
            self.release()

    def close(self) -> None:
        self.acquire()
        try:
            # QueueListener.stop() is not idempotent before Python 3.13.
            if self._listening:
                self._listening = False
                self._listener.stop()
                for handler in self._listener.handlers:
                    handler.close()
        finally:
            self.release()
        super().close()


def register_jsonl_handler(
    *,
    structured_logger: logging.Logger,
    rank: int,
    source: str,
    output_dir: str,
    **kw: Any,
) -> None:
    """Default factory: write structured records to a ``TraceJsonlHandler``
    through a ``_JsonlQueueHandler`` (format on the emitting thread, write on
    a listener thread)."""
    file_handler = TraceJsonlHandler(rank=rank, source=source, output_dir=output_dir)
    structured_logger.addHandler(_JsonlQueueHandler(file_handler))
    console_logger.info("Structured logging -> JSONL: %s", file_handler.baseFilename)
