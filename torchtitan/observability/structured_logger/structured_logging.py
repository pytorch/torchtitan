# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Structured-logging API: init_structured_logger, log_trace_span, log_trace_instant, log_trace_scalar.

Emits structured JSONL events for phase timing, scalars, and diagnostics.
Handler factories (JSONL, custom, etc.) are loaded dynamically via
the ``TITAN_STRUCT_LOGGER_HANDLERS`` env var.
"""

import asyncio
import enum
import functools
import importlib
import inspect
import logging
import os
from collections.abc import Callable
from timeit import default_timer as timer
from typing import Any, cast, TypeVar

import torch

from torchtitan.observability.structured_logger.step_state import get_step

F = TypeVar("F", bound=Callable[..., Any])

# Used by this module for regular logging
console_logger: logging.Logger = logging.getLogger(__name__)

# Dedicated logger for structured events. Uses a name distinct from
# ``__name__`` so we can set ``propagate = False`` here without also silencing
# the console logger above.
_structured_logger: logging.Logger = logging.getLogger("torchtitan.structured_logger")
_structured_logger.propagate = False

# Used to check if handler has been already initialized. If so, re-initializing
# is a no-op
_is_initialized: bool = False

# Set by ``init_structured_logger(enable=False)`` to make all trace calls no-ops.
_disabled: bool = False

_DEFAULT_HANDLER_FACTORY = (
    "torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler"
)


def _structured_logger_disabled() -> bool:
    """Whether structured logging is disabled.

    Driven by the ``enable`` flag passed to :func:`init_structured_logger`
    (sourced from ``DebugConfig.enable_structured_logging``).
    """
    return _disabled


class StrEnum(enum.Enum):
    """Stand-in for ``enum.StrEnum`` (added in Python 3.11)

    Mimics it for our use case: ``str(member)`` returns the value (e.g.
    ``"event"``), not ``"LogType.EVENT"``. Drop in favor of ``enum.StrEnum``
    once Python 3.10 support is no longer needed.
    """

    def __str__(self) -> str:
        return self.value


class LogType(StrEnum):
    """Record kind in the JSONL stream.

    - ``EVENT``: paired span record (``*_start`` / ``*_end`` from ``log_trace_span``).
    - ``INSTANT``: point-in-time record (``log_trace_instant``, ``log_trace_scalar``).
    - ``TEXT``: free-text log record (filtered out by ``TraceEventsOnlyFilter``).
    """

    EVENT = "event"
    INSTANT = "instant"
    TEXT = "text"


class ExtraFields(StrEnum):
    """Keys for the ``extra`` dict passed to logging calls."""

    LOG_TYPE = "log_type"
    LOG_TYPE_NAME = "log_type_name"
    EVENT_NAME = "event_name"
    STEP = "step"
    VALUE = "value"
    RELATIVE_STEP = "relative_step"
    TASK_NAME = "task_name"


def event_extra(
    event_type: str,
    event_name: str | None = None,
    step: int | None = None,
    relative_step: int | None = None,
    value: float | int | None = None,
    task_name: str | None = None,
    log_type: LogType = LogType.EVENT,
) -> dict[str, Any]:
    """Build the extra dict for a structured JSONL event record."""
    return {
        str(ExtraFields.LOG_TYPE): str(log_type),
        str(ExtraFields.LOG_TYPE_NAME): str(event_type),
        str(ExtraFields.EVENT_NAME): event_name,
        str(ExtraFields.STEP): step,
        str(ExtraFields.RELATIVE_STEP): relative_step,
        str(ExtraFields.VALUE): value,
        str(ExtraFields.TASK_NAME): task_name,
    }


class TraceEventsOnlyFilter(logging.Filter):
    """Defensive filter: drop any record on the structured logger that did not
    come through the ``log_trace_*`` API.

    How records get a ``log_type_name`` attribute:

    1. ``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar`` all
       call ``_structured_logger.info(msg, extra=event_extra(...))``.
    2. ``event_extra`` always sets ``log_type_name`` in the ``extra`` dict.
    3. Python's logging attaches ``extra`` keys as attributes on the
       ``LogRecord``, so ``record.log_type_name`` is populated.

    So any record reaching this filter WITHOUT ``log_type_name`` is a plain
    ``.info("text")`` call made directly on the structured logger — bypassing
    the API. That should not happen in this codebase (we never call
    ``_structured_logger`` outside the log_trace_* helpers). The filter exists
    as a safeguard against future accidents: if someone grabs the logger by
    name (``logging.getLogger("torchtitan.structured_logger")``) and writes
    free text, this filter keeps it out of the JSONL stream so the schema
    stays strict. The first drop emits a one-shot warning to make the trap
    discoverable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._warned = False

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is not None:
            return True
        if not self._warned:
            self._warned = True
            console_logger.warning(
                "Plain-text record on the structured logger was dropped. "
                "Use log_trace_span / log_trace_scalar / log_trace_instant."
            )
        return False


def init_structured_logger(
    source: str, output_dir: str, rank: int | None = None, enable: bool = True
) -> None:
    """Attach handlers to the structured logger. Call once per process.

    Handler factories come from the ``TITAN_STRUCT_LOGGER_HANDLERS`` env var
    (comma-separated ``module.path.factory_name``). When unset, a default
    JSONL handler is registered; when set, ONLY the listed factories run.

    ``rank`` defaults to ``$RANK`` (set by torchrun), so this can run
    before ``torch.distributed`` init. Idempotent: second and later calls
    are a no-op.

    When ``enable=False``, all subsequent ``log_trace_*`` calls become
    no-ops (no handlers are attached).

    Does not configure console output; call ``init_logger()`` for that.

    Example::

        init_structured_logger(source="trainer", output_dir="./outputs")
        log_trace_instant("binary_start")
    """
    global _is_initialized, _disabled

    if not enable:
        _disabled = True
        console_logger.info(
            "Structured logging disabled via DebugConfig.enable_structured_logging=False"
        )
        return

    # Avoids re-initializing
    if _is_initialized:
        return

    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    factories_env = os.environ.get("TITAN_STRUCT_LOGGER_HANDLERS", "")
    if factories_env.strip():
        factory_paths = [f.strip() for f in factories_env.split(",") if f.strip()]
    else:
        factory_paths = [_DEFAULT_HANDLER_FACTORY]

    for factory_path in factory_paths:
        module_path, func_name = factory_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        getattr(mod, func_name)(
            structured_logger=_structured_logger,
            rank=rank,
            source=source,
            output_dir=output_dir,
        )

    if (
        _structured_logger.level == logging.NOTSET
        or _structured_logger.level > logging.INFO
    ):
        _structured_logger.setLevel(logging.INFO)

    _is_initialized = True


def log_trace_scalar(scalars: dict[str, float | int], *, stacklevel: int = 2) -> None:
    """Emit a record per (name, value) pair. Useful when adding more context
    to the trace for debugging, e.g. registering `num_tokens_processed`.

    Step is read from ``set_step()``; non-numeric values are skipped
    with a warning. Bump ``stacklevel`` when wrapping in a helper so
    ``caller`` points at the real call site.

    Args:
        scalars: Mapping of scalar name to numeric value. Non-numeric
            values are skipped with a warning.
        stacklevel: Passed through to ``logger.info`` so the ``caller`` field
            in the emitted record points at the real call site. Increase from
            the default 2 if you wrap this function in a helper.

    Example::

        log_trace_scalar({"train.loss": 2.5, "train.tflops": 45.6})
    """
    if _structured_logger_disabled() or torch.compiler.is_compiling():
        return
    step = get_step()
    bad_keys: list[str] = []
    for name, value in scalars.items():
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            bad_keys.append(name)
            continue
        _structured_logger.info(
            f"[step {step if step is not None else 'N/A'}] {name}={value}",
            extra=event_extra(
                "metric_value",
                event_name=name,
                value=value,
                step=step,
                log_type=LogType.INSTANT,
            ),
            stacklevel=stacklevel,
        )
    if bad_keys:
        console_logger.warning(
            "log_trace_scalar skipped non-numeric values for keys: %s", bad_keys
        )


def log_trace_instant(event_type: str, *, stacklevel: int = 2) -> None:
    """Emit a zero-duration event or marker (e.g. ``"training_start"``).

    Use ``log_trace_span`` when you want start+end+duration.

    Args:
        event_type: Free-form string. Becomes ``log_type_name`` in the
            emitted record.
        stacklevel: Passed through to ``logger.info`` so the ``caller`` field
            in the emitted record points at the real call site. Increase from
            the default 2 if you wrap this function in a helper.

    Example::

        log_trace_instant("training_start")
    """
    if torch.compiler.is_compiling() or _structured_logger_disabled():
        return
    _structured_logger.info(
        str(event_type),
        extra=event_extra(event_type, log_type=LogType.INSTANT),
        stacklevel=stacklevel,
    )


class log_trace_span:  # noqa: N801
    """Time a block of work; emits ``_start`` and ``_end`` records.

    Usable as a context manager or decorator. On entry, captures the
    enclosing :class:`asyncio.Task`'s name via
    ``asyncio.current_task().get_name()`` (``None`` outside any task)
    and stamps it on both records. Analysis pairs ``_start`` / ``_end``
    via a LIFO stack on ``(source, task_name)``, so nested spans in one
    task pair correctly. The ``_end`` record's ``value`` is the elapsed
    wall-time in ms.

    On exception, emits an extra ``_error`` record (with exception type
    and message), then the normal ``_end`` -- so every ``_start`` has a
    matching close.

    Example::

        # context manager
        with log_trace_span("fwd_bwd"):
            loss = model(batch)
            loss.backward()

        # decorator of sync or async function
        @log_trace_span("rl_rollout")
        async def rollout(self, prompts):
            return await self.engine.generate(prompts)

    Args:
        event_type: Becomes ``log_type_name`` in the records.
        description: Human-readable label in the log line; doesn't
            affect ``log_type_name`` or filtering.
        stacklevel: Bump when wrapping in a helper so ``caller`` points
            at the real call site.
    """

    def __init__(
        self,
        event_type: str,
        description: str | None = None,
        *,
        stacklevel: int = 2,
    ):
        self.base_name = str(event_type)
        self.description = description
        self.stacklevel = stacklevel
        self.start_time: float = 0.0
        self._task_name: str | None = None
        self.start_type_name = self.base_name + "_start"
        self.end_type_name = self.base_name + "_end"

    def __enter__(self):
        if torch.compiler.is_compiling() or _structured_logger_disabled():
            return self

        # Cache the asyncio task name so __exit__ emits the same one as
        # __enter__; pairing relies on (source, task_name) being stable
        # across a span. None in SPMD / non-asyncio code.
        try:
            task = asyncio.current_task()
            self._task_name = task.get_name() if task else None
        except RuntimeError:
            self._task_name = None

        display_name = self.description or self.base_name
        self.start_time = timer()
        step = get_step()
        _structured_logger.info(
            f"[step {step if step is not None else 'N/A'}] {display_name} {self.start_type_name}",
            extra=event_extra(
                self.start_type_name,
                step=step,
                task_name=self._task_name,
            ),
            stacklevel=self.stacklevel,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # On success: emit ``_end``. On exception: emit ``_error`` then
        # ``_end``. The trailing ``_end`` carries elapsed-until-crash
        # and keeps pairing simple -- analysis tools don't have to
        # special-case exceptional spans.
        if torch.compiler.is_compiling() or _structured_logger_disabled():
            return None

        end_time = timer()
        step = get_step()
        duration_s = end_time - self.start_time
        delta_ms = duration_s * 1000

        if exc_type is not None:
            error_type_name = self.base_name + "_error"
            _structured_logger.info(
                f"[step {step if step is not None else 'N/A'}] {error_type_name}: {exc_type.__name__}: {exc_val}",
                extra=event_extra(
                    error_type_name,
                    step=step,
                    task_name=self._task_name,
                ),
                stacklevel=self.stacklevel,
            )

        _structured_logger.info(
            f"[step {step if step is not None else 'N/A'}] {self.end_type_name} took {delta_ms:.2f} ms",
            extra=event_extra(
                self.end_type_name,
                value=delta_ms,
                step=step,
                task_name=self._task_name,
            ),
            stacklevel=self.stacklevel,
        )
        return None

    def __call__(self, func: F) -> F:
        # Decorator support. Each invocation of the decorated function builds
        # a fresh ``log_trace_span`` so concurrent calls don't clobber each
        # other's ``self.start_time`` / ``self._task_name``.
        #
        # The async path wraps so __exit__ runs after ``await func(...)``
        # completes — a plain sync wrapper would close the context before the
        # coroutine runs and the recorded duration would be ~0ms.
        #
        # TODO: ``caller`` is inaccurate for *async* decorator use — it lands
        # on async_wrapper / asyncio / threading internals (incl. Monarch
        # endpoint dispatch). Use as a context manager when this matters.
        base_name, description, stacklevel = (
            self.base_name,
            self.description,
            self.stacklevel,
        )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with log_trace_span(base_name, description, stacklevel=stacklevel):
                    return await func(*args, **kwargs)

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # stacklevel + 1 skips this wrapper so caller points at the user.
            with log_trace_span(base_name, description, stacklevel=stacklevel + 1):
                return func(*args, **kwargs)

        return cast(F, sync_wrapper)
