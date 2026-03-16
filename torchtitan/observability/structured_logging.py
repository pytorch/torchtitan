# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""System JSONL logging: init_observability, record_span, record_event."""

import enum
import itertools
import json
import logging
import os
import socket
import threading
import time
from contextlib import ContextDecorator
from timeit import default_timer as timer
from typing import Any

from torchtitan.observability._constants import (
    EXPERIMENT_LOGGER_NAME,
    SYSTEM_LOGGER_NAME,
)
from torchtitan.observability.metrics import (
    ExperimentJSONFormatter,
    ExperimentLoggingHandler,
    MeanMetric,
    record_metric,
)
from torchtitan.observability.step_state import get_step, get_step_tags

MAX_MESSAGE_SIZE: int = 1000

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StrEnum(enum.Enum):
    """String-valued enum. str(member) returns the value, not 'Class.name'."""

    def __str__(self) -> str:
        return self.value


class EventType(StrEnum):
    """Standardized phase identifiers for structured JSONL events.
    Used for categorization and analysis.
    """

    TORCH_DISTRIBUTED_INIT = "torch_distributed_init"
    TORCH_DISTRIBUTED_TEARDOWN = "torch_distributed_teardown"
    MODEL_PARALLELISM_INIT = "model_parallelism_init"
    TOKENIZER_INIT = "tokenizer_init"
    DATA_ITERATOR_INIT = "data_iterator_init"
    RELOAD_DATA_LOADER_STATE = "reload_data_loader_state"
    BUILD_MODEL = "build_model"
    BUILD_LEARNER = "build_learner"
    OPTIMIZER_INIT = "optimizer_init"
    TRAINING_START = "training_start"
    STEP = "step"
    FETCHING_BATCH = "fetching_batch"
    FWD_BWD = "fwd_bwd"
    OPTIM = "optim"
    CHECKPOINT = "checkpoint"
    CHECKPOINT_INIT = "checkpoint_init"
    CHECKPOINT_STAGE = "checkpoint_stage"
    CHECKPOINT_LOAD = "checkpoint_load"
    EVAL_LAUNCH = "eval_launch"
    EVAL = "eval"
    SUMMARY_WRITER = "summary_writer"
    TORCH_MEMORY_BREAKDOWN = "torch_memory_breakdown"
    GC_COLLECT = "gc_collect"
    METRIC_VALUE = "metric_value"
    STATE_DICT_INIT = "state_dict_init"
    STATE_DICT_LOAD = "state_dict_load"

    # RL
    RL_GRADING = "rl_grading"
    RL_ROLLOUT = "rl_rollout"
    RL_GENERATE = "rl_generate"
    RL_ENV = "rl_env"


class LogType(StrEnum):
    """Distinguishes event records from free-text log records in JSONL."""

    EVENT = "event"
    TEXT = "text"


class ExtraFields(StrEnum):
    """Keys for the `extra` dict passed to logging calls."""

    LOG_TYPE = "log_type"
    LOG_TYPE_NAME = "log_type_name"
    EVENT_NAME = "event_name"
    STEP = "step"
    RELATIVE_STEP = "relative_step"
    CONTEXT = "context"
    VALUE = "value"


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def dict_to_str_list(d: dict[str, str] | None) -> list[str] | None:
    """Convert a dict to a list of "key:value" strings for JSONL normvector field."""
    if d is None:
        return None
    try:
        return [f"{k}:{v}" for k, v in d.items()]
    except Exception:
        return None


def event_extra(
    event_type: EventType | str,
    event_name: str | None = None,
    step: int | None = None,
    relative_step: int | None = None,
    value: float | int | None = None,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the extra dict for a structured JSONL event record."""
    return {
        str(ExtraFields.LOG_TYPE): str(LogType.EVENT),
        str(ExtraFields.LOG_TYPE_NAME): str(event_type),
        str(ExtraFields.EVENT_NAME): event_name,
        str(ExtraFields.STEP): step,
        str(ExtraFields.RELATIVE_STEP): relative_step,
        str(ExtraFields.VALUE): value,
        str(ExtraFields.CONTEXT): dict_to_str_list(context),
    }


# ---------------------------------------------------------------------------
# JSONL formatting
# ---------------------------------------------------------------------------


def to_structured_json(log_dict: dict[str, Any]) -> str:
    """Convert a log dict to 4-column JSON (int/normal/double/normvector)."""
    int_dict: dict[str, int] = {}
    str_dict: dict[str, str] = {}
    float_dict: dict[str, float] = {}
    vector_str_dict: dict[str, list[str]] = {}

    for k, v in log_dict.items():
        if v is None:
            continue
        if isinstance(v, bool):
            int_dict[k] = int(v)
        elif isinstance(v, int):
            int_dict[k] = v
        elif isinstance(v, str):
            str_dict[k] = v
        elif isinstance(v, float):
            float_dict[k] = v
        elif isinstance(v, (list, tuple)):
            vector_str_dict[k] = [str(e) for e in v]

    structured_dict = {
        "int": int_dict,
        "normal": str_dict,
        "double": float_dict,
        "normvector": vector_str_dict,
    }
    return json.dumps(structured_dict)


# ---------------------------------------------------------------------------
# StructuredJSONFormatter
# ---------------------------------------------------------------------------


class StructuredJSONFormatter(logging.Formatter):
    """Formats system log records as structured JSONL (4-column format).

    Each record becomes a JSON line with int/normal/double/normvector columns.
    Rank and source are constants (set once). Step and step_tags come from
    globals so each process has its own step.

    Example output (one JSON line):
        {"int": {"rank": 0, "step": 5, "time": 1709500000, "seq_id": 42},
         "normal": {"source": "trainer", "log_type_name": "fwd_bwd_end", ...},
         "double": {"value": 12.5, "delta_ms": 0.3},
         "normvector": {}}

    Usage:
        formatter = StructuredJSONFormatter(rank=0, source="trainer")
        handler.setFormatter(formatter)
    """

    _thread_local = threading.local()

    def __init__(self, rank: int, source: str):
        super().__init__()
        self.rank = rank
        self.source = source
        self._seq_counter = itertools.count()

    def format(self, record: logging.LogRecord) -> str:
        return to_structured_json(self._log_dict(record))

    def _log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build the flat dict that to_structured_json splits into 4 columns.

        Example output (before to_structured_json splits by type)::

            {"rank": 0, "source": "trainer", "step": 5,
             "log_type_name": "fwd_bwd_end", "value": 12.5, ...}
        """
        log_dict: dict[str, Any] = {}

        log_dict["delta_ms"] = self._refresh_event_delta()
        log_dict["tid"] = threading.get_native_id()
        log_dict["thread_time_ns"] = time.thread_time_ns()

        # Rank/source from self (constants, set once in init_observability)
        log_dict["rank"] = self.rank
        log_dict["source"] = self.source
        log_dict["host_name"] = socket.gethostname()
        log_dict["pid"] = os.getpid()

        # Step/step_tags from step_state globals (mutable, changes every step)
        step = get_step()
        if step is not None:
            log_dict["step"] = step
        step_tags = get_step_tags()
        if step_tags:
            log_dict["step_tags"] = list(step_tags)

        log_dict["time"] = int(record.created)
        log_dict["time_ms"] = int(record.created * 1000)
        log_dict["time_us"] = int(record.created * 1_000_000)

        log_dict["log_type"] = getattr(
            record, str(ExtraFields.LOG_TYPE), str(LogType.TEXT)
        )
        log_dict["log_type_name"] = getattr(
            record, str(ExtraFields.LOG_TYPE_NAME), None
        )

        # Per-record step override (from event_extra)
        record_step = getattr(record, str(ExtraFields.STEP), None)
        if record_step is not None:
            log_dict["step"] = record_step

        relative_step = getattr(record, str(ExtraFields.RELATIVE_STEP), None)
        if relative_step is not None:
            log_dict["relative_step"] = relative_step

        log_dict["event_name"] = getattr(record, str(ExtraFields.EVENT_NAME), None)

        value = getattr(record, str(ExtraFields.VALUE), None)
        if isinstance(value, (float, int)):
            log_dict["value"] = float(value)

        # Per-record context normvector
        record_context = getattr(record, str(ExtraFields.CONTEXT), None)
        if record_context:
            log_dict["context"] = record_context

        # Caller field for source traceability (file:line:function)
        log_dict["caller"] = (
            f"{os.path.relpath(record.pathname)}:{record.lineno}:{record.funcName}"
        )
        log_dict["log_file"] = record.filename
        log_dict["log_function"] = record.funcName
        log_dict["log_level"] = record.levelname
        log_dict["logger_name"] = record.name
        log_dict["stack_info"] = record.stack_info

        log_dict["seq_id"] = next(self._seq_counter)

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


# ---------------------------------------------------------------------------
# Filters and handlers
# ---------------------------------------------------------------------------


class EventsOnlyFilter(logging.Filter):
    """Filters logs, only passing events with LOG_TYPE_NAME set."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is not None


class StructuredLoggingHandler(logging.FileHandler):
    """Writes structured JSONL events to per-rank files.

    Creates the output directory if needed. Only passes events
    (records with LOG_TYPE_NAME set) — free-text logs are filtered out.
    """

    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.addFilter(EventsOnlyFilter())


# ---------------------------------------------------------------------------
# Crash forensics
# ---------------------------------------------------------------------------


class InflightEventTrackingHandler(logging.Handler):
    """Tracks the last structured event for crash forensics.

    On crash, ``handler.last_event`` tells you what phase the process was in:

        handler = InflightEventTrackingHandler()
        sys_logger.addHandler(handler)
        # ... training runs, then crashes during forward pass ...
        print(handler.last_event)  # "fwd_bwd_start"
        print(handler.last_event_time)  # 1708200121.5
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_event: str | None = None
        self.last_event_time: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event_name = getattr(record, str(ExtraFields.LOG_TYPE_NAME), None)
            if event_name is not None:
                self.last_event = str(event_name)
                self.last_event_time = time.time()
        except Exception:
            return  # Crash forensics handler must never itself crash


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

_system_logger = logging.getLogger(SYSTEM_LOGGER_NAME)


def init_observability(source: str, output_dir: str, rank: int | None = None) -> None:
    """Initialize observability JSONL file handlers.

    Adds per-rank system and experiment JSONL handlers for structured
    logging. Does NOT set up console logging — call ``init_logger()``
    from ``torchtitan.tools.logging`` for that.

    Idempotent: safe to call multiple times (skips existing handlers).
    Can be called before torch.distributed init — rank defaults to the
    RANK environment variable (set by torchrun).

    Example:

        init_logger()  # console
        init_observability(source="trainer", output_dir="./outputs")
        # Creates: ./outputs/system_logs/trainer_rank_0_system.jsonl

    Args:
        source: Component name (e.g., "trainer", "generator").
        output_dir: Root output directory.
        rank: Global rank. If None, reads from RANK env var (default 0).
    """
    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    # --- System handler ---
    sys_logger = logging.getLogger(SYSTEM_LOGGER_NAME)
    if not any(isinstance(h, StructuredLoggingHandler) for h in sys_logger.handlers):
        sys_path = os.path.join(
            output_dir, "system_logs", f"{source}_rank_{rank}_system.jsonl"
        )
        handler = StructuredLoggingHandler(filepath=sys_path)
        handler.setFormatter(StructuredJSONFormatter(rank=rank, source=source))
        sys_logger.addHandler(handler)
        sys_logger.addHandler(InflightEventTrackingHandler())
        # propagate=False prevents events from bubbling to the root logger
        # (which would duplicate them in stderr). Level ensures INFO records
        # are captured even if the root logger has a higher threshold.
        sys_logger.propagate = False
        if sys_logger.level == logging.NOTSET or sys_logger.level > logging.INFO:
            sys_logger.setLevel(logging.INFO)

    # --- Experiment handler (record_metric → per-rank JSONL) ---
    exp_logger = logging.getLogger(EXPERIMENT_LOGGER_NAME)
    # Skip if already initialized (idempotent)
    if not any(isinstance(h, ExperimentLoggingHandler) for h in exp_logger.handlers):
        exp_path = os.path.join(
            output_dir,
            "experiment_logs",
            f"{source}_rank_{rank}_experiment.jsonl",
        )
        handler = ExperimentLoggingHandler(filepath=exp_path)
        handler.setFormatter(ExperimentJSONFormatter(rank=rank, source=source))
        exp_logger.addHandler(handler)
    # Don't propagate to root logger (avoids duplicate console output)
    exp_logger.propagate = False
    if exp_logger.level == logging.NOTSET or exp_logger.level > logging.INFO:
        exp_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Public API: record_event, record_span
# ---------------------------------------------------------------------------


def record_event(metrics: dict[str, float | int]) -> None:
    """Log point-in-time scalars to system JSONL.

    Each key-value pair becomes a separate METRIC_VALUE event.
    Step is read from the global set by ``set_step()``.

    Example:

        record_event({"train.loss": 2.5, "train.tflops": 45.6})
        # system JSONL: {"normal": {"event_name": "train.loss"}, "double": {"value": 2.5}, ...}
    """
    step = get_step()
    for name, value in metrics.items():
        _system_logger.info(
            f"[step {step if step is not None else 'N/A'}] {name}={value}",
            extra=event_extra(
                EventType.METRIC_VALUE, event_name=name, value=value, step=step
            ),
            stacklevel=2,
        )


class record_span(ContextDecorator):  # noqa: N801
    """Context manager/decorator for timing phases.

    Logs START/END events to system JSONL. When ``log_to_metrics=True``
    (default), also records the duration as a MeanMetric (seconds) to
    experiment JSONL via ``record_metric``.

    Args:
        description (str): Human-readable label for log messages and metric key.
            Free-form string (e.g., "trainer_time/forward_backward_s").
            Appears in system JSONL messages and, when metrics are enabled,
            as the experiment metric key.
        event_type Optional([EventType,str]): Optional categorization for analysis tools. Can be an
            ``EventType`` enum for standardized phases, or any string.
            When omitted, the description is used as the event type.
        log_to_metrics: If True, record duration to experiment JSONL.
            Default True.

    Usage::
        with record_span("trainer_time/forward_backward_s", EventType.FWD_BWD):
            output = model(batch)
            loss.backward()

        # Without EventType (description used as event type):
        with record_span("rl_time/rollout_s"):
            rollouts = generate(prompts)
    """

    def __init__(
        self,
        description: str,
        event_type: EventType | str | None = None,
        *,
        log_to_metrics: bool = True,
    ):
        self.description = description
        self.log_to_metrics = log_to_metrics
        self.start_time: float = 0.0

        # Derive _start/_end type names from event_type or description.
        base_name = str(event_type) if event_type is not None else description
        self.start_type_name = base_name + "_start"
        self.end_type_name = base_name + "_end"

    def __enter__(self):
        self.start_time = timer()
        step = get_step()
        _system_logger.info(
            f"[step {step if step is not None else 'N/A'}] {self.description} {self.start_type_name}",
            extra=event_extra(self.start_type_name, event_name=self.description, step=step),
            stacklevel=2,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = timer()
        step = get_step()
        duration_s = end_time - self.start_time
        delta_ms = duration_s * 1000
        _system_logger.info(
            f"[step {step if step is not None else 'N/A'}] {self.description} {self.end_type_name} took {delta_ms:.2f} ms",
            extra=event_extra(self.end_type_name, event_name=self.description, value=delta_ms, step=step),
            stacklevel=2,
        )
        if self.log_to_metrics and step is not None:
            # stacklevel=3 adds to the metadata the actual call site, e.g. trainer.py:537
            record_metric(self.description, MeanMetric(sum=duration_s), _stacklevel=3)
        return False  # Don't suppress exceptions
