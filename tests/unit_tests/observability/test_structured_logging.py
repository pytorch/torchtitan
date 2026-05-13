# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for torchtitan.observability.structured_logger."""

import asyncio
import json
import logging
import os
import time
from unittest import mock

import pytest
from torchtitan.observability.structured_logger import step_state
from torchtitan.observability.structured_logger.gantt_generator import (
    generate_gantt_trace,
)
from torchtitan.observability.structured_logger.jsonl_handler import (
    MAX_MESSAGE_SIZE,
    register_jsonl_handler,
    TraceJsonlFormatter,
    TraceJsonlHandler,
)
from torchtitan.observability.structured_logger.step_state import (
    add_step_tag,
    clear_step_tags,
    get_relative_step,
    get_step,
    get_step_tags,
    set_step,
)
from torchtitan.observability.structured_logger.structured_logging import (
    _structured_logger,
    event_extra,
    ExtraFields,
    init_structured_logger,
    log_trace_instant,
    log_trace_scalar,
    log_trace_span,
    LogType,
    TraceEventsOnlyFilter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_context():
    """Reset step state before each test."""
    step_state._STEP_GLOBAL = None
    step_state._STEP_CV.set(None)
    step_state._RELATIVE_STEP_GLOBAL = None
    step_state._RELATIVE_STEP_CV.set(None)
    step_state._TAGS_GLOBAL = ()
    step_state._TAGS_CV.set(())
    yield
    step_state._STEP_GLOBAL = None
    step_state._STEP_CV.set(None)
    step_state._RELATIVE_STEP_GLOBAL = None
    step_state._RELATIVE_STEP_CV.set(None)
    step_state._TAGS_GLOBAL = ()
    step_state._TAGS_CV.set(())


@pytest.fixture
def structured_logger_fixture():
    """Provide a clean structured logger for testing."""
    import torchtitan.observability.structured_logger.structured_logging as sl_mod

    tl = _structured_logger
    orig = (tl.handlers[:], tl.level, tl.propagate)
    # Reset the module-level init sentinel so init_structured_logger re-runs
    # for each test (otherwise the second call short-circuits as "already
    # initialized").
    sl_mod._is_initialized = False
    yield tl
    tl.handlers, tl.level, tl.propagate = orig
    sl_mod._is_initialized = False


# ---------------------------------------------------------------------------
# Step context tests (hybrid ContextVar)
# ---------------------------------------------------------------------------


class TestSetStep:
    def test_set_step_stores_value(self):
        set_step(42)
        assert get_step() == 42

    def test_set_step_overwrites(self):
        set_step(1)
        set_step(2)
        assert get_step() == 2

    def test_step_default_is_none(self):
        assert get_step() is None

    def test_set_step_writes_to_both_global_and_cv(self):
        set_step(10)
        assert step_state._STEP_GLOBAL == 10
        assert step_state._STEP_CV.get() == 10

    def test_get_step_prefers_cv(self):
        """ContextVar takes priority over global."""
        step_state._STEP_GLOBAL = 1
        step_state._STEP_CV.set(2)
        assert get_step() == 2

    def test_get_step_falls_back_to_global(self):
        """If CV is None, falls back to global."""
        step_state._STEP_GLOBAL = 5
        step_state._STEP_CV.set(None)
        assert get_step() == 5


class TestStepTags:
    def test_add_step_tag_adds_tag(self):
        set_step(1)
        add_step_tag("gc")
        assert get_step_tags() == ("gc",)

    def test_add_step_tag_deduplicates(self):
        add_step_tag("gc")
        add_step_tag("gc")
        assert get_step_tags() == ("gc",)

    def test_add_step_tag_multiple_tags(self):
        add_step_tag("gc")
        add_step_tag("profiling")
        add_step_tag("checkpoint")
        assert get_step_tags() == ("gc", "profiling", "checkpoint")

    def test_clear_step_tags_clears_all(self):
        add_step_tag("gc")
        add_step_tag("profiling")
        clear_step_tags()
        assert get_step_tags() == ()

    def test_step_tags_are_tuples(self):
        """Step tags are tuples (immutable) to avoid shared-reference issues."""
        add_step_tag("a")
        tags = get_step_tags()
        assert isinstance(tags, tuple)

    def test_set_step_clears_tags(self):
        add_step_tag("gc")
        set_step(2)
        assert get_step_tags() == ()

    def test_add_step_tag_spmd_writes_global_only(self):
        # SPMD path: no asyncio task context. add_step_tag should write to
        # _TAGS_GLOBAL; _TAGS_CV stays at default (empty tuple).
        add_step_tag("gc")
        add_step_tag("checkpoint")
        assert step_state._TAGS_GLOBAL == ("gc", "checkpoint")
        assert step_state._TAGS_CV.get() == ()
        assert get_step_tags() == ("gc", "checkpoint")  # falls back to global

    def test_get_step_tags_reader_prefers_cv_over_global(self):
        # Invariant: the reader checks CV first (if non-empty) and falls
        # back to the global. Pins the contract so changing it would flip
        # which tag set async-task spans observe.
        step_state._TAGS_GLOBAL = ("gc",)
        step_state._TAGS_CV.set(("eval",))
        assert get_step_tags() == ("eval",)

    def test_add_step_tag_concurrent_async_tasks_are_isolated(self):
        # Async RL scenario: two actor tasks on the same process add
        # DIFFERENT tags for the same step. Each task's CV must be scoped;
        # neither should see the other's tag via the shared global. Also
        # verifies the exclusive-store semantic: a pre-existing global tag
        # is NOT visible inside the async tasks (actor contexts are
        # intentionally isolated; global tags don't bleed in).
        import asyncio

        # SPMD-side write before any async loop runs → goes to global.
        add_step_tag("pre_async_global_tag")
        assert step_state._TAGS_GLOBAL == ("pre_async_global_tag",)

        async def actor_gc():
            await asyncio.sleep(0.01)
            add_step_tag("gc")
            return get_step_tags()

        async def actor_eval():
            await asyncio.sleep(0.02)  # runs after actor_gc's write
            add_step_tag("eval")
            return get_step_tags()

        async def run():
            return await asyncio.gather(actor_gc(), actor_eval())

        gc_view, eval_view = asyncio.run(run())
        # Each actor sees ONLY its own tag -- no "gc" leak into actor_eval,
        # and (importantly) no "pre_async_global_tag" leak from global.
        assert gc_view == ("gc",), f"actor_gc saw {gc_view!r}, expected ('gc',)"
        assert eval_view == ("eval",), (
            f"actor_eval saw {eval_view!r}, expected ('eval',) "
            f"-- if 'gc' leaked in, add_step_tag is not task-isolated"
        )
        # Global is untouched by the tasks' writes.
        assert step_state._TAGS_GLOBAL == ("pre_async_global_tag",)

    def test_clear_step_tags_resets_both_stores(self):
        # clear_step_tags is called by set_step, so the SPMD happy path
        # cycles tags correctly between steps. Verify it also works when
        # both stores have content.
        step_state._TAGS_GLOBAL = ("gc",)
        step_state._TAGS_CV.set(("eval",))
        step_state.clear_step_tags()
        assert step_state._TAGS_GLOBAL == ()
        assert step_state._TAGS_CV.get() == ()
        assert get_step_tags() == ()

    def test_threading_thread_falls_back_to_global(self):
        # Plain threading.Thread does NOT inherit ContextVar state (that's
        # an asyncio-specific copy_context behavior). A reader on such a
        # thread must fall back to the global for anything to be visible.
        import threading

        add_step_tag("from_main")  # SPMD: writes global
        assert step_state._TAGS_GLOBAL == ("from_main",)

        result: list[tuple[str, ...]] = []

        def thread_reader() -> None:
            result.append(get_step_tags())

        t = threading.Thread(target=thread_reader)
        t.start()
        t.join()
        # Thread sees global via fallback (its CV is at default ()).
        assert result == [("from_main",)]


class TestRelativeStep:
    def test_relative_step_set_and_get(self):
        set_step(101, relative_step=1)
        assert get_relative_step() == 1
        assert get_step() == 101

    def test_relative_step_default_none(self):
        assert get_relative_step() is None

    def test_relative_step_defaults_to_step_when_omitted(self):
        """When set_step is called without relative_step, it defaults to step.

        Prevents a stale _RELATIVE_STEP_GLOBAL from leaking across calls
        (e.g. sync_step broadcasts that omit the kwarg). Correct for
        non-resumed runs; resumed trainers must pass relative_step
        explicitly.
        """
        set_step(1, relative_step=1)
        set_step(2)  # no relative_step -> defaults to step
        assert get_relative_step() == 2

    def test_relative_step_in_formatter_output(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        set_step(101, relative_step=1)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["relative_step"] == 1


class TestHybridContextVarAsync:
    def test_asyncio_task_inherits_step(self):
        """Asyncio tasks inherit the step from the parent context."""
        results = []

        async def worker():
            results.append(get_step())

        async def main():
            set_step(42)
            task = asyncio.create_task(worker())
            await task

        asyncio.run(main())
        assert results == [42]


# ---------------------------------------------------------------------------
# event_extra
# ---------------------------------------------------------------------------


class TestEventExtra:
    def test_basic_event(self):
        extra = event_extra("fwd_bwd")
        assert extra[str(ExtraFields.LOG_TYPE)] == str(LogType.EVENT)
        assert extra[str(ExtraFields.LOG_TYPE_NAME)] == str("fwd_bwd")

    def test_with_step_and_value(self):
        extra = event_extra("step", step=10, value=42.0)
        assert extra[str(ExtraFields.STEP)] == 10
        assert extra[str(ExtraFields.VALUE)] == 42.0

    def test_with_task_name(self):
        extra = event_extra("fwd_bwd", task_name="worker-0")
        assert extra[str(ExtraFields.TASK_NAME)] == "worker-0"


# ---------------------------------------------------------------------------
# TraceJsonlFormatter
# ---------------------------------------------------------------------------


class TestTraceJsonlFormatter:
    def test_format_produces_flat_json(self):
        """Output is flat JSON (not 4-column structured format)."""
        fmt = TraceJsonlFormatter(rank=0, source="trainer")
        set_step(5)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra(str("fwd_bwd") + "_start", step=5).items():
            setattr(record, k, v)

        output = fmt.format(record)
        parsed = json.loads(output)
        # Flat JSON: fields at top level, no "int"/"normal"/"double" keys
        assert "int" not in parsed
        assert "normal" not in parsed
        assert parsed["rank"] == 0
        assert parsed["source"] == "trainer"
        assert parsed["step"] == 5

    def test_rank_and_source_from_self(self):
        fmt = TraceJsonlFormatter(rank=3, source="generator")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["rank"] == 3
        assert parsed["source"] == "generator"

    def test_step_from_global(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        set_step(42)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["step"] == 42

    def test_task_name_in_output(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step", task_name="worker-0").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["task_name"] == "worker-0"

    def test_message_truncation(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        long_msg = "x" * (MAX_MESSAGE_SIZE + 100)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=long_msg,
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert "..." in parsed["message"]
        assert len(parsed["message"]) < len(long_msg)

    def test_step_tags_in_output(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        set_step(1)
        add_step_tag("gc")
        add_step_tag("profiling")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert set(parsed["step_tags"]) == {"gc", "profiling"}

    def test_seq_id_increments(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        seq_ids = []
        for i in range(3):
            r = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"msg{i}",
                args=None,
                exc_info=None,
            )
            for k, v in event_extra("step").items():
                setattr(r, k, v)
            parsed = json.loads(fmt.format(r))
            seq_ids.append(parsed["seq_id"])
        assert seq_ids == [0, 1, 2]

    def test_global_rank_in_output(self):
        """global_rank equals rank (needed for per-rank trace track assignment)."""
        fmt = TraceJsonlFormatter(rank=5, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert parsed["global_rank"] == 5
        assert parsed["rank"] == 5

    def test_local_rank_in_output(self):
        """local_rank from LOCAL_RANK env var (default 0)."""
        fmt = TraceJsonlFormatter(rank=0, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert "local_rank" in parsed
        assert isinstance(parsed["local_rank"], int)

    def test_has_time_us(self):
        fmt = TraceJsonlFormatter(rank=0, source="test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra("step").items():
            setattr(record, k, v)
        parsed = json.loads(fmt.format(record))
        assert "time_us" in parsed
        assert isinstance(parsed["time_us"], int)


# ---------------------------------------------------------------------------
# TraceEventsOnlyFilter
# ---------------------------------------------------------------------------


class TestTraceEventsOnlyFilter:
    def test_passes_event_records(self):
        f = TraceEventsOnlyFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        setattr(record, str(ExtraFields.LOG_TYPE_NAME), str("step"))
        assert f.filter(record) is True

    def test_blocks_non_event_records(self):
        f = TraceEventsOnlyFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="just text",
            args=None,
            exc_info=None,
        )
        assert f.filter(record) is False


# ---------------------------------------------------------------------------
# init_structured_logger
# ---------------------------------------------------------------------------


class TestInitStructuredLogger:
    def test_creates_trace_jsonl(self, tmp_path, structured_logger_fixture):
        output_dir = str(tmp_path)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)

        # Logger should have handlers
        assert any(
            isinstance(h, TraceJsonlHandler) for h in structured_logger_fixture.handlers
        )

        # File should exist after we log something
        set_step(1)
        structured_logger_fixture.info(
            "test event",
            extra=event_extra("step", step=1),
        )

        structured_logs_dir = os.path.join(output_dir, "structured_logs")
        assert os.path.exists(structured_logs_dir)
        jsonl_files = [
            f for f in os.listdir(structured_logs_dir) if f.endswith(".jsonl")
        ]
        assert len(jsonl_files) == 1

        with open(os.path.join(structured_logs_dir, jsonl_files[0])) as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["rank"] == 0
        assert parsed["source"] == "trainer"

    def test_idempotent(self, tmp_path, structured_logger_fixture):
        output_dir = str(tmp_path)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)
        handler_count = len(structured_logger_fixture.handlers)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)
        assert len(structured_logger_fixture.handlers) == handler_count

    def test_file_naming_pattern(self, tmp_path, structured_logger_fixture):
        """File name follows: {source}.global_rank_{rank}.{ts}-{rand}.jsonl"""
        init_structured_logger(rank=3, source="generator", output_dir=str(tmp_path))
        structured_logs_dir = os.path.join(str(tmp_path), "structured_logs")

        # Force a log event so file is created
        structured_logger_fixture.info("x", extra=event_extra("step"))

        files = os.listdir(structured_logs_dir)
        assert len(files) == 1
        assert files[0].startswith("generator.global_rank_3.")
        assert files[0].endswith(".jsonl")

    def test_second_call_is_noop(self, tmp_path, structured_logger_fixture):
        """init_structured_logger is idempotent: second call returns early and
        does not attach additional handlers."""
        output_dir = str(tmp_path)
        init_structured_logger(rank=0, source="trainer", output_dir=output_dir)
        handler_count = len(structured_logger_fixture.handlers)

        init_structured_logger(rank=0, source="other_actor", output_dir=output_dir)

        assert len(structured_logger_fixture.handlers) == handler_count


class TestFactoryMechanism:
    def test_default_creates_jsonl(self, tmp_path, structured_logger_fixture):
        """When TITAN_STRUCT_LOGGER_HANDLERS is unset, default JSONL factory runs."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TITAN_STRUCT_LOGGER_HANDLERS", None)
            init_structured_logger(rank=0, source="test", output_dir=str(tmp_path))
        assert any(
            isinstance(h, TraceJsonlHandler) for h in structured_logger_fixture.handlers
        )

    def test_custom_env_replaces_default(self, tmp_path, structured_logger_fixture):
        """When TITAN_STRUCT_LOGGER_HANDLERS is set, only specified factories run."""
        called = []

        def fake_factory(*, structured_logger, rank, source, output_dir, **kw):
            called.append((rank, source))

        with mock.patch.dict(
            os.environ,
            {
                "TITAN_STRUCT_LOGGER_HANDLERS": "tests.unit_tests.observability.test_structured_logging.fake_factory"
            },
        ):
            # Patch importlib to load our fake factory
            with mock.patch("importlib.import_module") as mock_import:
                mock_mod = mock.MagicMock()
                mock_mod.fake_factory = fake_factory
                mock_import.return_value = mock_mod
                init_structured_logger(rank=0, source="test", output_dir=str(tmp_path))

        assert len(called) == 1
        assert called[0] == (0, "test")
        # No TraceJsonlHandler since custom factory replaced default
        assert not any(
            isinstance(h, TraceJsonlHandler) for h in structured_logger_fixture.handlers
        )


# ---------------------------------------------------------------------------
# No-op flag
# ---------------------------------------------------------------------------


class TestNoOpFlag:
    def test_structured_logger_disabled_flag(self, tmp_path, structured_logger_fixture):
        """When the module-level ``_disabled`` flag is set, spans and instants produce no events."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with mock.patch(
            "torchtitan.observability.structured_logger.structured_logging._disabled",
            True,
        ):
            set_step(1)
            with log_trace_span("step"):
                pass
            log_trace_instant("binary_start")

        trace_dir = tmp_path / "structured_logs"
        lines = []
        if trace_dir.exists():
            for f in trace_dir.iterdir():
                lines.extend(f.read_text().strip().splitlines())
        assert (
            len(lines) == 0
        ), f"Expected no events when tracing disabled, got {len(lines)}"

    def test_init_with_enable_false_disables_logging(
        self, tmp_path, structured_logger_fixture
    ):
        """``init_structured_logger(enable=False)`` makes all subsequent trace calls no-ops."""
        import torchtitan.observability.structured_logger.structured_logging as sl_mod

        # Ensure clean state before this test
        sl_mod._disabled = False
        try:
            init_structured_logger(
                rank=0, source="trainer", output_dir=str(tmp_path), enable=False
            )
            set_step(1)
            with log_trace_span("step"):
                pass
            log_trace_instant("binary_start")
            log_trace_scalar({"x": 1.0})
        finally:
            sl_mod._disabled = False

        # No structured_logs directory should be created (no handlers attached)
        trace_dir = tmp_path / "structured_logs"
        if trace_dir.exists():
            for f in trace_dir.iterdir():
                assert (
                    f.read_text().strip() == ""
                ), "Expected no events when init'd with enable=False"


# ---------------------------------------------------------------------------
# log_trace_scalar
# ---------------------------------------------------------------------------


class TestLogTraceScalar:
    def test_writes_metric_events(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(5)
        log_trace_scalar({"train.loss": 2.5, "train.tflops": 45.6})

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["log_type_name"] == "metric_value"
        assert lines[0]["event_name"] == "train.loss"
        assert lines[0]["value"] == 2.5
        assert lines[0]["step"] == 5
        assert lines[1]["event_name"] == "train.tflops"
        assert lines[1]["value"] == 45.6

    def test_noop_when_disabled(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with mock.patch(
            "torchtitan.observability.structured_logger.structured_logging._disabled",
            True,
        ):
            log_trace_scalar({"should.not.appear": 1.0})

        trace_dir = tmp_path / "structured_logs"
        lines = []
        if trace_dir.exists():
            for f in trace_dir.iterdir():
                lines.extend(f.read_text().strip().splitlines())
        assert len(lines) == 0

    def test_empty_dict_is_noop(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        log_trace_scalar({})

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# log_trace_instant
# ---------------------------------------------------------------------------


class TestLogTraceInstant:
    def test_writes_instant_marker(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        log_trace_instant("binary_start")

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 1
        assert lines[0]["log_type_name"] == "binary_start"
        assert lines[0]["log_type"] == "instant"
        assert lines[0]["event_name"] is None

    def test_noop_when_disabled(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with mock.patch(
            "torchtitan.observability.structured_logger.structured_logging._disabled",
            True,
        ):
            log_trace_instant("training_start")

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# log_trace_span
# ---------------------------------------------------------------------------


class TestLogTraceSpan:
    def test_logs_start_and_end(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(5)

        with log_trace_span("fwd_bwd"):
            time.sleep(0.01)

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["log_type_name"] == str("fwd_bwd") + "_start"
        assert lines[1]["log_type_name"] == str("fwd_bwd") + "_end"
        assert lines[1]["value"] > 0
        assert lines[0]["step"] == 5
        assert lines[1]["step"] == 5
        # Span events must NOT have event_name (only in message field)
        assert lines[0].get("event_name") is None
        assert lines[1].get("event_name") is None

    def test_event_type_required(self):
        """event_type is required — cannot be omitted."""
        with pytest.raises(TypeError):
            log_trace_span()  # pyrefly: ignore[missing-argument]

    def test_event_type_accepts_plain_string(self, tmp_path, structured_logger_fixture):
        """Plain strings are accepted as log_type_name."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        with log_trace_span("rl_rollout"):
            pass

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert lines[0]["log_type_name"] == "rl_rollout_start"
        assert lines[1]["log_type_name"] == "rl_rollout_end"

    def test_two_spans_emit_start_end_pairs(self, tmp_path, structured_logger_fixture):
        """Sanity check: each `with log_trace_span(...)` writes exactly one
        start and one end record, identified by their ``log_type_name``."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        with log_trace_span("step"):
            pass
        with log_trace_span("optim"):
            pass

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        type_names = [line.get("log_type_name") for line in lines]
        assert type_names == ["step_start", "step_end", "optim_start", "optim_end"]

    def test_works_as_decorator(self, tmp_path, structured_logger_fixture):
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        @log_trace_span("optim")
        def optimizer_step():
            pass

        optimizer_step()

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2

    def test_async_decorator_brackets_coroutine_body(
        self, tmp_path, structured_logger_fixture
    ):
        # ContextDecorator's default __call__ produces a sync wrapper that
        # exits before `await func(...)` runs, so for an async function the
        # _end record fires before the body executes. Override __call__ for
        # coroutine functions so __exit__ runs after the awaitable completes.
        # Verified by event order in the JSONL stream — no wall-clock timing
        # needed, so the test is deterministic.
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))

        @log_trace_span("rl_rollout")
        async def rollout():
            log_trace_instant("body_marker")

        asyncio.run(rollout())

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            type_names = [
                json.loads(line)["log_type_name"] for line in f if line.strip()
            ]

        assert type_names == ["rl_rollout_start", "body_marker", "rl_rollout_end"]

    def test_exception_recording(self, tmp_path, structured_logger_fixture):
        """When an exception occurs, an _error event is emitted."""
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        with pytest.raises(ValueError):
            with log_trace_span("step"):
                raise ValueError("test error")

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # 3 events: start, error, end
        assert len(lines) == 3
        type_names = [line["log_type_name"] for line in lines]
        assert type_names[0].endswith("_start")
        assert type_names[1].endswith("_error")
        assert type_names[2].endswith("_end")

    def test_sync_decorator_caller_points_at_user_site(
        self, tmp_path, structured_logger_fixture
    ):
        """``@log_trace_span`` on a sync function must attribute caller to the
        user's call site (test method below), not to the internal sync_wrapper
        in structured_logging.py. Regression guard for the ``stacklevel + 1``
        adjustment in ``log_trace_span.__call__``.
        """
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))

        @log_trace_span("decorated_sync")
        def my_fn():
            pass

        my_fn()

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            start = next(
                json.loads(line)
                for line in f
                if line.strip()
                and json.loads(line)["log_type_name"] == "decorated_sync_start"
            )

        path, _, funcname = start["caller"].rsplit(":", 2)
        assert os.path.basename(path) == "test_structured_logging.py", start["caller"]
        # Without stacklevel+1, funcname would be "sync_wrapper".
        assert funcname == "test_sync_decorator_caller_points_at_user_site", start[
            "caller"
        ]

    def test_nested_async_decorators_emit_in_order(
        self, tmp_path, structured_logger_fixture
    ):
        """Mirrors the RL actor pattern: an outer ``@log_trace_span`` async
        method awaits an inner ``@log_trace_span`` async method. All four
        records must emit in nested order (outer_start, inner_start,
        inner_end, outer_end) — proves ``__exit__`` runs after ``await``
        completes for both layers, so durations are real and Gantt pairing
        works.
        """
        init_structured_logger(rank=0, source="trainer", output_dir=str(tmp_path))

        @log_trace_span("inner_async")
        async def inner():
            pass

        @log_trace_span("outer_async")
        async def outer():
            await inner()

        asyncio.run(outer())

        trace_dir = os.path.join(str(tmp_path), "structured_logs")
        jsonl_files = [f for f in os.listdir(trace_dir) if f.endswith(".jsonl")]
        with open(os.path.join(trace_dir, jsonl_files[0])) as f:
            type_names = [
                json.loads(line)["log_type_name"] for line in f if line.strip()
            ]

        assert type_names == [
            "outer_async_start",
            "inner_async_start",
            "inner_async_end",
            "outer_async_end",
        ]


# ---------------------------------------------------------------------------
# generate_gantt_trace (analysis.py)
# ---------------------------------------------------------------------------


class TestGenerateGanttTrace:
    def _write_jsonl(self, path, records):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_pairs_by_span_id(self, tmp_path):
        """Start/end events are paired by span_id."""
        records = [
            {
                "log_type_name": "fwd_bwd_start",
                "time_us": 1000,
                "rank": 0,
                "step": 1,
                "span_id": 0,
                "event_name": "fwd_bwd",
            },
            {
                "log_type_name": "fwd_bwd_end",
                "time_us": 2000,
                "rank": 0,
                "step": 1,
                "span_id": 0,
                "value": 1.0,
                "event_name": "fwd_bwd",
            },
        ]
        jsonl_path = os.path.join(str(tmp_path), "structured_logs", "test.jsonl")
        self._write_jsonl(jsonl_path, records)

        output_path = os.path.join(str(tmp_path), "gantt.json")
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"), output_path
        )

        # Filter to only "X" (complete) events
        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        assert len(x_events) == 1
        assert x_events[0]["name"] == "fwd_bwd"
        assert x_events[0]["ts"] == 1000
        assert x_events[0]["dur"] == 1000  # 1.0ms * 1000 = 1000us

    def test_empty_dir(self, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        trace = generate_gantt_trace(empty_dir, str(tmp_path / "out.json"))
        assert trace["traceEvents"] == []

    def test_metric_events(self, tmp_path):
        """metric_value events become instant events."""
        records = [
            {
                "log_type_name": "metric_value",
                "log_type": "instant",
                "time_us": 1000,
                "rank": 0,
                "step": 1,
                "event_name": "train.loss",
                "value": 2.5,
            },
        ]
        jsonl_path = os.path.join(str(tmp_path), "structured_logs", "test.jsonl")
        self._write_jsonl(jsonl_path, records)

        output_path = os.path.join(str(tmp_path), "gantt.json")
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"), output_path
        )

        instant_events = [e for e in trace["traceEvents"] if e.get("ph") == "i"]
        assert len(instant_events) == 1
        assert "train.loss" in instant_events[0]["name"]
        assert "2.5" in instant_events[0]["name"]

    def test_error_events(self, tmp_path):
        """_error events become instant error markers."""
        records = [
            {"log_type_name": "fwd_bwd_error", "time_us": 1500, "rank": 0, "step": 1},
        ]
        jsonl_path = os.path.join(str(tmp_path), "structured_logs", "test.jsonl")
        self._write_jsonl(jsonl_path, records)

        output_path = os.path.join(str(tmp_path), "gantt.json")
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"), output_path
        )

        error_events = [
            e
            for e in trace["traceEvents"]
            if e.get("ph") == "i" and "ERROR" in e.get("name", "")
        ]
        assert len(error_events) == 1

    def test_log_type_instant_routes_to_instant_regardless_of_suffix(self, tmp_path):
        """Records with ``log_type == "instant"`` render as instants even when
        their ``log_type_name`` ends in ``_start``. Confirms the classifier
        branches on emission intent, not name suffix — so future instants
        like ``binary_start`` / ``training_start`` work without an allowlist.
        """
        records = [
            {
                "log_type_name": "binary_start",
                "log_type": "instant",
                "time_us": 1000,
                "rank": 0,
            },
            {
                "log_type_name": "training_start",
                "log_type": "instant",
                "time_us": 2000,
                "rank": 0,
            },
        ]
        jsonl_path = os.path.join(str(tmp_path), "structured_logs", "test.jsonl")
        self._write_jsonl(jsonl_path, records)

        output_path = os.path.join(str(tmp_path), "gantt.json")
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"), output_path
        )

        instant_names = [
            e.get("name") for e in trace["traceEvents"] if e.get("ph") == "i"
        ]
        assert "binary_start" in instant_names
        assert "training_start" in instant_names
        # No span-pair events were created from these records.
        assert not any(e.get("ph") == "X" for e in trace["traceEvents"])

    def test_unknown_event_type_defaults_to_instant(self, tmp_path):
        """Records whose `log_type_name` doesn't match any branch render as
        bare instants. Matches Foundry's `ELSE 'instant'` fallthrough so no
        event is silently dropped.
        """
        records = [
            {"log_type_name": "ad_hoc_marker", "time_us": 1000, "rank": 0, "step": 3},
        ]
        jsonl_path = os.path.join(str(tmp_path), "structured_logs", "test.jsonl")
        self._write_jsonl(jsonl_path, records)

        output_path = os.path.join(str(tmp_path), "gantt.json")
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"), output_path
        )

        instant_events = [e for e in trace["traceEvents"] if e.get("ph") == "i"]
        assert len(instant_events) == 1
        assert instant_events[0]["name"] == "ad_hoc_marker"
        assert instant_events[0]["args"]["step"] == 3

    def test_null_event_name_uses_type_name(self, tmp_path):
        """When event_name is null (spans after P0 fix), display_name = type_name."""
        records = [
            {
                "log_type_name": "fwd_bwd_start",
                "time_us": 1000,
                "rank": 0,
                "step": 1,
                "span_id": 0,
            },
            {
                "log_type_name": "fwd_bwd_end",
                "time_us": 2000,
                "rank": 0,
                "step": 1,
                "span_id": 0,
                "value": 1.0,
            },
        ]
        jsonl_path = os.path.join(str(tmp_path), "structured_logs", "test.jsonl")
        self._write_jsonl(jsonl_path, records)

        output_path = os.path.join(str(tmp_path), "gantt.json")
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"), output_path
        )

        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        assert len(x_events) == 1
        # With null event_name, display_name falls back to type_name
        assert x_events[0]["name"] == "fwd_bwd"

    def test_fallback_pairing_without_span_id(self, tmp_path):
        """Falls back to (type_name, pid, rank) when span_id is absent."""
        records = [
            {
                "log_type_name": "step_start",
                "time_us": 1000,
                "rank": 0,
                "step": 1,
                "event_name": "step",
            },
            {
                "log_type_name": "step_end",
                "time_us": 2000,
                "rank": 0,
                "step": 1,
                "value": 1.0,
                "event_name": "step",
            },
        ]
        jsonl_path = os.path.join(str(tmp_path), "structured_logs", "test.jsonl")
        self._write_jsonl(jsonl_path, records)

        output_path = os.path.join(str(tmp_path), "gantt.json")
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"), output_path
        )

        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        assert len(x_events) == 1

    # ------------------------------------------------------------------
    # Pairing contract (post-span_id-removal): LIFO stack per (source,
    # task_name). Each test writes records WITHOUT a span_id field and
    # asserts pairing + tid placement by semantic outcome — which span
    # ends on which _end, and which tid each span lands on — not by
    # counting tids.
    # ------------------------------------------------------------------

    def _start_rec(self, type_name, time_us, task_name, rank=0, event_name=None):
        return {
            "log_type_name": f"{type_name}_start",
            "time_us": time_us,
            "rank": rank,
            "task_name": task_name,
            **({"event_name": event_name} if event_name else {}),
        }

    def _end_rec(self, type_name, time_us, duration_ms, task_name, rank=0):
        return {
            "log_type_name": f"{type_name}_end",
            "time_us": time_us,
            "rank": rank,
            "task_name": task_name,
            "value": duration_ms,
        }

    def test_lifo_pairs_nested_spans_single_task(self, tmp_path):
        """Nested spans within one task pair by LIFO stack on task_name.

        Outer wraps inner; inner ends first, outer ends second.
        Both spans must carry their original start timestamps and
        durations after pairing.
        """
        records = [
            self._start_rec("outer", 1000, "Task-1"),
            self._start_rec("inner", 1200, "Task-1"),
            self._end_rec("inner", 1400, duration_ms=0.2, task_name="Task-1"),
            self._end_rec("outer", 2000, duration_ms=1.0, task_name="Task-1"),
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        by_name = {e["name"]: e for e in trace["traceEvents"] if e.get("ph") == "X"}
        # Exactly two spans paired, with the right timings.
        assert "outer" in by_name and "inner" in by_name
        assert by_name["outer"]["ts"] == 1000
        assert by_name["outer"]["dur"] == 1000  # 1.0 ms -> 1000 us
        assert by_name["inner"]["ts"] == 1200
        assert by_name["inner"]["dur"] == 200

    def test_lifo_pairs_concurrent_non_nested_tasks(self, tmp_path):
        """Two tasks on the same source whose spans overlap in non-LIFO order.

        A_start, B_start, A_end, B_end — if the pair-by-(source, tid) scheme
        were used (as msl does), the first _end would pop B (last in stack)
        and mispair. Keying the stack on task_name isolates the two tasks.
        """
        records = [
            self._start_rec("A", 1000, "Task-1"),
            self._start_rec("B", 1100, "Task-2"),
            self._end_rec("A", 1500, duration_ms=0.5, task_name="Task-1"),
            self._end_rec("B", 2000, duration_ms=0.9, task_name="Task-2"),
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        by_name = {e["name"]: e for e in trace["traceEvents"] if e.get("ph") == "X"}
        # A must pair with its own _end (ts=1000, dur=500us), not B's.
        assert by_name["A"]["ts"] == 1000
        assert by_name["A"]["dur"] == 500
        assert by_name["B"]["ts"] == 1100
        assert by_name["B"]["dur"] == 900
        # Each task gets its own tid (both auto-named and overlapping).
        assert by_name["A"]["tid"] != by_name["B"]["tid"]

    def test_spmd_pairing_with_null_task_name(self, tmp_path):
        """SPMD code has task_name=None on every record; LIFO-per-source pairs
        nested spans correctly."""
        records = [
            self._start_rec("step", 1000, task_name=None),
            self._start_rec("fwd_bwd", 1100, task_name=None),
            self._end_rec("fwd_bwd", 1500, duration_ms=0.4, task_name=None),
            self._end_rec("step", 2000, duration_ms=1.0, task_name=None),
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        by_name = {e["name"]: e for e in trace["traceEvents"] if e.get("ph") == "X"}
        assert by_name["step"]["ts"] == 1000
        assert by_name["step"]["dur"] == 1000
        assert by_name["fwd_bwd"]["ts"] == 1100
        assert by_name["fwd_bwd"]["dur"] == 400

    def test_cross_source_pairing_is_isolated(self, tmp_path):
        """Two sources with identical task names must pair independently.

        Without source-scoped keys, source_b's records could consume
        source_a's pending starts (or vice versa) and report wrong
        durations.
        """
        # source_a: A_start at 1000, A_end at 9000 (dur 8 ms)
        records_a = [
            self._start_rec("A", 1000, "Task-1"),
            self._end_rec("A", 9000, duration_ms=8.0, task_name="Task-1"),
        ]
        # source_b: reuses "Task-1" but fires earlier and shorter
        records_b = [
            self._start_rec("A", 2000, "Task-1"),
            self._end_rec("A", 3000, duration_ms=1.0, task_name="Task-1"),
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records_a
        )
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "b.jsonl"), records_b
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        pid_to_source = {
            e["pid"]: e["args"]["name"]
            for e in trace["traceEvents"]
            if e.get("name") == "process_name"
        }
        by_source_and_dur = {
            (pid_to_source[e["pid"]], e["dur"])
            for e in trace["traceEvents"]
            if e.get("ph") == "X"
        }
        # Each source's span carries its own duration. If pairing leaked
        # across sources, one span would take the other's duration.
        assert ("a", 8000) in by_source_and_dur
        assert ("b", 1000) in by_source_and_dur

    def test_nested_spans_same_task_share_one_tid(self, tmp_path):
        """Nested spans within a single task land on ONE tid; Perfetto renders
        the nesting as a stacked flamegraph on that track, so splitting
        across multiple tids would produce a confusing "N parallel threads"
        rendering of what is actually a single thread doing work."""
        records = [
            self._start_rec("outer", 1000, "Task-1"),
            self._start_rec("middle", 1100, "Task-1"),
            self._start_rec("inner", 1200, "Task-1"),
            self._end_rec("inner", 1400, duration_ms=0.2, task_name="Task-1"),
            self._end_rec("middle", 1800, duration_ms=0.7, task_name="Task-1"),
            self._end_rec("outer", 2000, duration_ms=1.0, task_name="Task-1"),
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        assert len(x_events) == 3
        tids = {e["tid"] for e in x_events}
        assert tids == {0}, (
            f"Three nested spans in one task should share tid 0, "
            f"got tids={sorted(tids)}"
        )

    def test_sequential_tasks_reuse_one_tid(self, tmp_path):
        """Multiple auto-named tasks that do NOT overlap in wall-clock time
        should reuse a single tid (like the grader: 5 sequential
        ``score()`` endpoint calls should appear on one track, not five)."""
        records = []
        for i, task in enumerate(["Task-2", "Task-4", "Task-6", "Task-8", "Task-10"]):
            start = 1000 + i * 10_000
            end = start + 500
            records.append(self._start_rec("work", start, task_name=task))
            records.append(self._end_rec("work", end, duration_ms=0.5, task_name=task))
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        tids = {e["tid"] for e in trace["traceEvents"] if e.get("ph") == "X"}
        assert tids == {0}

    def test_two_concurrent_tasks_each_with_nested_spans(self, tmp_path):
        """Two tasks running in parallel, each with a nested inner span.

        Under TASK-RANGE slot-packing: each task's nesting collapses to one
        slot, the two tasks' ranges overlap -> 2 slots total (one per task).
        Under SPAN-LEVEL slot-packing (the current bug): all 4 spans are
        concurrent in wall-clock time -> 4 slots total. The failure mode
        exactly mirrors the 'grader shows 4 tids' bug the user reported.
        """
        records = [
            # Task-1: outer [1000..3000], inner [1200..1800]
            self._start_rec("A_outer", 1000, "Task-1"),
            self._start_rec("A_inner", 1200, "Task-1"),
            self._end_rec("A_inner", 1800, duration_ms=0.6, task_name="Task-1"),
            self._end_rec("A_outer", 3000, duration_ms=2.0, task_name="Task-1"),
            # Task-2: outer [2000..4000] (overlaps Task-1), inner [2200..2800]
            self._start_rec("B_outer", 2000, "Task-2"),
            self._start_rec("B_inner", 2200, "Task-2"),
            self._end_rec("B_inner", 2800, duration_ms=0.6, task_name="Task-2"),
            self._end_rec("B_outer", 4000, duration_ms=2.0, task_name="Task-2"),
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        tids = {e["tid"] for e in trace["traceEvents"] if e.get("ph") == "X"}
        assert len(tids) == 2, (
            f"Expected 2 tids (one per task), got {len(tids)}: {sorted(tids)}. "
            f"Span-level slot-packing would give 4 (all spans concurrent in "
            f"wall-clock); task-range gives 2 (nesting collapses per task)."
        )
        # Spans in the same task must share a tid.
        by_name = {e["name"]: e for e in trace["traceEvents"] if e.get("ph") == "X"}
        assert by_name["A_outer"]["tid"] == by_name["A_inner"]["tid"]
        assert by_name["B_outer"]["tid"] == by_name["B_inner"]["tid"]
        # Different tasks must get different tids.
        assert by_name["A_outer"]["tid"] != by_name["B_outer"]["tid"]

    def test_concurrent_tasks_get_separate_tids(self, tmp_path):
        """Two asyncio tasks whose time ranges overlap (e.g., from
        ``asyncio.gather``) must land on DIFFERENT tids so Perfetto
        renders them as parallel tracks rather than overlapping bars on
        one track.

        Under task-range slot-packing: Task-1's range [1000..3000] and
        Task-2's range [2000..4000] overlap -> slot 0 and slot 1.
        Under the (wrong) 'just use row.tid' approach, both would default
        to tid 0 in the fixture and the test would fail.
        """
        records = [
            # Task-1 span: 1000 -> 3000
            self._start_rec("A", 1000, "Task-1"),
            self._end_rec("A", 3000, duration_ms=2.0, task_name="Task-1"),
            # Task-2 span: 2000 -> 4000 (overlaps A in wall-clock)
            self._start_rec("B", 2000, "Task-2"),
            self._end_rec("B", 4000, duration_ms=2.0, task_name="Task-2"),
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        by_name = {e["name"]: e for e in trace["traceEvents"] if e.get("ph") == "X"}
        assert by_name["A"]["tid"] != by_name["B"]["tid"], (
            f"Concurrent tasks must get different tids; got A.tid="
            f"{by_name['A']['tid']}, B.tid={by_name['B']['tid']}"
        )

    def test_instant_uses_task_name_not_span_id_for_tid(self, tmp_path):
        """Instants (error, metric) with an explicit task_name land on that
        task's track; without one, they fall back to tid 0."""
        records = [
            # Named task "train_loop" has a real span so it gets a tid.
            self._start_rec("work", 1000, "train_loop"),
            self._end_rec("work", 2000, duration_ms=1.0, task_name="train_loop"),
            # Error instant inside the named task should go on its track.
            {
                "log_type_name": "work_error",
                "time_us": 1500,
                "rank": 0,
                "task_name": "train_loop",
            },
            # Metric with no task -> falls back to tid 0.
            {
                "log_type_name": "metric_value",
                "log_type": "instant",
                "time_us": 1700,
                "rank": 0,
                "event_name": "loss",
                "value": 2.5,
                "task_name": None,
            },
        ]
        self._write_jsonl(
            os.path.join(str(tmp_path), "structured_logs", "a.jsonl"), records
        )
        trace = generate_gantt_trace(
            os.path.join(str(tmp_path), "structured_logs"),
            os.path.join(str(tmp_path), "gantt.json"),
        )
        span = next(e for e in trace["traceEvents"] if e.get("ph") == "X")
        error = next(
            e
            for e in trace["traceEvents"]
            if e.get("ph") == "i" and "ERROR" in e["name"]
        )
        metric = next(
            e
            for e in trace["traceEvents"]
            if e.get("ph") == "i" and "loss" in e["name"]
        )
        # Error on explicit task's track, metric on main track.
        assert error["tid"] == span["tid"]
        assert metric["tid"] == 0


# ---------------------------------------------------------------------------
# register_jsonl_handler
# ---------------------------------------------------------------------------


class TestRegisterJsonlHandler:
    def test_creates_handler_with_correct_path(
        self, tmp_path, structured_logger_fixture
    ):
        register_jsonl_handler(
            structured_logger=structured_logger_fixture,
            rank=2,
            source="test_src",
            output_dir=str(tmp_path),
        )

        # Should have added a TraceJsonlHandler
        handlers = [
            h
            for h in structured_logger_fixture.handlers
            if isinstance(h, TraceJsonlHandler)
        ]
        assert len(handlers) == 1

        # File path should be in structured_logs/
        filepath = handlers[0].baseFilename
        assert "structured_logs" in filepath
        assert "test_src.global_rank_2" in filepath
        assert filepath.endswith(".jsonl")
