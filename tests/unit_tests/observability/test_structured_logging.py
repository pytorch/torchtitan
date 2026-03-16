# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for torchtitan.observability.structured_logging."""

import json
import logging
import os
import time

import pytest

from torchtitan.observability import step_state
from torchtitan.observability._constants import SYSTEM_LOGGER_NAME
from torchtitan.observability.analysis import generate_gantt_trace
from torchtitan.observability.step_state import (
    add_step_tag,
    clear_step_tags,
    get_step,
    get_step_tags,
    set_step,
)
from torchtitan.observability.structured_logging import (
    event_extra,
    EventsOnlyFilter,
    EventType,
    ExtraFields,
    InflightEventTrackingHandler,
    init_observability,
    LogType,
    MAX_MESSAGE_SIZE,
    record_event,
    record_span,
    StructuredJSONFormatter,
    StructuredLoggingHandler,
    to_structured_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_context():
    """Reset step state before each test."""
    step_state._STEP = None
    step_state._STEP_TAGS = ()
    yield
    step_state._STEP = None
    step_state._STEP_TAGS = ()


@pytest.fixture
def system_logger():
    """Provide a clean system logger for testing."""
    logger = logging.getLogger(SYSTEM_LOGGER_NAME)
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    yield logger
    # Restore
    logger.handlers = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate


# ---------------------------------------------------------------------------
# Step context tests
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

    def test_clear_step_tags_on_empty(self):
        clear_step_tags()
        assert get_step_tags() == ()

    def test_step_tags_are_tuples(self):
        """Step tags are tuples (immutable) to avoid shared-reference issues."""
        add_step_tag("a")
        tags = get_step_tags()
        assert isinstance(tags, tuple)


# ---------------------------------------------------------------------------
# event_extra
# ---------------------------------------------------------------------------


class TestEventExtra:
    def test_basic_event(self):
        extra = event_extra(EventType.FWD_BWD)
        assert extra[str(ExtraFields.LOG_TYPE)] == str(LogType.EVENT)
        assert extra[str(ExtraFields.LOG_TYPE_NAME)] == str(EventType.FWD_BWD)

    def test_with_step_and_value(self):
        extra = event_extra(EventType.STEP, step=10, value=42.0)
        assert extra[str(ExtraFields.STEP)] == 10
        assert extra[str(ExtraFields.VALUE)] == 42.0


# ---------------------------------------------------------------------------
# to_structured_json
# ---------------------------------------------------------------------------


class TestToStructuredJson:
    def test_int_field(self):
        result = json.loads(to_structured_json({"rank": 0}))
        assert result["int"]["rank"] == 0

    def test_float_field(self):
        result = json.loads(to_structured_json({"value": 3.14}))
        assert result["double"]["value"] == pytest.approx(3.14)

    def test_string_field(self):
        result = json.loads(to_structured_json({"source": "trainer"}))
        assert result["normal"]["source"] == "trainer"

    def test_list_field(self):
        result = json.loads(to_structured_json({"tags": ["a", "b"]}))
        assert result["normvector"]["tags"] == ["a", "b"]

    def test_none_values_skipped(self):
        result = json.loads(to_structured_json({"a": None, "b": 1}))
        assert "a" not in result["int"]
        assert "a" not in result["normal"]
        assert result["int"]["b"] == 1

    def test_bool_becomes_int(self):
        result = json.loads(to_structured_json({"flag": True}))
        assert result["int"]["flag"] == 1

    def test_empty_dict(self):
        result = json.loads(to_structured_json({}))
        assert result == {"int": {}, "normal": {}, "double": {}, "normvector": {}}


# ---------------------------------------------------------------------------
# StructuredJSONFormatter
# ---------------------------------------------------------------------------


class TestStructuredJSONFormatter:
    def test_format_produces_valid_json(self):
        fmt = StructuredJSONFormatter(rank=0, source="trainer")
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
        # Add event extra fields
        for k, v in event_extra(str(EventType.FWD_BWD) + "_start", step=5).items():
            setattr(record, k, v)

        output = fmt.format(record)
        parsed = json.loads(output)
        assert "int" in parsed
        assert "normal" in parsed
        assert parsed["int"]["rank"] == 0
        assert parsed["normal"]["source"] == "trainer"
        assert parsed["int"]["step"] == 5

    def test_rank_and_source_from_self(self):
        fmt = StructuredJSONFormatter(rank=3, source="generator")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["int"]["rank"] == 3
        assert parsed["normal"]["source"] == "generator"

    def test_step_from_global(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
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
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["int"]["step"] == 42

    def test_message_truncation(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
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
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "..." in parsed["normal"]["message"]
        assert len(parsed["normal"]["message"]) < len(long_msg)

    def test_step_tags_in_output(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
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
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "step_tags" in parsed["normvector"]
        assert set(parsed["normvector"]["step_tags"]) == {"gc", "profiling"}

    def test_seq_id_increments(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
        records = []
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
            for k, v in event_extra(EventType.STEP).items():
                setattr(r, k, v)
            records.append(r)

        seq_ids = []
        for r in records:
            parsed = json.loads(fmt.format(r))
            seq_ids.append(parsed["int"]["seq_id"])
        assert seq_ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# EventsOnlyFilter
# ---------------------------------------------------------------------------


class TestEventsOnlyFilter:
    def test_passes_event_records(self):
        f = EventsOnlyFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        setattr(record, str(ExtraFields.LOG_TYPE_NAME), str(EventType.STEP))
        assert f.filter(record) is True

    def test_blocks_non_event_records(self):
        f = EventsOnlyFilter()
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
# InflightEventTrackingHandler
# ---------------------------------------------------------------------------


class TestInflightEventTrackingHandler:
    def test_tracks_last_event(self):
        handler = InflightEventTrackingHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        setattr(
            record, str(ExtraFields.LOG_TYPE_NAME), str(EventType.FWD_BWD) + "_start"
        )
        handler.emit(record)
        assert handler.last_event == str(EventType.FWD_BWD) + "_start"
        assert handler.last_event_time is not None

    def test_ignores_non_event_records(self):
        handler = InflightEventTrackingHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="plain text",
            args=None,
            exc_info=None,
        )
        handler.emit(record)
        assert handler.last_event is None


# ---------------------------------------------------------------------------
# init_observability
# ---------------------------------------------------------------------------


class TestInitObservability:
    def test_creates_system_jsonl(self, tmp_path, system_logger):
        output_dir = str(tmp_path)
        init_observability(rank=0, source="trainer", output_dir=output_dir)

        # Logger should have handlers
        assert any(
            isinstance(h, StructuredLoggingHandler) for h in system_logger.handlers
        )

        # File should exist after we log something
        set_step(1)
        system_logger.info(
            "test event",
            extra=event_extra(EventType.STEP, step=1),
        )

        expected_path = os.path.join(
            output_dir, "system_logs", "trainer_rank_0_system.jsonl"
        )
        assert os.path.exists(expected_path)

        with open(expected_path) as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["int"]["rank"] == 0
        assert parsed["normal"]["source"] == "trainer"

    def test_idempotent(self, tmp_path, system_logger):
        output_dir = str(tmp_path)
        init_observability(rank=0, source="trainer", output_dir=output_dir)
        handler_count = len(system_logger.handlers)
        init_observability(rank=0, source="trainer", output_dir=output_dir)
        assert len(system_logger.handlers) == handler_count

    def test_creates_inflight_handler(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        assert any(
            isinstance(h, InflightEventTrackingHandler) for h in system_logger.handlers
        )


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------


class TestRecordEvent:
    def test_writes_metric_events(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(10)
        record_event({"train.loss": 2.5, "train.lr": 1e-4})

        jsonl_path = os.path.join(
            str(tmp_path), "system_logs", "trainer_rank_0_system.jsonl"
        )
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        # Check that event names are present
        event_names = {line["normal"].get("event_name") for line in lines}
        assert "train.loss" in event_names
        assert "train.lr" in event_names
        # Verify step appears in output
        assert all(line["int"].get("step") == 10 for line in lines)


# ---------------------------------------------------------------------------
# record_span
# ---------------------------------------------------------------------------


class TestRecordSpan:
    def test_logs_start_and_end(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(5)

        with record_span("Forward/Backward", EventType.FWD_BWD):
            time.sleep(0.01)

        jsonl_path = os.path.join(
            str(tmp_path), "system_logs", "trainer_rank_0_system.jsonl"
        )
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["normal"]["log_type_name"] == str(EventType.FWD_BWD) + "_start"
        assert lines[1]["normal"]["log_type_name"] == str(EventType.FWD_BWD) + "_end"
        # End event should have duration in value
        assert lines[1]["double"]["value"] > 0
        # Both events should have step=5
        assert lines[0]["int"]["step"] == 5
        assert lines[1]["int"]["step"] == 5

    def test_works_as_decorator(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        @record_span("Optimizer", EventType.OPTIM)
        def optimizer_step():
            pass

        optimizer_step()

        jsonl_path = os.path.join(
            str(tmp_path), "system_logs", "trainer_rank_0_system.jsonl"
        )
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["normal"]["log_type_name"] == str(EventType.OPTIM) + "_start"
        assert lines[1]["normal"]["log_type_name"] == str(EventType.OPTIM) + "_end"

    def test_does_not_suppress_exceptions(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        with pytest.raises(ValueError, match="test error"):
            with record_span("Test", EventType.STEP):
                raise ValueError("test error")

    def test_no_event_type_uses_description(self, tmp_path, system_logger):
        """When event_type is omitted, description is used as the base name."""
        init_observability(rank=0, source="reward", output_dir=str(tmp_path))
        set_step(1)
        with record_span("rl_time/scoring_s"):
            pass
        jsonl_path = os.path.join(
            str(tmp_path), "system_logs", "reward_rank_0_system.jsonl"
        )
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines[0]["normal"]["log_type_name"] == "rl_time/scoring_s_start"
        assert lines[1]["normal"]["log_type_name"] == "rl_time/scoring_s_end"

    def test_string_event_type(self, tmp_path, system_logger):
        """event_type can be a plain string."""
        init_observability(rank=0, source="reward", output_dir=str(tmp_path))
        set_step(1)
        with record_span("Grading", "rl_grading"):
            pass
        jsonl_path = os.path.join(
            str(tmp_path), "system_logs", "reward_rank_0_system.jsonl"
        )
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines[0]["normal"]["log_type_name"] == "rl_grading_start"
        assert lines[1]["normal"]["log_type_name"] == "rl_grading_end"

    def test_event_name_field_populated(self, tmp_path, system_logger):
        """record_span stores description in event_name field."""
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)
        with record_span("trainer_time/forward_backward_s", EventType.FWD_BWD):
            pass
        jsonl_path = os.path.join(
            str(tmp_path), "system_logs", "trainer_rank_0_system.jsonl"
        )
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        # event_name should be the description, not the EventType
        assert lines[0]["normal"]["event_name"] == "trainer_time/forward_backward_s"
        assert lines[1]["normal"]["event_name"] == "trainer_time/forward_backward_s"


class TestChromeTrace:
    """Tests for generate_gantt_trace in analysis.py."""

    def test_same_event_type_different_descriptions(self, tmp_path, system_logger):
        """Two sequential spans with the same EventType must both appear
        in the chrome trace (no key collision). Display name shows
        EventType when provided, description otherwise.

        This is a regression test: previously, pending_starts used
        EventType as the key, so the second span's start overwrote the
        first, losing one span from the output.
        """
        init_observability(rank=0, source="controller", output_dir=str(tmp_path))
        set_step(1)
        with record_span("rl_time/training_s", EventType.FWD_BWD):
            pass
        with record_span("rl_time/rollouts_to_train_batch_s"):
            pass
        with record_span("rl_time/scoring_s", EventType.FWD_BWD):
            pass

        log_dir = os.path.join(str(tmp_path), "system_logs")
        trace_path = os.path.join(str(tmp_path), "trace.json")
        trace = generate_gantt_trace(log_dir, trace_path)

        span_names = [e["name"] for e in trace["traceEvents"] if e.get("ph") == "X"]
        # Spans with EventType show the EventType name; without show description
        assert span_names.count("fwd_bwd") == 2
        assert "rl_time/rollouts_to_train_batch_s" in span_names
        assert len(span_names) == 3

    def test_no_event_type_uses_description_in_trace(self, tmp_path, system_logger):
        """Spans without EventType use description as the trace name."""
        init_observability(rank=0, source="test", output_dir=str(tmp_path))
        set_step(1)
        with record_span("my_custom/span_s"):
            pass

        log_dir = os.path.join(str(tmp_path), "system_logs")
        trace_path = os.path.join(str(tmp_path), "trace.json")
        trace = generate_gantt_trace(log_dir, trace_path)

        span_names = [e["name"] for e in trace["traceEvents"] if e.get("ph") == "X"]
        assert "my_custom/span_s" in span_names
