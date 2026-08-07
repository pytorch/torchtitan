# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.observability.vllm_stat_common import (
    build_stat_identity,
    extract_step_stats,
    StatLoggerContext,
    StepStats,
    VllmStatLoggerBase,
)


def _vllm_config(*, tp_size=1, dp_rank=0, model="test-model"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_rank=dp_rank
        ),
        model_config=SimpleNamespace(model=model),
    )


def _context(rank=0, tp_rank=0, dp_rank=0, generator_name="gen", output_dir="/tmp"):
    return StatLoggerContext(
        rank=rank,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        generator_name=generator_name,
        output_dir=output_dir,
    )


def test_build_stat_identity_tp_rank0_gate_with_context():
    tp_rank0 = build_stat_identity(
        _vllm_config(), 0, context=_context(rank=0, tp_rank=0, dp_rank=0)
    )
    assert tp_rank0.should_log is True
    off_head = build_stat_identity(
        _vllm_config(), 0, context=_context(rank=1, tp_rank=1, dp_rank=0)
    )
    assert off_head.should_log is False


def test_build_stat_identity_context_populates_attributes():
    ident = build_stat_identity(
        _vllm_config(model="m"),
        3,
        context=_context(rank=5, tp_rank=0, dp_rank=2, generator_name="gen-a"),
    )
    attrs = ident.attributes
    assert attrs["rank"] == 5
    assert attrs["tp_rank"] == 0
    assert attrs["dp_rank"] == 2
    assert attrs["engine_index"] == 3
    assert attrs["model_name"] == "m"
    assert attrs["generator_name"] == "gen-a"


def test_build_stat_identity_reads_env_tags(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    ident = build_stat_identity(
        _vllm_config(), 0, context=_context(rank=3, tp_rank=1, dp_rank=1)
    )
    assert ident.attributes["local_rank"] == 1
    assert ident.attributes["world_size"] == 8


def test_build_stat_identity_extra_attributes_merged_last():
    ident = build_stat_identity(
        _vllm_config(model="orig"),
        0,
        context=_context(rank=0, tp_rank=0, dp_rank=0),
        extra_attributes={"model_name": "override", "job": "j1"},
    )
    assert ident.attributes["model_name"] == "override"  # extra wins on conflict
    assert ident.attributes["job"] == "j1"


def test_extract_step_stats_both_none():
    step = extract_step_stats(None, None)
    assert step.kv_cache_usage is None
    assert step.num_running_reqs is None
    assert step.num_waiting_reqs is None
    assert step.prefix_queries == 0
    assert step.prefix_hits == 0
    assert step.num_generation_tokens == 0
    assert step.ttft_ms == []
    assert step.itl_ms == []
    assert step.finished == []


def test_extract_step_stats_populated_converts_seconds_to_ms():
    scheduler = SimpleNamespace(
        kv_cache_usage=0.5,
        num_running_reqs=4,
        num_waiting_reqs=2,
        prefix_cache_stats=SimpleNamespace(queries=10, hits=7),
    )
    iteration = SimpleNamespace(
        num_generation_tokens=12,
        prompt_token_stats=SimpleNamespace(total=100, cached_tokens=30),
        num_preempted_reqs=1,
        time_to_first_tokens_iter=[0.012],
        inter_token_latencies_iter=[0.01, 0.02],
        finished_requests=[
            SimpleNamespace(decode_time=0.03, queued_time=0.005, e2e_latency=0.05)
        ],
    )
    step = extract_step_stats(scheduler, iteration)
    assert step.kv_cache_usage == 0.5
    assert step.num_running_reqs == 4
    assert step.num_waiting_reqs == 2
    assert step.prefix_queries == 10
    assert step.prefix_hits == 7
    assert step.num_generation_tokens == 12
    assert step.num_prompt_tokens == 100
    assert step.num_cached_prompt_tokens == 30
    assert step.num_preempted_reqs == 1
    assert step.ttft_ms == pytest.approx([12.0])
    assert step.itl_ms == pytest.approx([10.0, 20.0])
    assert len(step.finished) == 1
    assert step.finished[0].decode_time_ms == pytest.approx(30.0)
    assert step.finished[0].queue_time_ms == pytest.approx(5.0)
    assert step.finished[0].e2e_latency_ms == pytest.approx(50.0)


def test_extract_step_stats_prefix_cache_none():
    scheduler = SimpleNamespace(
        kv_cache_usage=0.1,
        num_running_reqs=1,
        num_waiting_reqs=0,
        prefix_cache_stats=None,
    )
    step = extract_step_stats(scheduler, None)
    assert step.prefix_queries == 0
    assert step.prefix_hits == 0
    assert step.kv_cache_usage == 0.1


class _FakeLogger(VllmStatLoggerBase):
    """Minimal subclass exercising the base ``record()`` gate/guard."""

    def __init__(self, vllm_config, *, context, boom=False):
        super().__init__(vllm_config, 0, context=context)
        self.accumulated: list[StepStats] = []
        self._boom = boom

    def _accumulate(self, step):
        if self._boom:
            raise ValueError("simulated vLLM schema drift")
        self.accumulated.append(step)

    def log(self):
        pass

    def log_engine_initialized(self):
        pass


def test_record_noop_when_disabled():
    log = _FakeLogger(
        _vllm_config(), context=_context(rank=0, tp_rank=0, dp_rank=0)
    )
    assert log._enabled is False  # the subclass never flipped it True
    log.record(None, None)
    assert log.accumulated == []


def test_record_dispatches_to_accumulate_when_enabled():
    log = _FakeLogger(
        _vllm_config(), context=_context(rank=0, tp_rank=0, dp_rank=0)
    )
    log._enabled = True
    log.record(None, None)
    assert len(log.accumulated) == 1
    assert isinstance(log.accumulated[0], StepStats)


def test_record_disables_on_accumulate_failure_without_raising():
    log = _FakeLogger(
        _vllm_config(),
        context=_context(rank=0, tp_rank=0, dp_rank=0),
        boom=True,
    )
    log._enabled = True
    # Must not propagate: observability may never crash the engine loop.
    log.record(None, None)
    assert log._enabled is False
    # Once disabled, further calls are silent no-ops.
    log.record(None, None)
    assert log.accumulated == []


def test_record_disables_on_extract_failure_without_raising():
    log = _FakeLogger(
        _vllm_config(), context=_context(rank=0, tp_rank=0, dp_rank=0)
    )
    log._enabled = True
    # A malformed iteration_stats (missing fields) makes extract_step_stats raise;
    # the guard must swallow it and disable the logger.
    bad_iteration = SimpleNamespace()
    log.record(None, bad_iteration)
    assert log._enabled is False
    assert log.accumulated == []
