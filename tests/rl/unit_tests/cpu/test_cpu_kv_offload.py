# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPUKVOffload: its config, and `attach` with several vLLM workers (a scripted TP all-gather)."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch.distributed as dist

import torchtitan.rl.generator.vllm.cpu_kv_offload as cpu_kv_offload_module
import vllm.distributed.parallel_state as vllm_parallel_state
from torchtitan.rl.generator.vllm.cpu_kv_offload import CPUKVOffload
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput


def test_cpu_kv_offload_rejects_non_positive_sizes():
    with pytest.raises(ValueError, match="size_gib must be > 0, got 0"):
        CPUKVOffload.Config(size_gib=0)


class _ConnectorConfig(NamedTuple):
    num_workers: int


def _vllm_assign_request_id(request) -> None:
    """Stands in for vLLM's InputProcessor.assign_request_id."""


def _multi_worker_engine(
    monkeypatch,
    *,
    tp_size: int,
    peer_entry: Any = None,
    hash_randomization: int = 0,
    input_processor: Any = None,
    cpu_cache_policy: Any = None,
):
    """A fake engine set up by `CPUKVOffload.attach` as TP rank 0 of DP=2 x TP.

    `peer_entry(own_entry)` returns TP rank 1's part of each all-gather. Returns the engine, what
    the scheduler's `update_from_output` received, and what this rank contributed to each gather.
    """
    received, gathers = [], []
    connector_scheduler = SimpleNamespace(
        config=_ConnectorConfig(num_workers=2 * tp_size),
        _job_counter=0,
        manager=SimpleNamespace(
            _policy=cpu_cache_policy or LRUCachePolicy(cache_capacity=8)
        ),
    )
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=connector_scheduler),
        kv_cache_manager=SimpleNamespace(num_kv_cache_groups=1),
        requests={},
        update_from_output=lambda scheduler_output, output: received.append(output),
    )
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler)),
        input_processor=input_processor
        or SimpleNamespace(assign_request_id=_vllm_assign_request_id),
    )
    tp_group = SimpleNamespace(world_size=tp_size, rank_in_group=0, cpu_group=None)
    monkeypatch.setattr(vllm_parallel_state, "get_tp_group", lambda: tp_group)
    flags = SimpleNamespace(hash_randomization=hash_randomization)
    monkeypatch.setattr(cpu_kv_offload_module, "sys", SimpleNamespace(flags=flags))

    def all_gather_object(gathered, own_entry, group):
        gathers.append(own_entry)
        gathered[0], gathered[1] = own_entry, peer_entry(own_entry)

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    CPUKVOffload.Config(size_gib=1).build().attach(engine)
    return engine, received, gathers


def _finished(job_id: int, request_id: str | None = None) -> KVConnectorOutput:
    """One worker's finished transfers in a step: a load (of `request_id`) or a store."""
    meta = OffloadingWorkerMetadata()
    meta.mark_completed(job_id)
    return KVConnectorOutput(
        kv_connector_worker_meta=meta,
        finished_recving={request_id} if request_id else None,
    )


def test_multi_worker_kv_offload_finishes_a_load_once_every_tp_rank_did(monkeypatch):
    peer_outputs = iter([None, _finished(7, "r-0")])  # TP rank 1 finishes a step later
    engine, received, _ = _multi_worker_engine(
        monkeypatch, tp_size=2, peer_entry=lambda own: (own[0], next(peer_outputs))
    )
    scheduler = engine.engine_core.engine_core.scheduler
    step = SimpleNamespace(num_scheduled_tokens={"r-0": 8})

    own_output = replace(
        EMPTY_MODEL_RUNNER_OUTPUT, kv_connector_output=_finished(7, "r-0")
    )
    scheduler.update_from_output(step, own_output)
    scheduler.update_from_output(step, EMPTY_MODEL_RUNNER_OUTPUT)

    # vLLM's connector adds up these counts and finishes job 7 at 2 of 2.
    first, second = (output.kv_connector_output for output in received)
    assert first.kv_connector_worker_meta.completed_jobs == {7: 1}
    assert not first.finished_recving
    assert second.kv_connector_worker_meta.completed_jobs == {7: 1}
    assert second.finished_recving == {"r-0"}
    # The second step's own output was vLLM's shared EMPTY_MODEL_RUNNER_OUTPUT.
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None
    # 4 (DP=2 x TP=2) before: each scheduler waits for its own TP group only.
    assert scheduler.connector.connector_scheduler.config.num_workers == 2


def test_multi_worker_kv_offload_raises_when_tp_ranks_diverge(monkeypatch):
    peer = {"entry": None}
    engine, _, gathers = _multi_worker_engine(
        monkeypatch, tp_size=2, peer_entry=lambda own: peer["entry"] or own
    )
    update_from_output = engine.engine_core.engine_core.scheduler.update_from_output
    # TP rank 1 schedules another request with the same token count.
    update_from_output(
        SimpleNamespace(num_scheduled_tokens={"r-1": 8}), EMPTY_MODEL_RUNNER_OUTPUT
    )
    peer["entry"] = gathers[-1]

    with pytest.raises(RuntimeError, match="schedulers diverged"):
        update_from_output(
            SimpleNamespace(num_scheduled_tokens={"r-0": 8}), EMPTY_MODEL_RUNNER_OUTPUT
        )


def test_multi_worker_kv_offload_gives_every_tp_rank_the_same_request_ids(monkeypatch):
    internal_ids = []
    for _ in range(2):  # two TP ranks admit the same requests in the same order
        engine, _, _ = _multi_worker_engine(monkeypatch, tp_size=2)
        requests = [SimpleNamespace(request_id=request_id) for request_id in "aba"]
        for request in requests:
            engine.input_processor.assign_request_id(request)
        internal_ids.append(
            [(request.request_id, request.external_req_id) for request in requests]
        )
    assert (
        internal_ids[0] == internal_ids[1] == [("a-0", "a"), ("b-1", "b"), ("a-2", "a")]
    )


def test_multi_worker_kv_offload_with_tp1_only_sets_the_expected_count(monkeypatch):
    engine, _, _ = _multi_worker_engine(monkeypatch, tp_size=1)
    connector_scheduler = (
        engine.engine_core.engine_core.scheduler.connector.connector_scheduler
    )
    assert connector_scheduler.config.num_workers == 1  # 2 (DP=2) before
    assert engine.input_processor.assign_request_id is _vllm_assign_request_id


def test_multi_worker_kv_offload_needs_pythonhashseed_0(monkeypatch):
    with pytest.raises(ValueError, match="needs PYTHONHASHSEED=0"):
        _multi_worker_engine(monkeypatch, tp_size=2, hash_randomization=1)


def test_multi_worker_kv_offload_needs_vllms_assign_request_id(monkeypatch):
    """A vLLM that renamed it would keep random ids, and the first TP reload would hang."""
    with pytest.raises(RuntimeError, match="no assign_request_id to replace"):
        _multi_worker_engine(monkeypatch, tp_size=2, input_processor=SimpleNamespace())


def test_attach_needs_vllms_lru_policy(monkeypatch):
    """`evict_first` reorders the LRU's private `evictable_chunks`."""
    with pytest.raises(
        RuntimeError, match="needs vLLM's LRUCachePolicy, got ARCCachePolicy"
    ):
        _multi_worker_engine(
            monkeypatch, tp_size=1, cpu_cache_policy=ARCCachePolicy(cache_capacity=8)
        )
