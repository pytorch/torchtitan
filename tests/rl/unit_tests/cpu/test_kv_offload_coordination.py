# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise skewed KV events across real Gloo ranks without GPUs or a Store."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.rl.generator.vllm import cpu_kv_offload, simple_cpu_offload
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata


def test_cpu_store_waits_for_tp_workers_but_not_other_dp_replicas():
    manager = SimpleCPUOffloadScheduler.__new__(SimpleCPUOffloadScheduler)
    manager._expected_worker_count = 4  # TP=2, DP=2 external-launcher world.
    manager._store_event_pending_counts = {}
    manager._process_store_event = Mock()

    def init(connector, *_):
        connector.scheduler_manager = manager

    with (
        patch.object(
            simple_cpu_offload.VllmSimpleCPUOffloadConnector, "__init__", new=init
        ),
        patch.object(
            simple_cpu_offload,
            "get_tp_group",
            return_value=SimpleNamespace(world_size=2),
        ),
    ):
        connector = simple_cpu_offload.SimpleCPUOffloadConnector(Mock(), Mock(), Mock())
    aggregator = KVOutputAggregator(2)
    for step in range(3):
        outputs = [
            ModelRunnerOutput.with_kv_conn_output_only(
                KVConnectorOutput(
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                        {7: 1}
                        if (rank == 0 and step == 0 or rank == 1 and step == 2)
                        else {}
                    )
                )
            )
            for rank in range(2)
        ]
        connector.update_connector_output(
            aggregator.aggregate(outputs).kv_connector_output
        )
        if step < 2:
            manager._process_store_event.assert_not_called()
    manager._process_store_event.assert_called_once_with(7)


def test_cpu_receive_cleanup_preserves_lru_order_across_local_ids():
    pool = BlockPool(5, True, 16)
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        pool._insert_block_hash(
            block.block_id.to_bytes(8, "little"), block, num_tokens=16
        )
    scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
        cpu_kv_offload.KVOffloadScheduler
    )
    scheduler.requests = {
        local_id: SimpleNamespace(
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
            kv_transfer_params={"shared_request_id": shared_request_id},
        )
        for local_id, shared_request_id in (("local-z", "first"), ("local-a", "second"))
    }
    scheduler.finished_recving_kv_req_ids = set()
    assigned = dict(zip(scheduler.requests, blocks))

    def complete(output):
        for local_id in output.finished_recving or ():
            pool.free_blocks([assigned[local_id]])

    scheduler.connector = Mock(update_connector_output=Mock(side_effect=complete))
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={"local-a", "local-z"})
    )
    # Unallocated blocks precede the two released blocks; their order must be
    # shared request-ID order even though the local ID order is reversed.
    assert pool.get_new_blocks(4)[-2:] == blocks


def _coordinate_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        ranks=[0, 1],
        cpu_group=dist.group.WORLD,
    )
    try:
        with (patch.object(cpu_kv_offload, "get_tp_group", return_value=group),):
            # One shared work flag covers requests, pending KV and queued outputs.
            # A locally idle rank must keep stepping until every rank has drained.
            engine = VLLMEngine.__new__(VLLMEngine)
            engine._session_kv_holder = None
            engine._coordinate_kv_offload = True
            engine._uses_kv_offload = True
            engine._loaded_policy_version = 0
            engine._reset_kv_cache_on_model_state_dict_load = False
            model = SimpleNamespace(load_model_state_dict=Mock())
            fetch = Mock()
            engine._get_model = lambda: model
            state = SimpleNamespace(requests=False, scheduled=False)
            core = SimpleNamespace(
                scheduler=SimpleNamespace(has_requests=lambda: state.scheduled),
                batch_queue=None,
            )
            engine._engine = SimpleNamespace(
                output_processor=SimpleNamespace(
                    has_unfinished_requests=lambda: state.requests
                ),
                engine_core=SimpleNamespace(engine_core=core),
            )
            with patch(
                "torchtitan.rl.generator.vllm.engine.get_world_group",
                return_value=group,
            ):
                for source in ("requests", "scheduler", "queue", "idle"):
                    state.requests = rank == 1 and source == "requests"
                    state.scheduled = rank == 1 and source == "scheduler"
                    core.batch_queue = (
                        [object()] if rank == 1 and source == "queue" else None
                    )
                    assert engine.has_unfinished_requests() == (source != "idle")

                    model.load_model_state_dict.reset_mock()
                    if source == "requests":
                        # Only rank 1 is active, but both must reject before
                        # changing weights; the work flag alone is insufficient.
                        with pytest.raises(RuntimeError, match="hot_swap=False"):
                            engine.load_model_state_dict(fetch, policy_version=1)
                        model.load_model_state_dict.assert_not_called()
                        assert engine._loaded_policy_version == 0
                    else:
                        # Saving old, versioned KV does not block a drained load.
                        engine.load_model_state_dict(fetch, policy_version=1)
                        model.load_model_state_dict.assert_called_once_with(fetch)
                        assert engine._loaded_policy_version == 1

                    # Single-rank generators use the same combined local condition.
                    engine._coordinate_kv_offload = False
                    assert engine.has_unfinished_requests() == (
                        rank == 1 and source != "idle"
                    )
                    engine._coordinate_kv_offload = True

            # Use upstream completion accounting: the first save completion must not
            # release references, and a failed receive must wait for the other
            # worker before the scheduler can decide to retry/recompute.
            pool = BlockPool(8, False, 16)
            blocks = pool.get_new_blocks(2)
            store_scheduler = SimpleCPUOffloadScheduler.__new__(
                SimpleCPUOffloadScheduler
            )
            store_scheduler._expected_worker_count = 2
            store_scheduler._store_event_pending_counts = {}
            store_scheduler._cleanup_load_request = Mock()
            store_scheduler._process_store_event = Mock(
                side_effect=lambda event: pool.free_blocks(blocks)
            )
            scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
                cpu_kv_offload.KVOffloadScheduler
            )
            scheduler._kv_output_aggregator = KVOutputAggregator(2)
            local_ids = {
                shared_request_id: f"{shared_request_id}-rank{rank}"
                for shared_request_id in ("load", "failed", "send")
            }
            scheduler.requests = {
                local_id: SimpleNamespace(
                    kv_transfer_params={"shared_request_id": shared_request_id}
                )
                for shared_request_id, local_id in local_ids.items()
            }
            scheduler.shared_to_local_request_ids = local_ids.copy()
            observed = []

            def apply_output(_self, _scheduled, output):
                kv = output.kv_connector_output
                store_scheduler.update_connector_output(kv)
                observed.append(kv)
                # Sampling output stays local; only control metadata is gathered.
                assert output.sampled_token_ids == [[rank]]
                assert output.req_ids == [local_ids["load"]]
                return {}

            with patch.object(
                cpu_kv_offload.Scheduler, "update_from_output", new=apply_output
            ):
                for step in range(3):
                    report = rank == 0 and step == 0 or rank == 1 and step == 2
                    kv = KVConnectorOutput(
                        finished_recving=(
                            {local_ids["load"], local_ids["failed"]} if report else None
                        ),
                        finished_sending={local_ids["send"]} if report else None,
                        failed_recving=(
                            {local_ids["failed"]} if rank == 0 and step == 0 else set()
                        ),
                        kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                            {7: 1} if report else {}
                        ),
                    )
                    output = ModelRunnerOutput(
                        req_ids=[local_ids["load"]],
                        req_id_to_index={local_ids["load"]: 0},
                        sampled_token_ids=[[rank]],
                        kv_connector_output=kv,
                    )
                    scheduler.update_from_output(None, output)
                    if step < 2:
                        assert not observed[-1].finished_recving
                        assert not observed[-1].finished_sending
                        assert not observed[-1].failed_recving
                        assert all(b.ref_cnt == 1 for b in blocks)
                        store_scheduler._process_store_event.assert_not_called()
                    else:
                        assert observed[-1].finished_recving == {
                            local_ids["load"],
                            local_ids["failed"],
                        }
                        assert observed[-1].finished_sending == {local_ids["send"]}
                        assert observed[-1].failed_recving == {local_ids["failed"]}
                        assert all(b.ref_cnt == 0 for b in blocks)
                        store_scheduler._process_store_event.assert_called_once_with(7)
    finally:
        dist.destroy_process_group()


def test_progress_and_delayed_completion_coordination(tmp_path):
    mp.spawn(_coordinate_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2)


def test_idle_dp_forward_does_not_skip_queued_output_drain():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._coordinate_kv_offload = True
    engine._session_kv_holder = None
    calls = []

    def coordinate(has_requests):
        assert not has_requests
        engine._engine.should_execute_dummy_batch = True
        calls.append("coordinate")

    def drain():
        assert not engine._engine.should_execute_dummy_batch
        calls.append("drain")
        return []

    engine._engine = SimpleNamespace(
        dp_group=object(),
        has_unfinished_requests_dp=coordinate,
        engine_core=SimpleNamespace(
            engine_core=SimpleNamespace(
                scheduler=SimpleNamespace(has_requests=lambda: False),
                batch_queue=[object()],
            ),
            execute_dummy_batch=lambda: calls.append("dummy"),
        ),
        step=drain,
    )
    assert engine.step() == []
    assert calls == ["coordinate", "dummy", "drain"]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_scheduler_uses_resolved_async_setting(async_scheduling):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)
    )
    with (
        patch.object(cpu_kv_offload, "KVOffloadScheduler") as sync,
        patch.object(cpu_kv_offload, "AsyncKVOffloadScheduler") as async_,
    ):
        result = cpu_kv_offload.create_kv_offload_scheduler(config)
        selected, other = (async_, sync) if async_scheduling else (sync, async_)
        assert result is selected.return_value
        selected.assert_called_once_with(config)
        other.assert_not_called()


@pytest.mark.parametrize("suffix", ["00000000", "ffffffff"])
def test_priority_ties_use_shared_request_ids_across_waiting_queues(suffix):
    # Native vLLM orders these IDs differently depending on the random suffix.
    # Every TP scheduler must select the same shared request.
    scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
        cpu_kv_offload.KVOffloadScheduler
    )
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.shared_to_local_request_ids = {}
    scheduler.requests = {}
    scheduler.connector = None
    scheduler.log_stats = False
    scheduler.spec_decode_metrics_level = "none"
    scheduler.waiting = cpu_kv_offload._SharedIDRequestQueue()
    scheduler.skipped_waiting = cpu_kv_offload._SharedIDRequestQueue()
    requests = []
    for shared_request_id in ("request-5", "request"):
        request = Request(
            f"{shared_request_id}-{suffix}",
            [1],
            SamplingParams(
                max_tokens=1,
                extra_args={
                    "kv_transfer_params": {"shared_request_id": shared_request_id}
                },
            ),
            None,
            arrival_time=1.0,
        )
        scheduler.add_request(request)
        requests.append(request)
    earlier = requests[1]
    assert scheduler.waiting.pop_request() is earlier
    # The same tie-break must hold between the ready and blocked queue heads.
    earlier.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.add_request(earlier)
    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.skipped_waiting
    scheduler.waiting.prepend_requests(scheduler.skipped_waiting)
    scheduler.skipped_waiting.remove_request(earlier)
    assert list(scheduler.waiting) == [earlier, requests[0]]
    scheduler.waiting.remove_requests([earlier])
    scheduler.waiting.prepend_request(earlier)
    assert scheduler.waiting.pop_request() is earlier
    assert scheduler.waiting.pop_request() is requests[0]
    assert not scheduler.waiting


@pytest.mark.parametrize("late_receive", [False, True])
@pytest.mark.parametrize(
    "finished_status", [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR]
)
def test_terminal_cleanup_preserves_block_identity(late_receive, finished_status):
    # Use real queues and KV allocation/freeing without constructing a GPU worker.
    # Per-process string hash order must not change a later invalid block's owner.
    scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
        cpu_kv_offload.KVOffloadScheduler
    )
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.defer_block_free = False
    scheduler._inflight_prefills = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.requests = {}
    scheduler.shared_to_local_request_ids = {}
    scheduler.spec_decode_metrics_level = "none"
    scheduler.log_stats = False
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.encoder_cache_manager = EncoderCacheManager(0)
    scheduler.block_size = 16
    manager = scheduler.kv_cache_manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=5,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
                    ),
                )
            ],
        ),
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=False,
    )

    def allocate(request_id, shared_request_id):
        request = Request(
            request_id,
            [1] * 32,
            SamplingParams(
                max_tokens=1,
                extra_args={
                    "kv_transfer_params": {"shared_request_id": shared_request_id}
                },
            ),
            None,
        )
        scheduler.add_request(request)
        assert manager.allocate_slots(request, num_new_tokens=32) is not None
        request.num_computed_tokens = 32
        return request

    # Deliberately reverse local-ID ordering: block release must use shared request IDs.
    for request_id, shared_request_id in (
        ("local-B", "request-A"),
        ("local-A", "request-B"),
    ):
        request = allocate(request_id, shared_request_id)
        scheduler.waiting.remove_request(request)
        if late_receive:
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            scheduler.skipped_waiting.add_request(request)
        else:
            request.status = RequestStatus.RUNNING
            scheduler.running.append(request)
    original_b_blocks = manager.get_block_ids("local-A")[0].copy()
    scheduler.finish_requests({"local-A", "local-B", "unknown"}, finished_status)
    if late_receive:
        assert manager.block_pool.get_num_free_blocks() == 0
        assert len(scheduler.shared_to_local_request_ids) == 2
        # The shared ID remains reserved until the cancelled receive completes.
        with pytest.raises(AssertionError):
            allocate("too-early", "request-A")
        scheduler._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={"local-A", "local-B"})
        )
    assert not scheduler.requests
    assert not scheduler.shared_to_local_request_ids
    assert (
        not scheduler.running
        and not scheduler.waiting
        and not scheduler.skipped_waiting
    )

    # A shared request ID can be reused after its old request/receive is fully retired.
    following = [allocate("next-C", "request-A"), allocate("next-D", "request-B")]
    # Non-cached blocks are reused LIFO: canonical A-then-B cleanup puts B first.
    assert manager.get_block_ids("next-C")[0] == original_b_blocks[::-1]
    affected, _, _ = scheduler._update_requests_with_invalid_blocks(
        following, {original_b_blocks[-1]}, {}, evict_blocks=False
    )
    assert affected == {"next-C"}
