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
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.rl.generator.vllm import mooncake
from torchtitan.rl.generator.vllm.worker import TorchTitanGPUModelRunner
from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreWorker,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker import kv_connector_model_runner_mixin


def test_no_forward_submits_tail_once_and_polls_delayed_completion():
    # Exercise real Mooncake submission/release code without CUDA or a Store.
    # A finish-time job must be submitted even though no model forward runs.
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeStoreConnector", kv_role="kv_both"
    )
    worker = MooncakeStoreWorker.__new__(MooncakeStoreWorker)
    worker._capacity_only = False
    worker.can_put = True
    worker.kv_send_thread = Mock()
    connector = Mock(spec=mooncake.VllmMooncakeStoreConnector)
    connector.wait_for_save.side_effect = lambda: worker.wait_for_save(
        connector.bind_connector_metadata.call_args.args[0]
    )
    connector.get_transfer_results.return_value = SimpleNamespace(
        finished_sending=set(), finished_recving={"loaded"}, failed_recving=set()
    )
    connector.get_block_ids_with_load_errors.return_value = set()
    connector.get_kv_connector_stats.return_value = None
    connector.get_kv_connector_kv_cache_events.return_value = None
    connector.build_connector_worker_meta.side_effect = [
        MooncakeStoreWorkerMetadata(),
        MooncakeStoreWorkerMetadata(completed_saves={7: 1}),
        MooncakeStoreWorkerMetadata(completed_saves={7: 1}),
    ]
    pool = BlockPool(4, enable_caching=False, hash_block_size=16)
    blocks = pool.get_new_blocks(1)
    store = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
    store._gpu_block_pool = pool
    store._pinned_saves = {7: ([blocks[0].block_id], 2)}
    metadata = MooncakeStoreConnectorMetadata(set(), set())
    tail = ReqMeta(
        req_id="finished",
        token_len_chunk=0,
        block_ids=([blocks[0].block_id],),
        block_hashes=[],
        can_save=True,
        store_job_id=7,
        boundary_state_offloads=[(0, blocks[0].block_id, 24)],
    )
    metadata.add_request(tail)
    output = SchedulerOutput.make_empty()
    output.kv_connector_metadata = metadata
    with (
        patch.object(
            kv_connector_model_runner_mixin,
            "get_kv_transfer_group",
            return_value=connector,
        ),
        patch.object(
            kv_connector_model_runner_mixin, "has_kv_transfer_group", return_value=True
        ),
        patch("torch.cuda.Event") as event,
    ):
        first = TorchTitanGPUModelRunner.kv_connector_no_forward(output, config)
        worker.kv_send_thread.add_request.assert_called_once_with(tail)
        event.return_value.record.assert_called_once()
        assert tail.current_event is event.return_value
        assert first.kv_connector_output.finished_recving == {"loaded"}
        store.update_connector_output(first.kv_connector_output)
        assert blocks[0].ref_cnt == 1 and store.has_pending_push_work()

        # Later empty steps only poll; they must not resubmit the old metadata.
        output.kv_connector_metadata = MooncakeStoreConnectorMetadata(set(), set())
        second = TorchTitanGPUModelRunner.kv_connector_no_forward(output, config)
        worker.kv_send_thread.add_request.assert_called_once_with(tail)
        store.update_connector_output(second.kv_connector_output)
        assert blocks[0].ref_cnt == 1 and store.has_pending_push_work()

        # A second TP worker reports on a later step; only then release the save.
        third = TorchTitanGPUModelRunner.kv_connector_no_forward(output, config)
        store.update_connector_output(third.kv_connector_output)
        assert blocks[0].ref_cnt == 0 and not store.has_pending_push_work()
        worker.kv_send_thread.add_request.assert_called_once_with(tail)
        assert connector.clear_connector_metadata.call_count == 3


def _lookup_worker(rank, rendezvous):
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
        with (patch.object(mooncake, "get_tp_group", return_value=group),):
            # A follower must never query independently. Pending and hit results
            # are both shared, including the load metadata consumed by allocation.
            connector = mooncake.MooncakeStoreConnector.__new__(
                mooncake.MooncakeStoreConnector
            )
            connector.connector_worker = None
            spec = LoadSpec(16, 48, False)
            connector.connector_scheduler = SimpleNamespace(load_specs={})
            request = SimpleNamespace(request_id=f"lookup-rank{rank}")

            def lookup(*args):
                assert rank == 0
                connector.connector_scheduler.load_specs[request.request_id] = spec
                return 32, True

            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                side_effect=lookup,
            ):
                assert connector.get_num_new_matched_tokens(request, 16) == (32, True)
                assert (
                    connector.connector_scheduler.load_specs[request.request_id] == spec
                )
            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                return_value=(None, False),
            ) as pending:
                assert connector.get_num_new_matched_tokens(request, 16) == (
                    None,
                    False,
                )
                assert pending.call_count == (1 if rank == 0 else 0)
            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                side_effect=RuntimeError("lookup unavailable"),
            ):
                with pytest.raises(RuntimeError, match="lookup unavailable"):
                    connector.get_num_new_matched_tokens(request, 16)

    finally:
        dist.destroy_process_group()


def test_lookup_result_and_failure_broadcast(tmp_path):
    mp.spawn(_lookup_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2)


@pytest.mark.parametrize("rank", [0, 1])
def test_each_dp_replica_has_own_lookup_server_and_save_quorum(rank):
    worker = SimpleNamespace(lookup_server=None)
    scheduler = SimpleNamespace(_num_workers=4)

    def init(connector, *args):
        connector.connector_worker = worker
        connector.connector_scheduler = scheduler

    group = SimpleNamespace(rank_in_group=rank, world_size=2)
    with (
        patch.object(mooncake.VllmMooncakeStoreConnector, "__init__", new=init),
        patch.object(mooncake, "get_tp_group", return_value=group),
        patch.object(mooncake, "LookupKeyServer") as server,
    ):
        connector = mooncake.MooncakeStoreConnector(Mock(), Mock())
        assert connector.connector_scheduler._num_workers == 2
        assert server.call_count == (1 if rank == 0 else 0)
        # Prevent the upstream destructor from calling close on this fake worker.
        connector.connector_worker = None
