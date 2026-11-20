# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check Titan's Mooncake TP lookup sharing and no-forward save submission
without GPUs or a Store."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.rl.generator.vllm import mooncake
from torchtitan.rl.generator.vllm.worker import TorchTitanGPUModelRunner
from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreWorker,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker import kv_connector_model_runner_mixin


def test_no_forward_submits_finish_time_save_before_polling():
    # A finish-time save scheduled on a step without a forward must be submitted
    # once, before get_finished retires the request's save ledger.
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

    def get_transfer_results(finished_req_ids):
        worker.kv_send_thread.add_request.assert_called_once_with(tail)
        return SimpleNamespace(
            finished_sending=set(), finished_recving={"loaded"}, failed_recving=set()
        )

    connector.get_transfer_results.side_effect = get_transfer_results
    connector.get_block_ids_with_load_errors.return_value = set()
    connector.get_kv_connector_stats.return_value = None
    connector.get_kv_connector_kv_cache_events.return_value = None
    connector.build_connector_worker_meta.return_value = None
    metadata = MooncakeStoreConnectorMetadata(set(), set())
    tail = ReqMeta(
        req_id="finished",
        token_len_chunk=0,
        block_ids=([3],),
        block_hashes=[],
        can_save=True,
        store_job_id=7,
        boundary_state_offloads=[(0, 3, 24)],
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
        event.return_value.record.assert_called_once()
        assert tail.current_event is event.return_value
        assert first.kv_connector_output.finished_recving == {"loaded"}

        # Later empty steps only poll; they must not resubmit the old metadata.
        output.kv_connector_metadata = MooncakeStoreConnectorMetadata(set(), set())
        second = TorchTitanGPUModelRunner.kv_connector_no_forward(output, config)
        assert second.kv_connector_output.finished_recving == {"loaded"}
        assert connector.clear_connector_metadata.call_count == 2


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
        with patch.object(mooncake, "get_tp_group", return_value=group):
            # A follower must never query independently. Pending and hit results
            # are both shared, including the load metadata consumed by allocation.
            connector = mooncake.MooncakeStoreConnector.__new__(
                mooncake.MooncakeStoreConnector
            )
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

            # A later miss (e.g. the Store evicted the prefix) after the leader
            # dropped its spec must drop the follower's copy too.
            def miss(*args):
                connector.connector_scheduler.load_specs.pop(request.request_id)
                return 0, False

            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                side_effect=miss,
            ):
                assert connector.get_num_new_matched_tokens(request, 16) == (0, False)
                assert (
                    request.request_id not in connector.connector_scheduler.load_specs
                )
    finally:
        dist.destroy_process_group()


def test_lookup_result_broadcast(tmp_path):
    mp.spawn(_lookup_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2)
