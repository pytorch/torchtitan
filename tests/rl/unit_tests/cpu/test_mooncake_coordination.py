# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check Titan's Mooncake TP lookup sharing without GPUs or a Store."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.rl.generator.vllm import mooncake
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import LoadSpec


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
