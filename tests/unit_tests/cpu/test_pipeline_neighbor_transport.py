# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The opt-in pipeline transport: edge groups mapped to logical stages, tensor
P2P on the two-rank groups with the peer index, metadata on the CPU group, and
the collective mode vote."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch

from torchtitan.distributed import pipeline_parallel as pp


class TestPipelineNeighborTransport(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(
                pp._create_pipeline_transport_groups(
                    Mock(), Mock(), num_stages=2, pp_schedule="1F1B"
                )
            )

    def _groups(self, pp_ranks, my_pp_rank, num_stages, schedule):
        pairs = {(a, b): Mock(name=f"g{a}_{b}") for a, b in zip(pp_ranks, pp_ranks[1:])}
        if len(pp_ranks) > 2:
            pairs[(pp_ranks[0], pp_ranks[-1])] = Mock(name="g_wrap")
        parallel_dims = Mock()
        parallel_dims.get_pipeline_neighbor_groups.return_value = (Mock(name="meta"), pairs)
        with (
            patch.dict(os.environ, {pp.PIPELINE_NEIGHBOR_P2P_ENV: "1"}, clear=True),
            patch.object(pp.dist, "get_process_group_ranks", return_value=list(pp_ranks)),
            patch.object(pp.dist, "get_backend", return_value="nccl"),
            patch.object(pp.dist, "get_rank", return_value=my_pp_rank),
        ):
            return pp._create_pipeline_transport_groups(
                parallel_dims, Mock(), num_stages=num_stages, pp_schedule=schedule
            ), pairs

    def test_1f1b_edges_and_peer_indices(self):
        # PP ranks live on global ranks 3, 4, 5; this rank is pp rank 1 (global 4).
        transport, pairs = self._groups((3, 4, 5), 1, 3, "1F1B")
        assert transport is not None
        self.assertEqual(set(transport.edge_groups), {0, 1})
        self.assertIs(transport.edge_groups[0], pairs[(3, 4)])
        self.assertIs(transport.edge_groups[1], pairs[(4, 5)])
        # In group (3, 4) the peer 3 has index 0; in group (4, 5) the peer 5 has index 1.
        self.assertEqual(transport.edge_peers, {0: 0, 1: 1})

    def test_looped_schedule_wrap_edge(self):
        # Interleaved1F1B, 2 ranks x 2 virtual stages: stage 1 (rank 1) -> stage 2 (rank 0).
        transport, pairs = self._groups((0, 1), 0, 4, "Interleaved1F1B")
        assert transport is not None
        # Rank 0 holds stages 0 and 2, so it sits on edges 0 (0->1), 1 (1->2,
        # the wrap back to rank 0) and 2 (2->3); every peer is rank 1, index 1.
        self.assertEqual(set(transport.edge_groups), {0, 1, 2})
        self.assertEqual(transport.edge_peers, {0: 1, 1: 1, 2: 1})

    def test_forward_send_uses_the_edge_group_and_peer(self):
        stage_cls = type("S", (pp._NeighborP2PTransportMixin,), {})
        stage = stage_cls.__new__(stage_cls)
        stage.is_last = False
        stage.stage_index = 1
        stage.fwd_cache = {0: ((torch.zeros(2),), None)}
        stage.act_send_info = {0: [2]}
        stage._is_same_rank = lambda s: False
        stage._transport = pp._PipelineTransportGroups(Mock(), {1: "edge1"}, {1: 0})
        with patch.object(pp.dist, "P2POp", side_effect=lambda *a, **k: (a, k)):
            ops = stage.get_fwd_send_ops(0)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0][1], {"group_peer": 0, "group": "edge1"})

    def test_metadata_travels_on_the_cpu_group(self):
        meta_group = Mock(name="meta")
        stage = SimpleNamespace(
            _transport=pp._PipelineTransportGroups(meta_group, {}, {}),
            _resolve_peer_global_rank=lambda s: 7,
        )
        with patch.object(pp.dist, "send_object_list") as send:
            pp._NeighborP2PTransportMixin._send_meta(stage, {"k": 1}, 2)
        send.assert_called_once_with(
            [{"k": 1}], dst=7, group=meta_group, device=torch.device("cpu"), use_batch=False
        )

    def test_schedule_votes_with_one_collective(self):
        stage_cls = type("S", (pp._NeighborP2PTransportMixin,), {})
        stage = stage_cls.__new__(stage_cls)
        stage._user_meta = Mock()
        stage.device = torch.device("cpu")
        stage.group = Mock(name="pp_group")
        schedule = SimpleNamespace(_warmup_p2p=Mock(name="stock"))
        with (
            patch.object(pp.InferenceMode, "needs_dynamic", return_value=False),
            patch.object(pp.dist, "all_reduce") as all_reduce,
        ):
            pp._configure_neighbor_p2p_schedule(schedule)
            schedule._warmup_p2p([stage], True, False)
        all_reduce.assert_called_once_with(ANY, op=pp.dist.ReduceOp.MIN, group=stage.group)
        self.assertEqual(stage._inference_mode, pp.InferenceMode.STATIC)


if __name__ == "__main__":
    unittest.main()
