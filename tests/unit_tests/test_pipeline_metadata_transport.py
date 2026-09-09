# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch

from torchtitan.distributed import parallel_dims as parallel_dims_module
from torchtitan.distributed import pipeline_parallel
from torchtitan.distributed.parallel_dims import ParallelDims


class TestNeighborPipelineTransport(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(
                pipeline_parallel._create_pipeline_transport_groups(
                    Mock(), Mock(), num_stages=2, pp_schedule="1F1B"
                )
            )

    def test_adapts_early_pp_edge_groups_to_logical_stages(self):
        parallel_dims = Mock()
        local_metadata_group = Mock()
        local_first_edge = Mock()
        local_second_edge = Mock()
        parallel_dims.get_pipeline_neighbor_groups.return_value = (
            local_metadata_group,
            {0: local_first_edge, 1: local_second_edge},
        )
        with (
            patch.dict(
                os.environ,
                {pipeline_parallel._PIPELINE_NEIGHBOR_P2P_ENV: "1"},
                clear=True,
            ),
            patch.object(
                pipeline_parallel.dist,
                "get_process_group_ranks",
                return_value=[3, 4, 5],
            ),
            patch.object(
                pipeline_parallel.dist, "get_backend", return_value="nccl"
            ),
            patch.object(pipeline_parallel.dist, "get_rank", return_value=1),
            patch.object(
                pipeline_parallel,
                "_get_pp_rank_to_stage_indices_mapping",
                side_effect=[(0,), (1,), (2,)],
            ),
        ):
            transport = pipeline_parallel._create_pipeline_transport_groups(
                parallel_dims, Mock(), num_stages=3, pp_schedule="1F1B"
            )

        assert transport is not None
        self.assertIs(transport.metadata_group, local_metadata_group)
        self.assertEqual(
            transport.edge_groups,
            {0: local_first_edge, 1: local_second_edge},
        )
        parallel_dims.get_pipeline_neighbor_groups.assert_called_once_with((3, 4, 5))

    def test_creates_pp_scoped_groups_before_mesh_subgroups(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=3,
            ep=1,
            world_size=6,
        )
        groups = [Mock() for _ in range(6)]
        with (
            patch.dict(
                os.environ,
                {pipeline_parallel._PIPELINE_NEIGHBOR_P2P_ENV: "1"},
                clear=True,
            ),
            patch.object(parallel_dims_module.dist, "get_rank", return_value=3),
            patch.object(
                parallel_dims_module.dist, "new_group", side_effect=groups
            ) as new_group,
        ):
            parallel_dims._create_pipeline_neighbor_groups()

        metadata_group, edge_groups = parallel_dims.get_pipeline_neighbor_groups(
            (1, 3, 5)
        )
        self.assertIs(metadata_group, groups[3])
        self.assertEqual(edge_groups, {0: groups[4], 1: groups[5]})
        self.assertEqual(
            [call.kwargs for call in new_group.call_args_list],
            [
                {"ranks": (0, 2, 4), "backend": "gloo"},
                {
                    "ranks": [0, 2],
                    "backend": "nccl",
                    "group_desc": "pipeline_neighbor_0_0",
                },
                {
                    "ranks": [2, 4],
                    "backend": "nccl",
                    "group_desc": "pipeline_neighbor_0_1",
                },
                {"ranks": (1, 3, 5), "backend": "gloo"},
                {
                    "ranks": [1, 3],
                    "backend": "nccl",
                    "group_desc": "pipeline_neighbor_1_0",
                },
                {
                    "ranks": [3, 5],
                    "backend": "nccl",
                    "group_desc": "pipeline_neighbor_1_1",
                },
            ],
        )

    def test_schedule_uses_collective_vote_for_neighbor_stages(self):
        class NeighborStage:
            _user_meta = Mock()
            device = torch.device("cpu")
            group = Mock()
            _inference_mode = None

        stage = NeighborStage()
        original_warmup = Mock()
        schedule = SimpleNamespace(_warmup_p2p=original_warmup)

        with (
            patch.object(
                pipeline_parallel,
                "_NeighborP2PPipelineStage",
                NeighborStage,
            ),
            patch.object(
                pipeline_parallel.stage_lib.InferenceMode,
                "needs_dynamic",
                return_value=True,
            ),
            patch.object(pipeline_parallel.dist, "all_reduce") as all_reduce,
        ):
            pipeline_parallel._configure_neighbor_p2p_schedule(schedule)
            schedule._warmup_p2p([stage], has_backward=True, p2p_done=False)

        all_reduce.assert_called_once_with(
            ANY,
            op=pipeline_parallel.dist.ReduceOp.MIN,
            group=stage.group,
        )
        self.assertIs(
            stage._inference_mode, pipeline_parallel.stage_lib.InferenceMode.DYNAMIC
        )
        original_warmup.assert_not_called()

    def test_backward_receive_uses_next_two_rank_group_peer(self):
        edge_group = Mock()
        tensor = torch.ones(1)
        info = SimpleNamespace(
            is_root_arg=False,
            buffer=tensor,
            tensor_meta=Mock(),
            source=2,
        )
        stage = SimpleNamespace(stage_index=1)

        with patch.object(
            pipeline_parallel.dist, "P2POp", return_value="op"
        ) as p2p:
            ops = pipeline_parallel._NeighborP2PPipelineStage._recv_edge_ops(
                stage,
                (info,),
                edge_group,
                expected_source=2,
                group_peer=1,
            )

        self.assertEqual(ops, ["op"])
        p2p.assert_called_once_with(
            pipeline_parallel.dist.irecv,
            tensor,
            group_peer=1,
            group=edge_group,
        )

    def test_forward_send_uses_next_two_rank_group(self):
        edge_group = Mock()
        tensor = torch.ones(1)
        stage = SimpleNamespace(
            is_last=False,
            stage_index=0,
            fwd_cache={0: ((tensor,), None)},
            act_send_info={0: [1]},
            _next_edge_group=lambda: edge_group,
        )

        with (
            patch.object(
                pipeline_parallel.stage_lib,
                "to_local_if_dtensor",
                return_value=tensor,
            ),
            patch.object(pipeline_parallel.dist, "P2POp", return_value="op") as p2p,
        ):
            ops = pipeline_parallel._NeighborP2PPipelineStage.get_fwd_send_ops(
                stage, 0
            )

        self.assertEqual(ops, ["op"])
        p2p.assert_called_once_with(
            pipeline_parallel.dist.isend,
            tensor,
            group_peer=1,
            group=edge_group,
        )

    def test_send_metadata_uses_cpu_gloo_without_batching(self):
        metadata_group = Mock()
        stage = SimpleNamespace(
            _metadata_group=metadata_group,
            _resolve_peer_global_rank=lambda stage_idx: 17 + stage_idx,
        )

        with patch.object(pipeline_parallel.dist, "send_object_list") as send:
            pipeline_parallel._NeighborP2PPipelineStage._send_meta(
                stage, {"shape": (8, 16)}, 2
            )

        send.assert_called_once_with(
            [{"shape": (8, 16)}],
            dst=19,
            group=metadata_group,
            device=torch.device("cpu"),
            use_batch=False,
        )

    def test_receive_metadata_uses_cpu_gloo_without_batching(self):
        metadata_group = Mock()
        stage = SimpleNamespace(
            _metadata_group=metadata_group,
            _resolve_peer_global_rank=lambda stage_idx: 17 + stage_idx,
        )

        def recv(objects, **kwargs):
            objects[0] = {"shape": (8, 16)}

        with patch.object(
            pipeline_parallel.dist, "recv_object_list", side_effect=recv
        ) as receive:
            result = pipeline_parallel._NeighborP2PPipelineStage._recv_meta(stage, 1)

        self.assertEqual(result, {"shape": (8, 16)})
        receive.assert_called_once_with(
            ANY,
            src=18,
            group=metadata_group,
            device=torch.device("cpu"),
            use_batch=False,
        )


if __name__ == "__main__":
    unittest.main()
