# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torchtitan.components.distributed_optimizers.bucketed_redistribution import (
    _build_bucket_plans,
    _build_owned_redistribution_plan,
    _lower_packed_all_to_all,
    _MatrixBlock,
    _MatrixBlockRoute,
    _PackedAllToAllSchedule,
    _RedistributionGroup,
    _RedistributionPlan,
    BucketSpec,
)


class TestBucketedOptimizerRedistribution(unittest.TestCase):
    def test_bucket_planner_preserves_empty_local_storage_block(self):
        @dataclass(frozen=True)
        class Item:
            fqn: str
            tensor: DTensor

        tensor = Mock(spec=DTensor)
        tensor.shape = torch.Size((2, 3))
        tensor.to_local.return_value = torch.empty(0, 3)
        item = Item("layers.0.weight", tensor)
        blocks = (
            ((3,), _MatrixBlock(offsets=(0, 0), shape=(2, 3))),
            ((7,), _MatrixBlock(offsets=(2, 0), shape=(0, 3))),
        )
        group = _RedistributionGroup(
            process_group=object(),
            participants=(3, 7),
            local_participant=7,
        )
        mesh = Mock(spec=DeviceMesh)
        mesh.ndim = 1

        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "get_process_group_ranks",
            return_value=[3, 7],
        ), patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution."
            "_redistribution_group",
            return_value=group,
        ), patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution."
            "_dtensor_storage_blocks",
            return_value=blocks,
        ):
            result = _build_bucket_plans(
                (item,),
                (
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={item.fqn: 0},
                        mesh=mesh,
                    ),
                ),
                fqn=lambda value: value.fqn,
                compute_locally=lambda _value: False,
                storage_dtensor=lambda value: value.tensor,
            )

        plan = result.plans[0]
        self.assertEqual(result.ordered_items, (item,))
        self.assertEqual(
            plan.storage_to_compute_schedule.input_spans_by_parameter[0][0].numel,
            0,
        )
        self.assertEqual(
            plan.compute_to_storage_schedule.output_spans_by_parameter[0][0].numel,
            0,
        )

    def test_transport_neutral_routes_lower_to_packed_all_to_all(self):
        first = _MatrixBlock(offsets=(0, 0), shape=(2, 3))
        second = _MatrixBlock(offsets=(2, 0), shape=(2, 3))
        plan = _RedistributionPlan(
            participants=(3, 7),
            logical_shape=(4, 3),
            storage_to_compute_routes=(
                _MatrixBlockRoute(first, (3,), (7,)),
                _MatrixBlockRoute(second, (7,), (7,)),
            ),
            compute_to_storage_routes=(
                _MatrixBlockRoute(first, (7,), (3,)),
                _MatrixBlockRoute(second, (7,), (7,)),
            ),
        )

        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "get_process_group_ranks",
            return_value=[3, 7],
        ):
            forward = _lower_packed_all_to_all(
                (plan,),
                direction="storage_to_compute",
                process_group=object(),
                local_participant=7,
            )
            reverse = _lower_packed_all_to_all(
                (plan,),
                direction="compute_to_storage",
                process_group=object(),
                local_participant=7,
            )

        self.assertIsInstance(forward, _PackedAllToAllSchedule)
        self.assertEqual(forward.input_split_sizes, (0, 6))
        self.assertEqual(forward.output_split_sizes, (6, 6))
        self.assertEqual(
            tuple(span.block for span in forward.output_spans_by_parameter[0]),
            (first, second),
        )
        self.assertEqual(reverse.input_split_sizes, (6, 6))
        self.assertEqual(reverse.output_split_sizes, (0, 6))
        self.assertEqual(
            tuple(span.block for span in reverse.output_spans_by_parameter[0]),
            (second,),
        )

    def test_equivalent_replicas_prefer_local_copy_source(self):
        block = _MatrixBlock(offsets=(0, 0), shape=(2, 3))
        plan = _build_owned_redistribution_plan(
            (((3, 7), block),),
            participants=(3, 7),
            owner=7,
            logical_shape=(2, 3),
        )
        self.assertEqual(plan.storage_to_compute_routes[0].source_participants, (3, 7))
        self.assertEqual(plan.compute_to_storage_routes[0].destination_participants, (3, 7))

        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "get_process_group_ranks",
            return_value=[3, 7],
        ):
            schedule = _lower_packed_all_to_all(
                (plan,),
                direction="storage_to_compute",
                process_group=object(),
                local_participant=7,
            )

        self.assertEqual(schedule.input_split_sizes, (0, 6))
        self.assertEqual(schedule.output_split_sizes, (0, 6))

    def test_copy_fanout_and_reduction_routes_are_explicit(self):
        block = _MatrixBlock(offsets=(0, 0), shape=(2, 3))
        fanout = _RedistributionPlan(
            participants=(3, 7),
            logical_shape=(2, 3),
            storage_to_compute_routes=(
                _MatrixBlockRoute(block, (3,), (3, 7)),
            ),
            compute_to_storage_routes=(
                _MatrixBlockRoute(block, (3, 7), (3,)),
            ),
        )
        reduction = _RedistributionPlan(
            participants=(3, 7),
            logical_shape=(2, 3),
            storage_to_compute_routes=(
                _MatrixBlockRoute(
                    block,
                    (3, 7),
                    (3,),
                    reduce_op=dist.ReduceOp.SUM,
                ),
            ),
            compute_to_storage_routes=(
                _MatrixBlockRoute(block, (3,), (3, 7)),
            ),
        )

        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "get_process_group_ranks",
            return_value=[3, 7],
        ):
            schedule = _lower_packed_all_to_all(
                (fanout,),
                direction="storage_to_compute",
                process_group=object(),
                local_participant=3,
            )
            with self.assertRaisesRegex(ValueError, "cannot lower reduction"):
                _lower_packed_all_to_all(
                    (reduction,),
                    direction="storage_to_compute",
                    process_group=object(),
                    local_participant=3,
                )

        self.assertEqual(schedule.input_split_sizes, (6, 6))
        self.assertEqual(schedule.output_split_sizes, (6, 0))

    def test_routes_require_an_exact_nonoverlapping_partition(self):
        def plan(blocks):
            routes = tuple(
                _MatrixBlockRoute(block, (3,), (7,)) for block in blocks
            )
            return _RedistributionPlan(
                participants=(3, 7),
                logical_shape=(2, 3),
                storage_to_compute_routes=routes,
                compute_to_storage_routes=routes,
            )

        invalid_partitions = (
            (
                (_MatrixBlock((0, 0), (3, 3)),),
                ValueError,
                "outside",
            ),
            (
                (
                    _MatrixBlock((0, 0), (2, 3)),
                    _MatrixBlock((0, 0), (2, 3)),
                ),
                NotImplementedError,
                "overlapping",
            ),
            (
                (_MatrixBlock((0, 0), (1, 3)),),
                ValueError,
                "do not cover",
            ),
        )
        for blocks, error, message in invalid_partitions:
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                plan(blocks)

        split_routes = (
            _MatrixBlockRoute(_MatrixBlock((0, 0), (1, 3)), (3,), (3,)),
            _MatrixBlockRoute(_MatrixBlock((1, 0), (1, 3)), (7,), (7,)),
        )
        with self.assertRaisesRegex(ValueError, "compute destination"):
            _RedistributionPlan(
                participants=(3, 7),
                logical_shape=(2, 3),
                storage_to_compute_routes=split_routes,
                compute_to_storage_routes=split_routes,
            )
