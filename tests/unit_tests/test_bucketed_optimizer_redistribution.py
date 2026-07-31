# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.distributed as dist
from torchtitan.components.distributed_optimizers.bucketed_redistribution import (
    _build_bucket_plans,
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
            tensor: torch.Tensor

        item = Item("layers.0.weight", torch.empty(0, 3))
        blocks = (
            (3, _MatrixBlock(offsets=(0, 0), shape=(2, 3))),
            (7, _MatrixBlock(offsets=(2, 0), shape=(0, 3))),
        )
        group = _RedistributionGroup(
            process_group=object(),
            participants=(3, 7),
            local_participant=7,
        )

        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "get_process_group_ranks",
            return_value=[3, 7],
        ):
            result = _build_bucket_plans(
                (item,),
                (
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={item.fqn: 0},
                    ),
                ),
                fqn=lambda value: value.fqn,
                compute_locally=lambda _value: False,
                local_tensor=lambda value: value.tensor,
                redistribution_group=lambda _value: group,
                storage_blocks=lambda _value, _participants: blocks,
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
        plan = _RedistributionPlan(
            participants=(3, 7),
            storage_to_compute_routes=(
                _MatrixBlockRoute(block, (3, 7), (7,)),
            ),
            compute_to_storage_routes=(),
        )

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
            storage_to_compute_routes=(
                _MatrixBlockRoute(block, (3,), (3, 7)),
            ),
            compute_to_storage_routes=(),
        )
        reduction = _RedistributionPlan(
            participants=(3, 7),
            storage_to_compute_routes=(
                _MatrixBlockRoute(
                    block,
                    (3, 7),
                    (3,),
                    reduce_op=dist.ReduceOp.SUM,
                ),
            ),
            compute_to_storage_routes=(),
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
