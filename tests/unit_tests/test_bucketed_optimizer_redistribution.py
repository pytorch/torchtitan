# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torchtitan.components.distributed_optimizers.bucketed_redistribution import (
    _bind_bucket_configs,
    _build_owned_bucket_plans,
    _lower_packed_all_to_all,
    _MatrixBlock,
    _MatrixBlockRoute,
    _PackedAllToAllSchedule,
    _RedistributionGroup,
    _RedistributionPlan,
    assign_balanced_owners,
    BucketConfig,
    BucketSpec,
)


class TestBucketedOptimizerRedistribution(unittest.TestCase):
    def test_balanced_owner_assignment(self):
        self.assertEqual(
            assign_balanced_owners(
                [("a", "b"), ("c",)],
                {"a": 8, "b": 4, "c": 4},
                num_ranks=2,
                initial_memory_by_rank=(0, 4),
            ),
            ({"a": 0, "b": 1}, {"c": 0}),
        )

        owners = {"a": 0}
        config = BucketConfig(
            patterns=("a",),
            owner_rank_by_fqn=owners,
            mesh_axes=("optimizer",),
        )
        owners["a"] = 1
        self.assertEqual(config.owner_rank_by_fqn, {"a": 0})

    def test_bucket_config_requires_exactly_one_mesh_axis(self):
        for mesh_axes in ((), ("optimizer", "replicate")):
            with self.subTest(mesh_axes=mesh_axes), self.assertRaisesRegex(
                ValueError, "exactly one mesh axis"
            ):
                BucketConfig(
                    patterns=("*",),
                    owner_rank_by_fqn={},
                    mesh_axes=mesh_axes,
                )

    def test_bucket_config_uses_redistributed_parameters_to_resolve_mesh(self):
        redistributed_mesh = Mock(spec=DeviceMesh)
        redistributed_mesh.ndim = 1
        redistributed_storage_mesh = MagicMock(spec=DeviceMesh)
        redistributed_storage_mesh.__getitem__.return_value = redistributed_mesh
        compute_ready_storage_mesh = MagicMock(spec=DeviceMesh)
        redistributed = Mock(device_mesh=redistributed_storage_mesh)
        compute_ready = Mock(device_mesh=compute_ready_storage_mesh)

        specs = _bind_bucket_configs(
            (
                BucketConfig(
                    patterns=("layers.*",),
                    owner_rank_by_fqn={"layers.redistributed": 0},
                    mesh_axes=("optimizer",),
                ),
            ),
            {
                "layers.redistributed": redistributed,
                "layers.compute_ready": compute_ready,
            },
        )

        self.assertEqual(len(specs), 1)
        self.assertIs(specs[0].mesh, redistributed_mesh)
        redistributed_storage_mesh.__getitem__.assert_called_once_with(("optimizer",))
        compute_ready_storage_mesh.__getitem__.assert_not_called()

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
            result = _build_owned_bucket_plans(
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
                storage_to_compute=True,
                process_group=object(),
                local_participant=7,
            )
            reverse = _lower_packed_all_to_all(
                (plan,),
                storage_to_compute=False,
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

    def test_routes_require_an_exact_nonoverlapping_partition(self):
        def plan(blocks):
            routes = tuple(_MatrixBlockRoute(block, (3,), (7,)) for block in blocks)
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
                "in bounds",
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
