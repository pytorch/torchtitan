# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torchtitan.components.distributed_optimizers.flex_optimizer_reshard import (
    _bind_bucket_configs,
    _BucketedRedistributionRuntime,
    _BucketPlan,
    _BucketWork,
    _BufferSlot,
    _build_bucket_plans,
    _build_owned_bucket_plans,
    _compute_redistributed,
    _finalize_redistributed,
    _lower_packed_all_to_all,
    _PackedAllToAllSchedule,
    _ParticipantPartition,
    _prepare_redistributed,
    _RedistributionGroup,
    _RedistributionPlan,
    _tensor_region_view,
    _TensorRegion,
    _TensorRegionRoute,
    assign_balanced_owners,
    BucketConfig,
    BucketSpec,
)


def _fragmented_partition_plan(
    participants: tuple[int, ...],
    *,
    num_compute_units: int = 5,
    compute_unit_rows: int = 3,
    num_columns: int = 2,
) -> _RedistributionPlan:
    num_participants = len(participants)
    compute_ranges = tuple(
        Shard.local_shard_size_and_offset(
            num_compute_units,
            num_participants,
            rank,
        )
        for rank in range(num_participants)
    )
    compute_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=(num_local_units, compute_unit_rows, num_columns),
            logical_regions=(
                _TensorRegion(
                    offsets=(unit_offset, 0, 0),
                    shape=(num_local_units, compute_unit_rows, num_columns),
                ),
            )
            if num_local_units
            else (),
        )
        for participant, (num_local_units, unit_offset) in zip(
            participants,
            compute_ranges,
            strict=True,
        )
    )

    storage_partitions = []
    forward_routes = []
    reverse_routes = []
    for storage_rank, participant in enumerate(participants):
        num_local_rows, storage_offset = Shard.local_shard_size_and_offset(
            num_compute_units * compute_unit_rows,
            num_participants,
            storage_rank,
        )
        logical_regions = []
        fragment_start = storage_offset
        storage_end = storage_offset + num_local_rows
        while fragment_start < storage_end:
            compute_unit = fragment_start // compute_unit_rows
            row = fragment_start % compute_unit_rows
            num_rows = min(compute_unit_rows - row, storage_end - fragment_start)
            logical_region = _TensorRegion(
                offsets=(compute_unit, row, 0),
                shape=(1, num_rows, num_columns),
            )
            storage_region = _TensorRegion(
                offsets=(fragment_start - storage_offset, 0),
                shape=(num_rows, num_columns),
            )
            compute_rank = next(
                rank
                for rank, (num_local_units, unit_offset) in enumerate(compute_ranges)
                if unit_offset <= compute_unit < unit_offset + num_local_units
            )
            compute_unit_offset = compute_ranges[compute_rank][1]
            compute_region = _TensorRegion(
                offsets=(compute_unit - compute_unit_offset, row, 0),
                shape=(1, num_rows, num_columns),
            )
            compute_participant = participants[compute_rank]
            logical_regions.append(logical_region)
            forward_routes.append(
                _TensorRegionRoute(
                    logical_region=logical_region,
                    source_region=storage_region,
                    destination_region=compute_region,
                    source_participants=(participant,),
                    destination_participants=(compute_participant,),
                )
            )
            reverse_routes.append(
                _TensorRegionRoute(
                    logical_region=logical_region,
                    source_region=compute_region,
                    destination_region=storage_region,
                    source_participants=(compute_participant,),
                    destination_participants=(participant,),
                )
            )
            fragment_start += num_rows
        storage_partitions.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=(num_local_rows, num_columns),
                logical_regions=tuple(logical_regions),
            )
        )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=(num_compute_units, compute_unit_rows, num_columns),
        storage_partitions=tuple(storage_partitions),
        compute_partitions=compute_partitions,
        storage_to_compute_routes=tuple(forward_routes),
        compute_to_storage_routes=tuple(reverse_routes),
    )


class TestFlexOptimizerReshard(unittest.TestCase):
    def test_runtime_enqueues_return_before_next_compute(self):
        runtime = _BucketedRedistributionRuntime(torch.device("cuda"))
        caller_stream = Mock()
        context = Mock()
        context.device_handle.current_stream.return_value = caller_stream
        context.device_handle.stream.side_effect = lambda _stream: nullcontext()
        context.device_handle.Event.side_effect = Mock
        context.slots = (Mock(), Mock())
        for slot in context.slots:
            slot.communication_buffers.return_value = (object(), object())
        runtime._context = context

        plans = tuple(
            Mock(redistributed_items=(object(),), unredistributed_items=())
            for _ in range(3)
        )
        plan_names = {plan: f"bucket_{index}" for index, plan in enumerate(plans)}
        events = []
        for plan in plans:
            plan.storage_to_compute_schedule.execute.side_effect = (
                lambda *, _plan=plan, **_kwargs: events.append(
                    ("gather", plan_names[_plan])
                )
            )
            plan.compute_to_storage_schedule.execute.side_effect = (
                lambda *, _plan=plan, **_kwargs: events.append(
                    ("return", plan_names[_plan])
                )
            )

        def compute_redistributed(work, *_args, **_kwargs):
            events.append(("compute", plan_names[work.plan]))

        original_release = runtime._release

        def release(work, caller):
            events.append(("release", plan_names[work.plan]))
            original_release(work, caller)

        with patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._prepare_redistributed"
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._compute_redistributed",
            side_effect=compute_redistributed,
        ), patch(
            "torchtitan.components.distributed_optimizers."
            "flex_optimizer_reshard._finalize_redistributed"
        ), patch.object(
            runtime,
            "_release",
            side_effect=release,
        ):
            runtime.run(
                plans,
                local_tensor_spec=Mock(),
                prepare=Mock(),
                compute=Mock(),
                finalize=Mock(),
            )

        self.assertEqual(
            events,
            [
                ("gather", "bucket_0"),
                ("compute", "bucket_0"),
                ("gather", "bucket_1"),
                ("return", "bucket_0"),
                ("compute", "bucket_1"),
                ("release", "bucket_0"),
                ("gather", "bucket_2"),
                ("return", "bucket_1"),
                ("compute", "bucket_2"),
                ("release", "bucket_1"),
                ("return", "bucket_2"),
                ("release", "bucket_2"),
            ],
        )
        self.assertEqual(context.slots[0].communication_buffers.call_count, 2)
        self.assertEqual(context.slots[1].communication_buffers.call_count, 1)

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

    def test_bucket_planner_preserves_empty_local_storage_region(self):
        @dataclass(frozen=True)
        class Item:
            fqn: str
            tensor: DTensor

        tensor = Mock(spec=DTensor)
        tensor.shape = torch.Size((2, 3))
        tensor.to_local.return_value = torch.empty(0, 3)
        item = Item("layers.0.weight", tensor)
        regions = (
            ((3,), _TensorRegion(offsets=(0, 0), shape=(2, 3))),
            ((7,), _TensorRegion(offsets=(2, 0), shape=(0, 3))),
        )
        group = _RedistributionGroup(
            process_group=object(),
            participants=(3, 7),
            local_participant=7,
        )
        mesh = Mock(spec=DeviceMesh)
        mesh.ndim = 1

        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
            "get_process_group_ranks",
            return_value=[3, 7],
        ), patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "_redistribution_group",
            return_value=group,
        ), patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "_dtensor_storage_regions",
            return_value=regions,
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
                get_fqn=lambda value: value.fqn,
                storage_is_compute_ready=lambda _value: False,
                get_storage_dtensor=lambda value: value.tensor,
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

    def test_bucket_planner_accepts_owner_free_redistribution(self):
        @dataclass(frozen=True)
        class Item:
            fqn: str
            tensor: DTensor

        participants = (3, 7)
        first = _TensorRegion(offsets=(0, 0), shape=(1, 2))
        second = _TensorRegion(offsets=(1, 0), shape=(1, 2))
        local = _TensorRegion(offsets=(0, 0), shape=(1, 2))
        redistribution_plan = _RedistributionPlan(
            participants=participants,
            logical_shape=(2, 2),
            storage_partitions=(
                _ParticipantPartition(3, (1, 2), (first,)),
                _ParticipantPartition(7, (1, 2), (second,)),
            ),
            compute_partitions=(
                _ParticipantPartition(3, (1, 2), (second,)),
                _ParticipantPartition(7, (1, 2), (first,)),
            ),
            storage_to_compute_routes=(
                _TensorRegionRoute(first, local, local, (3,), (7,)),
                _TensorRegionRoute(second, local, local, (7,), (3,)),
            ),
            compute_to_storage_routes=(
                _TensorRegionRoute(first, local, local, (7,), (3,)),
                _TensorRegionRoute(second, local, local, (3,), (7,)),
            ),
        )
        tensor = Mock(spec=DTensor)
        tensor.to_local.return_value = torch.empty(1, 2)
        item = Item("layers.0.weight", tensor)
        group = _RedistributionGroup(
            process_group=object(),
            participants=participants,
            local_participant=3,
        )
        mesh = Mock(spec=DeviceMesh)
        mesh.ndim = 1
        received_owner_ranks = []

        def build_plan(_item, received_group, owner_rank):
            self.assertIs(_item, item)
            self.assertIs(received_group, group)
            received_owner_ranks.append(owner_rank)
            return redistribution_plan

        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard."
            "_redistribution_group",
            return_value=group,
        ), patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
            "get_process_group_ranks",
            return_value=list(participants),
        ):
            result = _build_bucket_plans(
                (item,),
                (
                    BucketSpec(
                        patterns=("layers.0.*",),
                        owner_rank_by_fqn={},
                        mesh=mesh,
                    ),
                ),
                get_fqn=lambda value: value.fqn,
                requires_owner=lambda _value: False,
                get_storage_dtensor=lambda value: value.tensor,
                build_redistribution_plan=build_plan,
            )

        self.assertEqual(received_owner_ranks, [None])
        self.assertEqual(result.ordered_items, (item,))
        plan = result.plans[0]
        self.assertEqual(plan.unredistributed_items, ())
        self.assertEqual(plan.redistributed_items, (item,))
        self.assertIs(plan.redistribution_plans[0], redistribution_plan)
        self.assertEqual(
            plan.storage_to_compute_schedule.input_split_sizes,
            (0, 2),
        )
        self.assertEqual(
            plan.storage_to_compute_schedule.output_split_sizes,
            (0, 2),
        )

    def test_transport_neutral_routes_lower_to_packed_all_to_all(self):
        first = _TensorRegion(offsets=(0, 0), shape=(2, 3))
        second = _TensorRegion(offsets=(2, 0), shape=(2, 3))
        local = _TensorRegion(offsets=(0, 0), shape=(2, 3))
        full = _TensorRegion(offsets=(0, 0), shape=(4, 3))
        plan = _RedistributionPlan(
            participants=(3, 7),
            logical_shape=(4, 3),
            storage_partitions=(
                _ParticipantPartition(3, (2, 3), (first,)),
                _ParticipantPartition(7, (2, 3), (second,)),
            ),
            compute_partitions=(
                _ParticipantPartition(3, (0,), ()),
                _ParticipantPartition(7, (4, 3), (full,)),
            ),
            storage_to_compute_routes=(
                _TensorRegionRoute(first, local, first, (3,), (7,)),
                _TensorRegionRoute(second, local, second, (7,), (7,)),
            ),
            compute_to_storage_routes=(
                _TensorRegionRoute(first, first, local, (7,), (3,)),
                _TensorRegionRoute(second, second, local, (7,), (7,)),
            ),
        )

        with patch(
            "torchtitan.components.distributed_optimizers.flex_optimizer_reshard.dist."
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
            tuple(span.region for span in forward.output_spans_by_parameter[0]),
            (first, second),
        )
        self.assertEqual(reverse.input_split_sizes, (6, 6))
        self.assertEqual(reverse.output_split_sizes, (0, 6))
        self.assertEqual(
            tuple(span.region for span in reverse.output_spans_by_parameter[0]),
            (local,),
        )

    def test_partitioned_compute_supports_fragmented_storage_and_empty_participant(
        self,
    ):
        participants = (3, 7, 11, 13)
        plan = _fragmented_partition_plan(participants)

        self.assertEqual(plan.logical_shape, (5, 3, 2))
        self.assertEqual(plan.storage_partition(13).tensor_shape, (3, 2))
        self.assertEqual(plan.compute_partition(13).tensor_shape, (0, 3, 2))
        self.assertEqual(plan.compute_partition(13).logical_regions, ())
        self.assertTrue(
            any(
                len(route.source_region.shape) == 2
                and len(route.destination_region.shape) == 3
                for route in plan.storage_to_compute_routes
            )
        )

        forward = _lower_packed_all_to_all(
            (plan,),
            storage_to_compute=True,
            process_group=object(),
            local_participant=13,
        )
        reverse = _lower_packed_all_to_all(
            (plan,),
            storage_to_compute=False,
            process_group=object(),
            local_participant=13,
        )

        self.assertEqual(forward.input_split_sizes, (0, 0, 6, 0))
        self.assertEqual(forward.output_split_sizes, (0, 0, 0, 0))
        self.assertEqual(reverse.input_split_sizes, (0, 0, 0, 0))
        self.assertEqual(reverse.output_split_sizes, (0, 0, 6, 0))

        item = object()
        group = _RedistributionGroup(
            process_group=object(),
            participants=participants,
            local_participant=13,
        )
        bucket = _BucketPlan(
            unredistributed_items=(),
            redistributed_items=(item,),
            redistribution_plans=(plan,),
            group=group,
            storage_to_compute_schedule=forward,
            compute_to_storage_schedule=reverse,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        slot = _BufferSlot()
        compute = Mock()
        _compute_redistributed(
            _BucketWork(
                plan=bucket,
                slot=slot,
                storage_buffer=torch.empty(reverse.output_buffer_numel),
                compute_fragment_buffer=torch.empty(0),
            ),
            slot,
            compute=compute,
        )
        compute.assert_not_called()

    def test_prepare_and_finalize_assemble_multiple_endpoint_spans(self):
        participants = (3, 7, 11, 13)
        redistribution_plan = _fragmented_partition_plan(participants)
        local_participant = 7
        forward = _lower_packed_all_to_all(
            (redistribution_plan,),
            storage_to_compute=True,
            process_group=object(),
            local_participant=local_participant,
        )
        reverse = _lower_packed_all_to_all(
            (redistribution_plan,),
            storage_to_compute=False,
            process_group=object(),
            local_participant=local_participant,
        )
        item = object()
        group = _RedistributionGroup(
            process_group=object(),
            participants=participants,
            local_participant=local_participant,
        )
        bucket = _BucketPlan(
            unredistributed_items=(),
            redistributed_items=(item,),
            redistribution_plans=(redistribution_plan,),
            group=group,
            storage_to_compute_schedule=forward,
            compute_to_storage_schedule=reverse,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        slot = _BufferSlot()
        prepared = torch.arange(8, dtype=torch.float32).view(4, 2)
        packed_forward = torch.empty(forward.input_buffer_numel)

        def prepare(_item, out):
            self.assertIs(_item, item)
            out.copy_(prepared)

        _prepare_redistributed(
            bucket,
            slot,
            packed_forward,
            prepare=prepare,
        )
        self.assertGreater(
            len(forward.input_spans_by_parameter[0]),
            1,
        )
        torch.testing.assert_close(packed_forward, prepared.flatten())

        expected_update = prepared.add(100)
        packed_reverse = torch.empty(reverse.output_buffer_numel)
        for span in reverse.output_spans_by_parameter[0]:
            packed_reverse[span.buffer_offset : span.buffer_offset + span.numel].copy_(
                _tensor_region_view(expected_update, span.region).reshape(-1)
            )
        work = _BucketWork(
            plan=bucket,
            slot=slot,
            storage_buffer=packed_reverse,
            compute_fragment_buffer=torch.empty(reverse.input_buffer_numel),
        )
        finalized = []
        _finalize_redistributed(
            work,
            slot,
            finalize=lambda _item, update: finalized.append((_item, update.clone())),
        )
        self.assertEqual(len(reverse.output_spans_by_parameter[0]), 2)
        self.assertEqual(len(finalized), 1)
        self.assertIs(finalized[0][0], item)
        torch.testing.assert_close(finalized[0][1], expected_update)

    def test_routes_require_an_exact_nonoverlapping_partition(self):
        def plan(regions):
            routes = tuple(
                _TensorRegionRoute(region, region, region, (3,), (3,))
                for region in regions
            )
            return _RedistributionPlan(
                participants=(3, 7),
                logical_shape=(2, 3),
                storage_partitions=(
                    _ParticipantPartition(3, (2, 3), regions),
                    _ParticipantPartition(7, (0,), ()),
                ),
                compute_partitions=(
                    _ParticipantPartition(3, (2, 3), regions),
                    _ParticipantPartition(7, (0,), ()),
                ),
                storage_to_compute_routes=routes,
                compute_to_storage_routes=routes,
            )

        invalid_partitions = (
            (
                (_TensorRegion((0, 0), (3, 3)),),
                ValueError,
                "in bounds",
            ),
            (
                (
                    _TensorRegion((0, 0), (2, 3)),
                    _TensorRegion((0, 0), (2, 3)),
                ),
                NotImplementedError,
                "overlapping",
            ),
            (
                (_TensorRegion((0, 0), (1, 3)),),
                ValueError,
                "do not cover",
            ),
        )
        for regions, error, message in invalid_partitions:
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                plan(regions)

        first = _TensorRegion((0, 0), (1, 3))
        second = _TensorRegion((1, 0), (1, 3))
        local = _TensorRegion((0, 0), (1, 3))
        split_routes = (
            _TensorRegionRoute(first, local, local, (3,), (3,)),
            _TensorRegionRoute(second, local, local, (7,), (7,)),
        )
        split_plan = _RedistributionPlan(
            participants=(3, 7),
            logical_shape=(2, 3),
            storage_partitions=(
                _ParticipantPartition(3, (1, 3), (first,)),
                _ParticipantPartition(7, (1, 3), (second,)),
            ),
            compute_partitions=(
                _ParticipantPartition(3, (1, 3), (first,)),
                _ParticipantPartition(7, (1, 3), (second,)),
            ),
            storage_to_compute_routes=split_routes,
            compute_to_storage_routes=split_routes,
        )
        self.assertEqual(
            tuple(
                partition.tensor_shape for partition in split_plan.compute_partitions
            ),
            ((1, 3), (1, 3)),
        )

        full = _TensorRegion((0, 0), (2, 3))
        forward_routes = (
            _TensorRegionRoute(first, first, first, (3,), (3,)),
            _TensorRegionRoute(second, second, second, (3,), (3,)),
        )
        swapped_reverse_routes = (
            _TensorRegionRoute(first, second, first, (3,), (3,)),
            _TensorRegionRoute(second, first, second, (3,), (3,)),
        )
        with self.assertRaisesRegex(ValueError, "must exactly invert"):
            _RedistributionPlan(
                participants=(3, 7),
                logical_shape=(2, 3),
                storage_partitions=(
                    _ParticipantPartition(3, (2, 3), (full,)),
                    _ParticipantPartition(7, (0,), ()),
                ),
                compute_partitions=(
                    _ParticipantPartition(3, (2, 3), (full,)),
                    _ParticipantPartition(7, (0,), ()),
                ),
                storage_to_compute_routes=forward_routes,
                compute_to_storage_routes=swapped_reverse_routes,
            )

        with self.assertRaisesRegex(ValueError, "participants must be unique"):
            _TensorRegionRoute(first, first, first, (3, 3), (3,))
