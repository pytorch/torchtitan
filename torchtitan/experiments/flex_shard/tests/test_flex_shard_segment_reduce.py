#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_fsdp import FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import (
    SegmentReduceDescriptor,
    SegmentReduceScratchPool,
    build_segment_reduce_plan,
    owned_segment_views,
    pack_segment_reduce_scatter_input,
    segment_descriptors_from_offsets,
    segment_reduce_to_owner,
)
from torchtitan.experiments.flex_shard.tests.common import single_rank_cpu_mesh


class TestSegmentReduceToOwner(TestCase):
    def test_builds_descriptors_from_end_offsets(self):
        descriptors = segment_descriptors_from_offsets(
            offsets=[4, 7, 7, 12],
            dst_ranks=[1, 1, 0, 1],
            names=["a", "b", "empty", "c"],
        )

        self.assertEqual(
            descriptors,
            (
                SegmentReduceDescriptor(0, 4, 1, "a"),
                SegmentReduceDescriptor(4, 3, 1, "b"),
                SegmentReduceDescriptor(7, 0, 0, "empty"),
                SegmentReduceDescriptor(7, 5, 1, "c"),
            ),
        )

    def test_coalesces_adjacent_segments_by_owner(self):
        descriptors = (
            SegmentReduceDescriptor(0, 4, 1, "a"),
            SegmentReduceDescriptor(4, 3, 1, "b"),
            SegmentReduceDescriptor(7, 2, 0, "c"),
            SegmentReduceDescriptor(9, 5, 1, "d"),
        )

        plan = build_segment_reduce_plan(descriptors, world_size=3)

        self.assertEqual(plan.rank_offsets, (0, 2, 14))
        self.assertEqual(plan.rank_numels, (2, 12, 0))
        self.assertEqual(plan.chunk_numel, 12)
        self.assertEqual(
            [(s.src_offset, s.numel, s.dst_rank, s.dst_offset) for s in plan.segments],
            [
                (0, 4, 1, 0),
                (4, 3, 1, 4),
                (7, 2, 0, 0),
                (9, 5, 1, 7),
            ],
        )
        self.assertEqual(
            [
                (s.src_offset, s.numel, s.dst_rank, s.dst_offset, s.original_indices)
                for s in plan.coalesced_segments
            ],
            [
                (0, 7, 1, 0, (0, 1)),
                (7, 2, 0, 0, (2,)),
                (9, 5, 1, 7, (3,)),
            ],
        )

    def test_packs_equal_size_owner_chunks_for_reduce_scatter(self):
        descriptors = (
            SegmentReduceDescriptor(0, 4, 1),
            SegmentReduceDescriptor(4, 3, 1),
            SegmentReduceDescriptor(7, 2, 0),
            SegmentReduceDescriptor(9, 5, 1),
        )
        plan = build_segment_reduce_plan(descriptors, world_size=3)
        flat_input = torch.arange(14, dtype=torch.float32)

        send = pack_segment_reduce_scatter_input(flat_input, plan)

        expected = torch.zeros(3, 12)
        expected[1, :7] = flat_input[:7]
        expected[0, :2] = flat_input[7:9]
        expected[1, 7:12] = flat_input[9:14]
        self.assertEqual(send.view(3, 12), expected)

    def test_packs_into_reusable_output_buffer(self):
        descriptors = (
            SegmentReduceDescriptor(0, 4, 1),
            SegmentReduceDescriptor(4, 3, 1),
            SegmentReduceDescriptor(7, 2, 0),
        )
        plan = build_segment_reduce_plan(descriptors, world_size=3)
        flat_input = torch.arange(9, dtype=torch.float32)
        reusable = torch.full((plan.send_numel + 5,), -1.0)

        send = pack_segment_reduce_scatter_input(flat_input, plan, out=reusable)

        self.assertEqual(send.data_ptr(), reusable.data_ptr())
        expected = torch.zeros(3, 7)
        expected[1, :7] = flat_input[:7]
        expected[0, :2] = flat_input[7:9]
        self.assertEqual(send.view(3, 7), expected)
        self.assertEqual(reusable[plan.send_numel :].tolist(), [-1.0] * 5)

    def test_owned_views_alias_output_buffer_in_descriptor_order(self):
        descriptors = (
            SegmentReduceDescriptor(0, 4, 1),
            SegmentReduceDescriptor(4, 3, 1),
            SegmentReduceDescriptor(7, 2, 0),
            SegmentReduceDescriptor(9, 5, 1),
        )
        plan = build_segment_reduce_plan(descriptors, world_size=3)
        output = torch.arange(12, dtype=torch.float32)

        views = owned_segment_views(output, plan, rank=1)

        self.assertEqual(
            [view.tolist() for view in views],
            [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11]],
        )
        for view in views:
            self.assertEqual(
                view.untyped_storage().data_ptr(),
                output.untyped_storage().data_ptr(),
            )

    def test_single_rank_reduce_scatter_fast_path(self):
        descriptors = (
            SegmentReduceDescriptor(0, 4, 0, "a"),
            SegmentReduceDescriptor(4, 3, 0, "b"),
        )
        flat_input = torch.arange(7, dtype=torch.float32)

        with single_rank_cpu_mesh() as mesh:
            result = segment_reduce_to_owner(
                flat_input,
                descriptors,
                group=mesh.get_group(),
                op=dist.ReduceOp.SUM,
            )

        self.assertEqual(result.output, flat_input)
        self.assertEqual([view.tolist() for view in result.owned_views], [[0, 1, 2, 3], [4, 5, 6]])

    def test_scratch_pool_reuses_released_reduce_scatter_buffers(self):
        descriptors = (
            SegmentReduceDescriptor(0, 4, 0, "a"),
            SegmentReduceDescriptor(4, 3, 0, "b"),
        )
        flat_input = torch.arange(7, dtype=torch.float32)
        scratch_pool = SegmentReduceScratchPool(max_slots=1)

        with single_rank_cpu_mesh() as mesh:
            first = segment_reduce_to_owner(
                flat_input,
                descriptors,
                group=mesh.get_group(),
                op=dist.ReduceOp.SUM,
                scratch_pool=scratch_pool,
            )
            self.assertIsNotNone(first.scratch)
            first_output_ptr = first.output.data_ptr()
            first_send_ptr = first.scratch.send.data_ptr()
            first.release_scratch()

            second = segment_reduce_to_owner(
                flat_input + 1,
                descriptors,
                group=mesh.get_group(),
                op=dist.ReduceOp.SUM,
                scratch_pool=scratch_pool,
            )
            self.assertIsNotNone(second.scratch)
            self.assertEqual(second.output.data_ptr(), first_output_ptr)
            self.assertEqual(second.scratch.send.data_ptr(), first_send_ptr)
            self.assertEqual(second.output, flat_input + 1)
            second.release_scratch()

    def test_scratch_pool_requires_release_when_bounded(self):
        scratch_pool = SegmentReduceScratchPool(max_slots=1)
        lease = scratch_pool.acquire(
            send_numel=4,
            output_numel=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(RuntimeError, "release_scratch"):
            scratch_pool.acquire(
                send_numel=4,
                output_numel=2,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        lease.release()

    def test_rejects_invalid_metadata(self):
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            segment_descriptors_from_offsets([4, 3], [0, 0])

        with self.assertRaisesRegex(ValueError, "dst_rank"):
            build_segment_reduce_plan(
                [SegmentReduceDescriptor(0, 1, 2)],
                world_size=2,
            )

        plan = build_segment_reduce_plan(
            [SegmentReduceDescriptor(3, 2, 0)],
            world_size=1,
        )
        with self.assertRaisesRegex(ValueError, "exceeds flat input"):
            pack_segment_reduce_scatter_input(torch.arange(4), plan)


class TestSegmentReduceToOwnerDistributed(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    def test_reduce_scatter_routes_owner_partitions(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        descriptors = (
            SegmentReduceDescriptor(0, 2, 0, "rank0-owned"),
            SegmentReduceDescriptor(2, 3, 1, "rank1-owned"),
        )
        flat_input = torch.arange(5, dtype=torch.float32) + 10 * self.rank

        result = segment_reduce_to_owner(
            flat_input,
            descriptors,
            group=mesh.get_group(),
            op=dist.ReduceOp.SUM,
        )

        reduced = torch.tensor([10, 12, 14, 16, 18], dtype=torch.float32)
        if self.rank == 0:
            self.assertEqual(result.owned_views[0], reduced[:2])
            self.assertEqual(result.output[:2], reduced[:2])
        else:
            self.assertEqual(result.owned_views[0], reduced[2:])
            self.assertEqual(result.output[:3], reduced[2:])


if __name__ == "__main__":
    run_tests()
