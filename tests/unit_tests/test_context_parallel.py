# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torchtitan.distributed.context_parallel as context_parallel
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.context_parallel import (
    apply_cp_to_forward,
    cp_shard,
    head_to_sequence_all_to_all,
    previous_sequence_shard_tail,
    sequence_to_head_all_to_all,
)
from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata


class FakeMesh:
    def __init__(self, size: int = 1, local_rank: int = 0, ndim: int = 1) -> None:
        self._size = size
        self._local_rank = local_rank
        self.ndim = ndim

    def size(self, *args, **kwargs) -> int:
        return self._size

    def get_group(self):
        return None

    def get_local_rank(self) -> int:
        return self._local_rank


class TestContextParallelHelpers(unittest.TestCase):
    def test_sequence_to_head_single_rank_preserves_autograd(self) -> None:
        mesh = FakeMesh(size=1)
        x = torch.randn(2, 4, 3, 5, requires_grad=True)

        y = sequence_to_head_all_to_all(x, mesh)

        self.assertIs(y, x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_head_to_sequence_single_rank_preserves_autograd(self) -> None:
        mesh = FakeMesh(size=1)
        x = torch.randn(2, 4, 3, 5, requires_grad=True)

        y = head_to_sequence_all_to_all(x, mesh)

        self.assertIs(y, x)
        y.square().sum().backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_all_to_all_validates_scatter_dim_before_distributed(self) -> None:
        mesh = FakeMesh(size=2)
        x = torch.randn(2, 4, 3, 5)

        with self.assertRaisesRegex(ValueError, "divisible by CP size 2"):
            sequence_to_head_all_to_all(x, mesh)

    def test_all_to_all_rejects_invalid_dims(self) -> None:
        mesh = FakeMesh(size=2)
        x = torch.randn(2, 4, 3, 5)

        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            sequence_to_head_all_to_all(x, mesh, head_dim=8)

    def test_all_to_all_rejects_identical_dims(self) -> None:
        mesh = FakeMesh(size=2)
        x = torch.randn(2, 4, 4, 5)

        with self.assertRaisesRegex(ValueError, "must be different"):
            sequence_to_head_all_to_all(x, mesh, sequence_dim=1, head_dim=1)

    def test_all_to_all_rejects_multidimensional_mesh(self) -> None:
        mesh = FakeMesh(size=2, ndim=2)
        x = torch.randn(2, 4, 4, 5)

        with self.assertRaisesRegex(ValueError, "require a 1D DeviceMesh"):
            sequence_to_head_all_to_all(x, mesh)

    def test_previous_sequence_shard_tail_single_rank_has_zero_grad(self) -> None:
        mesh = FakeMesh(size=1)
        tail = torch.randn(2, 3, 4, requires_grad=True)

        previous_tail = previous_sequence_shard_tail(tail, mesh)

        torch.testing.assert_close(previous_tail, torch.zeros_like(tail))
        previous_tail.sum().backward()
        torch.testing.assert_close(tail.grad, torch.zeros_like(tail))

    def test_varlen_metadata_rejects_sequence_reordering(self) -> None:
        mesh = FakeMesh(size=2)
        metadata = VarlenMetadata(
            cu_seq_q=torch.tensor([0, 4], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 4], dtype=torch.int32),
            max_q=4,
            max_k=4,
        )

        with self.assertRaisesRegex(ValueError, "contiguous sequence shards"):
            cp_shard(mesh, (torch.arange(4).unsqueeze(0),), metadata, "headtail")

    def test_apply_cp_to_forward_wraps_varlen_attention(self) -> None:
        mesh = FakeMesh(size=1)
        module = VarlenAttention(VarlenAttention.Config())
        calls = []

        def sequence_to_head(tensor, cp_mesh, *, sequence_dim=1, head_dim=2):
            calls.append(
                ("sequence_to_head", tuple(tensor.shape), sequence_dim, head_dim)
            )
            return tensor

        def head_to_sequence(tensor, cp_mesh, *, sequence_dim=1, head_dim=2):
            calls.append(
                ("head_to_sequence", tuple(tensor.shape), sequence_dim, head_dim)
            )
            return tensor

        def original_forward(q, k, v, **kwargs):
            calls.append(
                ("forward", tuple(q.shape), tuple(k.shape), tuple(v.shape), kwargs)
            )
            return q + k + v

        metadata = VarlenMetadata(
            cu_seq_q=torch.tensor([0, 4], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 4], dtype=torch.int32),
            max_q=4,
            max_k=4,
        )
        module.forward = original_forward
        with (
            patch.object(
                context_parallel,
                "sequence_to_head_all_to_all",
                side_effect=sequence_to_head,
            ),
            patch.object(
                context_parallel,
                "head_to_sequence_all_to_all",
                side_effect=head_to_sequence,
            ),
        ):
            apply_cp_to_forward([module], mesh)
            q = torch.ones(2, 4, 3, 5)
            output = module(q, q, q, attention_masks=metadata)

        torch.testing.assert_close(output, q * 3)
        self.assertEqual(
            [call[0] for call in calls],
            [
                "sequence_to_head",
                "sequence_to_head",
                "sequence_to_head",
                "forward",
                "head_to_sequence",
            ],
        )
        self.assertIs(calls[3][-1]["attention_masks"], metadata)


class TestContextParallelCollectives(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_sequence_to_head_forward_and_backward(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("cp",),
        )
        base = torch.arange(8, device=self.device_type, dtype=torch.float32).reshape(
            1, 2, 4, 1
        )
        local = (base + self.rank * 100).requires_grad_()

        output = sequence_to_head_all_to_all(local, mesh)

        heads_per_rank = local.shape[2] // self.world_size
        expected = torch.cat(
            [
                (base + source_rank * 100)[
                    :, :, self.rank * heads_per_rank : (self.rank + 1) * heads_per_rank
                ]
                for source_rank in range(self.world_size)
            ],
            dim=1,
        )
        torch.testing.assert_close(output, expected)

        (output * (self.rank + 1)).sum().backward()
        expected_grad = torch.cat(
            [
                torch.full_like(local[:, :, :heads_per_rank], destination_rank + 1)
                for destination_rank in range(self.world_size)
            ],
            dim=2,
        )
        torch.testing.assert_close(local.grad, expected_grad)

    @with_comms
    def test_sequence_head_exchange_round_trip(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("cp",),
        )
        local = (
            torch.arange(8, device=self.device_type, dtype=torch.float32).reshape(
                1, 2, 4, 1
            )
            + self.rank * 100
        ).requires_grad_()

        full_sequence = sequence_to_head_all_to_all(local, mesh)
        round_trip = head_to_sequence_all_to_all(full_sequence, mesh)

        torch.testing.assert_close(round_trip, local)
        round_trip.square().sum().backward()
        torch.testing.assert_close(local.grad, 2 * local.detach())

    @with_comms
    def test_previous_sequence_shard_tail_forward_and_backward(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("cp",),
        )
        local_tail = torch.full(
            (2, 3),
            self.rank + 1.0,
            device=self.device_type,
            requires_grad=True,
        )

        previous_tail = previous_sequence_shard_tail(local_tail, mesh)

        expected = (
            torch.zeros_like(local_tail)
            if self.rank == 0
            else torch.ones_like(local_tail)
        )
        torch.testing.assert_close(previous_tail, expected)

        (previous_tail * (self.rank + 1)).sum().backward()
        expected_grad = (
            torch.full_like(local_tail, 2.0)
            if self.rank == 0
            else torch.zeros_like(local_tail)
        )
        torch.testing.assert_close(local_tail.grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
