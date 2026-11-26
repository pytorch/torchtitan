# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed import context_parallel
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.ragged_tensor import RaggedTensor
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.decoder_sharding import (
    decoder_input_sharding,
    loss_kwarg_sharding,
)

# Kept sets of 8 tokens; capacity 16.
_ROWS = [[5, 7], [], [2], [4, 4, 4], [], [9], [1], [3, 3]]


def _ragged(capacity: int = 16) -> RaggedTensor:
    return RaggedTensor.from_rows(
        torch.tensor([v for row in _ROWS for v in row]),
        torch.tensor([len(row) for row in _ROWS]),
        capacity=capacity,
    )


def _rows(ragged: RaggedTensor) -> list[list[int]]:
    rows, start = [], 0
    for length in ragged.lengths.tolist():
        rows.append(ragged.values[start : start + length].tolist())
        start += length
    return rows


def _padding(ragged: RaggedTensor) -> list[int]:
    return ragged.values[int(ragged.lengths.sum()) :].tolist()


class TestRaggedTensor(unittest.TestCase):
    def test_row_indices_send_padding_past_the_last_row(self):
        ragged = RaggedTensor.from_rows(
            torch.tensor([5, 7, 2]), torch.tensor([2, 0, 1]), capacity=6
        )
        self.assertEqual(ragged.values.tolist(), [5, 7, 2, 0, 0, 0])
        self.assertEqual(ragged.row_indices().tolist(), [0, 0, 2, 3, 3, 3])

    def test_take_rows_and_split_repack_into_their_capacity(self):
        ragged = _ragged()
        taken = ragged.take_rows(torch.tensor([3, 0, 7]), capacity=8)
        self.assertEqual(_rows(taken), [_ROWS[3], _ROWS[0], _ROWS[7]])
        self.assertEqual(_padding(taken), [0])
        parts = ragged.split(2)
        self.assertEqual([p.values.shape[0] for p in parts], [8, 8])
        self.assertEqual([_rows(p) for p in parts], [_ROWS[:4], _ROWS[4:]])
        self.assertEqual(_padding(parts[1]), [0] * 4)

    def test_rows_over_the_capacity_raise(self):
        with self.assertRaisesRegex(RuntimeError, "don't fit"):
            _ragged().take_rows(torch.tensor([0, 3]), capacity=4)


class TestShardRaggedTensor(unittest.TestCase):
    """CP keeps exactly each rank's tokens' rows, in its token order, padded to C // cp."""

    def _shard(self, cp_rank: int, permutation: torch.Tensor | None, capacity: int):
        dist.init_process_group("fake", store=FakeStore(), rank=cp_rank, world_size=2)
        try:
            parallelism_context = ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=2,
                tp=1,
                pp=1,
                ep=1,
                world_size=2,
                enable_sequence_parallel=False,
            )
            with mock.patch(
                "torchtitan.distributed.parallelism_context.device_type", "cpu"
            ):
                parallelism_context.build_mesh()
            input_dict = {"labels": torch.arange(8), "kept": _ragged(capacity)}
            shardings = decoder_input_sharding()
            shardings.update(loss_kwarg_sharding({"kept": None}, shardings["labels"]))
            # As TrainingEngine preprocesses: sharding runs on the raw inputs, which are
            # annotated afterwards.
            with parallelism_context.activate_spmd():
                sharded = context_parallel.shard_tensors(
                    input_dict, input_shardings=shardings, permutation=permutation
                )
                # The kept sets take the labels' layout.
                annotate_input_spmd_types(
                    parallelism_context,
                    {"labels": sharded["labels"], "kept": sharded["kept"]},
                    shardings,
                )
            return sharded
        finally:
            dist.destroy_process_group()

    def test_contiguous_and_permuted_sharding(self):
        permutation = torch.tensor([[0, 7, 1, 6, 2, 5, 3, 4]])
        for perm in (None, permutation):
            for cp_rank in (0, 1):
                with self.subTest(permuted=perm is not None, cp_rank=cp_rank):
                    sharded = self._shard(cp_rank, perm, capacity=16)
                    kept = sharded["kept"]
                    self.assertEqual(kept.values.shape, (8,))
                    tokens = sharded["labels"].tolist()
                    self.assertEqual(_rows(kept), [_ROWS[t] for t in tokens])
                    self.assertTrue(all(v == 0 for v in _padding(kept)))

    def test_a_rank_over_its_share_of_the_capacity_raises(self):
        # Rank 0's tokens hold 6 of the 10 values; capacity 10 leaves each rank 5.
        with self.assertRaisesRegex(RuntimeError, "don't fit"):
            self._shard(0, None, capacity=10)


if __name__ == "__main__":
    unittest.main()
