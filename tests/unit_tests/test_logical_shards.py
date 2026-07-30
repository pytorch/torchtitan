# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointableTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.logical_shards import normalize_logical_tensor


def _set_checkpoint_metadata(
    tensor: torch.Tensor,
    *,
    global_shape,
    global_offsets,
    local_offsets,
    local_sizes,
) -> torch.Tensor:
    setattr(tensor, "global_shape", global_shape)  # noqa: B010
    setattr(tensor, "global_offsets", global_offsets)  # noqa: B010
    setattr(tensor, "local_offsets", local_offsets)  # noqa: B010
    setattr(tensor, "local_sizes", local_sizes)  # noqa: B010
    return tensor


class TestLogicalShards(unittest.TestCase):
    def test_dense_and_empty(self):
        tensor = torch.empty(3, 5)
        logical = normalize_logical_tensor(tensor)
        self.assertIs(logical.local_tensor, tensor)
        self.assertEqual(logical.layout.global_shape, (3, 5))
        self.assertEqual(logical.layout.global_offsets, ((0, 0),))
        self.assertEqual(logical.layout.local_offsets, ((0, 0),))
        self.assertEqual(logical.layout.local_sizes, ((3, 5),))

        scalar = normalize_logical_tensor(torch.tensor(1.0))
        self.assertEqual(scalar.layout.global_offsets, ((),))
        self.assertEqual(scalar.layout.local_sizes, ((),))

        empty = normalize_logical_tensor(torch.empty(0, 5))
        self.assertEqual(empty.layout.global_shape, (0, 5))
        self.assertEqual(empty.layout.global_offsets, ())
        self.assertEqual(empty.layout.local_offsets, ())
        self.assertEqual(empty.layout.local_sizes, ())

    def test_checkpointable_padded_multiple_pieces_and_empty_owner(self):
        tensor = _set_checkpoint_metadata(
            torch.empty(5, 8),
            global_shape=(6, 6),
            global_offsets=((0, 0), (4, 2)),
            local_offsets=((1, 1), (3, 4)),
            local_sizes=((2, 3), (2, 2)),
        )
        self.assertIsInstance(tensor, CheckpointableTensor)
        logical = normalize_logical_tensor(tensor)
        self.assertEqual(logical.layout.global_shape, (6, 6))
        self.assertEqual(logical.layout.global_offsets, ((0, 0), (4, 2)))
        self.assertEqual(logical.layout.local_offsets, ((1, 1), (3, 4)))
        self.assertEqual(logical.layout.local_sizes, ((2, 3), (2, 2)))

        empty_owner = _set_checkpoint_metadata(
            torch.empty(2, 8),
            global_shape=(6, 6),
            global_offsets=(),
            local_offsets=(),
            local_sizes=(),
        )
        empty_layout = normalize_logical_tensor(empty_owner).layout
        self.assertEqual(empty_layout.global_shape, (6, 6))
        self.assertEqual(empty_layout.global_offsets, ())

        explicit_empty = _set_checkpoint_metadata(
            torch.empty(0, 6),
            global_shape=(6, 6),
            global_offsets=((6, 0),),
            local_offsets=((0, 0),),
            local_sizes=((0, 6),),
        )
        self.assertEqual(
            normalize_logical_tensor(explicit_empty).layout.global_offsets, ()
        )

    def test_checkpointable_metadata_validation(self):
        incomplete = torch.empty(2, 2)
        setattr(incomplete, "global_shape", (2, 2))  # noqa: B010
        with self.assertRaisesRegex(ValueError, "incomplete CheckpointableTensor"):
            normalize_logical_tensor(incomplete)

        cases = {
            "same length": dict(
                global_offsets=((0, 0),), local_offsets=(), local_sizes=((1, 1),)
            ),
            "outside global": dict(
                global_offsets=((3, 3),),
                local_offsets=((0, 0),),
                local_sizes=((2, 2),),
            ),
            "outside the physical": dict(
                global_offsets=((0, 0),),
                local_offsets=((3, 3),),
                local_sizes=((2, 2),),
            ),
            "global rectangles": dict(
                global_offsets=((0, 0), (1, 1)),
                local_offsets=((0, 0), (2, 2)),
                local_sizes=((2, 2), (2, 2)),
            ),
            "local rectangles": dict(
                global_offsets=((0, 0), (2, 2)),
                local_offsets=((0, 0), (1, 1)),
                local_sizes=((2, 2), (2, 2)),
            ),
        }
        for error, metadata in cases.items():
            with self.subTest(error=error):
                tensor = _set_checkpoint_metadata(
                    torch.empty(4, 4),
                    global_shape=(4, 4),
                    **metadata,
                )
                with self.assertRaisesRegex(ValueError, error):
                    normalize_logical_tensor(tensor)

        wrong_rank = _set_checkpoint_metadata(
            torch.empty(0),
            global_shape=(2, 2),
            global_offsets=(),
            local_offsets=(),
            local_sizes=(),
        )
        with self.assertRaisesRegex(ValueError, "same number of dimensions"):
            normalize_logical_tensor(wrong_rank)

        non_integer = _set_checkpoint_metadata(
            torch.empty(2, 2),
            global_shape=(2.5, 2),
            global_offsets=((0, 0),),
            local_offsets=((0, 0),),
            local_sizes=((2, 2),),
        )
        with self.assertRaisesRegex(ValueError, "sequence of integers"):
            normalize_logical_tensor(non_integer)

        private_metadata = torch.empty(2, 2)
        setattr(private_metadata, "_global_shape", (4, 2))  # noqa: B010
        with self.assertRaisesRegex(ValueError, "public CheckpointableTensor"):
            normalize_logical_tensor(private_metadata)


class TestLogicalShardsDTensor(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_shard_replicate_and_empty_owner(self):
        mesh = init_device_mesh("cpu", (self.world_size,))
        rank = dist.get_rank()

        shard0 = distribute_tensor(torch.empty(7, 5), mesh, [Shard(0)])
        logical0 = normalize_logical_tensor(shard0)
        expected_rows = (2, 2, 2, 1)[rank]
        self.assertNotIsInstance(logical0.local_tensor, DTensor)
        self.assertEqual(logical0.local_tensor.shape, shard0.to_local().shape)
        self.assertEqual(logical0.layout.global_shape, (7, 5))
        self.assertEqual(logical0.layout.global_offsets, ((2 * rank, 0),))
        self.assertEqual(logical0.layout.local_offsets, ((0, 0),))
        self.assertEqual(logical0.layout.local_sizes, ((expected_rows, 5),))

        shard1 = distribute_tensor(torch.empty(5, 7), mesh, [Shard(1)])
        logical1 = normalize_logical_tensor(shard1)
        expected_columns = (2, 2, 2, 1)[rank]
        self.assertEqual(logical1.layout.global_offsets, ((0, 2 * rank),))
        self.assertEqual(logical1.layout.local_sizes, ((5, expected_columns),))

        replicated = distribute_tensor(torch.empty(3, 5), mesh, [Replicate()])
        replicated_layout = normalize_logical_tensor(replicated).layout
        self.assertEqual(replicated_layout.global_offsets, ((0, 0),))
        self.assertEqual(replicated_layout.local_sizes, ((3, 5),))

        sparse = distribute_tensor(torch.empty(2, 5), mesh, [Shard(0)])
        sparse_layout = normalize_logical_tensor(sparse).layout
        if rank < 2:
            self.assertEqual(sparse_layout.global_offsets, ((rank, 0),))
            self.assertEqual(sparse_layout.local_sizes, ((1, 5),))
        else:
            self.assertEqual(sparse_layout.global_offsets, ())
            self.assertEqual(sparse_layout.local_sizes, ())

        partial = DTensor.from_local(
            torch.empty(2, 3),
            mesh,
            [Partial()],
            run_check=False,
            shape=torch.Size((2, 3)),
            stride=torch.Size((3, 1)),
        )
        with self.assertRaisesRegex(ValueError, "Partial"):
            normalize_logical_tensor(partial)


if __name__ == "__main__":
    unittest.main()
