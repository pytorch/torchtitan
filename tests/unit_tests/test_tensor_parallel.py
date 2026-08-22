# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

from torch.distributed.tensor import Replicate, Shard

from torchtitan.distributed import tensor_parallel
from torchtitan.distributed.tensor_parallel import NoParallel


class _FakeDTensor:
    def __init__(self, placements, local_value=None):
        self.placements = placements
        self.local_value = local_value
        self.redistribute_calls = []

    def redistribute(self, *, placements, async_op):
        self.redistribute_calls.append((placements, async_op))
        self.placements = placements
        return self

    def to_local(self):
        return self.local_value


class TestNoParallel(unittest.TestCase):
    def test_prepare_output_handles_nested_outputs(self):
        output_layout = Replicate()
        first = _FakeDTensor((Shard(0),))
        second = _FakeDTensor((Replicate(),))
        metadata = object()
        outputs = (first, {"nested": [second, metadata]})

        with patch.object(tensor_parallel, "DTensor", _FakeDTensor):
            result = NoParallel._prepare_output_fn(
                output_layout,
                False,
                None,
                outputs,
                None,
            )

        self.assertIs(result[0], first)
        self.assertEqual(first.placements, (output_layout,))
        self.assertEqual(first.redistribute_calls, [((output_layout,), True)])
        self.assertIs(result[1]["nested"][0], second)
        self.assertEqual(second.redistribute_calls, [])
        self.assertIs(result[1]["nested"][1], metadata)

    def test_prepare_output_converts_each_dtensor_leaf_to_local(self):
        output_layout = Replicate()
        first_local = object()
        second_local = object()
        first = _FakeDTensor((Replicate(),), first_local)
        second = _FakeDTensor((Shard(0),), second_local)

        with patch.object(tensor_parallel, "DTensor", _FakeDTensor):
            result = NoParallel._prepare_output_fn(
                output_layout,
                True,
                None,
                [first, (second,)],
                None,
            )

        self.assertIs(result[0], first_local)
        self.assertIs(result[1][0], second_local)
        self.assertEqual(second.redistribute_calls, [((output_layout,), True)])


if __name__ == "__main__":
    unittest.main()
