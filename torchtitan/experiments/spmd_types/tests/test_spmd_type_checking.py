# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Experimental test: spmd_types integration with torchtitan-style parallelism.

Demonstrates using spmd_types to type-check a tensor-parallel linear layer
forward pass, similar to how torchtitan shards model weights across TP ranks.

Requires: pip install spmd_types  (or pip install -e '.[spmd-types]')
"""

import unittest

import torch
import torch.distributed as dist

try:
    from spmd_types import (
        all_gather,
        all_reduce,
        assert_type,
        P,
        R,
        reduce_scatter,
        V,
    )
    from spmd_types._checker import typecheck
    from spmd_types._type_attr import get_axis_local_type
    from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    HAS_SPMD_TYPES = True
except ImportError:
    HAS_SPMD_TYPES = False


@unittest.skipUnless(HAS_SPMD_TYPES, "spmd_types not installed")
class TestColumnParallelLinear(unittest.TestCase):
    """Type-check a column-parallel linear layer (weight sharded on output dim).

    In torchtitan's tensor parallelism, a column-parallel linear shards the
    weight along the output dimension. The input is Replicated across TP ranks,
    and after the local matmul, the output is Varying (each rank holds a
    different shard of the output features).

    Type flow: input(R) @ weight(V) -> output(V)
    """

    WORLD_SIZE = 4

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        cls.pg = dist.distributed_c10d._get_default_group()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()

    def tearDown(self):
        self.mode.__exit__(None, None, None)

    def _rank_map(self, cb):
        return self.mode.rank_map(cb)

    def test_column_parallel_forward(self):
        """R input @ V weight -> V output (column-parallel matmul)."""
        batch, in_features, out_features = 2, 8, 16
        shard_size = out_features // self.WORLD_SIZE

        # Input: replicated across all ranks
        input_data = torch.randn(batch, in_features)
        x = self._rank_map(lambda r: input_data.clone())
        assert_type(x, {self.pg: R})

        # Weight: each rank holds a different column shard (Varying)
        weight = self._rank_map(
            lambda r: torch.randn(shard_size, in_features)
        )
        assert_type(weight, {self.pg: V})

        with typecheck():
            # Local matmul: R @ V^T -> V
            out = torch.nn.functional.linear(x, weight)

        self.assertIs(get_axis_local_type(out, self.pg), V)
        # Each rank should have shape (batch, shard_size)
        for r in range(self.WORLD_SIZE):
            self.assertEqual(out._local_tensors[r].shape, (batch, shard_size))

    def test_column_parallel_with_all_gather(self):
        """Column-parallel output (V) -> all_gather -> full output (R)."""
        batch, in_features, out_features = 2, 8, 16
        shard_size = out_features // self.WORLD_SIZE

        input_data = torch.randn(batch, in_features)
        x = self._rank_map(lambda r: input_data.clone())
        assert_type(x, {self.pg: R})

        weight = self._rank_map(
            lambda r: torch.randn(shard_size, in_features)
        )
        assert_type(weight, {self.pg: V})

        with typecheck():
            out = torch.nn.functional.linear(x, weight)
            # Gather sharded output across ranks
            gathered = all_gather(out, self.pg, src=V, dst=R)

        self.assertIs(get_axis_local_type(gathered, self.pg), R)


@unittest.skipUnless(HAS_SPMD_TYPES, "spmd_types not installed")
class TestRowParallelLinear(unittest.TestCase):
    """Type-check a row-parallel linear layer (weight sharded on input dim).

    In torchtitan's tensor parallelism, a row-parallel linear shards the
    weight along the input dimension. Each rank computes a partial result
    that must be summed via all_reduce.

    Type flow: input(V) @ weight(V) -> partial(P) -> all_reduce -> output(R)
    """

    WORLD_SIZE = 4

    @classmethod
    def setUpClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=cls.WORLD_SIZE, store=store
        )
        cls.pg = dist.distributed_c10d._get_default_group()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        self.mode = LocalTensorMode(self.WORLD_SIZE)
        self.mode.__enter__()

    def tearDown(self):
        self.mode.__exit__(None, None, None)

    def _rank_map(self, cb):
        return self.mode.rank_map(cb)

    def test_row_parallel_forward(self):
        """V input @ V weight -> P output -> all_reduce -> R output."""
        batch, in_features, out_features = 2, 16, 8
        shard_size = in_features // self.WORLD_SIZE

        # Input: each rank holds a different shard of the input features
        x = self._rank_map(lambda r: torch.randn(batch, shard_size))
        assert_type(x, {self.pg: V})

        # Weight: each rank holds a different row shard
        weight = self._rank_map(
            lambda r: torch.randn(out_features, shard_size)
        )
        assert_type(weight, {self.pg: V})

        with typecheck():
            # Local matmul: V @ V^T -> V (each rank computes partial result)
            partial = torch.nn.functional.linear(x, weight)
            self.assertIs(get_axis_local_type(partial, self.pg), V)

            # Reinterpret V as P for all_reduce (the partial results need summing)
            from spmd_types import reinterpret

            partial_p = reinterpret(partial, self.pg, src=V, dst=P)
            self.assertIs(get_axis_local_type(partial_p, self.pg), P)

            # All-reduce to get the final replicated output
            result = all_reduce(partial_p, self.pg, src=P, dst=R)

        self.assertIs(get_axis_local_type(result, self.pg), R)
        for r in range(self.WORLD_SIZE):
            self.assertEqual(result._local_tensors[r].shape, (batch, out_features))

    def test_row_parallel_type_error(self):
        """Verify type errors are caught: adding P + R is invalid."""
        batch, in_features, out_features = 2, 16, 8
        shard_size = in_features // self.WORLD_SIZE

        x = self._rank_map(lambda r: torch.randn(batch, shard_size))
        assert_type(x, {self.pg: V})

        weight = self._rank_map(
            lambda r: torch.randn(out_features, shard_size)
        )
        assert_type(weight, {self.pg: V})

        bias_r = self._rank_map(lambda r: torch.randn(out_features))
        assert_type(bias_r, {self.pg: R})

        with typecheck():
            partial = torch.nn.functional.linear(x, weight)

            from spmd_types import reinterpret
            from spmd_types.types import SpmdTypeError

            partial_p = reinterpret(partial, self.pg, src=V, dst=P)
            # P + R is invalid — spmd_types should catch this
            with self.assertRaises(SpmdTypeError):
                _ = partial_p + bias_r


if __name__ == "__main__":
    unittest.main()
