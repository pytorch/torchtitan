# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestDTensorStatefulRngInit(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    def _dense_init(self, shape: tuple[int, ...], init_fn):
        torch.manual_seed(123)
        tensor = torch.empty(shape, device=self.device_type)
        init_fn(tensor)
        state = torch.cuda.get_rng_state()
        next_draw = torch.rand(17, device=self.device_type)
        return tensor.cpu(), state, next_draw.cpu()

    def _dtensor_init(self, shape: tuple[int, ...], shard_dim: int, init_fn):
        torch.manual_seed(123)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = distribute_tensor(
            torch.empty(shape, device=self.device_type),
            mesh,
            [Shard(shard_dim)],
        )
        init_fn(tensor)
        state = torch.cuda.get_rng_state()
        next_draw = torch.rand(17, device=self.device_type)
        return tensor.full_tensor().cpu(), state, next_draw.cpu()

    def _assert_init_matches_dense(
        self, shape: tuple[int, ...], shard_dim: int, init_fn
    ):
        expected, expected_state, expected_next = self._dense_init(shape, init_fn)
        actual, actual_state, actual_next = self._dtensor_init(
            shape, shard_dim, init_fn
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)

    @with_comms
    def test_normal_matches_dense(self):
        init_fn = partial(nn.init.normal_, mean=0.1, std=0.02)
        for shard_dim in range(3):
            with self.subTest(shard_dim=shard_dim):
                self._assert_init_matches_dense((5, 37, 19), shard_dim, init_fn)

    @with_comms
    def test_uniform_matches_dense(self):
        init_fn = partial(nn.init.uniform_, a=-0.2, b=0.3)
        for shard_dim in range(3):
            with self.subTest(shard_dim=shard_dim):
                self._assert_init_matches_dense((7, 41, 17), shard_dim, init_fn)

    @with_comms
    def test_trunc_normal_matches_dense(self):
        init_fn = partial(
            nn.init.trunc_normal_,
            mean=0.0,
            std=0.02,
            a=-0.06,
            b=0.06,
        )
        for shard_dim in range(3):
            with self.subTest(shard_dim=shard_dim):
                self._assert_init_matches_dense((5, 43, 13), shard_dim, init_fn)


if __name__ == "__main__":
    unittest.main()
