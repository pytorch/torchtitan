# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import unittest

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
        return tensor.cpu()

    def _dtensor_init(self, shape: tuple[int, ...], init_fn):
        torch.manual_seed(123)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = distribute_tensor(
            torch.empty(shape, device=self.device_type),
            mesh,
            [Shard(0)],
        )
        init_fn(tensor)
        return tensor.full_tensor().cpu()

    @with_comms
    def test_normal_matches_dense(self):
        init_fn = partial(nn.init.normal_, mean=0.1, std=0.02)
        expected = self._dense_init((17, 11), init_fn)
        actual = self._dtensor_init((17, 11), init_fn)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @with_comms
    def test_uniform_matches_dense(self):
        init_fn = partial(nn.init.uniform_, a=-0.2, b=0.3)
        expected = self._dense_init((13, 17), init_fn)
        actual = self._dtensor_init((13, 17), init_fn)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @with_comms
    def test_trunc_normal_matches_dense(self):
        init_fn = partial(
            nn.init.trunc_normal_,
            mean=0.0,
            std=0.02,
            a=-0.06,
            b=0.06,
        )
        expected = self._dense_init((19, 13), init_fn)
        actual = self._dtensor_init((19, 13), init_fn)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
