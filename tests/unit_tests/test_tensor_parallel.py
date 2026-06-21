# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.tensor_parallel import NoParallel


class _SingleOutput(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.gate(x)


class _MultiOutput(nn.Module):
    """Mirrors the MoE router gate, which returns ``(logits, ...)``."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        return logits, logits * 2.0, x.shape[0]


class TestNoParallelOutput(DTensorTestBase):
    """``NoParallel`` must handle modules that return more than one output."""

    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("tp",)
        )

    @with_comms
    def test_single_output_unchanged(self):
        torch.manual_seed(0)
        dim = 8
        mod = _SingleOutput(dim).to(self.device_type)
        ref = mod.gate.weight.detach().clone()

        mod = parallelize_module(mod, self._mesh(), NoParallel(use_local_output=True))
        x = torch.randn(4, dim, device=self.device_type)
        out = mod(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertNotIsInstance(out, DTensor)
        torch.testing.assert_close(out, x @ ref.t())

    @with_comms
    def test_multi_output(self):
        torch.manual_seed(0)
        dim = 8
        mod = _MultiOutput(dim).to(self.device_type)
        ref = mod.gate.weight.detach().clone()

        mod = parallelize_module(mod, self._mesh(), NoParallel(use_local_output=True))
        x = torch.randn(4, dim, device=self.device_type)

        # Without the fix this raises: 'tuple' object has no attribute 'placements'
        logits, doubled, batch = mod(x)

        for t in (logits, doubled):
            self.assertIsInstance(t, torch.Tensor)
            self.assertNotIsInstance(t, DTensor)
        self.assertEqual(batch, 4)

        expected = x @ ref.t()
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(doubled, expected * 2.0)

    @with_comms
    def test_multi_output_keeps_dtensor_when_not_local(self):
        torch.manual_seed(0)
        dim = 8
        mod = _MultiOutput(dim).to(self.device_type)
        mod = parallelize_module(mod, self._mesh(), NoParallel(use_local_output=False))
        x = torch.randn(4, dim, device=self.device_type)

        logits, doubled, batch = mod(x)
        for t in (logits, doubled):
            self.assertIsInstance(t, DTensor)
        self.assertEqual(batch, 4)


if __name__ == "__main__":
    unittest.main()
