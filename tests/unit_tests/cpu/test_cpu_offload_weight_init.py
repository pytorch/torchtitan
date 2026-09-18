# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard

from torchtitan.protocols.module import Module


@dataclass(kw_only=True, slots=True)
class _TestModuleConfig(Module.Config):
    dim: int = 16


class _TestModule(Module):
    Config = _TestModuleConfig

    def __init__(self, config: _TestModuleConfig):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(config.dim, config.dim))
        self.w2 = nn.Parameter(torch.empty(config.dim, config.dim))


class TestCPUOffloadWeightInit(unittest.TestCase):
    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                rank=0,
                world_size=1,
                store=dist.HashStore(),
            )
        self.mesh = init_device_mesh("cpu", (1,))

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_cpu_dtensor_trunc_normal_init(self):
        """Verify trunc_normal_ completes and enforces bounds on CPU DTensor without hanging."""
        std = 0.02
        a, b = -0.04, 0.04
        config = _TestModuleConfig(
            param_init={
                "w1": partial(nn.init.trunc_normal_, std=std, a=a, b=b),
                "w2": nn.init.zeros_,
            }
        )
        mod = config.build()
        mod.w1 = nn.Parameter(distribute_tensor(torch.empty(16, 16), self.mesh, [Shard(0)]))
        mod.w2 = nn.Parameter(distribute_tensor(torch.empty(16, 16), self.mesh, [Shard(0)]))
        mod._param_init = config.param_init

        self.assertIsInstance(mod.w1, DTensor)
        self.assertEqual(mod.w1.device.type, "cpu")

        # Initialize weights
        mod.init_states()

        w1_local = mod.w1.to_local()
        self.assertTrue(torch.all(w1_local >= a))
        self.assertTrue(torch.all(w1_local <= b))
        self.assertFalse(torch.all(w1_local == 0))
        self.assertTrue(torch.all(mod.w2.to_local() == 0))

    def test_cpu_dtensor_sequential_random_init_distinct(self):
        """Verify sequential random initializations on CPU DTensors produce distinct values."""
        config = _TestModuleConfig(
            param_init={
                "w1": partial(nn.init.normal_, mean=0.0, std=1.0),
                "w2": partial(nn.init.normal_, mean=0.0, std=1.0),
            }
        )
        mod = config.build()
        mod.w1 = nn.Parameter(distribute_tensor(torch.empty(32, 32), self.mesh, [Shard(0)]))
        mod.w2 = nn.Parameter(distribute_tensor(torch.empty(32, 32), self.mesh, [Shard(0)]))
        mod._param_init = config.param_init

        mod.init_states()

        w1_data = mod.w1.to_local()
        w2_data = mod.w2.to_local()

        # Both parameters have identical shapes; verify they are NOT identical
        self.assertFalse(
            torch.allclose(w1_data, w2_data),
            "w1 and w2 received identical weights due to RNG state reset!",
        )


if __name__ == "__main__":
    unittest.main()
