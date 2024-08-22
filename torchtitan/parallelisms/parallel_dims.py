# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh
from torchtitan.logging import logger


@dataclass
class ParallelDims:
    dp: int
    world_size: int
    enable_loss_parallel: bool
    dp_type: str

    def __post_init__(self):
        self.dp_type = self.dp_type.lower()
        self._validate()

    def _validate(self):
        dp = self.dp
        if dp == -1:
            self.dp = dp = self.world_size
        assert dp >= 1, dp
        assert (
            dp == self.world_size
        ), f"Invalid parallel dims: dp({dp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        assert self.dp_type in ("fsdp", "ddp")

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip([self.dp], ["dp"], strict=True):
            if d > 1:
                dims.append(d)
                names.append(name)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def dp_enabled(self):
        return self.dp > 1

    @property
    def loss_parallel_enabled(self):
        return False  # requires tensor parallelism

    @cached_property
    def model_parallel_size(self):
        return 1
