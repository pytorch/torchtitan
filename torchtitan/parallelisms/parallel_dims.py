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
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool
    dp_type: str = "fsdp"
    dp_replicate: int = 1  # Only used when dp_type is hsdp

    def __post_init__(self):
        self.dp_type = self.dp_type.lower()
        self._validate()

    def _validate(self):
        dp, dp_replicate, tp, pp = self.dp, self.dp_replicate, self.tp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (tp * pp)
        assert dp >= 1, dp
        assert dp_replicate >= 1 and dp % dp_replicate == 0, (dp, dp_replicate)
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert (
            dp * tp * pp == self.world_size,
        ), f"Invalid parallel dims: dp({dp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        assert self.dp_type in ("fsdp", "ddp", "hsdp")
        assert self.dp_type != "hsdp" or dp_replicate > 1, (self.dp_type, dp_replicate)

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp, self.tp], ["pp", "dp", "tp"], strict=True
        ):
            if d <= 1:
                continue

            if name != "dp" or self.dp_replicate <= 1:
                dims.append(d)
                names.append(name)
                continue

            dp_shard = self.dp // self.dp_replicate
            dims.extend([self.dp_replicate, dp_shard])
            names.extend(["dp_replicate", "dp_shard"])

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        # Create all the submesh here to ensure all required process groups are
        # initialized
        if self.dp_replicate > 1:
            mesh["dp_replicate", "dp_shard"]._flatten()
        return mesh

    @property
    def dp_enabled(self):
        return self.dp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def model_parallel_size(self):
        return self.tp * self.pp
