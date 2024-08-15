# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple, Union

from torch.distributed.device_mesh import init_device_mesh
from torchtitan.logging import logger


@dataclass
class ParallelDims:
    dp: Union[int, List[int]]
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool
    dp_type: str = "fsdp"

    def __post_init__(self):
        self.dp_type = self.dp_type.lower()
        self._validate()

    def _get_dp(self) -> Tuple[int, int]:
        if isinstance(self.dp, (tuple, list)):
            return self.dp[0], self.dp[1]
        elif self.dp_type == "fsdp":
            return 1, self.dp
        else:
            return self.dp, 1

    def _validate(self):
        dp, tp, pp = self.dp, self.tp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (tp * pp)

        dp_replicate, dp_shard = self._get_dp()
        assert dp_replicate >= 1, self.dp
        assert dp_shard >= 1, self.dp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert (
            dp_replicate * dp_shard * tp * pp == self.world_size,
        ), f"Invalid parallel dims: dp({dp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        assert self.dp_type in ("fsdp", "ddp", "hsdp")
        assert self.dp_type != "hsdp" or dp_replicate > 1, (self.dp_type, dp_replicate)

    def build_mesh(self, device_type):
        dims = []
        names = []
        dp_replicate, dp_shard = self._get_dp()
        for d, name in zip(
            [self.pp, dp_replicate, dp_shard, self.tp],
            ["pp", "dp_replicate", "dp_shard", "tp"],
            strict=True,
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        # Create all the submesh here to ensure all required process groups are
        # initialized
        if dp_replicate > 1:
            mesh["dp_replicate", "dp_shard"]._flatten()
        return mesh

    @property
    def dp_enabled(self):
        return isinstance(self.dp, list) or self.dp > 1

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
