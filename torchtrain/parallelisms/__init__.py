# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh

from torchtrain.parallelisms.parallelize_llama import parallelize_llama

logger = logging.getLogger(__name__)


models_parallelize_fns = {
    "llama": parallelize_llama,
}


@dataclass
class ParallelDims:
    dp: int
    sp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, sp, pp = self.dp, self.sp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (sp * pp)
        assert dp >= 1, dp
        assert sp >= 1, sp
        assert pp >= 1, pp
        assert (
            dp * sp * pp == self.world_size
        ), f"Invalid parallel dims: dp({dp}) * sp({sp}) * pp({pp}) != WORLD_SIZE({self.world_size})"

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.dp, self.sp, self.pp], ["dp", "sp", "pp"], # requires 3.10 - strict=True
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        names = tuple(names)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def dp_enabled(self):
        return self.dp > 1

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.sp > 1 and self.enable_loss_parallel

    @cached_property
    def model_parallel_size(self):
        return self.sp * self.pp
