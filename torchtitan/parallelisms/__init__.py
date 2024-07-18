# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh
from torchtitan.logging_utils import logger
from torchtitan.parallelisms.parallelize_llama import parallelize_llama, pipeline_llama

models_parallelize_fns = {
    "llama2": parallelize_llama,
    "llama3": parallelize_llama,
}
models_pipelining_fns = {
    "llama2": pipeline_llama,
    "llama3": pipeline_llama,
}


@dataclass
class ParallelDims:
    dp: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool
    dp_type: str
    dp_replicate: int

    def __post_init__(self):
        self.dp_type = self.dp_type.lower()
        self._validate()

    def _validate(self):
        dp, cp, tp, pp = self.dp, self.cp, self.tp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (cp * tp * pp)
        assert dp >= 1, dp
        assert dp % self.dp_replicate, (self.dp_replicate, dp)
        assert cp >= 1, cp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert dp * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp({dp}) * cp ({cp}) * tp({tp}) * pp({pp}) "
            f"!= WORLD_SIZE({self.world_size})"
        )
        assert self.dp_type in ("fsdp", "ddp", "hsdp")

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp * self.cp, self.tp], ["pp", "dp", "tp"], strict=True
        ):
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
    def cp_enabled(self):
        return self.cp > 1

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
