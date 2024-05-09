# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh
from torchtitan.logging_utils import logger
from torchtitan.parallelisms.parallelize_llama import parallelize_llama

models_parallelize_fns = {
    "llama2": parallelize_llama,
    "llama3": parallelize_llama,
}


@dataclass
class ParallelDims:
    dp: int
    dp_replicate: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, dp_replicate, tp, pp = self.dp, self.dp_replicate, self.tp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (dp_replicate * tp * pp)
        assert dp >= 1, dp
        assert dp_replicate >= 1, dp_replicate
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert (
            dp * dp_replicate * tp * pp == self.world_size
        ), (
            f"Invalid parallel dims: dp({dp}) * dp_replicate({dp_replicate}) * "
            f"tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})."
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp, self.tp],
            ["pp", "dp_replicate", "dp", "tp"],
            strict=True
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
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

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
