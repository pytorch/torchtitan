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
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    ep_mode: str
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp, ep = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
        )
        for d in (dp_replicate, cp, tp, pp, ep):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        dp = dp_replicate * dp_shard
        if dp < 0:
            dp = self.world_size // (cp * tp * pp)
            self.dp_shard = dp_shard = dp // dp_replicate

        assert dp_shard >= 1
        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        if ep > 1:
            assert self.ep_mode in ["tp", "tp2ep", "dp2ep"]
            if self.ep_mode == "tp" or self.ep_mode == "tp2ep":
                assert ep == tp
            elif self.ep_mode == "dp2ep":
                # EP would borrow all cp and some dp_shard degree
                assert ep % cp == 0 and (dp_shard * cp) % ep == 0
        else:
            self.ep_mode = "none"

    def build_mesh_with_dp2ep(self, device_type):
        # In dp2ep, dp_shard and ep are derived submeshes:
        # dp_shard = dp_shard_1 * dp_shard_2
        # ep = dp_shard_2 * cp
        dp_shard_1 = self.dp_shard * self.cp // self.ep
        dp_shard_2 = self.ep // self.cp

        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, dp_shard_1, dp_shard_2, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard_1", "dp_shard_2", "cp", "tp"],
        ):
            # dp_shard_1 is needed even if it's 1, whose FSDP wrapping
            # helps the MoE layers do mixed precision training
            if d > 1 or name == "dp_shard_1":
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading
        dp_mesh_dim_names = []
        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
        dp_mesh_dim_names.append("dp_shard_1")
        if "dp_shard_2" in names:
            dp_mesh_dim_names.append("dp_shard_2")
        mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

        # Mesh for param sharding
        dp_shard_cp_mesh_dim_name = []
        dp_shard_cp_mesh_dim_name.append("dp_shard_1")
        if "dp_shard_2" in names:
            dp_shard_cp_mesh_dim_name.append("dp_shard_2")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_name.append("cp")
        mesh[tuple(dp_shard_cp_mesh_dim_name)]._flatten(mesh_dim_name="dp_shard_cp")

        # Mesh for ep
        ep_mesh_dim_names = []
        if "dp_shard_2" in names:
            ep_mesh_dim_names.append("dp_shard_2")
        if self.cp_enabled:
            ep_mesh_dim_names.append("cp")
        assert len(ep_mesh_dim_names) > 0
        mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        return mesh

    def build_mesh(self, device_type):
        if self.ep_mode == "dp2ep":
            return self.build_mesh_with_dp2ep(device_type)

        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading
        dp_mesh_dim_names = []
        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")

        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

        # Mesh for param sharding
        dp_shard_cp_mesh_dim_name = []
        if self.dp_shard_enabled:
            dp_shard_cp_mesh_dim_name.append("dp_shard")

        if self.cp_enabled:
            dp_shard_cp_mesh_dim_name.append("cp")

        if dp_shard_cp_mesh_dim_name != []:
            mesh[tuple(dp_shard_cp_mesh_dim_name)]._flatten(mesh_dim_name="dp_shard_cp")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

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
    def ep_enabled(self):
        return self.ep > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp
