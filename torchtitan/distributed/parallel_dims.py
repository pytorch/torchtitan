# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type


__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int

    _world_mesh: DeviceMesh = None

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
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        if ep > 1:
            # EP would borrow all cp and some dp_shard degree
            assert ep % cp == 0 and (dp_shard * cp) % ep == 0

    def build_mesh(self) -> DeviceMesh:
        # TODO: Current implementation of ParallelDims for dp2ep Expert Parallel
        #       is not very clean, due to the limited support from DeviceMesh
        #       for creating two staggered meshes. Will improve.
        if self.ep > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # With ep, dp_shard and ep are derived submeshes:
        # dp_shard = dp_shard_mod_ep * dp_shard_in_ep
        # ep = dp_shard_in_ep * cp
        dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
        dp_shard_in_ep = self.ep // self.cp

        dims = []
        names = []
        for d, name in zip(
            [
                self.pp,
                self.dp_replicate,
                dp_shard_mod_ep,
                dp_shard_in_ep,
                self.cp,
                self.tp,
            ],
            ["pp", "dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"],
        ):
            # dp_shard_mod_ep is needed even if it's 1, whose FSDP wrapping
            # helps the MoE layers do mixed precision training
            if d > 1 or name == "dp_shard_mod_ep":
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")

        # Handle the case where all parallelism dimensions are 1
        # We still need to create a mesh with named dimensions for submesh access
        if not dims:
            dims = [1]
            names = ["world"]

        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []
        # Mesh for ep
        ep_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        # dp_shard_mod_ep is always needed, even if it's 1
        dp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_shard_cp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_cp_mesh_dim_names.append("dp_shard_mod_ep")
        if "dp_shard_in_ep" in names:
            dp_mesh_dim_names.append("dp_shard_in_ep")
            dp_shard_cp_mesh_dim_names.append("dp_shard_in_ep")
            dp_cp_mesh_dim_names.append("dp_shard_in_ep")
            ep_mesh_dim_names.append("dp_shard_in_ep")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")
            ep_mesh_dim_names.append("cp")

        # Ensure dp_cp submesh exists even when all parallelism dimensions are 1
        # This is required for fault tolerance functionality
        if not dp_cp_mesh_dim_names and "world" in names:
            dp_cp_mesh_dim_names = ["world"]

        mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        return mesh

    def _build_mesh_without_ep(self) -> DeviceMesh:
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

        # Handle the case where all parallelism dimensions are 1
        # We still need to create a mesh with named dimensions for submesh access
        if not dims:
            dims = [1]
            names = ["world"]

        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        # Ensure dp_cp submesh exists even when all parallelism dimensions are 1
        # This is required for fault tolerance functionality
        if not dp_cp_mesh_dim_names and "world" in names:
            dp_cp_mesh_dim_names = ["world"]


        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        return mesh

    @property
    def world_mesh(self) -> str:
        # doing late init so ParallelDims can still be used as a lightweight
        # dataclass without having to initialize the world mesh
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

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
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @cached_property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # https://github.com/pytorch/torchtitan/pull/640#discussion_r1849481001

        # Context Parallel requires that seq_len be divisible by 2 * CP degree,
        # when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.tp * (self.cp * 2)

    @cached_property
    def dense_params_mesh_ndim(self):
        # Note: In dp2ep EP, EP params mesh ndim is 1 more due to the 'ep' mesh
        return self.dp_replicate_enabled + self.fsdp_enabled + self.tp_enabled
