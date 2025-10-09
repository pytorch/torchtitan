# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass

import torch
import torchcomms
from torch.distributed.device_mesh import DeviceMesh
from torchcomms.device_mesh import init_device_mesh

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger

__all__ = ["ParallelDimsForComms"]


@dataclass
class ParallelDimsForComms(ParallelDims):
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
        backend = os.environ["TEST_BACKEND"]
        device = torch.device("cuda")
        # TODO:
        # - Extend support for additional parallelism strategies (e.g., pipeline, context)
        # - Refactor and modularize initialization logic for communication objects and device mesh construction.
        if self.dp_shard_enabled and not self.tp_enabled:
            comms = (torchcomms.new_comm(backend, device, name="dp_shard_cp"),)
            mesh = init_device_mesh(mesh_dim_comms=comms, mesh_dim_names=names)
        elif self.dp_shard_enabled and self.tp_enabled:
            comm = torchcomms.new_comm(backend, device, name="main")
            mesh_arrange = torch.arange(
                self.world_size, dtype=torch.int, device="cpu"
            ).view(self.dp_shard, self.dp)
            tp_comm = comm.split(mesh_arrange.tolist(), "tp")
            dp_comm = comm.split(mesh_arrange.transpose(0, 1).tolist(), "dp_shard")
            mesh = init_device_mesh(
                mesh_dim_comms=(dp_comm, tp_comm),
                mesh_dim_names=("dp_shard", "tp"),
                _global_comm=comm,
            )

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

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        return mesh
