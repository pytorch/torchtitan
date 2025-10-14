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

__all__ = ["TorchCommsParallelDims"]


@dataclass
class TorchCommsParallelDims(ParallelDims):
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
        if (
            self.dp_shard > 1
            and self.pp == 1
            and self.dp_replicate == 1
            and self.cp == 1
            and self.tp == 1
        ):
            self.comms = []
            comm = torchcomms.new_comm(backend, device, name="main")
            # TODO: it's a hacky solution for now and we will update it in a week
            mesh = init_device_mesh(
                mesh_dim_comms=(comm, comm, comm, comm),
                mesh_dim_names=("dp_shard", "dp", "dp_cp", "dp_shard_cp"),
                _global_comm=comm,
            )
            self.comms.append(comm)
            return mesh
        else:
            raise NotImplementedError("Only support FSDP 1D parallelism for now.")
