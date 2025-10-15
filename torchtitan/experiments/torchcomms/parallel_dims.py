# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torchcomms
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.device_mesh import DeviceMesh
from torchcomms.device_mesh import _flatten_with_comm, init_device_mesh

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger

__all__ = ["TorchCommsParallelDims"]


def _calculate_ranks_per_dimension(
    meshes: List[torch.Tensor],
    dim_names: List[str],
    dim_sizes: List[int],
    cur_rank: int,
) -> Dict[str, List[int]]:
    """Util function to calculate global ranks mapping for each mesh dimension.

    Args:
        meshes: List of mesh tensors to calculate ranks from
        dim_names: List of dimension names corresponding to each mesh
        dim_sizes: List of dimension sizes corresponding to each mesh
        cur_rank: The current rank to find in the global ranks

    Returns:
        Dictionary mapping dimension names to the list of ranks that share the same dimension group as cur_rank
    """
    ranks_per_dim = {}
    for idx, dim_name in enumerate(dim_names):
        global_ranks = (
            meshes[idx].transpose(idx, -1).reshape(-1, dim_sizes[idx]).tolist()
        )
        for row in global_ranks:
            if cur_rank in row:
                ranks_per_dim[dim_name] = row
                break
    return ranks_per_dim


@dataclass
class TorchCommsParallelDims(ParallelDims):
    def _build_mesh_without_ep(self) -> DeviceMesh:
        # TODO: support EP
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
        mesh = torch.arange(self.world_size, dtype=torch.int, device="cpu").view(
            self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp
        )
        comm = torchcomms.new_comm(
            backend,
            device,
            name="comms_test_n_d_parallel",
        )

        # Get current rank to determine which groups this rank belongs to
        cur_rank = comm.get_rank()

        mesh_dim_names = ["pp", "dp_replicate", "dp_shard", "cp", "tp"]
        mesh_sizes = [mesh.size(idx) for idx in range(len(mesh_dim_names))]
        meshes = [mesh] * len(mesh_dim_names)
        ranks_per_dim = _calculate_ranks_per_dimension(
            meshes, mesh_dim_names, mesh_sizes, cur_rank
        )
        comm_per_dim = {}

        # Create communicators using the new single-list API
        for dim_name, ranks in ranks_per_dim.items():
            comm_per_dim[dim_name] = comm.split(ranks, dim_name)

        try:
            device_mesh = init_device_mesh(
                mesh_dim_comms=(
                    comm_per_dim["pp"],
                    comm_per_dim["dp_replicate"],
                    comm_per_dim["dp_shard"],
                    comm_per_dim["cp"],
                    comm_per_dim["tp"],
                ),
                mesh_dim_names=tuple(mesh_dim_names),
                _global_comm=comm,
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                for sub_comm in comm_per_dim.values():
                    sub_comm.finalize()
                comm.finalize()
                return
            raise

        flatten_mesh = [
            mesh.view(self.pp, self.dp_replicate * self.dp_shard, self.cp, self.tp),
            mesh.view(self.pp, self.dp_replicate, self.dp_shard * self.cp, self.tp),
            mesh.view(self.pp, self.dp_replicate * self.dp_shard * self.cp, self.tp),
        ]

        flattened_mesh_dim_names = ["dp", "dp_shard_cp", "dp_cp"]
        flatten_mesh_dim_names = {
            "dp": ["dp_replicate", "dp_shard"],
            "dp_shard_cp": ["dp_shard", "cp"],
            "dp_cp": ["dp_replicate", "dp_shard", "cp"],
        }

        reshape_size = [
            self.dp_replicate * self.dp_shard,
            self.dp_shard * self.cp,
            self.dp_replicate * self.dp_shard * self.cp,
        ]

        flatten_ranks_per_dim = _calculate_ranks_per_dimension(
            flatten_mesh, flattened_mesh_dim_names, reshape_size, cur_rank
        )

        for flatten_dim_name, ranks in flatten_ranks_per_dim.items():
            comm_per_dim[flatten_dim_name] = comm.split(ranks, flatten_dim_name)
            sizes = []
            strides = []
            # This is important because we need to make sure the layout is correct
            for dim_name in flatten_mesh_dim_names[flatten_dim_name]:
                layout = device_mesh[dim_name]._layout
                sizes.append(layout.sizes)
                strides.append(layout.strides)
            flatten_layout = _MeshLayout(tuple(sizes), tuple(strides))
            _flatten_with_comm(
                device_mesh,
                flatten_dim_name,
                comm_per_dim[flatten_dim_name],
                ranks,
                flatten_layout,
            )

        # call .finalize() to release the sub comm before the root comm
        self.comms = [*comm_per_dim.values(), comm]

        return device_mesh
