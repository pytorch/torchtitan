# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass

import torch
import torchcomms
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.device_mesh import DeviceMesh
from torchcomms.device_mesh import _flatten_with_comm, init_device_mesh

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger

__all__ = ["TorchCommsParallelDims"]


def _calculate_ranks_per_dimension(
    meshes: list[torch.Tensor],
    dim_names: list[str],
    dim_sizes: list[int],
    cur_rank: int,
) -> dict[str, list[int]]:
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


def _create_device_mesh(
    world_size: int,
    mesh_shape: tuple,
    mesh_dim_names: list[str],
) -> dict:
    """Util function to create device mesh with communicators for each dimension.

    Args:
        world_size: Total number of ranks in the world
        mesh_shape: Shape of the device mesh
        mesh_dim_names: List of dimension names for the mesh

    Returns:
        Dictionary containing:
            - comm: Root communicator
            - device_mesh: Initialized DeviceMesh object
            - mesh: Tensor representation of the mesh
            - comm_per_dim: Communicators for each dimension
        Returns empty dict if initialization fails
    """
    backend = os.environ["TEST_BACKEND"]
    device = torch.device("cuda")
    mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(mesh_shape)
    comm = torchcomms.new_comm(
        backend,
        device,
        name="comms_test_n_d_parallel",
    )

    cur_rank = comm.get_rank()

    mesh_sizes = [mesh.size(idx) for idx in range(len(mesh_dim_names))]
    meshes = [mesh] * len(mesh_dim_names)
    ranks_per_dim = _calculate_ranks_per_dimension(
        meshes, mesh_dim_names, mesh_sizes, cur_rank
    )

    # Create sub-communicators for each dimension
    comm_per_dim = {}
    for dim_name, ranks in ranks_per_dim.items():
        comm_per_dim[dim_name] = comm.split(ranks, dim_name)

    # Initialize device mesh with communicators
    mesh_dim_comms = tuple(comm_per_dim[name] for name in mesh_dim_names)
    try:
        device_mesh = init_device_mesh(
            mesh_dim_comms=mesh_dim_comms,
            mesh_dim_names=tuple(mesh_dim_names),
            _global_comm=comm,
        )
    except TypeError as e:
        # TODO: remove this once PT 2.10 is released
        if "_rank" in str(e):
            for sub_comm in comm_per_dim.values():
                sub_comm.finalize()
            comm.finalize()
            return {}
        raise

    return {
        "comm": comm,
        "device_mesh": device_mesh,
        "mesh": mesh,
        "comm_per_dim": comm_per_dim,
    }


def _flatten_comms(
    flatten_ranks_per_dim: dict[str, list[int]],
    comm,
    flatten_mesh_dim_names: dict[str, list[str]],
    device_mesh: DeviceMesh,
    comm_per_dim: dict[str, any],
) -> None:
    """Util function to flatten mesh dimensions and create corresponding communicators.

    Args:
        flatten_ranks_per_dim: Mapping of flattened dimension names to ranks
        comm: Base communicator
        flatten_mesh_dim_names: Mapping of flattened names to original dimension names
        device_mesh: Device mesh to flatten
        comm_per_dim: Dictionary to store the created communicators
    """
    for flatten_dim_name, ranks in flatten_ranks_per_dim.items():
        comm_per_dim[flatten_dim_name] = comm.split(ranks, flatten_dim_name)
        sizes = []
        strides = []
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


@dataclass
class TorchCommsParallelDims(ParallelDims):
    def _build_mesh_without_ep(self) -> DeviceMesh:
        mesh_shape = (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp)
        mesh_dim_names = ["pp", "dp_replicate", "dp_shard", "cp", "tp"]

        dims = [d for d in mesh_shape if d > 1]
        names = [name for d, name in zip(mesh_shape, mesh_dim_names) if d > 1]

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")

        result = _create_device_mesh(self.world_size, mesh_shape, mesh_dim_names)
        comm = result.get("comm", None)
        device_mesh = result.get("device_mesh", None)
        mesh = result.get("mesh", None)
        comm_per_dim = result.get("comm_per_dim", None)
        assert (
            comm is not None
            and device_mesh is not None
            and mesh is not None
            and comm_per_dim is not None
        ), "fail to init device mesh"

        cur_rank = comm.get_rank()

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

        _flatten_comms(
            flatten_ranks_per_dim,
            comm,
            flatten_mesh_dim_names,
            device_mesh,
            comm_per_dim,
        )

        # Call .finalize() in train.py after training but before destroying the process group
        # to release sub-communicators before the root communicator.
        self.comms = [*comm_per_dim.values(), comm]
        return device_mesh

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # With ep, dp_shard and ep are derived submeshes:
        # dp_shard = dp_shard_mod_ep * dp_shard_in_ep
        if self.etp == self.tp:
            # ep = dp_shard_in_ep * cp
            dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
            dp_shard_in_ep = self.ep // self.cp
        else:
            assert self.etp == 1
            # ep = dp_shard_in_ep * cp * tp
            dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep
            dp_shard_in_ep = self.ep // (self.cp * self.tp)

        mesh_shape = (
            self.pp,
            self.dp_replicate,
            dp_shard_mod_ep,
            dp_shard_in_ep,
            self.cp,
            self.tp,
        )
        mesh_dim_names = [
            "pp",
            "dp_replicate",
            "dp_shard_mod_ep",
            "dp_shard_in_ep",
            "cp",
            "tp",
        ]

        dims = [
            d
            for d, name in zip(mesh_shape, mesh_dim_names)
            if d > 1 or name == "dp_shard_mod_ep"
        ]
        names = [
            name
            for d, name in zip(mesh_shape, mesh_dim_names)
            if d > 1 or name == "dp_shard_mod_ep"
        ]

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")

        result = _create_device_mesh(self.world_size, mesh_shape, mesh_dim_names)
        comm = result.get("comm", None)
        device_mesh = result.get("device_mesh", None)
        mesh = result.get("mesh", None)
        comm_per_dim = result.get("comm_per_dim", None)
        assert (
            comm is not None
            and device_mesh is not None
            and mesh is not None
            and comm_per_dim is not None
        ), "fail to init device mesh"

        cur_rank = comm.get_rank()

        flatten_mesh = [
            mesh.view(
                self.pp,
                self.dp_replicate * dp_shard_mod_ep * dp_shard_in_ep,
                self.cp,
                self.tp,
            ),
            mesh.view(
                self.pp,
                self.dp_replicate,
                dp_shard_mod_ep * dp_shard_in_ep * self.cp,
                self.tp,
            ),
            mesh.view(
                self.pp,
                self.dp_replicate * dp_shard_mod_ep * dp_shard_in_ep * self.cp,
                self.tp,
            ),
            mesh.view(
                self.pp,
                self.dp_replicate,
                dp_shard_mod_ep,
                dp_shard_in_ep * self.cp * self.tp,
            ),
        ]

        flattened_mesh_dim_names = ["dp", "dp_shard_cp", "dp_cp", "ep"]
        flatten_mesh_dim_names = {
            "dp": ["dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep"],
            "dp_shard_cp": ["dp_shard_mod_ep", "dp_shard_in_ep", "cp"],
            "dp_cp": ["dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp"],
            "ep": ["dp_shard_in_ep", "cp", "tp"],
        }

        reshape_size = [
            self.dp_replicate * dp_shard_mod_ep * dp_shard_in_ep,
            dp_shard_mod_ep * dp_shard_in_ep * self.cp,
            self.dp_replicate * dp_shard_mod_ep * dp_shard_in_ep * self.cp,
            dp_shard_in_ep * self.cp * self.tp,
        ]

        flatten_ranks_per_dim = _calculate_ranks_per_dimension(
            flatten_mesh, flattened_mesh_dim_names, reshape_size, cur_rank
        )

        _flatten_comms(
            flatten_ranks_per_dim,
            comm,
            flatten_mesh_dim_names,
            device_mesh,
            comm_per_dim,
        )

        # Call .finalize() in train.py after training but before destroying the process group
        # to release sub-communicators before the root communicator.
        self.comms = [*comm_per_dim.values(), comm]
        return device_mesh
