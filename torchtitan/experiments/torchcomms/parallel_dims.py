# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Any

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


def _flatten_comms(
    flatten_ranks_per_dim: dict[str, list[int]],
    comm: Any,
    flatten_mesh_dim_names: dict[str, list[str]],
    device_mesh: DeviceMesh,
    comm_per_dim: dict[str, Any],
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
    """ParallelDims implementation using torchcomms for device mesh initialization."""

    # Store communicators for cleanup
    comms: list[Any] = field(default_factory=list)

    def build_mesh(self) -> DeviceMesh:
        """
        This method follows the same mesh structure as the base ParallelDims but uses
        torchcomms for communicator initialization instead of torch.distributed.
        """
        # Calculate derived dimensions
        batch = self.dp_replicate * self.dp_shard
        fsdp = self.dp_shard * self.cp
        efsdp = fsdp * self.tp // (self.etp * self.ep)

        # Build mesh shape and names based on EP configuration
        if self.ep > 1:
            # With EP, we need to split dp_shard for expert parallelism
            if self.etp == self.tp:
                dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
                dp_shard_in_ep = self.ep // self.cp
            else:
                assert self.etp == 1
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
        else:
            mesh_shape = (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp)
            mesh_dim_names = ["pp", "dp_replicate", "dp_shard", "cp", "tp"]

        # Log active dimensions
        active_dims = [d for d in mesh_shape if d > 1]
        active_names = [name for d, name in zip(mesh_shape, mesh_dim_names) if d > 1]
        logger.info(f"Building {len(active_dims)}-D device mesh with {active_names}, {active_dims}")

        # Initialize torchcomms communicators
        backend = os.environ["TEST_BACKEND"]
        device = torch.device("cuda")
        mesh = torch.arange(self.world_size, dtype=torch.int, device="cpu").view(mesh_shape)

        comm = torchcomms.new_comm(
            backend,
            device,
            name="torchcomms_parallel_dims",
        )
        cur_rank = comm.get_rank()

        # Calculate ranks per dimension
        mesh_sizes = [mesh.size(idx) for idx in range(len(mesh_dim_names))]
        meshes = [mesh] * len(mesh_dim_names)
        ranks_per_dim = _calculate_ranks_per_dimension(
            meshes, mesh_dim_names, mesh_sizes, cur_rank
        )

        # Create sub-communicators for each dimension
        comm_per_dim: dict[str, Any] = {}
        for dim_name, ranks in ranks_per_dim.items():
            comm_per_dim[dim_name] = comm.split(ranks, dim_name)

        # Initialize device mesh with torchcomms
        mesh_dim_comms = tuple(comm_per_dim[name] for name in mesh_dim_names)
        try:
            device_mesh = init_device_mesh(
                mesh_dim_comms=mesh_dim_comms,
                mesh_dim_names=tuple(mesh_dim_names),
                _global_comm=comm,
            )
        except TypeError as e:
            if "_rank" in str(e):
                for sub_comm in comm_per_dim.values():
                    sub_comm.finalize()
                comm.finalize()
                raise RuntimeError("Failed to init device mesh due to torchcomms API mismatch") from e
            raise

        # Create flattened mesh dimensions for compatibility with ParallelDims API
        if self.ep > 1:
            flatten_mesh = [
                mesh.view(self.pp, batch, self.cp, self.tp),
                mesh.view(self.pp, self.dp_replicate, fsdp, self.tp),
                mesh.view(self.pp, batch * self.cp, self.tp),
                mesh.view(self.pp, self.dp_replicate, efsdp, self.ep, self.etp),
            ]
            flattened_mesh_dim_names = ["batch", "fsdp", "loss", "ep"]
            flatten_mesh_dim_names_map = {
                "batch": ["dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep"],
                "fsdp": ["dp_shard_mod_ep", "dp_shard_in_ep", "cp"],
                "loss": ["dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp"],
                "ep": ["dp_shard_in_ep", "cp"] if self.etp == self.tp else ["dp_shard_in_ep", "cp", "tp"],
            }
            reshape_sizes = [batch, fsdp, batch * self.cp, self.ep]
        else:
            flatten_mesh = [
                mesh.view(self.pp, batch, self.cp, self.tp),
                mesh.view(self.pp, self.dp_replicate, fsdp, self.tp),
                mesh.view(self.pp, batch * self.cp, self.tp),
            ]
            flattened_mesh_dim_names = ["batch", "fsdp", "loss"]
            flatten_mesh_dim_names_map = {
                "batch": ["dp_replicate", "dp_shard"],
                "fsdp": ["dp_shard", "cp"],
                "loss": ["dp_replicate", "dp_shard", "cp"],
            }
            reshape_sizes = [batch, fsdp, batch * self.cp]

        flatten_ranks_per_dim = _calculate_ranks_per_dimension(
            flatten_mesh, flattened_mesh_dim_names, reshape_sizes, cur_rank
        )

        _flatten_comms(
            flatten_ranks_per_dim,
            comm,
            flatten_mesh_dim_names_map,
            device_mesh,
            comm_per_dim,
        )

        # Store world mesh
        self._world_mesh = device_mesh

        # Build internal mesh references following ParallelDims convention
        self._meshes = {
            "pp": device_mesh["pp"],
            "batch": device_mesh["batch"],
            "loss": device_mesh["loss"],
            "dp_replicate": device_mesh["dp_replicate"],
            "fsdp": device_mesh["fsdp"],
            "cp": device_mesh["cp"],
            "tp": device_mesh["tp"],
        }

        if self.ep > 1:
            self._meshes["ep"] = device_mesh["ep"]
            self._meshes["efsdp"] = device_mesh["efsdp"] if "efsdp" in device_mesh.mesh_dim_names else device_mesh["dp_shard_mod_ep"]
            self._meshes["etp"] = device_mesh["etp"] if "etp" in device_mesh.mesh_dim_names else device_mesh["tp"]
        else:
            # Create fake meshes for EP-related dimensions when EP is not enabled
            self._meshes["ep"] = device_mesh["pp"]  # placeholder
            self._meshes["efsdp"] = device_mesh["fsdp"]
            self._meshes["etp"] = device_mesh["tp"]

        logger.info(
            f"Successfully created torchcomms meshes with dimensions: "
            f"{list(comm_per_dim.keys())}"
        )

        # Call .finalize() in train.py after training but before destroying the process group
        # to release sub-communicators before the root communicator.
        self.comms = [*comm_per_dim.values(), comm]
        return device_mesh
