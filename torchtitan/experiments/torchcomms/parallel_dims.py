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
from torchcomms.device_mesh import (
    _flatten_with_comm,
    init_device_mesh,
    init_native_device_mesh,
)

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


def _create_device_mesh(
    world_size: int,
    mesh_shape: tuple,
    mesh_dim_names: List[str],
) -> Dict:
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
        if os.environ.get("TORCHCOMMS_PATCH_FOR_COMPILE", "0") == "1":
            logger.info("calling init_native_device_mesh")
            device_mesh = init_native_device_mesh(
                mesh_dim_comms=mesh_dim_comms,
                mesh_dim_names=tuple(mesh_dim_names),
            )
        else:
            logger.info("calling init_device_mesh")
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
    flatten_ranks_per_dim: Dict[str, List[int]],
    comm,
    flatten_mesh_dim_names: Dict[str, List[str]],
    device_mesh: DeviceMesh,
    comm_per_dim: Dict[str, any],
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
        # Create the torchcomms communicator for the flattened dimension
        comm_per_dim[flatten_dim_name] = comm.split(ranks, flatten_dim_name)

        if os.environ.get("TORCHCOMMS_PATCH_FOR_COMPILE", "0") == "1":
            root_mesh = device_mesh._get_root_mesh()
            root_mesh.register_comm_backend(
                "torchcomms", {flatten_dim_name: comm_per_dim[flatten_dim_name]}
            )
            dim_names_to_flatten = tuple(flatten_mesh_dim_names[flatten_dim_name])
            submesh = device_mesh[dim_names_to_flatten]
            submesh._flatten(mesh_dim_name=flatten_dim_name)
        else:
            # Use torchcomms _flatten_with_comm (uses process groups)
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
    def _mesh_exist(self, name: str, degree: int) -> bool:
        """Override to handle EP-related dimensions correctly.

        For EP-related dimensions (ep, etp, efsdp), we check the actual
        parallelism degree rather than relying on the mesh size, since we
        may be reusing other meshes as placeholders.
        """
        if name == "ep":
            return self.ep > 1
        if name == "etp":
            return self.etp > 1
        if name == "efsdp":
            return self.ep > 1
        return degree > 1

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Override to handle multi-dimensional EP meshes.

        TorchComms creates EP meshes differently from the base class.
        For ["ep", "etp"] requests, we return the 2D mesh that spans
        both the EP and TP dimensions in the torchcomms mesh structure.
        """
        # Ensure the mesh is built first (same check as base class)
        if not self._meshes:
            self.build_mesh()

        if isinstance(dims, list) and set(dims) & {"ep", "etp", "efsdp"}:
            # For EP-related multi-dimensional meshes, return None if EP not enabled
            if not self.ep_enabled:
                return None

            # When EP is enabled, return the 2D ep_etp mesh for ep+etp combinations
            if set(dims) == {"ep", "etp"}:
                # Return the cached 2D ep_etp mesh
                return getattr(self, "_ep_etp_mesh", None)

            # Handle ["dp_replicate", "efsdp"] or ["efsdp"] for expert data parallelism
            if set(dims) == {"efsdp"}:
                return self._meshes.get("efsdp")
            if set(dims) == {"dp_replicate", "efsdp"}:
                # Return a 2D mesh combining dp_replicate and efsdp
                # In torchcomms, efsdp is mapped to fsdp (dp_shard_cp)
                # We need to create a proper 2D mesh
                return getattr(self, "_dp_replicate_efsdp_mesh", None)

            # For other EP-related combinations, return None for now
            return None

        return super().get_optional_mesh(dims)

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

    def build_mesh(self) -> DeviceMesh:
        """Build the device mesh using torchcomms communicators.

        This overrides the base class to use torchcomms for collective communication.
        The mesh structure matches the base class expectations for compatibility.
        """
        logger.info(
            f"Building torchcomms device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}, etp={self.etp}"
        )

        # Build the torchcomms mesh
        if self.ep_enabled:
            device_mesh = self._build_mesh_with_ep()
        else:
            device_mesh = self._build_mesh_without_ep()

        self._world_mesh = device_mesh

        # Map torchcomms dimension names to base class expectations:
        # - "dp" -> "batch" (dp_replicate * dp_shard)
        # - "dp_shard_cp" -> "fsdp" (dp_shard * cp)
        # - "dp_cp" -> "loss" (dp_replicate * dp_shard * cp)

        # Create self._meshes with expected dimension names
        # Note: For flattened dimensions, access via __getitem__ works through _flatten_mapping
        # Get the root mesh which has the flatten mapping
        root_mesh = device_mesh._get_root_mesh() if hasattr(device_mesh, '_get_root_mesh') else device_mesh
        flatten_mapping = getattr(root_mesh, '_flatten_mapping', {})

        def get_flattened_mesh(name: str):
            """Get a flattened mesh dimension, trying both direct access and flatten mapping."""
            try:
                return device_mesh[name]
            except (KeyError, ValueError):
                mesh = flatten_mapping.get(name)
                if mesh is None:
                    logger.warning(f"Could not find {name} dimension in device mesh")
                return mesh

        fsdp_mesh = get_flattened_mesh("dp_shard_cp")
        dp_mesh = get_flattened_mesh("dp")
        dp_cp_mesh = get_flattened_mesh("dp_cp")
        ep_mesh = get_flattened_mesh("ep") if self.ep_enabled else None
        efsdp_mesh = get_flattened_mesh("efsdp") if self.ep_enabled else None

        self._meshes = {
            "pp": device_mesh["pp"],
            "batch": dp_mesh,  # dp = dp_replicate * dp_shard
            "loss": dp_cp_mesh,  # dp_cp = dp_replicate * dp_shard * cp
            "dp_replicate": device_mesh["dp_replicate"],
            "fsdp": fsdp_mesh,  # dp_shard_cp = dp_shard * cp
            "cp": device_mesh["cp"],
            "tp": device_mesh["tp"],
        }

        # For EP-enabled meshes, add EP-related dimensions
        if self.ep_enabled:
            self._meshes["ep"] = ep_mesh if ep_mesh is not None else get_flattened_mesh("ep")
            # efsdp is the FSDP dimension for expert parameters
            # Use the properly named efsdp mesh
            self._meshes["efsdp"] = efsdp_mesh if efsdp_mesh is not None else self._meshes["fsdp"]
            # etp is the tensor parallelism dimension for experts
            # In torchcomms, we use tp as the equivalent
            self._meshes["etp"] = self._meshes["tp"]

            # Create the 2D ep_etp mesh for expert parallelism
            # The torchcomms mesh has: ["pp", "dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"]
            # When etp == tp and cp == 1:
            #   - ep = dp_shard_in_ep
            #   - etp = tp
            #   - device_mesh["dp_shard_in_ep", "tp"] gives the 2D (ep, etp) mesh
            # When cp > 1, we need to flatten dp_shard_in_ep and cp
            if self.etp == self.tp:
                if self.cp == 1:
                    # Simple case: ep = dp_shard_in_ep, etp = tp
                    self._ep_etp_mesh = device_mesh["dp_shard_in_ep", "tp"]
                else:
                    # Need to flatten dp_shard_in_ep and cp into one dimension
                    # For now, use dp_shard_in_ep and tp directly
                    logger.warning(
                        "EP with cp > 1: ep_etp_mesh handling may be incomplete"
                    )
                    self._ep_etp_mesh = device_mesh["dp_shard_in_ep", "tp"]
            else:
                # etp == 1: ep includes tp, so no separate 2D mesh
                # The code path that needs ["ep", "etp"] shouldn't hit this case
                self._ep_etp_mesh = None

            # Create dp_replicate + efsdp mesh for expert data parallelism
            if self.dp_replicate > 1:
                self._dp_replicate_efsdp_mesh = device_mesh["dp_replicate", "dp_shard_cp"]
            else:
                self._dp_replicate_efsdp_mesh = None
        else:
            # For non-EP case, reuse fsdp/tp meshes for ep/efsdp/etp
            self._meshes["ep"] = self._meshes["tp"]
            self._meshes["efsdp"] = self._meshes["fsdp"]
            self._meshes["etp"] = self._meshes["tp"]
            self._ep_etp_mesh = None
            self._dp_replicate_efsdp_mesh = None

        # Create global meshes (these are multi-dimensional mesh views)
        self._global_meshes = {
            "dataloading": device_mesh,  # Use root mesh
            "loss": dp_cp_mesh,
            "dense": device_mesh,
            "sparse": device_mesh,
        }

        logger.info(
            f"Successfully created torchcomms meshes with active dimensions: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

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
            # efsdp mesh: dp_shard_mod_ep dimension
            # efsdp = fsdp * tp // (etp * ep) = dp_shard_mod_ep
            mesh.view(
                self.pp,
                self.dp_replicate,
                dp_shard_mod_ep,
                dp_shard_in_ep,
                self.cp,
                self.tp,
            ),
        ]

        flattened_mesh_dim_names = ["dp", "dp_shard_cp", "dp_cp", "ep", "efsdp"]
        flatten_mesh_dim_names = {
            "dp": ["dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep"],
            "dp_shard_cp": ["dp_shard_mod_ep", "dp_shard_in_ep", "cp"],
            "dp_cp": ["dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp"],
            "ep": ["dp_shard_in_ep", "cp", "tp"],
            "efsdp": ["dp_shard_mod_ep"],  # efsdp = dp_shard_mod_ep
        }

        reshape_size = [
            self.dp_replicate * dp_shard_mod_ep * dp_shard_in_ep,
            dp_shard_mod_ep * dp_shard_in_ep * self.cp,
            self.dp_replicate * dp_shard_mod_ep * dp_shard_in_ep * self.cp,
            dp_shard_in_ep * self.cp * self.tp,
            dp_shard_mod_ep,  # efsdp size
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
