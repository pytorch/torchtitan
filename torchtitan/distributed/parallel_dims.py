# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

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
    etp: int
    world_size: int

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp, ep, etp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.etp,
        )
        for d in (dp_replicate, cp, tp, pp, ep, etp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        if ep > 1:
            assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"

    def _mesh_exist(self, name: str, degree: int) -> bool:
        if name == "efsdp":
            # We always keep the efsdp if EP is larger than 1 because we need
            # FSDP wrapping to help the MoE layers do mixed precision training.
            return True if self.ep > 1 else False
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        """
        Build the device mesh with the required mesh dimensions.

        The following mesh dimensions will be created:

            pp:      Pipeline Parallelism (PP).
            batch:   Used by data loading to determine the global batch size and which
                     part of the data each rank should read. This dimension includes both
                     ``dp_replicate`` and ``dp_shard``.
            loss:    Used by all-reduce when computing the loss. Includes ``dp_replicate``,
                     ``dp_shard``, and ``cp`` degrees, as all of them parallelize the data,
                     essentially require the weight gradients reduction.
            dp_replicate: For DDP or HSDP replicate dimension.
            fsdp:    For FSDP dimension. This includes ``dp_shard`` and ``cp``. Note that
                     we always assume that when ``cp`` is used, FSDP is also applied to
                     utilize its weight all-gather and gradients reduce_scatter even if
                     there may be no data parallelism (e.g., global batch size is 1).
            cp:      Context Parallelism (CP).
            tp:      Tensor Parallelism (TP).
            ep:      Expert Parallelism (EP).
            efsdp:   FSDP in the EP region.
            etp:     TP in the EP region.

        Note: Most dimensions above are created by unflattening the world mesh, except for loss,
        which is created by flattening the batch and cp dimensions.
        This API performs the following unflatten operations from the world mesh:

            ["pp", "batch", "cp", "tp"]  # dataloading_mesh
            ["pp", "dp_replicate", "fsdp", "tp"]  # dense_mesh
            ["pp", "dp_replicate", "efsdp", "ep", "etp"]  # sparse_mesh

        Note: DeviceMesh currently recreates the process group for each dimension.
        It should share the process group for the same dim group to avoid unnecessary
        process group creation. We can also use Fake to achieve a similar goal.
        However, using Fake to avoid redundancy messing up the code. We only use Fake
        when it is necessary. For now, we just let DeviceMesh create redundant process
        group and wait for DeviceMesh to fix the issue.
        """

        def unflatten_mesh(
            world_mesh: DeviceMesh,
            dim_names: tuple[str, ...],
            dim_degrees: tuple[int, ...],
        ):
            """Unflatten the world mesh to create the required mesh dimensions.

            Uses fake backend for dimensions with degree 1 or for 'batch' dimension
            to avoid unnecessary process group creation.
            """
            backend_override = {}
            for name, degree in zip(dim_names, dim_degrees, strict=True):
                if not self._mesh_exist(name, degree):
                    backend_override[name] = "fake"

            return world_mesh._unflatten(
                0,
                dim_degrees,
                dim_names,
                # pyrefly: ignore [bad-argument-type]
                backend_override=backend_override,
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}, etp={self.etp}"
        )

        batch = self.dp_replicate * self.dp_shard
        fsdp = self.dp_shard * self.cp
        efsdp = fsdp * self.tp // (self.etp * self.ep)

        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "batch", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        dense_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "fsdp", "tp"),
            (self.pp, self.dp_replicate, fsdp, self.tp),
        )
        sparse_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "efsdp", "ep", "etp"),
            (self.pp, self.dp_replicate, efsdp, self.ep, self.etp),
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": dense_mesh,
            "sparse": sparse_mesh,
        }

        self._meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh,
            "dp_replicate": dense_mesh["dp_replicate"],
            "fsdp": dense_mesh["fsdp"],
            "cp": dataloading_mesh["cp"],
            "tp": dataloading_mesh["tp"],
            "ep": sparse_mesh["ep"],
            "efsdp": sparse_mesh["efsdp"],
            "etp": sparse_mesh["etp"],
        }

        # Validate mesh sizes
        self._validate_meshes()

        logger.info(
            f"Successfully created meshes with active dimensions: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        """Validate that created meshes have the expected sizes."""
        expected_sizes = {
            "pp": self.pp,
            "batch": self.dp_replicate * self.dp_shard,
            "loss": self.dp_replicate * self.dp_shard * self.cp,
            "dp_replicate": self.dp_replicate,
            "fsdp": self.dp_shard * self.cp,
            "cp": self.cp,
            "tp": self.tp,
            "ep": self.ep,
            "efsdp": self.dp_shard * self.cp * self.tp // (self.etp * self.ep),
            "etp": self.etp,
        }

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._meshes[mesh_name].size()
            assert actual_size == expected_size, (
                f"Mesh '{mesh_name}' has unexpected size: "
                f"expected {expected_size}, got {actual_size}"
            )

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Get a device mesh by dimension name(s), returning None if not enabled.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s), or None if:
            - The dimension size is 1 (parallelism not enabled)
            - The dimension doesn't exist (except efsdp which can exist even if size is 1 when ep > 1)

        Raises:
            ValueError: If the requested dimension name(s) is not valid.
        """
        if not self._meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for mesh_name in dims:
            if mesh_name not in self._meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. "
                    f"Valid dimensions are: {list(self._meshes.keys())}"
                )

        if any(not self._mesh_exist(dim, self._meshes[dim].size()) for dim in dims):
            return None

        if len(dims) == 1:
            return self._meshes[dims[0]]
        else:
            for global_mesh in self._global_meshes.values():
                assert global_mesh.mesh_dim_names is not None
                if not set(dims).issubset(set(global_mesh.mesh_dim_names)):
                    continue
                return global_mesh[tuple(dims)]
            raise ValueError(f"Invalid mesh name combinations {dims}.")

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s).

        Raises:
            ValueError: If the mesh is not available (dimension size = 1 or not enabled),
                or if the requested dimension name(s) is not valid.
        """
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            enabled_str = (
                "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            )
            raise ValueError(
                f"Mesh '{dims}' is not available. "
                f"Ensure the corresponding parallelism dimension is {enabled_str}."
            )
        return mesh

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        """Get all enabled one-dimensional device meshes.

        Returns a dictionary of enabled one-dimensional device meshes, allowing you to
        access their process groups.

        Note:
            Device meshes created with the Fake backend are still included in the results.

        Returns:
            dict[str, DeviceMesh]: A dictionary mapping mesh dimension names to their
                corresponding DeviceMesh objects. Only includes meshes where:
                - ndim == 1 (one-dimensional)
                - parallelism is enabled (size > 1)

        Example:
            >>> parallel_dims = ParallelDims(
            ...     dp_replicate=2, dp_shard=2, cp=1, tp=2, pp=1, ep=1, etp=1, world_size=8
            ... )
            >>> meshes = parallel_dims.get_all_one_dimensional_meshes()
            >>> print(meshes.keys())
            dict_keys(['dp_replicate', 'fsdp', 'tp', 'batch', 'loss', 'efsdp'])
        """
        if not self._meshes:
            self.build_mesh()
        return {k: v for k, v in self._meshes.items() if v.ndim == 1 and v.size() > 1}

    @property
    def world_mesh(self) -> DeviceMesh:
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

    @property
    def etp_enabled(self):
        return self.etp > 1

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        return self.dp_replicate * self.dp_shard * self.cp

    @property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # https://github.com/pytorch/torchtitan/pull/640#discussion_r1849481001

        # Context Parallel requires that seq_len be divisible by 2 * CP degree,
        # when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.tp * (self.cp * 2)
