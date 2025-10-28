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

    def build_mesh(self) -> DeviceMesh:
        """
        Build the device mesh with the required mesh dimensions.

        The following mesh dimensions will be created:

            pp:      Pipeline Parallelism (PP).
            batch:   Used by data loading to determine the global batch size and which
                     part of the data each rank should read. This dimension includes both
                     ``dp_replicate`` and ``dp_shard``. The backend is set to ``fake`` for
                     this dimension to avoid unnecessary process group creation.
            loss:    Used by all-reduce when computing the loss. Includes ``dp_replicate``,
                     ``dp_shard``, and ``cp`` degrees, as all are data parallelisms.
            dp_replicate: For DDP or HSDP replicate dimension.
            fsdp:    For FSDP dimension. This includes ``dp_shard`` and ``cp``.
            cp:      Context Parallelism (CP).
            tp:      Tensor Parallelism (TP).
            ep:      Expert Parallelism (EP).
            efsdp:   FSDP in the EP region.
            etp:     TP in the EP region.

        Note: All the dimensions above are created by unflattening the world mesh.
        This API performs the following unflatten operations:

            ["pp", "batch", "cp", "tp"]
            ["pp", "loss", "tp"]
            ["pp", "dp_replicate", "fsdp", "tp"]
            ["pp", "dp_replicate", "efsdp", "ep", "etp"]

        Note: DeviceMesh currently recreates the process group for each dimension.
        It should share the process group for the same dim group to avoid unnecessary
        process group creation.
        """

        def unflatten_mesh(
            world_mesh: DeviceMesh, dim_names: tuple[str], dim_degrees: tuple[int]
        ):
            """Unflatten the world mesh to create the required mesh dimensions.

            Uses fake backend for dimensions with degree 1 or for 'batch' dimension
            to avoid unnecessary process group creation.
            """
            backend_override = {}
            for name, degree in zip(dim_names, dim_degrees, strict=True):
                if degree == 1 or name == "batch":
                    backend_override[name] = "fake"

            return world_mesh._unflatten(
                0, dim_degrees, dim_names, backend_override=backend_override
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}, etp={self.etp}"
        )

        batch = self.dp_replicate * self.dp_shard
        loss = self.dp_replicate * self.dp_shard * self.cp
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
        loss_mesh = dataloading_mesh["batch", "cp"].flatten("loss_mesh")
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

        self._meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh["loss"],
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
            f"{list(self.get_all_meshes().keys())}"
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

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Get a device mesh by dimension names.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'etp', 'efsdp'

        Returns:
            DeviceMesh for the requested dimension(s), or None if any of
            dimension(s) has size 1 (i.e., parallelism is disabled for that dimension).

        Raises:
            ValueError: If the requested dimension name(s) is not valid.
        """
        if not self._meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        if not all(dim in self._meshes for dim in dims):
            valid_dims = sorted(self._meshes.keys())
            raise ValueError(
                f"Invalid mesh dim: '{dims}'. Valid dimensions are: {valid_dims}"
            )

        if any(self._meshes[dim].size() == 1 for dim in dims):
            return None

        meshes = [self._meshes[dim] for dim in dims]
        return meshes[0] if len(meshes) == 1 else DeviceMesh._concatenate(meshes)

    def get_all_meshes(self) -> dict[str, DeviceMesh]:
        if not self._meshes:
            self.build_mesh()
        return {k: v for k, v in self._meshes.items() if v.size() > 1}

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self.build_mesh()
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
