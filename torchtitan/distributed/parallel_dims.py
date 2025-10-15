# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import dataclass

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
    mesh_dim_names: tuple[str] = tuple()

    _world_mesh: DeviceMesh = None

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
            if etp == tp:
                # EP would borrow all cp and some dp_shard degree
                assert ep % cp == 0 and (dp_shard * cp) % ep == 0
            elif etp == 1:
                # EP would borrow all cp and tp and some dp_shard degree
                assert ep % (cp * tp) == 0 and (dp_shard * cp * tp) % ep == 0

    def build_mesh(self) -> "ParallelDims":
        """Build the device mesh with the required mesh dimensions.

        The following mesh dimensions may be created based on the parallel configuration:

            pp: For PP.
            dp_replicate: For DDP or HSDP replicate dimension.
            dp_shard_cp: For FSDP or HSDP shard dimension. This includes
                         ``cp`` even if ``cp`` is 1. As a result, we always
                         use the name ``dp_shard_cp``, and ``dp_shard`` is not
                         created as a dimension.
            dp_cp: This is used by loss all-reduce. It includes ``dp_replicate``,
                   ``dp_shard``, and ``cp`` as all of them are data parallelisms.
            dp: This is used by data loading to decide the global batch size and
                which part of data this raunk should read.  This dim includes both
                ``dp_replicate`` and ``dp_shard``.
                The name is confusing; ``batch`` could be a better name.
            cp: For CP.
            tp: For TP.
            ep: For EP.
            dp_shard_in_ep: For FSDP or HSDP shard dimension in the EP region.

        Note: These dimensions won't exist at the same time. If we consider
        the unflatten() operator only, the following are all the meshes required
        assuming all degrees are > 1 except for ``pp``:

            ["dp", "cp", "tp"]: The ``dp`` process group is wasted as the dataloader
                                doesn't need it for communication.
            ["dp_cp", "tp"]: Loss computation.
            ["dp_replicate", "dp_shard_cp", "tp"]: Non-EP region computation.
            ["dp_replicate", "dp_shard_in_ep", "ep", "tp"]: EP region computation if etp == tp.
            ["dp_replicate", "dp_shard_in_ep", "ep"]: EP region computation if etp == 1.

        In reality, we don't actually need to create all of these meshes.
        For example, ``dp_cp`` can be sliced and flattened from ["dp", "cp", "tp"].
        So we don't actually need to create ["dp_cp", "tp"].

        But there are some meshes we MUST create if that mesh will be used for a
        parameter. So Non-EP-region-computation mesh and EP-region-computation mesh
        are required.
        """

        def add_dim(name, degree, config):
            config["name"].append(name)
            config["degree"].append(degree)

        world_mesh = init_device_mesh(device_type, [self.world_size])
        dp_shard_in_ep = (
            self.dp_shard * self.cp // self.ep
            if self.etp == self.tp
            else self.dp_shard * self.cp * self.tp // self.ep
        )

        data_mesh_dims = defaultdict(list)
        non_ep_computation_dims = defaultdict(list)
        ep_computation_dims = defaultdict(list)

        if self.pp_enabled:
            add_dim("pp", self.pp, data_mesh_dims)
            add_dim("pp", self.pp, non_ep_computation_dims)
            add_dim("pp", self.pp, ep_computation_dims)

        if self.dp_enabled:
            add_dim("dp", self.dp_replicate * self.dp_shard, data_mesh_dims)
            if self.dp_replicate_enabled:
                add_dim("dp_replicate", self.dp_replicate, non_ep_computation_dims)
                add_dim("dp_replicate", self.dp_replicate, ep_computation_dims)
            if self.dp_shard_enabled:
                add_dim("dp_shard_cp", self.dp_shard * self.cp, non_ep_computation_dims)
                add_dim("dp_shard_in_ep", dp_shard_in_ep, ep_computation_dims)

        if self.cp_enabled:
            add_dim("cp", self.cp, data_mesh_dims)

        if self.tp_enabled:
            add_dim("tp", self.tp, data_mesh_dims, non_ep_computation_dims)
            if self.etp == self.tp:
                add_dim("tp", self.tp, ep_computation_dims)

        self._all_meshes = []

        if self.dp_enabled:
            data_mesh = world_mesh._unflatten(
                0, data_mesh_dims["degree"], data_mesh_dims["name"]
            )
            self._all_meshes.append(data_mesh)
            # Note that we don't create loss_mesh as it is easier to flatten
            # from data_mesh
            if self.cp_enabled:
                self._all_meshes[-1]["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
            else:
                self._all_meshes[-1]["dp"]._flatten(mesh_dim_name="dp_cp")

        if self.dp_cp_enabled or self.tp_enabled or self.pp_enabled:
            self._all_meshes.append(
                world_mesh._unflatten(
                    0,
                    non_ep_computation_dims["degree"],
                    non_ep_computation_dims["name"],
                )
            )

        if self.ep_enabled:
            add_dim("ep", self.ep, ep_computation_dims)
            self._all_meshes.append(
                world_mesh._unflatten(
                    0, ep_computation_dims["degree"], ep_computation_dims["name"]
                )
            )

        self._world_mesh = world_mesh
        self.mesh_dim_names = tuple(
            name for m in self._all_meshes for name in m.mesh_dim_names
        )
        return self

    def __getitem__(self, name):
        # This is a hack to make ParallelDims behave like a DeviceMesh.
        # We will need to change trainer if design is concluded. For now,
        # this is just a quick hack to make it work with unflatten()

        if "mesh_dim_names" == name:
            return [name for m in self._all_meshes for name in m.mesh_dim_names]

        for mesh in self._all_meshes:
            try:
                submesh = mesh[name]
                return submesh
            except KeyError:
                pass
        raise AttributeError(f"ParallelDims has no attribute {name}")

    """
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
        if self.etp == self.tp:
            # ep = dp_shard_in_ep * cp
            dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
            dp_shard_in_ep = self.ep // self.cp
        else:
            assert self.etp == 1
            # ep = dp_shard_in_ep * cp * tp
            dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep
            dp_shard_in_ep = self.ep // (self.cp * self.tp)

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
        if self.etp == 1 and self.tp_enabled:
            ep_mesh_dim_names.append("tp")

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

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        return mesh
    """

    @property
    def world_mesh(self) -> "ParallelDims":
        # This is a hack to make ParallelDims behave like a DeviceMesh.
        # We will need to change trainer if design is concluded. For now,
        # this is just a quick hack to make it work with unflatten()

        # doing late init so ParallelDims can still be used as a lightweight
        # dataclass without having to initialize the world mesh
        if self._world_mesh is None:
            self.build_mesh()
        return self

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
