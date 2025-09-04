# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
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

    _world_mesh: DeviceMesh = None
    intermediate_num: int = 0

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

    def _merge_mesh_config(
        self,
        config_l: dict[str, int],
        config_r: dict[str, int],
        inter_flatten_names_map_list: list[dict[str, list[str]]],
    ) -> tuple[list[int], list[str]]:
        final_mesh_shape: list[int] = []
        final_mesh_dim_names: list[str] = []
        it_l = iter(config_l.items())
        it_r = iter(config_r.items())
        val_l: int = 0
        val_r: int = 0
        name_l: str = ""
        name_r: str = ""
        while True:
            if not val_l and (next_item := next(it_l, None)):
                name_l, val_l = next_item
            if not val_r and (next_item := next(it_r, None)):
                name_r, val_r = next_item
            if not bool(val_l) and not bool(val_r):
                return final_mesh_shape, final_mesh_dim_names
            if bool(val_l) ^ bool(val_r):
                raise ValueError("Cannot merge two mesh configuration")
            if val_l == val_r:
                # Case one: both mesh dim names are the same and same dim shape
                if val_l == val_r:
                    final_mesh_shape.append(val_l)
                    final_mesh_dim_names.append(name_l)
                # Case two: both mesh dim names happen to have same dim shape, we unflatten it into (n, 1)
                else:
                    final_mesh_shape.append(val_l)
                    final_mesh_shape.append(1)
                    final_mesh_dim_names.append(
                        f"inter_dim_{ParallelDims.intermediate_num}"
                    )
                    ParallelDims.intermediate_num += 1
                    final_mesh_dim_names.append(
                        f"inter_dim_{ParallelDims.intermediate_num}"
                    )
                    inter_flatten_names_map_list.append(
                        {
                            name_l: [
                                f"inter_dim_{ParallelDims.intermediate_num}",
                                f"inter_dim_{ParallelDims.intermediate_num-1}",
                            ],
                            name_r: [
                                f"inter_dim_{ParallelDims.intermediate_num}",
                                f"inter_dim_{ParallelDims.intermediate_num-1}",
                            ],
                        }
                    )
                    ParallelDims.intermediate_num += 1
                val_l = 0
                val_r = 0
            else:
                # Case three: we need to further unflatten one of the mesh dim.
                # If left and right (after divided by their gcd) are coprime with each other,
                # there is no way we can unflatten into a common mesh configuration.
                gcd = math.gcd(val_l, val_r)
                if gcd != min(val_l, val_r):
                    raise ValueError(
                        "Cannot merge two mesh configuration because there are dims which are coprime with each other."
                    )
                elif gcd == val_l:
                    final_mesh_shape.append(val_l)
                    final_mesh_dim_names.append(name_l)
                    val_r = val_r // gcd
                    inter_flatten_names_map_list.append(
                        {
                            name_r: [
                                f"inter_dim_{ParallelDims.intermediate_num}",
                                name_l,
                            ],
                        }
                    )
                    name_r = f"inter_dim_{ParallelDims.intermediate_num}"
                    ParallelDims.intermediate_num += 1
                else:
                    final_mesh_shape.append(val_r)
                    final_mesh_dim_names.append(name_r)
                    val_l = val_l // gcd
                    inter_flatten_names_map_list.append(
                        {
                            name_r: [
                                f"inter_dim_{ParallelDims.intermediate_num}",
                                name_l,
                            ],
                        }
                    )
                    name_r = f"inter_dim_{ParallelDims.intermediate_num}"
                    ParallelDims.intermediate_num += 1

    def _build_mesh_with_mega_ctr(
        self,
        device_type: str,
        mesh_configurations: list[dict[str, int]],
        flatten_names_map: list[dict[str, list[str]]],
    ) -> DeviceMesh:
        """
        Build a mesh with multiple mesh configurations and flatten names map.
        The flatten names map is a list of maps, each map is a mapping from a mesh dim name to a list of flatten names.
        One possible example can be:
        mesh_configurations = {
            {
            "pp": 2,
            "cp": 4,
            "dp_shard": 4,
            "dp_replicate": 2,
            "tp": 8,
            },
            {
            "pp": 2,
            "dp": 4,
            "ep": 16,
            "ep_tp": 4,
            }
        }
        flatten_names_map = {
            "dp": {
                "dp_shard",
                "dp_replicate",
            }
            "loss_update": {
                "cp",
                "dp",
            }
            "dp_shard_cp": {
                "cp",
                "dp_shard",
            }
            ...
        }
        """
        logger.info(
            f"Building a mesh with {len(mesh_configurations)} layouts with {mesh_configurations}, {flatten_names_map}"
        )
        assert (
            len(mesh_configurations) >= 1
        ), "mesh_configurations should have at least one map."
        assert (
            len({len(c) for c in mesh_configurations}) == 1
        ), "All maps within mesh_configurations should be equal."
        flatten_names = {
            name
            for flatten_map in flatten_names_map
            for names in flatten_map.values()
            for name in names
        }
        valid_names = {
            name for flatten_map in flatten_names_map for name in flatten_map.keys()
        } | {key for c in mesh_configurations for key in c.keys()}
        assert (
            flatten_names <= valid_names
        ), f"Invalid dim names {flatten_names - valid_names} are specified in flatten_names_map"

        # For now we only support that all mesh configurations can be unflatten into one common mesh configuration
        # For example, config [8, 4, 4] and [4, 2, 2, 8] can be all flatten from [4, 2, 2, 2, 4] but [5, 2] and [2, 5] cannot.
        final_mesh_shape: list[int] = []
        final_mesh_dim_names: list[str] = []
        inter_flatten_names_map_list: list[dict[str, list[str]]] = []
        if len(mesh_configurations) == 1:
            final_mesh_shape = list(mesh_configurations[0].values())
            final_mesh_dim_names = list(mesh_configurations[0].keys())
        else:
            config_iter = iter(mesh_configurations)
            first = next(config_iter)
            second = next(config_iter)
            final_mesh_shape, final_mesh_dim_names = self._merge_mesh_config(
                first, second, inter_flatten_names_map_list
            )
            while next_one := next(config_iter, None):
                final_mesh_shape, final_mesh_dim_names = self._merge_mesh_config(
                    dict(zip(final_mesh_dim_names, final_mesh_shape)),
                    next_one,
                    inter_flatten_names_map_list,
                )

        logger.info(
            f"Building intermediate {len(final_mesh_shape)}-D device mesh with {final_mesh_dim_names}, {final_mesh_shape}"
        )
        mesh = init_device_mesh(
            device_type, final_mesh_shape, mesh_dim_names=final_mesh_dim_names
        )
        inter_flatten_names_map_list.reverse()
        for flatten_map in inter_flatten_names_map_list:
            for key, din_names in flatten_map.items():
                mesh[tuple(din_names)]._flatten(mesh_dim_name=key)
        for flatten_map in flatten_names_map:
            for key, din_names in flatten_map.items():
                mesh[tuple(din_names)]._flatten(mesh_dim_name=key)

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
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
