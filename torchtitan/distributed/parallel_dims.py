# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import spmd_types as spmd
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchtitan.config.configs import ParallelismConfig
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type


__all__ = ["MeshAxisName", "ParallelDims", "SpmdLayout", "unfold_dp_axes"]


class StrEnum(str, Enum):
    """str + Enum for Python < 3.11 compatibility."""

    pass


class MeshAxisName(StrEnum):
    """Names for axes of a ``DeviceMesh``.

    Naming convention: throughout torchtitan code, comments, and docstrings
    we say ``axis`` for a ``DeviceMesh`` axis and ``dim`` for a tensor
    dimension. This avoids the ambiguity of ``dim`` referring to both.

    Note that PyTorch upstream's ``DeviceMesh`` API still uses the older
    ``mesh_dim_names`` attribute and ``mesh_dim`` parameter names; we keep
    those exact spellings when calling into PyTorch APIs (we cannot rename
    upstream surface), but use ``axis`` for any name we own.
    """

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    FSDP = "fsdp"
    TP = "tp"
    CP = "cp"
    PP = "pp"
    EP = "ep"
    EFSDP = "efsdp"


@dataclass(frozen=True, slots=True)
class SpmdLayout:
    """Temporary SPMD layout annotations keyed by logical mesh axis name.

    TODO(pianpwk): Replace this with ``spmd_types.SpmdLayout`` once that API is
    available in TorchTitan's minimum ``spmd_types`` version.
    """

    axis_types: dict[MeshAxisName, spmd.PerMeshAxisSpmdType]
    partition_spec: spmd.PartitionSpec | tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        sharded_dims: dict[int, MeshAxisName] = {}
        for axis_name, axis_type in self.axis_types.items():
            if not isinstance(axis_type, spmd.Shard):
                continue
            if self.partition_spec is not None:
                raise ValueError(
                    "SpmdLayout with PartitionSpec should use spmd.V instead "
                    "of spmd.S(dim) in per-axis-types, and express tensor dim "
                    "sharding in the provided PartitionSpec."
                )
            if axis_type.dim in sharded_dims:
                raise ValueError(
                    "SpmdLayout has multiple mesh axes sharding tensor dim "
                    f"{axis_type.dim}; provide partition_spec to make shard "
                    "ordering explicit."
                )
            sharded_dims[axis_type.dim] = axis_name

    def axes(self) -> tuple[MeshAxisName, ...]:
        return tuple(self.axis_types)

    def per_axis_spmd_types(self) -> dict[MeshAxisName, spmd.PerMeshAxisSpmdType]:
        """
        Return per-axis types with PartitionSpec sharding represented as S(i).
        e.g. {DP: R, CP: V} + PartitionSpec(None, CP) -> {DP: R, CP: S(1)}

        This is not meant as a minimal description of the SPMD layout; shard order
        cannot be expressed. Specifically, shard order information will be lost in
        this representation. This is purely a helper for calling spmd.redistribute,
        which takes per-axis types (e.g. redistribute(S(1) -> R)).

        This manually handles ``MeshAxisName``, because spmd_types normalization
        functions often attempt to resolve to concrete runtime mesh axes, even
        without a set current mesh.
        """
        result = dict(self.axis_types)
        if self.partition_spec is not None:
            for dim, entry in enumerate(self.partition_spec):
                if entry is None:
                    continue
                axes = entry if isinstance(entry, tuple) else (entry,)
                for axis_name in axes:
                    if not isinstance(axis_name, MeshAxisName):
                        raise TypeError(
                            f"Expected MeshAxisName in partition_spec, "
                            f"got {axis_name!r}."
                        )
                    result[axis_name] = spmd.S(dim)
        return result


def unfold_dp_axes(axes: Iterable[MeshAxisName | str]) -> list[str]:
    """Expand logical ``dp`` into concrete dense storage mesh axes."""
    result: list[str] = []
    for axis in axes:
        axis_value = axis.value if isinstance(axis, MeshAxisName) else axis
        if axis_value == "dp":
            result.extend(("dp_replicate", "dp_shard"))
        else:
            result.append(axis_value)
    return result


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int
    spmd_backend: Literal["default", "full_dtensor", "spmd_types"] = "default"
    # Cache by axis name(s); DeviceMesh equality is by identity, so reuse
    # is required for ``mesh in spmd_meshes()`` checks.
    _single_axis_meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _multi_axis_meshes: dict[tuple[str, ...], DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None
    _spmd_meshes: list[DeviceMesh] = field(default_factory=list)

    @classmethod
    def from_config(
        cls, parallelism_config: ParallelismConfig, world_size: int
    ) -> ParallelDims:
        return cls(
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            dp_shard=parallelism_config.data_parallel_shard_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            world_size=world_size,
            spmd_backend=parallelism_config.spmd_backend,
        )

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp, ep = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
        )
        for d in (dp_replicate, cp, tp, pp, ep):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def _mesh_exist(self, name: str, degree: int) -> bool:
        if name == "fsdp":
            # Always keep fsdp mesh with real backend so fully_shard()
            # can apply MixedPrecisionPolicy even at degree 1.
            return True
        if name == "dp_shard" and self.spmd_backend in ("full_dtensor", "spmd_types"):
            # Under full_dtensor/spmd_types, ``dp_shard`` is the DP storage axis
            # (no flattened ``fsdp``); keep alive at size 1 so ``fully_shard``
            # can install MixedPrecisionPolicy and FSDP can discriminate the DP
            # submesh on TP/DDP/PP-only.
            return True
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

        Note: Most dimensions above are created by unflattening the world mesh, except for loss,
        which is created by flattening the batch and cp dimensions.
        This API performs the following unflatten operations from the world mesh:

            ["pp", "batch", "cp", "tp"]  # dataloading_mesh
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"]  # full_dtensor dense_mesh
            ["pp", "dp_replicate", "fsdp", "tp"]  # legacy dense_mesh
            ["pp", "dp", "cp", "tp"]  # spmd_types dense_mesh
            ["pp", "dp_replicate", "efsdp", "ep"]  # sparse_mesh

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
                backend_override=backend_override,
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}"
        )

        batch = self.dp_replicate * self.dp_shard
        fsdp = self.dp_shard * self.cp
        efsdp = fsdp * self.tp // self.ep

        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "batch", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        spmd_dense_mesh_for_fwdbwd = None
        if self.spmd_backend == "full_dtensor":
            # Under full_dtensor, ``dp_shard`` and ``cp`` cannot be folded
            # together: activations carry a ``cp`` dimension, so parameters
            # need a ``cp`` axis as well. ``fully_shard`` folds ``dp_shard``
            # and ``cp`` internally at initialization time.
            candidate_spmd_dense_axes = ["dp_replicate", "dp_shard", "cp", "tp"]
            full_dense_mesh_for_fsdp = unflatten_mesh(
                self._world_mesh,
                tuple(["pp"] + candidate_spmd_dense_axes),
                (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp),
            )
        elif self.spmd_backend == "spmd_types":
            # Two mesh views over the same devices:
            #
            # full_dense_mesh_for_fsdp (dp_replicate, dp_shard, cp, tp) -- passed to
            #   fully_shard() so FSDP can shard parameters along dp_shard.
            #   The SPMD type system never sees these axes, but they are passed
            #   to fully_shard() via DataParallelMeshDims to specify shard/replicate axes.
            #
            # spmd_dense_mesh_for_fwdbwd (dp, cp, tp) -- the mesh the SPMD type
            # system uses for forward/backward typechecking.
            # dp folds dp_replicate * dp_shard into one logical axis.
            # TODO(pianpwk): Clean up mesh construction once SPMD no longer
            # shares codepaths with DTensor/default backends.
            candidate_spmd_dense_axes = ["dp", "cp", "tp"]
            full_dense_mesh_for_fsdp = unflatten_mesh(
                self._world_mesh,
                ("pp", "dp_replicate", "dp_shard", "cp", "tp"),
                (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp),
            )
            full_dense_mesh_for_fwdbwd = unflatten_mesh(
                self._world_mesh,
                tuple(["pp"] + candidate_spmd_dense_axes),
                (self.pp, batch, self.cp, self.tp),
            )
            spmd_dense_mesh_for_fwdbwd = full_dense_mesh_for_fwdbwd["dp", "cp", "tp"]
        else:
            # Legacy path folds ``dp_shard`` and ``cp`` into ``fsdp``.
            candidate_spmd_dense_axes = ["dp_replicate", "fsdp", "tp"]
            full_dense_mesh_for_fsdp = unflatten_mesh(
                self._world_mesh,
                ("pp", "dp_replicate", "fsdp", "tp"),
                (self.pp, self.dp_replicate, fsdp, self.tp),
            )

        full_sparse_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "efsdp", "ep"),
            (self.pp, self.dp_replicate, efsdp, self.ep),
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": full_dense_mesh_for_fsdp,
            "sparse": full_sparse_mesh,
        }
        if spmd_dense_mesh_for_fwdbwd is not None:
            self._global_meshes["spmd_dense_for_fwdbwd"] = spmd_dense_mesh_for_fwdbwd
        self._single_axis_meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh,
            "dp_replicate": full_dense_mesh_for_fsdp["dp_replicate"],
            "cp": dataloading_mesh["cp"],
            "tp": dataloading_mesh["tp"],
            "ep": full_sparse_mesh["ep"],
            "efsdp": full_sparse_mesh["efsdp"],
        }
        if self.spmd_backend == "full_dtensor":
            self._single_axis_meshes["dp_shard"] = full_dense_mesh_for_fsdp["dp_shard"]
        elif self.spmd_backend == "spmd_types":
            assert spmd_dense_mesh_for_fwdbwd is not None
            self._single_axis_meshes["dp"] = spmd_dense_mesh_for_fwdbwd["dp"]
            self._single_axis_meshes["dp_shard"] = full_dense_mesh_for_fsdp["dp_shard"]
        else:
            self._single_axis_meshes["fsdp"] = full_dense_mesh_for_fsdp["fsdp"]

        self._validate_meshes()

        candidate_spmd_sparse_axes = ["dp_replicate", "efsdp", "ep"]
        activated_spmd_dense_mesh = self.get_activated_mesh(candidate_spmd_dense_axes)
        activated_spmd_sparse_mesh = self.get_activated_mesh(candidate_spmd_sparse_axes)
        self._spmd_meshes = [
            m
            for m in (activated_spmd_dense_mesh, activated_spmd_sparse_mesh)
            if m is not None
        ]

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
            "cp": self.cp,
            "tp": self.tp,
            "ep": self.ep,
            "efsdp": self.dp_shard * self.cp * self.tp // self.ep,
        }
        if self.spmd_backend == "full_dtensor":
            expected_sizes["dp_shard"] = self.dp_shard
        elif self.spmd_backend == "spmd_types":
            expected_sizes["dp"] = self.dp_replicate * self.dp_shard
            expected_sizes["dp_shard"] = self.dp_shard
        else:
            expected_sizes["fsdp"] = self.dp_shard * self.cp

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._single_axis_meshes[mesh_name].size()
            assert actual_size == expected_size, (
                f"Mesh '{mesh_name}' has unexpected size: "
                f"expected {expected_size}, got {actual_size}"
            )

    def get_optional_mesh(
        self,
        dims: str | list[str],
        *,
        include_singleton_axes: bool = False,
    ) -> DeviceMesh | None:
        """Get a device mesh by dimension name(s), returning None if not enabled.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'efsdp'.
            include_singleton_axes: Include axes with size 1 in the returned
                 submesh. Only current usecase is in spmd_types backend distributed
                 param/buffer registration (assert_type call), so that size-1 axis
                 filtering is handled internally by spmd_types.
                 TODO(pianpwk): let spmd_types handle all size-1 mesh axis filtering
                 once migration to spmd_types backend is complete.

        Returns:
            DeviceMesh for the requested dimension(s), or None if:
            - The dimension size is 1 (parallelism not enabled)
            - The dimension doesn't exist
            Note: 'fsdp' always exists (for mixed precision via fully_shard()),
            and 'efsdp' exists when ep > 1, even if their size is 1.

        Raises:
            ValueError: If the requested dimension name(s) is not valid.
        """
        if not self._single_axis_meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for mesh_name in dims:
            if mesh_name not in self._single_axis_meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. "
                    f"Valid dimensions are: {list(self._single_axis_meshes.keys())}"
                )

        if not include_singleton_axes and any(
            not self._mesh_exist(dim, self._single_axis_meshes[dim].size())
            for dim in dims
        ):
            return None

        if len(dims) == 1:
            return self._single_axis_meshes[dims[0]]

        # Cache to ensure mesh equality by object identity.
        key = tuple(dims)
        if key in self._multi_axis_meshes:
            return self._multi_axis_meshes[key]

        candidates = [
            (name, global_mesh)
            for name, global_mesh in self._global_meshes.items()
            if global_mesh.mesh_dim_names is not None
            and set(dims).issubset(set(global_mesh.mesh_dim_names))
        ]
        if not candidates:
            raise ValueError(f"Invalid mesh name combinations {dims}.")
        submesh = candidates[0][1][key]
        self._multi_axis_meshes[key] = submesh
        return submesh

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'efsdp'.

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

    def spmd_meshes(self) -> list[DeviceMesh]:
        """Valid full-SPMD meshes, restricted to enabled axes.

        Returns the full-SPMD meshes; today we have dense and sparse.
        """
        if not self._spmd_meshes:
            self.build_mesh()
        return self._spmd_meshes

    def get_activated_mesh(self, axes: list[str]) -> DeviceMesh | None:
        """Submesh of ``axes`` filtered to those actually enabled in this run.

        Returns a mesh containing the axes in ``axes`` that are enabled. If
        none of the axes in ``axes`` is enabled, returns ``None``. This
        differs from ``get_optional_mesh``, which returns ``None`` as soon
        as any axis in ``axes`` is not enabled.
        """
        if not self._single_axis_meshes:
            self.build_mesh()
        axes = [
            axis
            for axis in axes
            if axis in self._single_axis_meshes
            and self.get_optional_mesh(axis) is not None
        ]
        return self.get_optional_mesh(axes) if axes else None

    def resolve_mesh(self, axes: Iterable[MeshAxisName | str]) -> DeviceMesh | None:
        """Resolve the device mesh for a set of mesh axis names.

        Given the axes, query ``parallel_dims`` for the corresponding SPMD
        mesh (dense or sparse).

        ``axes`` is always a superset of the resolved mesh's axes: we always
        specify every axis. Under full_dtensor the resolved mesh contains
        every activated axis; under non-full_dtensor only ``tp`` and ``ep``
        are kept (DP/CP stay out-of-band).

        Returns ``None`` when no axis is enabled under non-``full_dtensor``.
        Raises ``ValueError`` under ``full_dtensor`` if the resolved mesh is
        not one of ``parallel_dims.spmd_meshes()``.
        """
        axes_list = [
            axis.value if isinstance(axis, MeshAxisName) else axis for axis in axes
        ]
        if self.spmd_backend == "default":
            in_band = ("tp", "ep")
            axes_list = [axis for axis in axes_list if axis in in_band]
        elif self.spmd_backend == "full_dtensor":
            axes_list = unfold_dp_axes(axes_list)
        elif self.spmd_backend == "spmd_types":
            in_band = ("dp", "cp", "tp", "ep")
            axes_list = [axis for axis in axes_list if axis in in_band]
        mesh = self.get_activated_mesh(axes_list)
        if mesh is None:
            return None
        assert mesh.mesh_dim_names is not None, "DeviceMesh must have named axes"
        if self.spmd_backend == "full_dtensor" and mesh not in self.spmd_meshes():
            raise ValueError(
                f"Resolved mesh {list(mesh.mesh_dim_names)} does not match any "
                f"SPMD mesh. Valid meshes: "
                f"{[list(m.mesh_dim_names or ()) for m in self.spmd_meshes()]}."
            )
        return mesh

    def resolve_shared_mesh(
        self, placements: Iterable["SpmdLayout | None"]
    ) -> DeviceMesh | None:
        """Resolve the mesh shared by a list of SpmdLayouts.

        All non-``None`` entries must reference the same axis keys (placement
        values may differ -- "redistribute on the same mesh" is exactly the
        case of same axes, different placements). ``None`` entries are
        skipped (e.g. LocalMapConfig non-tensor args, optional in/dst/grad
        placements).

        Returns ``None`` when every entry is ``None`` or when ``resolve_mesh``
        filters every axis out (legacy non-``full_dtensor`` path); callers
        should treat this as a no-op for the corresponding boundary.
        """
        non_none = [p for p in placements if p is not None]
        if not non_none:
            return None
        axes = non_none[0].axes()
        for p in non_none[1:]:
            p_axes = p.axes()
            assert p_axes == axes, (
                f"Inconsistent mesh axes within a boundary: "
                f"{sorted(k.value for k in axes)} vs "
                f"{sorted(k.value for k in p_axes)}"
            )
        return self.resolve_mesh(axes)

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
            ...     dp_replicate=2, dp_shard=2, cp=1, tp=2, pp=1, ep=1, world_size=8
            ... )
            >>> meshes = parallel_dims.get_all_one_dimensional_meshes()
            >>> print(meshes.keys())
            dict_keys(['dp_replicate', 'fsdp', 'tp', 'batch', 'loss', 'efsdp'])

        Note:
            Under ``spmd_backend="full_dtensor"`` the dense shard axis appears as
            ``'dp_shard'`` instead of the pre-flattened ``'fsdp'``.
        """
        if not self._single_axis_meshes:
            self.build_mesh()
        return {
            k: v
            for k, v in self._single_axis_meshes.items()
            if v.ndim == 1 and v.size() > 1
        }

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
