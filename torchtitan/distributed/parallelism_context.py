# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device mesh construction for training parallelisms.

See ``torchtitan/distributed/PARALLELISM_CONTEXT.md`` for axis names, the world_size
product, and how EP reuses ranks from the dense mesh.
"""

from __future__ import annotations

import contextlib
import logging

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torchtitan.tools.utils import device_type

if TYPE_CHECKING:
    from torchtitan.config.parallelism import ParallelismConfig

logger = logging.getLogger(__name__)


__all__ = [
    "MeshAxisName",
    "DistributedTopology",
    "ParallelismContext",
    "unfold_dp_axis",
    "unfold_dp_axes",
]


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
    TP = "tp"
    CP = "cp"
    PP = "pp"
    EP = "ep"
    EDP_SHARD = "edp_shard"


@dataclass(frozen=True)
class DistributedTopology:
    """Logical world and optional real PP group supplied by initialization.

    Args:
        world_size: Logical world size used to construct model meshes.
        real_pp_group_for_fake_spmd: Real PP process group in an otherwise fake
            logical world. ``None`` means all mesh axes use the default group
            backend.
    """

    world_size: int
    real_pp_group_for_fake_spmd: dist.ProcessGroup | None = None


def unfold_dp_axis(axis: MeshAxisName | str) -> tuple[MeshAxisName, ...]:
    """Expand logical ``dp`` into concrete dense storage mesh axes."""
    axis_name = MeshAxisName(axis)
    if axis_name == MeshAxisName.DP:
        return (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD)
    return (axis_name,)


def unfold_dp_axes(axes: Iterable[MeshAxisName | str]) -> list[str]:
    """Expand logical ``dp`` into concrete dense storage mesh axes."""
    return [
        concrete_axis.value for axis in axes for concrete_axis in unfold_dp_axis(axis)
    ]


@dataclass
class ParallelismContext:
    """Parallelism degrees and the device meshes derived from them."""

    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int
    enable_sequence_parallel: bool
    _real_pp_group_for_fake_spmd: dist.ProcessGroup | None = None
    # Cache by axis name(s); DeviceMesh equality is by identity, so reuse the
    # same object instead of re-slicing a submesh on every lookup.
    _single_axis_meshes: dict[str, DeviceMesh] = field(
        default_factory=dict, init=False, repr=False
    )
    _multi_axis_meshes: dict[tuple[str, ...], DeviceMesh] = field(
        default_factory=dict, init=False, repr=False
    )
    _global_meshes: dict[str, DeviceMesh] = field(
        default_factory=dict, init=False, repr=False
    )
    _world_mesh: DeviceMesh | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        parallelism_config: ParallelismConfig,
        topology: DistributedTopology,
    ) -> ParallelismContext:
        """Construct the parallelism context from config and topology."""
        return cls(
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            dp_shard=parallelism_config.data_parallel_shard_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            world_size=topology.world_size,
            enable_sequence_parallel=parallelism_config.enable_sequence_parallel,
            _real_pp_group_for_fake_spmd=topology.real_pp_group_for_fake_spmd,
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
            f"Invalid parallelism: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        sparse_region = dp_shard * cp * tp
        if sparse_region % ep != 0:
            raise ValueError(
                f"expert_parallel_degree ({ep}) must divide "
                f"dp_shard * cp * tp ({sparse_region})"
            )

    def _axis_is_active(self, axis: str, degree: int) -> bool:
        if axis == "dp_shard":
            # Keep the DP storage axis alive at size 1 so ``fully_shard`` can
            # install MixedPrecisionPolicy and discriminate the DP submesh on
            # TP/DDP/PP-only.
            return True
        if axis == "edp_shard":
            # Keep the expert DP shard axis alive whenever EP is enabled so FSDP
            # can apply mixed precision to routed experts, including at size 1.
            return self.ep > 1
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        """
        Build the required device-mesh views.

        The following mesh axes will be created:

            pp:      Pipeline Parallelism (PP).
            dp:      Logical data parallelism used by data loading, forward/backward,
                     and data-parallel reductions. This axis folds ``dp_replicate``
                     and ``dp_shard``.
            loss:    Used by all-reduce when computing the loss. Includes ``dp_replicate``,
                     ``dp_shard``, and ``cp`` degrees, as all of them parallelize the data,
                     essentially require the weight gradients reduction.
            dp_replicate: DDP or HSDP replicate axis.
            cp:      Context Parallelism (CP).
            tp:      Tensor Parallelism (TP).
            ep:      Expert Parallelism (EP).
            edp_shard: FSDP shard axis in the expert-parallel region.

        Note: Most axes above are created by unflattening the world mesh, except for loss,
        which is created by flattening the dp and cp axes.
        This API performs the following unflatten operations from the world mesh:

            ["pp", "dp_replicate", "dp_shard", "cp", "tp"]  # storage mesh
            ["pp", "dp", "cp", "tp"]  # fwd/bwd dense mesh
            ["pp", "dp_replicate", "edp_shard", "ep"]  # sparse_mesh

        Note: DeviceMesh currently recreates the process group for each axis.
        It should share the process group for the same axis group to avoid unnecessary
        process group creation. We can also use Fake to achieve a similar goal.
        However, using Fake to avoid redundancy messing up the code. We only use Fake
        when it is necessary. For now, we just let DeviceMesh create redundant process
        group and wait for DeviceMesh to fix the issue.
        """

        def unflatten_mesh(
            world_mesh: DeviceMesh,
            axis_names: tuple[str, ...],
            axis_degrees: tuple[int, ...],
        ):
            """Unflatten the world mesh to create the required mesh axes.

            Uses the fake backend for inactive axes to avoid unnecessary
            process-group creation.
            """
            backend_override = {}
            for axis, degree in zip(axis_names, axis_degrees, strict=True):
                if not self._axis_is_active(axis, degree):
                    backend_override[axis] = "fake"

            return world_mesh._unflatten(
                0,
                axis_degrees,
                axis_names,
                backend_override=backend_override,
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}"
        )

        batch = self.dp_replicate * self.dp_shard
        edp_shard = self.dp_shard * self.cp * self.tp // self.ep

        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )
        # Two mesh views over the same devices:
        #
        # full_dense_mesh_for_fsdp (dp_replicate, dp_shard, cp, tp) is passed to
        # fully_shard() so FSDP can shard parameters along dp_shard.
        # spmd_dense_mesh_for_fwdbwd (dp, cp, tp) is used for forward/backward
        # typechecking, with dp folding dp_replicate * dp_shard into one axis.
        full_dense_mesh_for_fsdp = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "dp_shard", "cp", "tp"),
            (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp),
        )
        full_dense_mesh_for_fwdbwd = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )
        spmd_dense_mesh_for_fwdbwd = full_dense_mesh_for_fwdbwd["dp", "cp", "tp"]
        loss_mesh = full_dense_mesh_for_fwdbwd["dp", "cp"]._flatten("loss_mesh")

        full_sparse_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "edp_shard", "ep"),
            (self.pp, self.dp_replicate, edp_shard, self.ep),
        )

        self._global_meshes = {
            "loss": loss_mesh,
            "dense": full_dense_mesh_for_fsdp,
            "sparse": full_sparse_mesh,
        }
        self._global_meshes["spmd_dense_for_fwdbwd"] = spmd_dense_mesh_for_fwdbwd
        if self.ep > 1:
            self._global_meshes["spmd_sparse_for_fwdbwd"] = full_sparse_mesh[
                "dp_replicate", "edp_shard", "ep"
            ]
        pp_mesh = full_dense_mesh_for_fwdbwd["pp"]
        pp_group = self._real_pp_group_for_fake_spmd
        if pp_group is not None:
            if dist.get_world_size(pp_group) != self.pp:
                raise ValueError(
                    "The real PP process group size must match the configured PP "
                    f"degree: {dist.get_world_size(pp_group)} != {self.pp}"
                )
            pp_mesh = DeviceMesh.from_group(
                pp_group,
                device_type,
                mesh_dim_names=(MeshAxisName.PP.value,),
            )

        self._single_axis_meshes = {
            "pp": pp_mesh,
            "loss": loss_mesh,
            "dp_replicate": full_dense_mesh_for_fsdp["dp_replicate"],
            "cp": full_dense_mesh_for_fwdbwd["cp"],
            "tp": full_dense_mesh_for_fwdbwd["tp"],
            "ep": full_sparse_mesh["ep"],
            "edp_shard": full_sparse_mesh["edp_shard"],
        }
        self._single_axis_meshes["dp"] = spmd_dense_mesh_for_fwdbwd["dp"]
        self._single_axis_meshes["dp_shard"] = full_dense_mesh_for_fsdp["dp_shard"]

        self._validate_meshes()

        logger.info(
            f"Successfully created meshes with active axes: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        """Validate that created meshes have the expected sizes."""
        expected_sizes = {
            "pp": self.pp,
            "loss": self.dp_replicate * self.dp_shard * self.cp,
            "dp_replicate": self.dp_replicate,
            "cp": self.cp,
            "tp": self.tp,
            "ep": self.ep,
            "edp_shard": self.dp_shard * self.cp * self.tp // self.ep,
        }
        expected_sizes["dp"] = self.dp_replicate * self.dp_shard
        expected_sizes["dp_shard"] = self.dp_shard

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._single_axis_meshes[mesh_name].size()
            assert actual_size == expected_size, (
                f"Mesh '{mesh_name}' has unexpected size: "
                f"expected {expected_size}, got {actual_size}"
            )

    def get_optional_mesh(
        self,
        axes: str | list[str],
        *,
        include_singleton_axes: bool = False,
    ) -> DeviceMesh | None:
        """Get a device mesh by axis name(s), returning None if not enabled.

        Args:
            axes: Names of the mesh axes. Valid options include:
                 'pp', 'loss', 'dp_replicate', 'dp', 'dp_shard',
                 'cp', 'tp', 'ep', 'edp_shard'.
            include_singleton_axes: Include axes with size 1 in the returned
                 submesh. This is used for distributed parameter and buffer
                 registration so spmd_types can handle size-1 axis filtering.

        Returns:
            DeviceMesh for the requested axes, or None if:
            - An axis is not active
            - An axis does not exist
            Note: 'dp_shard' always exists (for mixed precision via
            fully_shard()), and 'edp_shard' exists when ep > 1, even if their
            size is 1.

        Raises:
            ValueError: If a requested axis name is not valid.
        """
        if not self._single_axis_meshes:
            self.build_mesh()

        if isinstance(axes, str):
            axes = [axes]

        for mesh_name in axes:
            if mesh_name not in self._single_axis_meshes:
                raise ValueError(
                    f"Invalid mesh axis: '{mesh_name}'. "
                    f"Valid axes are: {list(self._single_axis_meshes.keys())}"
                )

        if not include_singleton_axes and any(
            not self._axis_is_active(axis, self._single_axis_meshes[axis].size())
            for axis in axes
        ):
            return None

        if len(axes) == 1:
            return self._single_axis_meshes[axes[0]]

        # Cache to ensure mesh equality by object identity.
        key = tuple(axes)
        if key in self._multi_axis_meshes:
            return self._multi_axis_meshes[key]

        candidates = [
            (name, global_mesh)
            for name, global_mesh in self._global_meshes.items()
            if global_mesh.mesh_dim_names is not None
            and set(axes).issubset(set(global_mesh.mesh_dim_names))
        ]
        if not candidates:
            raise ValueError(f"Invalid mesh axis combination {axes}.")
        submesh = candidates[0][1][key]
        self._multi_axis_meshes[key] = submesh
        return submesh

    def get_mesh(self, axes: str | list[str]) -> DeviceMesh:
        """Get a device mesh by axis name(s), raising if not available.

        Args:
            axes: Names of the mesh axes. Valid options include:
                 'pp', 'loss', 'dp_replicate', 'dp', 'dp_shard',
                 'cp', 'tp', 'ep', 'edp_shard'.

        Returns:
            DeviceMesh for the requested axes.

        Raises:
            ValueError: If the mesh is not available or an axis name is invalid.
        """
        mesh = self.get_optional_mesh(axes)
        if mesh is None:
            enabled_str = "active" if isinstance(axes, str) else "all active"
            raise ValueError(
                f"Mesh '{axes}' is not available. "
                f"Ensure the corresponding parallelism axes are {enabled_str}."
            )
        return mesh

    @contextlib.contextmanager
    def activate_spmd(self, *, typechecking: bool = False) -> Iterator[None]:
        """Activate this context's meshes for an SPMD runtime region."""
        # Import lazily because spmd_types imports ParallelismContext for its
        # mesh resolution helpers.
        from torchtitan.distributed.spmd_types import (
            set_current_spmd_mesh,
            set_spmd_meshes,
        )

        dense_mesh = self.spmd_dense_mesh()
        set_spmd_meshes(
            dense_mesh=dense_mesh,
            sparse_mesh=self.spmd_sparse_mesh(),
            dense_sp_enabled=self.sp_enabled,
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(set_current_spmd_mesh(dense_mesh))
            if typechecking:
                from spmd_types.checker import typecheck

                stack.enter_context(typecheck(local=False))
            yield

    def spmd_dense_mesh(self) -> DeviceMesh:
        """Dense SPMD mesh used for forward/backward typechecking."""
        if not self._single_axis_meshes:
            self.build_mesh()
        return self._global_meshes["spmd_dense_for_fwdbwd"]

    def spmd_sparse_mesh(self) -> DeviceMesh | None:
        """Sparse SPMD mesh used inside expert dispatch."""
        if not self._single_axis_meshes:
            self.build_mesh()
        return self._global_meshes.get("spmd_sparse_for_fwdbwd")

    def get_dense_tp_mesh(self) -> DeviceMesh:
        """Return the TP-axis mesh used by dense forward/backward computation."""
        return self.spmd_dense_mesh()["tp"]

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

        Given the axes, query ``parallelism_context`` for the corresponding SPMD
        mesh (dense or sparse).

        ``axes`` is always a superset of the resolved mesh's axes: we always
        specify every axis. Unsupported axes are dropped.

        Returns ``None`` when none of the kept axes is enabled.
        """
        in_band = ("dp", "cp", "tp", "ep")
        axes_list = [
            axis.value if isinstance(axis, MeshAxisName) else axis for axis in axes
        ]
        return self.get_activated_mesh([axis for axis in axes_list if axis in in_band])

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        """Get all enabled one-dimensional device meshes.

        Returns a dictionary of enabled one-dimensional device meshes, allowing you to
        access their process groups.

        Note:
            Axes that ``build_mesh`` created with the Fake backend are excluded,
            because their process groups cannot carry collectives. For example,
            ``edp_shard`` when EP is disabled: its size is ``dp_shard * cp * tp``,
            but ``_axis_is_active`` marks it inactive so it is unflattened with a
            fake backend.

        Returns:
            dict[str, DeviceMesh]: A dictionary mapping mesh axis names to their
                corresponding DeviceMesh objects. Only includes meshes where:
                - ndim == 1 (one-dimensional)
                - parallelism is enabled (size > 1)
                - the axis exists, i.e. it is not backed by the Fake backend

        Example:
            >>> parallelism_context = ParallelismContext(
            ...     dp_replicate=2, dp_shard=2, cp=1, tp=2, pp=1, ep=1,
            ...     world_size=8, enable_sequence_parallel=True
            ... )
            >>> meshes = parallelism_context.get_all_one_dimensional_meshes()
            >>> print(meshes.keys())
            dict_keys(['loss', 'dp_replicate', 'tp', 'dp', 'dp_shard'])

        """
        if not self._single_axis_meshes:
            self.build_mesh()
        return {
            k: v
            for k, v in self._single_axis_meshes.items()
            if v.ndim == 1 and v.size() > 1 and self._axis_is_active(k, v.size())
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
    def sp_enabled(self):
        return self.tp_enabled and self.enable_sequence_parallel

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
