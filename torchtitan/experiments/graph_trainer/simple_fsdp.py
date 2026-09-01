# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from typing import cast

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F

from spmd_types.types import partition_spec_get_shard
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.protocols.module import Module

_active_parametrization = True


@contextmanager
def disable_active_parametrization() -> Generator[None, None, None]:
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class FSDPShardMetadata:
    """Static logical/storage shape contract for one FSDP-sharded parameter."""

    shard_dim: int
    param_ndim: int
    logical_dim_size: int
    padded_dim_size: int
    shard_degree: int

    @property
    def padding(self) -> int:
        return self.padded_dim_size - self.logical_dim_size


def _pad_tensor_dim(
    tensor: torch.Tensor,
    dim: int,
    padded_dim_size: int,
) -> torch.Tensor:
    """Right-pad one tensor dimension to ``padded_dim_size``."""
    dim = dim % tensor.ndim
    padding = padded_dim_size - tensor.shape[dim]
    if padding < 0:
        raise ValueError(
            f"Cannot pad dimension {dim} from {tensor.shape[dim]} down to "
            f"{padded_dim_size}."
        )
    if padding == 0:
        return tensor
    pad = [0] * (2 * tensor.ndim)
    pad[2 * (tensor.ndim - dim - 1) + 1] = padding
    return F.pad(tensor, tuple(pad))


def _validate_dp_storage_config(
    device_mesh: DeviceMesh,
    dp_storage_placements: tuple[Placement, ...],
    mode: str,
    shard_metadata: FSDPShardMetadata | None,
) -> None:
    """Validate explicit SPMD parameter storage at initialization time."""
    if len(dp_storage_placements) != device_mesh.ndim:
        raise ValueError(
            f"DP storage has {len(dp_storage_placements)} placements for a "
            f"{device_mesh.ndim}D mesh."
        )
    if not all(
        isinstance(placement, (Shard, Replicate)) for placement in dp_storage_placements
    ):
        raise ValueError(
            "Explicit SimpleFSDP supports only Shard and Replicate storage "
            f"placements, got {dp_storage_placements}."
        )

    shard_axes_and_placements = [
        (mesh_axis_index, placement)
        for mesh_axis_index, placement in enumerate(dp_storage_placements)
        if isinstance(placement, Shard)
    ]
    if mode == "replicate":
        if shard_axes_and_placements or shard_metadata is not None:
            raise ValueError(
                "Replicated SimpleFSDP storage must not have shard metadata."
            )
        return

    if len(shard_axes_and_placements) != 1 or shard_metadata is None:
        raise ValueError(
            f"{mode} SimpleFSDP storage requires exactly one Shard placement "
            "and its metadata."
        )
    mesh_axis_index, shard_placement = shard_axes_and_placements[0]
    storage_shard_dim = shard_placement.dim % shard_metadata.param_ndim
    if storage_shard_dim != shard_metadata.shard_dim:
        raise ValueError(
            f"Storage placement shards dimension {storage_shard_dim}, but "
            f"metadata tracks dimension {shard_metadata.shard_dim}."
        )
    if device_mesh.size(mesh_axis_index) != shard_metadata.shard_degree:
        raise ValueError(
            f"Storage shard degree {device_mesh.size(mesh_axis_index)} does not "
            f"match metadata degree {shard_metadata.shard_degree}."
        )


@spmd.register_autograd_function
class _FSDPPaddedParamUnshard(torch.autograd.Function):
    """Explicit FSDP all-gather with a reduce-scatter backward.

    Storage is evenly sharded from a globally padded tensor. Forward gathers
    that padded tensor and removes the global tail. Backward restores the tail
    before reduce-scatter. All ranks therefore execute the same static-shape
    operations, including the slice and pad.
    """

    @staticmethod
    def spmd_typecheck(outputs, *, tensor, group, shard_dim):
        axis = spmd.MeshAxis.of(group)
        spmd.assert_type(tensor, {axis: spmd.S(shard_dim)})
        # pyrefly: ignore [bad-argument-type]
        spmd.assert_type_like(outputs, tensor, {axis: spmd.R})

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        tensor: torch.Tensor,
        group: dist.ProcessGroup,
        shard_dim: int,
        unpadded_dim_size: int,
        forward_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        shard_dim = shard_dim % tensor.ndim
        ctx.group = group
        ctx.shard_dim = shard_dim
        ctx.unpadded_dim_size = unpadded_dim_size
        ctx.padded_dim_size = tensor.shape[shard_dim] * dist.get_world_size(group)
        ctx.input_dtype = tensor.dtype
        ctx.reduce_dtype = reduce_dtype

        if unpadded_dim_size > ctx.padded_dim_size:
            raise ValueError(
                f"Logical dimension size {unpadded_dim_size} exceeds gathered "
                f"padded size {ctx.padded_dim_size}."
            )

        collective_input = (
            tensor.to(forward_dtype)
            if forward_dtype is not None and tensor.dtype != forward_dtype
            else tensor
        )
        gathered = funcol.all_gather_single(
            collective_input,
            gather_dim=shard_dim,
            group=group,
        )
        if isinstance(gathered, funcol.AsyncCollectiveTensor):
            gathered = gathered.wait()
        # Keep this slice on every rank, including when it is a no-op. Its
        # arguments depend only on static global metadata, never local rank.
        return gathered.narrow(shard_dim, 0, unpadded_dim_size)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        collective_input = (
            grad_output.to(ctx.reduce_dtype)
            if ctx.reduce_dtype is not None
            else grad_output
        )
        collective_input = _pad_tensor_dim(
            collective_input,
            ctx.shard_dim,
            ctx.padded_dim_size,
        )
        local_grad = funcol.reduce_scatter_single(
            collective_input,
            reduceOp=dist.ReduceOp.SUM.name,
            scatter_dim=ctx.shard_dim,
            group=ctx.group,
        )
        if isinstance(local_grad, funcol.AsyncCollectiveTensor):
            local_grad = local_grad.wait()
        if collective_input.dtype != ctx.input_dtype:
            local_grad = local_grad.to(ctx.input_dtype)
        return local_grad, None, None, None, None, None


"""
[Note: SimpleFSDP parameter representation]

GraphTrainer SimpleFSDP supports only the spmd_types backend. Under
model-parallel (TP/EP), parameters arrive as annotated plain tensors,
pre-sharded in module.parallelize, instead of DTensors.

`data_parallel()` first pads the local storage to an even FSDP shard size, shards
the plain tensor across DP, then wraps that storage once on the full mesh. The
resulting DTensor keeps the original logical global shape while its local tensor
includes padding.
Parameters therefore remain DTensors at rest and DCP sees logical shapes. The
logical, padded, and local shard sizes are static metadata.

Pre-forward unwraps the local storage without communication and uses custom
autograd functions around functional collectives. A sharded axis performs
all-gather then global-tail slice in forward, and global-tail padding then
reduce-scatter in backward. A replicated data-parallel axis is an identity in
forward and all-reduces in backward. This covers DDP, FSDP, and HSDP without a
DTensor redistribution in the compute graph. Every rank traces the same static
operations. We also handle BWD reductions FSDP is expected to do on
model-parallel axes, as the sharding annotations (R/I@TP for SP on/off) assume
FSDP does its job. We look up parameter typing on non-FSDP axes, and
`convert(I->R)` in pre-forward (`P->I` all-reduce in post-backward) if annotated
as R.
For other types: I is no-op, S(i) is sharded at rest, P is banned in titan for now.
"""


def _prepare_spmd_parameter_for_fsdp(
    tensor: torch.Tensor,
    param_name: str,
    non_dp_mesh: DeviceMesh | None,
) -> tuple[
    dict[spmd.MeshAxis, spmd.PerMeshAxisSpmdType],
    DTensorSpec | None,
]:
    """Prepare an SPMD-annotated parameter for SimpleFSDP.

    Record the parameter's model-parallel axis types and the DTensor layout that
    its final storage wrapper must preserve. The parameter itself remains a
    plain, rank-local tensor until DP storage is constructed.
    """
    non_dp_mesh_types = {}
    if non_dp_mesh is None:
        return non_dp_mesh_types, None

    if not spmd.has_local_type(tensor):
        raise ValueError(
            f"Parameter {param_name!r} must have an SPMD type before "
            "applying SimpleFSDP with a non-DP mesh."
        )
    assert non_dp_mesh.mesh_dim_names is not None
    local_type = spmd.get_local_type(tensor)
    partition_spec = spmd.get_partition_spec(tensor)
    for axis_name in non_dp_mesh.mesh_dim_names:
        axis = spmd.MeshAxis.of(non_dp_mesh.get_group(axis_name))
        non_dp_mesh_types[axis] = (
            partition_spec_get_shard(partition_spec, axis) or local_type[axis]
        )
    placements = tuple(
        spmd.spmd_type_to_dtensor_placement(
            non_dp_mesh_types[spmd.MeshAxis.of(non_dp_mesh.get_group(axis_name))]
        )
        for axis_name in non_dp_mesh.mesh_dim_names
    )
    non_dp_spec = DTensor.from_local(
        tensor,
        non_dp_mesh,
        placements,
        run_check=False,
    )._spec
    return (
        non_dp_mesh_types,
        non_dp_spec,
    )


def _compose_storage_layout(
    device_mesh: DeviceMesh,
    dp_storage_placements: tuple[Placement, ...],
    non_dp_spec: DTensorSpec | None,
) -> tuple[DeviceMesh, tuple[Placement, ...]]:
    """Compose DP storage axes with an optional model-parallel layout."""
    if non_dp_spec is None:
        return device_mesh, dp_storage_placements

    dp_placements = []
    for placement in dp_storage_placements:
        if isinstance(placement, Shard):
            tensor_meta = non_dp_spec.tensor_meta
            assert tensor_meta is not None
            shard_dim = placement.dim % len(tensor_meta.shape)
            split_factor = non_dp_spec.num_shards_map[shard_dim]
            placement = (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else placement
            )
        dp_placements.append(placement)
    storage_mesh = DeviceMesh._concatenate([device_mesh, non_dp_spec.mesh])
    return storage_mesh, tuple(dp_placements) + non_dp_spec.placements


def _create_fsdp_param_dtensor(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    dp_storage_placements: tuple[Placement, ...],
    *,
    mode: str,
    shard_dim: int,
    non_dp_spec: DTensorSpec | None,
) -> tuple[DTensor, FSDPShardMetadata | None]:
    """Create DP storage, then wrap it as one logical rest-time DTensor."""
    metadata = None
    logical_dim_size = tensor.shape[shard_dim]
    padded_dim_size = logical_dim_size
    if mode in ("fully_shard", "hybrid_shard"):
        shard_mesh_axis = 0 if mode == "fully_shard" else 1
        shard_degree = device_mesh.size(shard_mesh_axis)
        padded_dim_size = (
            (logical_dim_size + shard_degree - 1) // shard_degree
        ) * shard_degree
        metadata = FSDPShardMetadata(
            shard_dim=shard_dim,
            param_ndim=tensor.ndim,
            logical_dim_size=logical_dim_size,
            padded_dim_size=padded_dim_size,
            shard_degree=shard_degree,
        )

    if non_dp_spec is None:
        logical_global_shape = tensor.shape
        logical_global_stride = tensor.stride()
    else:
        tensor_meta = non_dp_spec.tensor_meta
        assert tensor_meta is not None
        logical_global_shape = tensor_meta.shape
        logical_global_stride = tensor_meta.stride

    padded_tensor = _pad_tensor_dim(tensor, shard_dim, padded_dim_size)

    # ``distribute_tensor`` accepts leaf tensors only; storage padding is an
    # initialization-time transform.
    padded_tensor = padded_tensor.detach().requires_grad_(tensor.requires_grad)
    dp_storage = distribute_tensor(
        padded_tensor,
        device_mesh,
        dp_storage_placements,
    )
    storage_mesh, storage_placements = _compose_storage_layout(
        device_mesh,
        dp_storage_placements,
        non_dp_spec,
    )
    # Padding is private to local storage. DCP and other DTensor consumers see
    # the logical model shape across both DP and model-parallel axes.
    distributed_param = DTensor.from_local(
        dp_storage.to_local(),
        storage_mesh,
        storage_placements,
        run_check=False,
        shape=logical_global_shape,
        stride=logical_global_stride,
    )
    return distributed_param, metadata


_wrap_class_id = count()


def _register_parametrization(
    module: nn.Module,
    param_names: list[str],
    parametrization_init: Callable[[str], nn.Module],
) -> None:
    """
    It works with state_dict without incurring parametrization calls because
    state_dict accesses parameters directly from self._parameters, not from getters
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L2141
    TODO: In checkpoint saving/loading, avoid parametrization calls when calling
    get_model_state_dict func in torchtitan/components/checkpointer/dcp.py.
    """
    param_name_to_property = {}
    for param_name in param_names:
        parametrization = parametrization_init(param_name)
        param_name_to_property[param_name] = property(
            lambda self, pn=param_name, p=parametrization: p(self._parameters[pn])
        )
    module_cls = type(
        f"SimpleFSDP{module.__class__.__name__}_{next(_wrap_class_id)}",
        (module.__class__,),
        param_name_to_property,
    )
    # Expose the dynamically created class as a real, importable symbol
    # so that pickle/GraphPickler can resolve it during serialization.
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls
    module.__class__ = module_cls


class MaterializeParamForCompute(Module):
    def __init__(
        self,
        device_mesh: DeviceMesh,
        dp_storage_placements: tuple[Placement, ...],
        mode: str,
        mp_policy: MixedPrecisionPolicy | None,
        non_dp_mesh_types: dict[spmd.MeshAxis, spmd.PerMeshAxisSpmdType],
        shard_metadata: FSDPShardMetadata | None = None,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.dp_storage_placements = dp_storage_placements
        self.mode = mode
        if self.mode not in ("replicate", "fully_shard", "hybrid_shard"):
            raise ValueError(f"Unsupported SimpleFSDP mode {self.mode!r}.")
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype: torch.dtype | None = mp_policy.param_dtype
        self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype
        self.shard_metadata = shard_metadata

        # non_dp_mesh_types stores local type for non-FSDP (model-parallel) axes
        # (e.g. TP on dense, EP on sparse), so SimpleFSDP handles any TP/EP grad
        # reductions it's responsible for.
        _validate_dp_storage_config(
            self.device_mesh,
            self.dp_storage_placements,
            self.mode,
            self.shard_metadata,
        )
        self.non_dp_mesh_types = non_dp_mesh_types

    def materialize_for_compute(self, x: DTensor) -> torch.Tensor:
        """Materialize a DTensor storage shard using explicit collectives."""
        # This boundary only changes representation. Matching grad placements
        # ensure its backward wraps the local gradient without communication.
        local_shard = x.to_local(grad_placements=x.placements)

        output = local_shard
        # DP-replicated storage is I -> R before the FSDP unshard. Autograd
        # therefore applies this all-reduce after the FSDP reduce-scatter,
        # which is the required HSDP/DDP gradient ordering.
        for mesh_axis_index, placement in enumerate(self.dp_storage_placements):
            if isinstance(placement, Replicate):
                group = self.device_mesh.get_group(mesh_axis_index)
                output = spmd.convert(
                    output,
                    group,
                    src=spmd.I,
                    dst=spmd.R,
                    op_dtype=self.param_dtype,
                    backward_options={"op_dtype": self.reduce_dtype},
                )
        for mesh_axis_index, placement in enumerate(self.dp_storage_placements):
            if isinstance(placement, Shard):
                metadata = cast(FSDPShardMetadata, self.shard_metadata)
                output = _FSDPPaddedParamUnshard.apply(
                    output,
                    self.device_mesh.get_group(mesh_axis_index),
                    metadata.shard_dim,
                    metadata.logical_dim_size,
                    self.param_dtype,
                    self.reduce_dtype,
                )
        # Model-parallel I -> R follows the FSDP unshard in forward, so its
        # TP/EP gradient all-reduce precedes the FSDP reduce-scatter backward.
        for axis, axis_type in self.non_dp_mesh_types.items():
            if axis_type is spmd.R:
                output = spmd.convert(
                    output,
                    axis,
                    src=spmd.I,
                    dst=spmd.R,
                    op_dtype=self.param_dtype,
                    backward_options={"op_dtype": self.reduce_dtype},
                )
        return output

    def forward(self, x: DTensor) -> torch.Tensor:
        global _active_parametrization
        # This should never be set to true during forward, only outside for model
        # inspection / debugging / initialization
        # model initialization can be done now through
        # with disable_active_parametrization():
        #     model.init_states()
        if not _active_parametrization:
            return x

        return self.materialize_for_compute(x)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    # non_dp_mesh: model-parallel (TP/EP) mesh so SimpleFSDP constructs DTensor params on full-mesh
    non_dp_mesh: DeviceMesh | None = None,
) -> nn.Module:
    if get_spmd_backend() != "spmd_types":
        raise ValueError(
            "GraphTrainer SimpleFSDP requires spmd_backend='spmd_types'; "
            "the partial_dtensor backend is not supported."
        )

    dp_storage_placements: tuple[Placement, ...]
    if mode == "replicate":
        dp_storage_placements = (Replicate(),)
    elif mode == "fully_shard":
        dp_storage_placements = (Shard(shard_dim),)
    elif mode == "hybrid_shard":
        # replicate inter-host, fully shard intra-host
        dp_storage_placements = (Replicate(), Shard(shard_dim))
        assert (
            device_mesh.ndim == 2
        ), "hybrid sharded data parallel requires 2D DeviceMesh"
    else:
        raise ValueError(f"Unsupported mode {mode}")

    modules = list(model.modules())

    for mod in modules:
        params_dict = dict(mod.named_parameters(recurse=False))
        # we shouldn't apply data parallel to the modules that are already
        # sharded by data parallel
        if "SimpleFSDP" in mod.__class__.__name__:
            continue

        param_non_dp_mesh_types = {}
        param_shard_metadata = {}

        for p_name, p in params_dict.items():
            if p is not None and p.numel() > 0:
                canonical_shard_dim = shard_dim % p.ndim
                # TP/EP has already produced a rank-local annotated plain
                # tensor here. FSDP padding and logical sizes belong to that
                # local model-parallel partition, not the later DTensor's
                # reconstructed global shape.
                non_dp_mesh_types, non_dp_spec = _prepare_spmd_parameter_for_fsdp(
                    p,
                    p_name,
                    non_dp_mesh,
                )
                param_non_dp_mesh_types[p_name] = non_dp_mesh_types
                distributed_param, metadata = _create_fsdp_param_dtensor(
                    p,
                    device_mesh,
                    dp_storage_placements,
                    mode=mode,
                    shard_dim=canonical_shard_dim,
                    non_dp_spec=non_dp_spec,
                )
                if metadata is not None:
                    param_shard_metadata[p_name] = metadata
                mod.register_parameter(
                    p_name,
                    nn.Parameter(distributed_param),
                )

        _register_parametrization(
            mod,
            list(params_dict.keys()),
            lambda param_name: MaterializeParamForCompute(
                device_mesh=device_mesh,
                dp_storage_placements=dp_storage_placements,
                mode=mode,
                mp_policy=mp_policy,
                shard_metadata=param_shard_metadata.get(param_name),
                non_dp_mesh_types=param_non_dp_mesh_types.get(param_name, {}),
            ),
        )
    return model
