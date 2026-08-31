# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count

import spmd_types as spmd
import torch
import torch.nn as nn

from spmd_types.types import partition_spec_get_shard
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
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


"""
[Note: SimpleFSDP and spmd_types]

Under spmd_types backend, SimpleFSDP differs slightly under model-parallel (TP/EP).
Params arrive as annotated plain tensors, pre-sharded in module.parallelize,
instead of DTensors.

`data_parallel()` first performs full-mesh DTensor translation & FSDP shards,
so rest-time params match DTensor backend (mesh is FSDP + TP), this is mostly
so DCP integration / grad norm impl remains the same.

In pre-forward (ReplicateComputation.forward), we additionally handle any BWD reductions
FSDP is expected to do, as the sharding annotations (R/I@TP for SP on/off) are assuming
FSDP does its job (R FWD <-> P BWD assumes FSDP redistributes to I).
We lookup parameter typing on non-FSDP axes, and `convert(I->R)` in pre-forward
(P->I all-reduce in post-backward) if annotated as R.
For other types: I is no-op, S(i) is sharded at rest, P is banned in titan for now.
"""


def _spmd_local_tensor_to_dtensor(
    tensor: torch.Tensor,
    param_name: str,
    non_dp_mesh: DeviceMesh | None,
    param_non_dp_mesh_types: dict[str, dict[spmd.MeshAxis, spmd.PerMeshAxisSpmdType]],
) -> torch.Tensor:
    """Prepare an SPMD-annotated parameter for SimpleFSDP.

    For the spmd_types backend, record the parameter's model-parallel axis types
    for ReplicateComputation and restore its DTensor wrapper on ``non_dp_mesh``.
    """
    non_dp_mesh_types = {}
    param_non_dp_mesh_types[param_name] = non_dp_mesh_types
    if non_dp_mesh is None:
        return tensor

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
    return DTensor.from_local(tensor, non_dp_mesh, placements, run_check=False)


def _distribute_dtensor(
    tensor: DTensor,
    device_mesh: DeviceMesh,
    dp_placements: Sequence[Placement],
) -> DTensor:
    """
    Below are experimental enhancements to distribute a DTensor.
    This helps enable Simple FSDP + TP/EP, in which
        inner spec/mesh is TP/EP spec/mesh
        outer spec/mesh is FSDP/DDP/HSDP spec/mesh
    """
    inner_spec = tensor._spec
    outer_mesh, inner_mesh = device_mesh, inner_spec.mesh
    spanned_mesh = DeviceMesh._concatenate([outer_mesh, inner_mesh])

    if len(dp_placements) == 1:
        assert dp_placements[0].is_replicate() or dp_placements[0].is_shard()
        if dp_placements[0].is_shard():
            # For FSDP + EP/TP/EP+TP
            assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
            shard_dim = dp_placements[0].dim
            split_factor = inner_spec.num_shards_map[shard_dim]
            tensor_placement = (
                (
                    _StridedShard(shard_dim, split_factor=split_factor)
                    if split_factor > 1
                    else dp_placements[0]
                ),
            ) + inner_spec.placements
        else:
            # For DDP + TP/EP
            assert len(inner_spec.placements) == 1
            tensor_placement = (dp_placements[0], inner_spec.placements[0])
    elif len(dp_placements) == 2:
        assert dp_placements[0].is_replicate() and dp_placements[1].is_shard()
        # For HSDP + EP/TP/EP+TP
        assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
        shard_dim = dp_placements[1].dim
        split_factor = inner_spec.num_shards_map[shard_dim]
        tensor_placement = (
            dp_placements[0],
            (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else dp_placements[1]
            ),
        ) + inner_spec.placements
    else:
        raise ValueError(
            f"Unsupported placement {dp_placements} for distributing DTensor {tensor}"
        )

    # HSDP case needs 2 placements for 2D outer_mesh
    current_placements = (Replicate(),) * len(dp_placements)
    target_placements = tuple(dp_placements)

    current_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=current_placements,
        tensor_meta=inner_spec.tensor_meta,
    )
    target_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=target_placements,
        tensor_meta=inner_spec.tensor_meta,
    )
    result_tensor = redistribute_local_tensor(
        tensor._local_tensor,
        current_spec=current_spec,
        target_spec=target_spec,
    )
    return DTensor(
        result_tensor.requires_grad_(tensor.requires_grad),
        DTensorSpec(
            mesh=spanned_mesh,
            placements=tensor_placement,
            tensor_meta=inner_spec.tensor_meta,
        ),
        requires_grad=tensor.requires_grad,
    )


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


class ReplicateComputation(Module):
    def __init__(
        self,
        device_mesh: DeviceMesh,
        param_sharding: tuple[Placement, ...],
        mode: str,
        mp_policy: MixedPrecisionPolicy | None,
        non_dp_mesh_types: dict[spmd.MeshAxis, spmd.PerMeshAxisSpmdType] | None = None,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.mode = mode
        self.compute_placements: list[Placement] = [Replicate()] * self.device_mesh.ndim
        self.grad_placements: list[Placement] = [
            Partial(reduce_op="sum")
        ] * self.device_mesh.ndim
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype: torch.dtype | None = mp_policy.param_dtype
        self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype

        # non_dp_mesh_types stores local type for non-FSDP (model-parallel) axes
        # (e.g. TP on dense, EP on sparse), so SimpleFSDP handles any TP/EP grad
        # reductions it's responsible for.
        if get_spmd_backend() == "spmd_types":
            assert non_dp_mesh_types is not None
        self.non_dp_mesh_types = non_dp_mesh_types

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        # data parallel runtime replicate parameters and do local compute
        # the gradients are partial tensors that needs to perform reduction
        # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)
        # support FSDP/DDP/HSDP + EP + TP (assuming TP shards the inner-most dim)
        non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
        assert non_dp_mesh_dims <= 2, "Only DP + EP/TP/EP+TP is supported"
        if non_dp_mesh_dims > 0:
            dp_mesh = self.device_mesh
            # re-wrap 2D DTensor to 1D DTensor on dp_mesh for efficient FSDP all-gather
            sharded_local_tensor = x.to_local()
            sharded_dtensor = DTensor.from_local(
                sharded_local_tensor, dp_mesh, self.param_sharding
            )

            # the actual FSDP's fwd all-gather & bwd reduce-scatter
            # DDP's bwd all-reduce on dp_mesh
            replicated_dtensor = sharded_dtensor.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            )

            # re-wrap all-gathered DTensor on dp_mesh to be on non_dp_mesh
            # TODO: DTensor should support this mesh collapsing operation
            replicated_local_tensor = replicated_dtensor.to_local(
                grad_placements=self.grad_placements
            )

            non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
            non_dp_mesh_dim_names = tuple(
                x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:]
            )
            non_dp_mesh = x._spec.mesh[non_dp_mesh_dim_names]

            if self.non_dp_mesh_types is not None:
                output = replicated_local_tensor
                for axis, axis_type in self.non_dp_mesh_types.items():
                    if axis_type is spmd.R:
                        # handle any BWD all-reduces on non-FSDP-axes that FSDP is responsible for.
                        # e.g. TP RMSNorm w/ SP on, is annotated as spmd.R, we add P->I in BWD.
                        # if SP off, annotation is spmd.I, no effect.
                        output = spmd.convert(
                            output,
                            axis,
                            src=spmd.I,
                            dst=spmd.R,
                            op_dtype=self.param_dtype,
                            backward_options={"op_dtype": self.reduce_dtype},
                        )
            else:
                output = DTensor.from_local(
                    replicated_local_tensor, non_dp_mesh, non_dp_placements
                )
        elif non_dp_mesh_dims == 0:
            output = x.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            )
            output = output.to_local(grad_placements=self.grad_placements)
        else:
            raise AssertionError(
                f"Unsupported replicate compute on placement {x._spec.placements} for DTensor {x}"
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

        output = self.replicate_compute(x)
        return output


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    # non_dp_mesh: model-parallel (TP/EP) mesh so SimpleFSDP constructs DTensor params on full-mesh
    non_dp_mesh: DeviceMesh | None = None,
) -> nn.Module:
    param_sharding: tuple[Placement, ...]
    if mode == "replicate":
        param_sharding = (Replicate(),)
    elif mode == "fully_shard":
        param_sharding = (Shard(shard_dim),)
    elif mode == "hybrid_shard":
        # replicate inter-host, fully shard intra-host
        param_sharding = (Replicate(), Shard(shard_dim))
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

        for p_name, p in params_dict.items():
            if p is not None and p.numel() > 0:
                if get_spmd_backend() == "spmd_types":
                    p = _spmd_local_tensor_to_dtensor(
                        p,
                        p_name,
                        non_dp_mesh,
                        param_non_dp_mesh_types,
                    )
                distribute_tensor_func = (
                    _distribute_dtensor if isinstance(p, DTensor) else distribute_tensor
                )
                mod.register_parameter(
                    p_name,
                    nn.Parameter(
                        distribute_tensor_func(p, device_mesh, param_sharding)
                    ),
                )

                # to be compatible with DCP, we use a customized _register_parametrization
                # instead of nn.utils.parametrize.register_parametrization here
                # nn.utils.parametrize.register_parametrization(
                #     mod,
                #     p_name,
                #     ReplicateComputation(
                #         device_mesh,
                #         param_sharding,
                #         mode,
                #         mp_policy=mp_policy,
                #     ),
                #     unsafe=True,
                # )

        _register_parametrization(
            mod,
            list(params_dict.keys()),
            lambda param_name: ReplicateComputation(
                device_mesh=device_mesh,
                param_sharding=param_sharding,
                mode=mode,
                mp_policy=mp_policy,
                non_dp_mesh_types=(
                    param_non_dp_mesh_types.get(param_name, {})
                    if get_spmd_backend() == "spmd_types"
                    else None
                ),
            ),
        )
    return model
