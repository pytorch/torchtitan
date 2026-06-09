# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

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


# Cache of (original_class, param_names) -> wrapper class, so all instances
# of the same module type share one SimpleFSDP class for torch.compile reuse.
_wrap_class_cache: dict[tuple[type, frozenset[str]], type] = {}


def _register_parametrization(
    module: nn.Module, param_names: list[str], parametrization: nn.Module
) -> None:
    """
    It works with state_dict without incurring parametrization calls because
    state_dict accesses parameters directly from self._parameters, not from getters
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L2141
    TODO: In checkpoint saving/loading, avoid parametrization calls when calling
    get_model_state_dict func in torchtitan's torchtitan/components/checkpoint.py.
    """
    param_name_to_property = {
        param_name: property(
            lambda self, pn=param_name: parametrization(self._parameters[pn])
        )
        for param_name in param_names
    }
    cache_key = (module.__class__, frozenset(param_names))
    if cache_key in _wrap_class_cache:
        module_cls = _wrap_class_cache[cache_key]
    else:
        module_cls = type(
            f"SimpleFSDP{module.__class__.__name__}",
            (module.__class__,),
            param_name_to_property,
        )
        # Expose the dynamically created class as a real, importable symbol
        # so that pickle/GraphPickler can resolve it during serialization.
        sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls
        _wrap_class_cache[cache_key] = module_cls
    module.__class__ = module_cls


class ReplicateComputation(Module):
    def __init__(
        self,
        device_mesh: DeviceMesh,
        param_sharding: tuple[Placement, ...],
        mp_policy: MixedPrecisionPolicy | None,
        dp_axis_names_in_full_mesh: frozenset[str] | None = None,
        full_spmd_mesh: DeviceMesh | None = None,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.compute_placements: list[Placement] = [Replicate()] * self.device_mesh.ndim
        self.grad_placements: list[Placement] = [
            Partial(reduce_op="sum")
        ] * self.device_mesh.ndim
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype: torch.dtype | None = mp_policy.param_dtype
        self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype
        # full_spmd_mesh is the mesh to lift the all-gathered param onto; it is
        # set only under full_dtensor, so its presence is the full_dtensor switch.
        # dp_axis_names: SPMD DP axes to all-gather (see replicate_compute).
        self.full_dtensor = full_spmd_mesh is not None
        self.full_spmd_mesh = full_spmd_mesh
        self.dp_axis_names: frozenset[str] = (
            dp_axis_names_in_full_mesh
            if dp_axis_names_in_full_mesh is not None
            else frozenset(device_mesh.mesh_dim_names or ())
        )

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        if self.full_dtensor:
            # x is stored on the folded DP submesh (+ tp/ep) exactly like the
            # legacy nD path, so the all-gather below is bit-identical to legacy.
            non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
            if non_dp_mesh_dims > 0:
                sharded_local = x.to_local()
                sharded_on_dp = DTensor.from_local(
                    sharded_local, self.device_mesh, self.param_sharding
                )
                replicated_on_dp = sharded_on_dp.redistribute(
                    placements=self.compute_placements,
                    forward_dtype=self.param_dtype,
                    backward_dtype=self.reduce_dtype,
                )
                replicated_local = replicated_on_dp.to_local(
                    grad_placements=self.grad_placements
                )
                non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
                non_dp_names = tuple(x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:])
            else:
                replicated = x.redistribute(
                    placements=self.compute_placements,
                    forward_dtype=self.param_dtype,
                    backward_dtype=self.reduce_dtype,
                )
                replicated_local = replicated.to_local(
                    grad_placements=self.grad_placements
                )
                non_dp_placements = ()
                non_dp_names = ()

            # Lift the all-gathered local back onto the full SPMD mesh so it
            # composes with cp-carrying activations: DP axes become Replicate,
            # tp/ep keep their placement. grad_placements = Partial on the DP
            # axes (the R->P duality) so the backward reduce-scatters once with
            # no spurious reduction -- no custom autograd function needed.
            non_dp_map = dict(zip(non_dp_names, non_dp_placements))
            full_names = self.full_spmd_mesh.mesh_dim_names or ()
            fwd_placements = tuple(
                Replicate() if n in self.dp_axis_names else non_dp_map[n]
                for n in full_names
            )
            grad_placements = tuple(
                Partial() if n in self.dp_axis_names else non_dp_map[n]
                for n in full_names
            )
            return DTensor.from_local(
                replicated_local,
                self.full_spmd_mesh,
                fwd_placements,
                run_check=False,
                grad_placements=grad_placements,
            )

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


def _simple_fsdp_mode(dp_mesh: DeviceMesh) -> str:
    """Pick the simple_fsdp mode from a DP mesh's axis names."""
    names = dp_mesh.mesh_dim_names or ()
    if "dp_replicate" in names:
        return "hybrid_shard" if len(names) > 1 else "replicate"
    return "fully_shard"


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
) -> nn.Module:
    """Apply simple_fsdp parametrization, the graph_trainer analog of fully_shard.

    Legacy (``dp_mesh_dims is None``): ``device_mesh`` is the DP-only mesh
    (ParallelDims pre-flattens ``dp_shard``/``cp`` into it); params are plain
    tensors or TP/EP DTensors.

    Full DTensor (``dp_mesh_dims`` set): ``device_mesh`` is the full SPMD mesh and
    ``dp_mesh_dims`` names its DP axes, like fully_shard. The named shard axes are
    flattened into one FSDP storage axis; each param is reduced to its pre-FSDP
    form, stored there by the same path as legacy, and lifted onto the full SPMD
    mesh at compute time.

    The sharding mode is inferred from the mesh (like fully_shard), not passed in.
    """
    full_dtensor = dp_mesh_dims is not None
    full_spmd_mesh: DeviceMesh | None = None
    shard_names: tuple[str, ...] = ()
    dp_axis_names: frozenset[str] = frozenset()
    if dp_mesh_dims is not None:
        # Like fully_shard: flatten the named DP shard axes (e.g. dp_shard + cp)
        # into one FSDP storage axis, then prepend the optional replicate axis.
        full_spmd_mesh = device_mesh
        shard_names = dp_mesh_dims.shard_names
        replicate_names = dp_mesh_dims.replicate_names
        fsdp_mesh = (
            full_spmd_mesh[shard_names]._flatten("_".join(shard_names) + "_fsdp")
            if len(shard_names) > 1
            else full_spmd_mesh[shard_names[0]]
        )
        device_mesh = (
            DeviceMesh._concatenate([full_spmd_mesh[replicate_names], fsdp_mesh])
            if replicate_names
            else fsdp_mesh
        )
        dp_axis_names = frozenset(shard_names + replicate_names)
        # A replicate axis means HSDP; otherwise pure FSDP.
        mode = "hybrid_shard" if replicate_names else "fully_shard"
    else:
        mode = _simple_fsdp_mode(device_mesh)

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

    logger.info("simple_fsdp: %s on mesh %s", mode, device_mesh.mesh_dim_names)

    for mod in list(model.modules()):
        # Skip modules already parametrized by an earlier data_parallel pass.
        if "SimpleFSDP" in mod.__class__.__name__:
            continue
        params_dict = dict(mod.named_parameters(recurse=False))

        wrapped_names: list[str] = []
        for p_name, p in params_dict.items():
            if p is None or p.numel() == 0:
                continue
            to_store: torch.Tensor = p
            if full_dtensor:
                if not isinstance(p, DTensor):
                    raise TypeError(
                        f"{type(mod).__name__}.{p_name} must already be a DTensor "
                        "before apply_simple_fsdp under full_dtensor "
                        "(model.parallelize must run first)."
                    )
                if not set(shard_names) <= set(p._spec.mesh.mesh_dim_names or ()):
                    # Param on a different SPMD family (dense vs sparse); a
                    # sibling data_parallel pass covers it.
                    continue
                # Reduce to the pre-FSDP (tp/ep-only) form the storage expects.
                to_store = _strip_dp_axes(p, dp_axis_names)
            # Custom _register_parametrization (not nn.utils.parametrize) keeps
            # the state_dict DCP-compatible by reading params directly.
            distribute = (
                _distribute_dtensor
                if isinstance(to_store, DTensor)
                else distribute_tensor
            )
            mod.register_parameter(
                p_name,
                nn.Parameter(
                    distribute(to_store, device_mesh, param_sharding),
                    requires_grad=p.requires_grad,
                ),
            )
            wrapped_names.append(p_name)

        # Register only the params this pass took ownership of, so the storage
        # decision and the parametrization stay in lockstep. Legacy wraps every
        # param, so it registers all names.
        if wrapped_names or not full_dtensor:
            _register_parametrization(
                mod,
                wrapped_names if full_dtensor else list(params_dict.keys()),
                ReplicateComputation(
                    device_mesh,
                    param_sharding,
                    mp_policy=mp_policy,
                    full_spmd_mesh=full_spmd_mesh,
                    dp_axis_names_in_full_mesh=dp_axis_names if full_dtensor else None,
                ),
            )
    return model


def _strip_dp_axes(p: DTensor, dp_axis_names: frozenset[str]) -> torch.Tensor:
    """Reduce an fdt param to its pre-FSDP form by dropping the DP axes.

    model.parallelize leaves params Replicate on every DP axis (FSDP has not
    sharded them yet), so the local tensor is the full tp/ep-sharded param.
    Returns a plain tensor when there is no tp/ep, else a DTensor on the tp/ep
    submesh -- the same starting point the legacy storage path expects.
    """
    full_mesh = p._spec.mesh
    names = full_mesh.mesh_dim_names or ()
    non_dp = [(n, plc) for n, plc in zip(names, p.placements) if n not in dp_axis_names]
    for n, plc in zip(names, p.placements):
        if n in dp_axis_names:
            assert (
                plc.is_replicate()
            ), f"full_dtensor param must be Replicate on DP axis {n}, got {plc}"
    # detach: at setup the param is not in an autograd graph, and
    # distribute_tensor / _distribute_dtensor require a leaf tensor.
    local = p.to_local().detach()
    if not non_dp:
        return local
    non_dp_mesh = full_mesh[tuple(n for n, _ in non_dp)]
    return DTensor.from_local(local, non_dp_mesh, tuple(plc for _, plc in non_dp))
