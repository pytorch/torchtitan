# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

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
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from torchtitan.protocols.module import Module

_active_parametrization = True


@dataclass(frozen=True)
class _FSDPPostAllGatherState:
    """Values produced by pre-all-gather and consumed by post-all-gather.

    metadata: Hook-defined auxiliary values needed to reconstruct the tensor
        subclass after all-gather, e.g. torchao Float8 scale tensors.
    param_dtype: Logical parameter dtype for the reconstructed wrapper. This
        is not necessarily the communication tensor dtype.
    """

    metadata: Any
    param_dtype: torch.dtype


class _PostAllGather(torch.autograd.Function):
    """Call a tensor subclass's FSDP post-all-gather hook differentiably.

    The hook reconstructs the wrapper subclass around the all-gathered local
    tensor. When the gathered input is a DTensor, this also restores the DTensor
    spec around the hook-produced local wrapper.
    """

    @staticmethod
    def forward(
        ctx,
        gathered: torch.Tensor,
        wrapper: torch.Tensor,
        post_all_gather_state: _FSDPPostAllGatherState,
    ) -> torch.Tensor:
        gathered_local = (
            gathered._local_tensor if isinstance(gathered, DTensor) else gathered
        )
        # FSDP2 uses the returned inner tensors to manage storage lifetime for
        # its cached unsharded parameter. SimpleFSDP keeps the returned wrapper
        # live directly through autograd, so there is no separate storage to free.
        output, _ = wrapper.fsdp_post_all_gather(
            (gathered_local,),
            post_all_gather_state.metadata,
            post_all_gather_state.param_dtype,
        )
        if isinstance(gathered, DTensor):
            return DTensor(
                output,
                gathered._spec,
                requires_grad=gathered.requires_grad,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _PreAllGatherDTensor(torch.autograd.Function):
    """Call a DTensor-local wrapper's FSDP pre-all-gather hook differentiably.

    torchao pre-all-gather hooks return plain tensors to communicate. Taking the
    original wrapper-containing DTensor as the Function input preserves the
    autograd edge from the eventual loss back to the wrapped parameter.
    """

    @staticmethod
    def forward(
        ctx,
        dtensor: DTensor,
        device_mesh: DeviceMesh,
        mp_policy: "MixedPrecisionPolicy",
    ) -> tuple[DTensor, _FSDPPostAllGatherState]:
        wrapper = dtensor._local_tensor
        assert mp_policy.param_dtype is not None, (
            "FSDP hook mixed precision policy should have a concrete param dtype "
            "before calling fsdp_pre_all_gather"
        )
        all_gather_inputs, metadata = wrapper.fsdp_pre_all_gather(
            device_mesh,
            dtensor.size(),
            dtensor.stride(),
            None,
            mp_policy,
        )
        assert len(all_gather_inputs) == 1, (
            "simple_fsdp currently supports tensor subclass FSDP hooks with "
            f"one all-gather input; got {len(all_gather_inputs)}"
        )
        return (
            DTensor(
                all_gather_inputs[0],
                dtensor._spec,
                requires_grad=dtensor.requires_grad,
            ),
            _FSDPPostAllGatherState(
                metadata=metadata,
                param_dtype=mp_policy.param_dtype,
            ),
        )

    @staticmethod
    def backward(ctx, grad_output, _grad_post_all_gather_state):
        return grad_output, None, None


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


def _has_fsdp_all_gather_hooks(tensor: torch.Tensor) -> bool:
    has_pre_hook = hasattr(tensor, "fsdp_pre_all_gather")
    has_post_hook = hasattr(tensor, "fsdp_post_all_gather")
    if has_pre_hook != has_post_hook:
        raise AssertionError(
            "simple_fsdp requires tensor subclasses to define both "
            "fsdp_pre_all_gather and fsdp_post_all_gather"
        )
    return has_pre_hook


def _uses_fsdp_all_gather_hooks(param: torch.Tensor) -> bool:
    if isinstance(param, DTensor):
        return _has_fsdp_all_gather_hooks(param._local_tensor)
    return _has_fsdp_all_gather_hooks(param)


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
            lambda self, pn=param_name: parametrization(self._parameters[pn], pn)
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
        mode: str,
        mp_policy: MixedPrecisionPolicy | None,
        full_dtensor: bool = False,
        fsdp_hook_param_names: set[str] | None = None,
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
        self.full_dtensor = full_dtensor
        self.fsdp_hook_param_names = fsdp_hook_param_names or set()

    def _fsdp_pre_all_gather_dtensor(
        self,
        dtensor: DTensor,
    ) -> tuple[DTensor, _FSDPPostAllGatherState]:
        wrapper = dtensor._local_tensor
        # Match FSDP2's post-hook dtype contract: use the mixed precision
        # param dtype when configured, otherwise fall back to the original
        # wrapper dtype as the logical reconstructed parameter dtype.
        param_dtype = self.param_dtype or wrapper.dtype
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=self.reduce_dtype,
        )
        pre_all_gather_dtensor, post_all_gather_state = _PreAllGatherDTensor.apply(
            dtensor,
            self.device_mesh,
            mp_policy,
        )
        return pre_all_gather_dtensor, post_all_gather_state

    def replicate_compute(self, x: DTensor, param_name: str = "") -> torch.Tensor:
        original_dtensor = x
        post_all_gather_state = None
        forward_dtype = self.param_dtype
        use_fsdp_hooks = param_name in self.fsdp_hook_param_names
        if use_fsdp_hooks:
            x, post_all_gather_state = self._fsdp_pre_all_gather_dtensor(x)
            # The pre-all-gather hook applies mp_policy.param_dtype when
            # producing the communication tensor; avoid a second DTensor cast.
            forward_dtype = None

        # data parallel runtime replicate parameters and do local compute
        # the gradients are partial tensors that needs to perform reduction
        # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)
        # support FSDP/DDP/HSDP + EP + TP (assuming TP shards the inner-most dim)
        non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
        assert non_dp_mesh_dims <= 2, "Only DP + EP/TP/EP+TP is supported"
        if non_dp_mesh_dims > 0:
            if self.full_dtensor:
                raise NotImplementedError(
                    "full_dtensor not implemented for nD parallelisms"
                )
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
                forward_dtype=forward_dtype,
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
                forward_dtype=forward_dtype,
                backward_dtype=self.reduce_dtype,
            )

            if not self.full_dtensor:
                output = output.to_local(grad_placements=self.grad_placements)
        else:
            raise AssertionError(
                f"Unsupported replicate compute on placement {x._spec.placements} for DTensor {x}"
            )

        if not use_fsdp_hooks:
            return output

        assert post_all_gather_state is not None
        return _PostAllGather.apply(
            output,
            original_dtensor._local_tensor,
            post_all_gather_state,
        )

    def forward(self, x: DTensor, param_name: str = "") -> torch.Tensor:
        global _active_parametrization
        # This should never be set to true during forward, only outside for model
        # inspection / debugging / initialization
        # model initialization can be done now through
        # with disable_active_parametrization():
        #     model.init_states()
        if not _active_parametrization:
            return x

        return self.replicate_compute(x, param_name)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    full_dtensor: bool = False,
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

        fsdp_hook_param_names: set[str] = set()
        for p_name, p in params_dict.items():
            if p is not None and p.numel() > 0:
                if _uses_fsdp_all_gather_hooks(p):
                    fsdp_hook_param_names.add(p_name)

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
            ReplicateComputation(
                device_mesh,
                param_sharding,
                mode,
                mp_policy=mp_policy,
                full_dtensor=full_dtensor,
                fsdp_hook_param_names=fsdp_hook_param_names,
            ),
        )
    return model
