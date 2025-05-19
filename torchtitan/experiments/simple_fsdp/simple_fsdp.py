# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


_active_parametrization = True


@contextmanager
def disable_data_parallel():
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None


def _distribute_dtensor(
    tensor: DTensor,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> DTensor:
    """
    Below are experimental enhancements to distribute a DTensor.
    This helps enable Simple FSDP + TP, in which
        inner spec/mesh is TP spec/mesh
        outer spec/mesh is FSDP spec/mesh
    The logic follows
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/_fsdp_param.py#L261
    """
    inner_spec = tensor._spec
    outer_mesh, inner_mesh = device_mesh, inner_spec.mesh
    outer_global_mesh = _mesh_resources.get_root_mesh(outer_mesh)
    inner_global_mesh = _mesh_resources.get_root_mesh(inner_mesh)
    if outer_global_mesh != inner_global_mesh or (
        outer_global_mesh is None or inner_global_mesh is None
    ):
        raise AssertionError(
            "Cannot distribute tensor across two meshes without the same root mesh: \n"
            f"outer global mesh: {outer_global_mesh}\ninner global mesh: {inner_global_mesh}"
        )
    assert outer_mesh.mesh_dim_names is not None
    assert inner_mesh.mesh_dim_names is not None
    submesh_names = outer_mesh.mesh_dim_names + inner_mesh.mesh_dim_names
    spanned_mesh = outer_global_mesh[submesh_names]
    shard_dim = placements[0].dim
    split_factor = inner_spec.num_shards_map[shard_dim]
    tensor_placement = (
        (
            _StridedShard(shard_dim, split_factor=split_factor)
            if split_factor > 1
            else placements[0]
        ),
        inner_spec.placements[0],
    )

    current_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=(Replicate(),),
        tensor_meta=inner_spec.tensor_meta,
    )
    target_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=(placements[0],),
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


def fsdp_policy():
    def _fsdp_recomp_policy():
        def _custom_policy(ctx, func, *args, **kwargs):
            to_recompute = func in {
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                torch.ops._c10d_functional.wait_tensor.default,
                torch.ops.aten._to_copy.default,  # for dtype cast in FSDP
            }
            return (
                CheckpointPolicy.MUST_RECOMPUTE
                if to_recompute
                else CheckpointPolicy.MUST_SAVE
            )

        return _custom_policy

    return create_selective_checkpoint_contexts(_fsdp_recomp_policy())


class ReplicateComputation(torch.nn.Module):
    def __init__(
        self, device_mesh, param_sharding, mode, regional_ac, mp_policy, tp_mesh
    ):
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.mode = mode
        self.compute_placements = [Replicate()] * self.device_mesh.ndim
        self.grad_placements = [Partial(reduce_op="avg")] * self.device_mesh.ndim
        self.regional_ac = regional_ac
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype = mp_policy.param_dtype
        self.reduce_dtype = mp_policy.reduce_dtype
        self.tp_mesh = tp_mesh

    def replicate_compute(self, x):
        # data parallel runtime replicate parameters and do local compute
        # the gradients are partial tensors that needs to perform reduction
        # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)

        # NOTE: specifying mixed precision is only available in pytorch_intern24
        #       https://github.com/tianyu-l/pytorch_intern24/pull/20
        # support for FSDP + TP (assuming TP shards the inner-most dim)
        if self.mode == "fully_shard" and x._spec.mesh.ndim == 2:
            dp_placement, tp_placement = x._spec.placements
            # TODO: remove tp_mesh as an input arg to data_parallel API and use x._spec.mesh["tp"]
            #       after DeviceMesh supports slicing a non-root mesh
            # dp_mesh, tp_mesh = self.device_mesh, x._spec.mesh["tp"]
            dp_mesh, tp_mesh = self.device_mesh, self.tp_mesh

            # re-wrap 2D DTensor to 1D DTensor on dp_mesh for efficient FSDP all-gather
            sharded_local_tensor = x.to_local()
            sharded_dtensor = DTensor.from_local(
                sharded_local_tensor, dp_mesh, self.param_sharding
            )

            # the actuall FSDP all-gather on dp_mesh
            replicated_dtensor = sharded_dtensor.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            )

            # re-wrap 1D all-gathered DTensor on dp_mesh to 1D DTensor on tp_mesh
            # TODO: DTensor should support this mesh collasping operation
            replicated_local_tensor = replicated_dtensor.to_local(
                grad_placements=self.grad_placements
            )
            output = DTensor.from_local(
                replicated_local_tensor, tp_mesh, (tp_placement,)
            )
        else:
            output = x.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            ).to_local(grad_placements=self.grad_placements)

        return output

    def forward(self, x):
        global _active_parametrization
        # This should never be set to true during forward, only outside for model
        # inspection / debugging / initialization
        # model initialization can be done now through
        # with disable_data_parallel():
        #     model.init_weights()
        if not _active_parametrization:
            return x

        if self.regional_ac and self.mode in ("fully_shard", "hybrid_shard"):
            # apply checkpointing to implement reshard_after_forward
            output = checkpoint(
                self.replicate_compute, x, use_reentrant=False, context_fn=fsdp_policy
            )
        else:
            output = self.replicate_compute(x)

        return output


def data_parallel(
    model,
    device_mesh,
    mode="replicate",
    ac_mode: str = "none",
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    tp_mesh: Optional[DeviceMesh] = None,
):
    if mode == "replicate":
        param_sharding = (Replicate(),)
    elif mode == "fully_shard":
        param_sharding = (Shard(0),)
    elif mode == "hybrid_shard":
        # replicate inter-host, fully shard intra-host
        param_sharding = (Replicate(), Shard(0))
        assert (
            device_mesh.ndim == 2
        ), "hybrid sharded data parallel requires 2D DeviceMesh"
    else:
        raise ValueError(f"Unsupported mode {mode}")

    modules = list(model.modules())

    # apply regional ac (with fsdp_policy) if no global ac is to be applied
    regional_ac = ac_mode == "none"

    for mod in modules:
        params_dict = dict(mod.named_parameters(recurse=False))
        for p_name, p in params_dict.items():
            if p is not None and p.numel() > 0:
                distribute_tensor_func = (
                    _distribute_dtensor if isinstance(p, DTensor) else distribute_tensor
                )
                mod.register_parameter(
                    p_name,
                    nn.Parameter(
                        distribute_tensor_func(p, device_mesh, param_sharding)
                    ),
                )
                nn.utils.parametrize.register_parametrization(
                    mod,
                    p_name,
                    ReplicateComputation(
                        device_mesh,
                        param_sharding,
                        mode,
                        regional_ac,
                        mp_policy=mp_policy,
                        tp_mesh=tp_mesh,
                    ),
                    unsafe=True,
                )

    return model
