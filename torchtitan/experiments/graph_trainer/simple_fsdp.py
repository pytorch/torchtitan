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
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
)
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch.fx.traceback import annotate

from torchtitan.distributed.fsdp import linear_param_shard_placements
from torchtitan.protocols.module import Module

from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)

_active_parametrization = True
FSDP_PARAM_FQNS_META = "fsdp_param_fqns"
FSDP_MESH_AXIS_NAMES_META = "fsdp_mesh_axis_names"


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
[Note: SimpleFSDP parameter sharding]

Params arrive as annotated plain tensors, pre-sharded in module.parallelize,
instead of DTensors.

`data_parallel()` first performs full-mesh DTensor translation & FSDP shards,
so rest-time parameters use the FSDP + model-parallel mesh. This keeps DCP
integration and grad norm behavior unchanged.

In pre-forward (ReplicateComputation.forward), we additionally handle any BWD reductions
FSDP is expected to do, as the sharding annotations (R/I@TP for SP on/off) are assuming
FSDP does its job (R FWD <-> P BWD assumes FSDP redistributes to I).
We lookup parameter typing on non-FSDP axes, and P->I all-reduce in post-backward
if annotated as R.
For other types: I is no-op, S(i) is sharded at rest, P is banned in titan for now.
"""


def _prepare_spmd_parameter_for_fsdp(
    tensor: torch.Tensor,
    param_name: str,
    non_dp_mesh: DeviceMesh | None,
) -> tuple[torch.Tensor, dict[spmd.MeshAxis, spmd.PerMeshAxisSpmdType]]:
    """Prepare an SPMD-annotated parameter for SimpleFSDP.

    Record the parameter's model-parallel axis types for ReplicateComputation
    and restore its DTensor wrapper on ``non_dp_mesh``.
    """
    non_dp_mesh_types = {}
    if non_dp_mesh is None:
        return tensor, non_dp_mesh_types

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
    return (
        DTensor.from_local(tensor, non_dp_mesh, placements, run_check=False),
        non_dp_mesh_types,
    )


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
    # The outer mesh distributes the TP/EP-local tensor, not the global tensor.
    local_meta = TensorMeta(
        tensor._local_tensor.shape, tensor._local_tensor.stride(), tensor.dtype
    )

    current_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=current_placements,
        tensor_meta=local_meta,
    )
    target_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=target_placements,
        tensor_meta=local_meta,
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


class _BuildUnshardedTensorFunction(torch.autograd.Function):
    """Own the gradient edge for an unsharded tensor built outside FSDP2.

    FSDP2 creates this edge internally for its post-all-gather output.
    GraphTrainer's SimpleFSDP reconstructs the unsharded weight itself and so
    has no such edge, and this routes its logical tensor gradient straight
    back through the high-precision gather.

    Lives here, with its only user, rather than beside the FSDP2 edge it
    mirrors. ``forward`` does the same two steps ``fsdp_post_all_gather`` does
    on its first unshard -- quantize, then wrap.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, weight: torch.Tensor, wrapper: _ShardedFSDPTensor):
        del ctx
        # replicate_compute returns a plain local tensor under spmd_types. Keep
        # this invariant explicit so a future storage format cannot silently
        # quantize a DTensor's local shard.
        assert not isinstance(weight, DTensor), (
            "unsharded tensor construction received a DTensor instead of a "
            "plain local tensor"
        )
        with torch.no_grad():
            return _UnshardedFSDPTensor(weight, wrapper._build_operands(weight))

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_weight: torch.Tensor):
        del ctx
        return grad_weight, None


class _UnshardAndReduceGradFunction(torch.autograd.Function):
    """Unshard a parameter in forward and reduce its gradient in backward.

    Forward (compute time), for one parameter:
      1. Wrap ``sharded_local_tensor`` (the parameter's local FSDP shard, in
         the master dtype) as a DTensor on ``dp_mesh`` with ``dp_placements``.
      2. Cast it to ``param_dtype``, then unshard it over DP (all-gather for
         FSDP/HSDP, no-op for DDP), and return the local unsharded weight.
      3. Declare ``reduce_dtype`` as the gradient dtype of the returned
         unsharded weight (``ctx.set_output_grad_dtype``), so autograd casts
         any gradient flowing into it to ``reduce_dtype``. FSDP2 does the same
         with ``grad_dtype``, which works there because its unsharded weight
         is a leaf; here it is an intermediate.

    Backward (compute time):
      1. The gradient of the unsharded weight arrives in ``param_dtype``; the
         ``ctx.set_output_grad_dtype`` call from forward step 3 casts it to
         ``reduce_dtype``.
      2. All-reduce it over the model-parallel axes (e.g. TP/EP) where the
         parameter is replicated, if any.
      3. Reduce it over DP: reduce-scatter for FSDP, all-reduce for DDP, both
         for HSDP.
      4. Cast the local shard to the master dtype and return it as the
         gradient of ``sharded_local_tensor``.

    Why one Function: FSDP2 runs its gradient reductions outside autograd, on
    the ``.grad`` of its unsharded parameter, a leaf with ``grad_dtype =
    reduce_dtype``. SimpleFSDP runs them inside autograd, where the unsharded
    weight is an intermediate and ``grad_dtype`` cannot be set; with each
    collective in its own autograd node, autograd would round the gradient to
    ``param_dtype`` between them.

    Args:
        sharded_local_tensor: The parameter's local FSDP shard.
        dp_mesh: The data-parallel mesh (DP axes only; TP/EP are excluded).
        dp_placements: The parameter's placements on ``dp_mesh``.
        unsharded_shape: Shape of the FSDP-unsharded weight on this rank. Not
            the global shape: still sharded over model-parallel axes (TP/EP).
        unsharded_stride: Stride of the FSDP-unsharded weight on this rank.
        param_dtype: Dtype of the all-gather and the unsharded weight. ``None``
            keeps the master dtype.
        reduce_dtype: Dtype of the gradient reductions. ``None`` uses the
            master dtype.
        non_dp_replicate_axes: The model-parallel (e.g. TP/EP) mesh axes where
            the parameter is replicated (R).
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        sharded_local_tensor: torch.Tensor,
        dp_mesh: DeviceMesh,
        dp_placements: tuple[Placement, ...],
        unsharded_shape: torch.Size,
        unsharded_stride: tuple[int, ...],
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
        non_dp_replicate_axes: list[spmd.MeshAxis],
    ) -> torch.Tensor:
        ctx.dp_mesh = dp_mesh
        ctx.dp_placements = dp_placements
        ctx.reduce_dtype = reduce_dtype
        ctx.non_dp_replicate_axes = non_dp_replicate_axes
        ctx.sharded_dtype = sharded_local_tensor.dtype
        # TODO: Drop the guard once the nightly has Dynamo support for
        # set_output_grad_dtype. Until then, under torch.compile a gradient
        # computed in a wider dtype is rounded to param_dtype on entry.
        if not torch.compiler.is_dynamo_compiling():
            # FSDP2 resolves the unsharded grad_dtype the same way.
            ctx.set_output_grad_dtype(reduce_dtype or sharded_local_tensor.dtype)
        sharded_dtensor = DTensor.from_local(
            sharded_local_tensor,
            dp_mesh,
            dp_placements,
            shape=unsharded_shape,
            stride=unsharded_stride,
        )
        unsharded_dtensor = sharded_dtensor.redistribute(
            placements=[Replicate()] * dp_mesh.ndim, forward_dtype=param_dtype
        )
        return unsharded_dtensor.to_local()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad: torch.Tensor):
        for axis in ctx.non_dp_replicate_axes:
            grad = spmd.all_reduce(
                grad, axis, src=spmd.P, dst=spmd.I, op_dtype=ctx.reduce_dtype
            )
        partial_grad = DTensor.from_local(
            grad,
            ctx.dp_mesh,
            [Partial(reduce_op="sum")] * ctx.dp_mesh.ndim,
            run_check=False,
        )
        sharded_grad = partial_grad.redistribute(
            placements=ctx.dp_placements, forward_dtype=ctx.reduce_dtype
        )
        return (
            sharded_grad.to_local().to(ctx.sharded_dtype),
            *([None] * 7),
        )


class ReplicateComputation(Module):
    def __init__(
        self,
        param_fqn: str,
        device_mesh: DeviceMesh,
        param_sharding: tuple[Placement, ...],
        mode: str,
        mp_policy: MixedPrecisionPolicy | None,
        non_dp_mesh_types: dict[spmd.MeshAxis, spmd.PerMeshAxisSpmdType],
    ) -> None:
        super().__init__()
        self.param_fqn = param_fqn
        self.device_mesh = device_mesh
        self.mesh_axis_names = tuple(device_mesh.mesh_dim_names or ())
        self.param_sharding = param_sharding
        self.mode = mode
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype: torch.dtype | None = mp_policy.param_dtype
        self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype

        # Non-FSDP (model-parallel) axes where the parameter is R. FSDP is
        # responsible for their backward all-reduces: e.g. TP RMSNorm with SP
        # uses R, so add P->I in backward. I remains a no-op.
        self.non_dp_replicate_axes = [
            axis for axis, axis_type in non_dp_mesh_types.items() if axis_type is spmd.R
        ]

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        # data parallel runtime replicate parameters and do local compute
        # the gradients are partial tensors that needs to perform reduction
        # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)
        # support FSDP/DDP/HSDP + EP + TP (assuming TP shards the inner-most dim)
        non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
        assert non_dp_mesh_dims <= 2, "Only DP + EP/TP/EP+TP is supported"
        if non_dp_mesh_dims > 0:
            non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
            non_dp_mesh_axis_names = tuple(
                x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:]
            )
            non_dp_mesh = x._spec.mesh[non_dp_mesh_axis_names]
            # Recover the TP/EP-local shape from global metadata, including uneven
            # shards. DP-local sizes alone cannot recover uneven shard lengths.
            local_shape, _ = compute_local_shape_and_global_offset(
                x.shape, non_dp_mesh, non_dp_placements, skip_offset=True
            )
            # Preserve the input's logical dimension order when projecting its
            # layout onto the TP/EP-local shape.
            local_stride = compute_local_stride(x.stride(), local_shape)
        elif non_dp_mesh_dims == 0:
            local_shape, local_stride = x.shape, x.stride()
        else:
            raise AssertionError(
                f"Unsupported replicate compute on placement {x._spec.placements} for DTensor {x}"
            )

        # Re-wrap the TP/EP-local tensor as a DTensor on dp_mesh for an efficient
        # FSDP all-gather.
        # TODO: DTensor should support this mesh collapsing operation
        return _UnshardAndReduceGradFunction.apply(
            x.to_local(),
            self.device_mesh,
            self.param_sharding,
            torch.Size(local_shape),
            tuple(local_stride),
            self.param_dtype,
            self.reduce_dtype,
            self.non_dp_replicate_axes,
        )

    def forward(self, x: DTensor) -> torch.Tensor:
        with annotate(
            {
                FSDP_PARAM_FQNS_META: (self.param_fqn,),
                FSDP_MESH_AXIS_NAMES_META: self.mesh_axis_names,
            }
        ):
            return self._forward(x)

    def _forward(self, x: DTensor) -> torch.Tensor:
        global _active_parametrization
        # This should never be set to true during forward, only outside for model
        # inspection / debugging / initialization
        # model initialization can be done now through
        # with disable_active_parametrization():
        #     model.init_states()
        if not _active_parametrization:
            return x

        unsharded_weight = self.replicate_compute(x)
        # Which operands to build is determined by the tensor subclass on the
        # *sharded* parameter, and replicate_compute does not preserve it: its
        # return value is a plain local tensor under spmd_types. Its input is
        # not -- ``x`` is the sharded parameter, still a DTensor, because
        # data_parallel translates parameters to full-mesh DTensors before
        # this runs (replicate_compute reads ``x._spec`` on entry for the same
        # reason). So take the subclass from ``x`` and the values from the
        # return value. A parameter with no operands passes straight through,
        # so this costs nothing when nothing is quantized.
        #
        # Read the local tensor directly rather than through ``to_local``:
        # only the subclass type is wanted, so the autograd edge and the extra
        # traced node that ``to_local`` adds would both be dead weight.
        source = x._local_tensor
        if isinstance(source, _UnshardedFSDPTensor):
            raise RuntimeError(
                "The data parallel parametrization received an already-"
                "unsharded weight. FSDP2 builds the unsharded tensor in "
                "fsdp_post_all_gather and owns its gradient edge, so building "
                "one here as well would quantize the weight a second time."
            )
        if not isinstance(source, _ShardedFSDPTensor):
            return unsharded_weight
        return _BuildUnshardedTensorFunction.apply(unsharded_weight, source)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    # Model-parallel (TP/EP) mesh used to construct DTensor parameters on the
    # full mesh.
    # TODO: Unify this with device_mesh as a global data- and model-parallel mesh.
    non_dp_mesh: DeviceMesh | None = None,
    param_shard_placements: dict[nn.Parameter, Shard] | None = None,
) -> nn.Module:
    """Shard ``model`` and install the data-parallel parametrization.

    ``param_shard_placements`` overrides the per-parameter shard placement;
    parameters it omits use ``Shard(shard_dim)``. It defaults to
    ``linear_param_shard_placements(model)``.
    """
    if mode == "hybrid_shard":
        # replicate inter-host, fully shard intra-host
        assert (
            device_mesh.ndim == 2
        ), "hybrid sharded data parallel requires 2D DeviceMesh"
    elif mode not in ("replicate", "fully_shard"):
        raise ValueError(f"Unsupported mode {mode}")

    if param_shard_placements is None:
        param_shard_placements = linear_param_shard_placements(model)

    def get_param_sharding(param: nn.Parameter) -> tuple[Placement, ...]:
        if mode == "replicate":
            return (Replicate(),)
        placement = param_shard_placements.get(param, Shard(shard_dim))
        if mode == "fully_shard":
            return (placement,)
        return (Replicate(), placement)

    for module_fqn, mod in model.named_modules():
        params_dict = dict(mod.named_parameters(recurse=False))
        # we shouldn't apply data parallel to the modules that are already
        # sharded by data parallel
        if "SimpleFSDP" in mod.__class__.__name__:
            continue

        param_non_dp_mesh_types = {}
        param_shardings: dict[str, tuple[Placement, ...]] = {}

        for p_name, p in params_dict.items():
            if p is not None:
                param_sharding = param_shardings[p_name] = get_param_sharding(p)
            if p is not None and p.numel() > 0:
                p, non_dp_mesh_types = _prepare_spmd_parameter_for_fsdp(
                    p,
                    p_name,
                    non_dp_mesh,
                )
                param_non_dp_mesh_types[p_name] = non_dp_mesh_types
                distribute_tensor_func = (
                    _distribute_dtensor if isinstance(p, DTensor) else distribute_tensor
                )
                mod.register_parameter(
                    p_name,
                    nn.Parameter(
                        distribute_tensor_func(p, device_mesh, param_sharding),
                        requires_grad=p.requires_grad,
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
                param_fqn=(f"{module_fqn}.{param_name}" if module_fqn else param_name),
                device_mesh=device_mesh,
                param_sharding=param_shardings[param_name],
                mode=mode,
                mp_policy=mp_policy,
                non_dp_mesh_types=param_non_dp_mesh_types.get(param_name, {}),
            ),
        )
    return model
