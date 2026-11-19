# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import Shard

from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.parallelism import FlexShardConfig, FSDP2Config, FSDPSymmMemScope
from torchtitan.distributed.flex_shard_fsdp import (
    declare_state_dict_layouts,
    flex_shard_bucket_mesh,
)
from torchtitan.distributed.fsdp_lifecycle import complete_fsdp_grouped_forward
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.param_init import skip_param_init

__all__ = [
    "apply_fsdp_to_decoder",
    "apply_fsdp_to_multimodal_encoder",
    "disable_fsdp_gradient_division",
    "enable_fsdp_symm_mem",
    "post_validate_fsdp_for_accelerator_graph",
    "get_fsdp_reshard_after_forward_policy",
    "linear_param_shard_placements",
    "mixed_precision_policy",
    "resolve_fsdp_mesh",
    "resolve_sparse_fsdp_mesh",
    "maybe_store_frozen_params_in_param_dtype_",
]

logger = logging.getLogger(__name__)


def post_validate_fsdp_for_accelerator_graph(
    model_parts: list[nn.Module], *, flex_shard_enabled: bool
) -> None:
    """Finish grouped forwards and release full parameters for pool reuse.

    Eager and captured validation use the same cleanup. With ChunkedLossWrapper,
    the model skips lm_head, then the loss calls it after the root forward.
    A group containing norm and lm_head can still have norm pending. Validation
    has no backward callback to clear this state. With FlexShard
    (``flex_shard_enabled``), FlexShard's ``reshard()`` on each model part
    closes the same forward of norm and lm_head's bucket.
    """
    for model in model_parts:
        if flex_shard_enabled:
            cast(Any, model).reshard()
        for module in model.modules():
            if not isinstance(module, FSDPModule):
                continue
            complete_fsdp_grouped_forward(module, None)
            module.reshard()


def _fsdp_unit_holds_only_embedding_weights(
    fsdp_module: FSDPModule, model: nn.Module
) -> bool:
    """Whether every parameter managed by ``fsdp_module``'s FSDP unit is an
    ``nn.Embedding`` weight in ``model``.

    Reads the unit's own parameters rather than ``fsdp_module.parameters()``,
    so modules grouped into the same ``fully_shard`` call are also covered.
    """
    # TODO: Use a public FSDP API to list the parameters of an FSDP unit.
    unit_params = [
        fsdp_param.sharded_param
        for param_group in fsdp_module._get_fsdp_state()._fsdp_param_groups
        for fsdp_param in param_group.fsdp_params
    ]
    embedding_weights = {
        m.weight for m in model.modules() if isinstance(m, nn.Embedding)
    }
    return all(p in embedding_weights for p in unit_params)


if TYPE_CHECKING:
    from torchtitan.models.common.decoder import Decoder


_DENSE_STORAGE_AXES = ["dp_replicate", "dp_shard", "cp", "tp"]
_SPARSE_STORAGE_AXES = ["dp_replicate", "edp_shard", "ep"]


def linear_param_shard_placements(
    module: nn.Module,
    *,
    include_unstacked_grouped: bool = False,
) -> dict[nn.Parameter, Shard]:
    """Shard linear parameters along their matrix-row dimension.

    A stacked Linear stores weight as ``[N, F, D]`` and bias as ``[N, F]``.
    A GroupedLinear prepends its group dimension, producing ``[E, N, F, D]``
    for stacked weights. FSDP's default ``Shard(0)`` would split a logical
    projection or group dimension, so shard the matrix rows instead.

    Args:
        module: Module tree whose linear parameters should be inspected.
        include_unstacked_grouped: Also return ordinary ``[E, F, D]`` grouped
            weights. This is used when the expert dimension is too small for
            the E/FSDP mesh and expert weights must shard their matrix rows.
    """
    placements: dict[nn.Parameter, Shard] = {}
    for child in module.modules():
        if isinstance(child, GroupedLinear):
            if child.num_linears == 1 and not include_unstacked_grouped:
                continue
        elif not isinstance(child, nn.Linear) or getattr(child, "num_linears", 1) == 1:
            continue
        weight = cast(nn.Parameter, child.weight)
        placements[weight] = Shard(weight.ndim - 2)
        if (bias := getattr(child, "bias", None)) is not None:
            placements[bias] = Shard(bias.ndim - 1)
    return placements


class _FSDPGroup(NamedTuple):
    """Modules that ``apply_fsdp_to_decoder`` passes to one ``fully_shard`` call.

    ``placements`` holds every parameter whose placement isn't ``Shard(0)``,
    plus all routed-expert parameters. ``routed_experts`` is the block's routed
    experts, which expert parallelism shards over a separate mesh.
    """

    modules: list[nn.Module]
    reshard_after_forward: bool
    placements: dict[nn.Parameter, Shard]
    routed_experts: nn.Module | None = None


def _decoder_blocks(model: "Decoder") -> list[nn.Module]:
    """Return a decoder's transformer blocks in forward order: ``layers``, then
    the MTP layers of a multi-token prediction decoder."""
    return [*model.layers.values(), *(getattr(model, "mtp_layers", None) or ())]


def _decoder_fsdp_groups(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    *,
    reshard_after_forward_policy: str,
    pp_enabled: bool,
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
) -> tuple[_FSDPGroup | None, _FSDPGroup | None, list[_FSDPGroup]]:
    """Return a decoder's FSDP groups as ``(embedding, head, blocks)``.

    The embedding group also holds the norm and output projection when weights
    are tied, and there is no head group then. Each transformer block is a
    group, MTP layers included. The root group, ``fully_shard(model)``, takes
    the parameters left over and doesn't reshard after forward.

    Args:
        model: The decoder.
        dp_mesh: The FSDP mesh, as for ``apply_fsdp_to_decoder``.
        reshard_after_forward_policy: "default", "always" or "never".
        pp_enabled: Whether pipeline parallelism is enabled.
        ep_degree: The expert-parallel degree.
        edp_mesh: The FSDP mesh for routed experts when ``ep_degree > 1``.
    """
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )
    embedding = head = None
    if model.enable_weight_tying:
        # When weights are tied, tok_embeddings and output share the same parameter.
        # Group them together in one FSDP unit to avoid duplicate all-gathers.
        embedding = _FSDPGroup(
            [
                m
                for m in (model.tok_embeddings, model.norm, model.lm_head)
                if m is not None
            ],
            reshard_after_forward_policy == "always",
            {},
        )
    else:
        if model.tok_embeddings is not None:
            embedding = _FSDPGroup([model.tok_embeddings], reshard_after_forward, {})
        # As an optimization, do not reshard_after_forward the last layers
        # by default since FSDP would prefetch them immediately.
        if model.norm is not None and model.lm_head is not None:
            head = _FSDPGroup(
                [model.norm, model.lm_head],
                reshard_after_forward_policy == "always",
                {},
            )

    blocks = []
    for transformer_block in _decoder_blocks(model):
        # A stacked Linear keeps W1/W3 separate from the matrix-row dimension.
        # Shard matrix rows so every rank retains both projections.
        placements = linear_param_shard_placements(transformer_block)
        moe = next(
            (m for m in transformer_block.children() if isinstance(m, MoE)), None
        )
        if moe is not None:
            if ep_degree > 1:
                assert edp_mesh is not None
                expert_sharding_size = edp_mesh["edp_shard"].size() * ep_degree
            else:
                # FSDP cuts dim 0 only over its shard axes: ``dp_shard``,
                # plus ``cp`` when CP is on (see ``resolve_fsdp_mesh``).
                # ``dp_replicate`` replicates, and ``tp`` shards other dims
                # via the TP plan, so neither divides dim 0.
                expert_sharding_size = dp_mesh["dp_shard"].size()
                if "cp" in (dp_mesh.mesh_dim_names or ()):
                    expert_sharding_size *= dp_mesh["cp"].size()
            placements.update(
                {param: Shard(0) for param in moe.routed_experts.parameters()}
            )
            # Experts may shard their output features when there are more
            # expert shards than experts.
            if expert_sharding_size > moe.num_experts:
                placements.update(
                    linear_param_shard_placements(
                        moe.routed_experts,
                        include_unstacked_grouped=True,
                    )
                )
        blocks.append(
            _FSDPGroup(
                [transformer_block],
                reshard_after_forward,
                placements,
                None if moe is None else moe.routed_experts,
            )
        )
    return embedding, head, blocks


def _decoder_prefetch_order(model: "Decoder") -> list[nn.Module]:
    """Return the embedding, each transformer block and the output projection,
    in forward order, each standing for its FSDP group.

    With explicit prefetching, each group prefetches the next one in forward and
    the previous one in backward.
    """
    order = _decoder_blocks(model)
    if not order:
        return []
    if model.tok_embeddings is not None:
        order.insert(0, model.tok_embeddings)
    if model.norm is not None and model.lm_head is not None:
        order.append(model.lm_head)
    return order


def resolve_fsdp_mesh(
    parallelism_context: ParallelismContext,
    *,
    shard_on_cp_axis: bool = True,
) -> tuple[DeviceMesh, DataParallelMeshDims | None]:
    """Select the dense storage mesh and DataParallelMeshDims.

    ``dp_shard`` is always included (force-kept-alive in the dense storage mesh
    even at size 1) so FSDP can pick the DP submesh out of the multi-axis
    storage mesh inside ``DeviceMesh._concatenate([dp_mesh, tp_mesh])``.

    ``shard_on_cp_axis=False`` leaves CP in the storage mesh but outside FSDP's
    reduction axes. Parameters must then be CP-invariant, with their gradient
    contributions already reduced across CP before FSDP reduction.
    """
    storage_mesh = parallelism_context.get_activated_mesh(_DENSE_STORAGE_AXES)
    assert storage_mesh is not None

    if storage_mesh.size() == 1:
        # ``assert_type`` filters out inactive size-1 axes, so params get no
        # annotations under a size-1 full mesh. That leaves ``fully_shard()``
        # with no SPMD annotations to translate to DTensor params, so do not
        # pass a DataParallelMeshDims object to FSDP.
        return storage_mesh, None

    shard_axes = ["dp_shard"]
    if parallelism_context.cp_enabled and shard_on_cp_axis:
        shard_axes.append("cp")
    shard: str | tuple[str, ...] = (
        tuple(shard_axes) if len(shard_axes) > 1 else shard_axes[0]
    )
    replicate = "dp_replicate" if parallelism_context.dp_replicate_enabled else None

    return storage_mesh, DataParallelMeshDims(shard=shard, replicate=replicate)


def resolve_sparse_fsdp_mesh(
    parallelism_context: ParallelismContext,
) -> tuple[DeviceMesh | None, DataParallelMeshDims | None]:
    """Sparse counterpart of ``resolve_fsdp_mesh`` for routed experts.

    Returns ``(None, None)`` when EP is disabled; otherwise the sparse
    storage mesh + sparse DP axes. The FSDP axis is ``edp_shard`` and
    ``dp_replicate`` is shared with the dense path.
    """
    if not parallelism_context.ep_enabled:
        return None, None
    sparse_mesh = parallelism_context.get_activated_mesh(_SPARSE_STORAGE_AXES)
    assert sparse_mesh is not None
    replicate = "dp_replicate" if parallelism_context.dp_replicate_enabled else None
    return sparse_mesh, DataParallelMeshDims(shard="edp_shard", replicate=replicate)


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """
    Disable FSDP's automatic gradient division for all FSDP modules.

    Set gradient_divide_factor=1.0 to disable FSDP's automatic gradient division.
    We handle gradient scaling ourselves in the training loop with global token count.

    Note: This also works for ReplicateModule since it inherits from FSDPModule.

    Args:
        model: The model containing FSDP-wrapped or Replicate-wrapped modules
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def enable_fsdp_symm_mem(model: nn.Module, scope: FSDPSymmMemScope) -> None:
    """Enable symmetric-memory communication for the FSDP modules ``scope`` selects."""
    if scope is None:
        return
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        if scope == "dense" and any(isinstance(m, MoE) for m in module.children()):
            continue
        module.set_force_sum_reduction_for_comms(True)
        module.set_symm_mem_for_comm()


def get_fsdp_reshard_after_forward_policy(
    reshard_after_forward_policy: str, pp_enabled: bool
) -> bool:
    """Resolve fsdp_reshard_after_forward policy string to a boolean.

    Args:
        reshard_after_forward_policy: One of "always", "never", or "default".
        pp_enabled: Whether pipeline parallelism is enabled.

    Returns:
        Boolean indicating whether to reshard after forward.
    """
    match reshard_after_forward_policy:
        case "always":
            return True
        case "never":
            return False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            return not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )


def mixed_precision_policy(
    module: Module,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cast_forward_inputs: bool = True,
) -> MixedPrecisionPolicy:
    """Build the FSDP mixed precision policy for the parameters of ``module``.

    Parameters listed in their module's ``preserve_dtype_params`` are
    all-gathered and computed in the wider of their own dtype and
    ``param_dtype``; the rest in ``param_dtype``.
    Call before ``fully_shard`` replaces the parameters: the override matches
    parameters by identity.
    """
    preserved = {param for _, param in module.named_preserve_dtype_params()}

    def param_dtype_override_fn(param: nn.Parameter) -> torch.dtype | None:
        if param not in preserved:
            return None
        return torch.promote_types(param.dtype, param_dtype)

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=cast_forward_inputs,
        param_dtype_override_fn=param_dtype_override_fn if preserved else None,
    )


def maybe_store_frozen_params_in_param_dtype_(
    module: Module, precision: PrecisionConfig
) -> None:
    """Store the frozen parameters of ``module`` in ``precision.param_dtype`` when
    it is narrower than ``precision.dtype``.

    A frozen parameter is never updated and FSDP computes it in ``param_dtype``,
    so a wider dtype only costs memory. Parameters listed in
    ``preserve_dtype_params`` keep their dtype. Each tensor is
    swapped rather than assigned to ``param.data``, because setting ``.data`` would
    leave a quantized tensor subclass's inner tensor in the old dtype. Parameter
    identity (tied weights) and attributes (e.g. the SPMD type) are kept. Call
    before ``fully_shard``, which records parameter dtypes.
    """
    param_dtype = TORCH_DTYPE_MAP[precision.param_dtype]
    if param_dtype.itemsize >= TORCH_DTYPE_MAP[precision.dtype].itemsize:
        return
    preserved = {param for _, param in module.named_preserve_dtype_params()}
    for param in module.parameters():
        if param.requires_grad or param in preserved or param.dtype == param_dtype:
            continue
        new_param = nn.Parameter(param.detach().to(param_dtype), requires_grad=False)
        for name, value in param.__dict__.items():
            new_param.__dict__.setdefault(name, value)
        torch.utils.swap_tensors(param, new_param)


def apply_fsdp_to_multimodal_encoder(
    encoder: Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
    pp_enabled: bool = False,
    cpu_offload: bool = False,
    *,
    dp_mesh_dims: DataParallelMeshDims | None = None,
    unshard_in_current_stream: bool = False,
) -> None:
    """Apply FSDP to a multimodal encoder as a single unit.

    One all-gather for all encoder parameters is more efficient than per-layer
    sharding for the relatively small modality tower. Call before
    ``apply_fsdp_to_decoder`` so the encoder is already sharded.

    ``cpu_offload`` must match what the caller passes to ``apply_fsdp_to_decoder``.
    Under ``parallelism.fsdp_enable_cpu_offloading`` the trainer materializes the whole model
    on CPU, so an encoder sharded without ``CPUOffloadPolicy`` keeps CPU
    parameters while FSDP produces CUDA gradients for them, and backward dies with
    "attempting to assign a gradient with device type 'cuda' to a tensor with
    device type 'cpu'".
    """
    if unshard_in_current_stream:
        # The explicit prefetching chain in apply_fsdp_to_decoder does not
        # cover encoders, so their all-gathers would not overlap with compute.
        raise ValueError(
            "parallelism.fsdp_unshard_in_current_stream is not supported for models "
            "with multimodal encoders."
        )
    mp_policy = mixed_precision_policy(
        encoder, param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=pp_enabled
    )
    fsdp_config: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
        "dp_mesh_dims": dp_mesh_dims,
    }
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    fully_shard(encoder, **fsdp_config)


def _declare_tensor_parallelism(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    *,
    routed_expert_params: set[nn.Parameter],
) -> None:
    """Declare to FlexShard how tensor parallelism splits each parameter of
    ``model`` and its grad. Call before ``flex_shard()``.

    ``Module._parallelize`` leaves each parameter as its tp rank's local shard,
    with the SPMD type that FSDP2 lays it out by:

    - A parameter that TP splits declares where its shard sits in the full
      parameter, as its outer layout. flex_shard composes it with the bucket's
      split in the layouts the wrapped ``param_init`` functions use.
    - A parameter typed ``R`` on tp, e.g. a norm weight under sequence
      parallelism, has a grad that's partial over tp. FSDP2 types the grad
      ``P`` and all-reduces it over tp before its reduce-scatter. FlexShard does
      the same for a parameter that declares tp as its partial-grad group.
    - A parameter typed ``I`` on tp declares nothing: each tp rank computes
      its whole grad.
    - Any other type on tp, or none, raises. Untyped, a parameter would declare
      itself whole, and the grad norm would count it once per tp rank.

    Routed experts are skipped. MoE models need ``ep >= tp``, so the routed
    experts are typed on the sparse mesh, and keep their EP layout.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import set_partial_grad_group

    # pyrefly: ignore [missing-import]
    from flex_shard.layout_adapters.spmd_types import spmd_types_to_global_layout

    tp_group = tp_mesh.get_group()
    tp_axis = spmd.MeshAxis.of(tp_group)
    layouts = {}
    for fqn, param in model.named_parameters():
        if param in routed_expert_params:
            continue
        local_type = spmd.get_local_type(param)
        tp_type = local_type.get(tp_axis)
        if tp_type is spmd.R:
            set_partial_grad_group(param, tp_group)
        elif tp_type is spmd.V or isinstance(tp_type, spmd.Shard):
            # Split by TP: V with its dim in the PartitionSpec, or S(dim). The
            # adapter raises for V without a dim.
            layouts[fqn] = spmd.SpmdType(
                dict(local_type), spmd.get_partition_spec(param)
            )
        elif tp_type is not spmd.I:
            raise ValueError(
                "parallelism.fsdp_backend=FlexShardConfig() supports only "
                "parameters typed R, I, V or S(dim) on tp under tensor "
                f"parallelism, but {fqn} is {tp_type}."
            )
    spmd_types_to_global_layout(model, layouts, tp_mesh)


def _apply_flex_shard(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    groups: list[_FSDPGroup],
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    dp_mesh_dims: DataParallelMeshDims | None,
    edp_mesh: DeviceMesh | None,
) -> None:
    """Shard ``model`` with FlexShard (meta-pytorch/flex_shard), one
    ``fsdp2_compatible`` bucket per FSDP group.

    Each bucket names the group's modules, with its placements and reshard
    setting, and a root bucket takes the parameters left over, as
    ``fully_shard(model)`` does. ``dp_mesh`` and ``dp_mesh_dims`` are the dense
    storage mesh and its data-parallel axes, as ``fully_shard`` gets them. The
    buckets shard over the mesh ``fully_shard`` shards over: ``dp_shard``, or
    under context parallelism, ``dp_shard`` and ``cp`` flattened into one axis.
    Gradients are summed over it without division, as
    ``disable_fsdp_gradient_division`` makes FSDP2 do. The buckets' collectives
    then match FSDP2's byte for byte. Each sharded parameter's ``param_init``
    function is wrapped with flex_shard's ``init_from_full``, so the shards start
    as one device's initialization. ``model.state_dict()`` declares each local
    shard's place in the full parameter, so DCP saves the same checkpoints as for
    FSDP2.

    Under tensor parallelism, each parameter is its tp rank's local shard, and
    declares before ``flex_shard()`` how TP splits it and its grad (see
    ``_declare_tensor_parallelism``).

    With expert parallelism, ``edp_mesh`` is the sparse mesh, with the
    ``edp_shard`` and ``ep`` axes. ``fully_shard`` splits an MoE block into one
    group per mesh, so the block gets two buckets: its routed experts on
    ``edp_shard``, and its other parameters on the dense buckets' mesh. The
    buckets prefetch as ``apply_fsdp_to_decoder`` sets FSDP2's groups to.
    """
    # flex_shard is an optional dependency, needed only for this backend.
    # pyrefly: ignore [missing-import]
    from flex_shard import (
        BucketSpec,
        flex_shard,
        get_flex_shard_global_layouts,
        GlobalLayout,
        init_from_full,
        MixedPrecisionPolicy,
        set_global_layout,
    )

    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.shard import Shard as FlexShardShard

    # flex_shard_reduce_meshes gives the mesh to reduce the grad norm over for
    # the parameters of buckets that another parallelism split first.
    bucket_mesh = flex_shard_bucket_mesh(dp_mesh, dp_mesh_dims)
    tp_mesh = dp_mesh["tp"] if "tp" in (dp_mesh.mesh_dim_names or ()) else None
    if tp_mesh is not None:
        _declare_tensor_parallelism(
            model,
            tp_mesh,
            routed_expert_params={
                param
                for group in groups
                if group.routed_experts is not None
                for param in group.routed_experts.parameters()
            },
        )
    expert_mesh = None
    if edp_mesh is not None:
        assert edp_mesh.mesh_dim_names == ("edp_shard", "ep"), edp_mesh.mesh_dim_names
        expert_mesh = edp_mesh["edp_shard"]
        # EP has split each routed-expert parameter's dim 0, the experts, over
        # ep. Declaring this rank's part makes the layouts that the wrapped
        # param_init functions use cover all experts, so each shard is its slice
        # of one device's init, which briefly holds the parameter with all experts.
        ep, ep_rank = edp_mesh["ep"].size(), edp_mesh["ep"].get_local_rank()
        for group in groups:
            if group.routed_experts is None:
                continue
            for param in group.routed_experts.parameters():
                num_local_experts, *rest = param.shape
                set_global_layout(
                    param,
                    GlobalLayout(
                        global_shape=(num_local_experts * ep, *rest),
                        global_offsets=(
                            (num_local_experts * ep_rank, *[0] * len(rest)),
                        ),
                        local_offsets=((0,) * param.ndim,),
                        local_sizes=(tuple(param.shape),),
                    ),
                )
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    # Shard dims other than 0, which FSDP2 gets from shard_placement_fn.
    shard_dims = {
        param: placement.dim
        for group in groups
        for param, placement in group.placements.items()
    }

    def placement_fn(named_params, mesh):
        del mesh
        return {
            fqn: (FlexShardShard(shard_dims.get(param, 0)),)
            for fqn, param in named_params
        }

    def bucket(
        patterns: list[str],
        reshard_after_forward: bool,
        mesh: DeviceMesh = bucket_mesh,
    ) -> BucketSpec:
        return BucketSpec(
            patterns,
            placement_fn=placement_fn,
            mesh=mesh,
            mp_policy=mp_policy,
            gradient_divide_factor=1.0,
            reshard_after_forward=reshard_after_forward,
            fsdp2_compatible=True,
        )

    module_fqns = {module: fqn for fqn, module in model.named_modules()}
    param_fqns = {param: fqn for fqn, param in model.named_parameters()}
    group_buckets: list[list[BucketSpec]] = []
    for group in groups:
        if expert_mesh is None or group.routed_experts is None:
            patterns = [module_fqns[module] for module in group.modules]
            group_buckets.append([bucket(patterns, group.reshard_after_forward)])
            continue
        # The expert bucket hooks the routed experts. The other bucket names
        # the rest of the block's parameters, so it hooks their deepest common
        # module, the block.
        expert_params = set(group.routed_experts.parameters())
        dense_fqns = [
            param_fqns[param]
            for module in group.modules
            for param in module.parameters()
            if param not in expert_params
        ]
        group_buckets.append(
            [
                bucket(dense_fqns, group.reshard_after_forward),
                bucket(
                    [module_fqns[group.routed_experts]],
                    group.reshard_after_forward,
                    expert_mesh,
                ),
            ]
        )
    buckets = [spec for specs in group_buckets for spec in specs]
    # fully_shard(model) puts the remaining parameters in the root group, which
    # FSDP2 does not reshard after forward.
    grouped_params = {
        param
        for group in groups
        for module in group.modules
        for param in module.parameters()
    }
    root_fqns = [
        fqn for fqn, param in model.named_parameters() if param not in grouped_params
    ]
    if root_fqns:
        buckets.append(bucket(root_fqns, reshard_after_forward=False))

    flex_shard_module = flex_shard(model, buckets=buckets)
    # FlexShard parameters are plain local shards. Wrap each one's param_init
    # function so that it fills the full parameter and the shard keeps its part:
    # one device's initialization, whatever the sharding. to_empty() replaces
    # the parameters but keeps the modules, so the wrapped functions survive it.
    # validate_flex_shard_training rejects modules without param_init, whose
    # reset_parameters() would bypass this.
    for fqn, layout in get_flex_shard_global_layouts(model).items():
        module_fqn, _, name = fqn.rpartition(".")
        module = model.get_submodule(module_fqn)
        assert isinstance(module, Module) and module._param_init is not None, fqn
        # A skipped init, e.g. tok_embeddings' with tied weights, which lm_head
        # initializes, stays a no-op on the shard.
        if module._param_init.get(name, skip_param_init) is not skip_param_init:
            # A new dict: modules built from one Config share its param_init.
            module._param_init = {
                **module._param_init,
                name: partial(
                    init_from_full, layout=layout, init_fn=module._param_init[name]
                ),
            }
    declare_state_dict_layouts(model)
    if expert_mesh is not None:
        # Every bucket holds parameters, so there is one storage per bucket.
        assert len(flex_shard_module.sharded_bucket_storages) == len(buckets)
        storages = iter(flex_shard_module.sharded_bucket_storages)
        group_storages = {}
        for group, specs in zip(groups, group_buckets):
            storages_of_group = [next(storages) for _ in specs]
            group_storages.update(dict.fromkeys(group.modules, storages_of_group))
            # FSDP2 unshards a block's expert group with the block, so the
            # expert bucket prefetches nothing itself.
            for storage in storages_of_group[1:]:
                storage.set_buckets_to_forward_prefetch([])
                storage.set_buckets_to_backward_prefetch([])
        # FlexShard's default prefetches the next bucket in forward order. A
        # block's expert bucket would then gather the next block's dense bucket
        # before EP's device-to-host sync, but the next block's expert bucket
        # only from that block's start, after the sync. These lists match
        # FSDP2's schedule instead, which gathers the whole next block from this
        # block's start.
        order = [group_storages[module] for module in _decoder_prefetch_order(model)]
        for current, following in zip(order, order[1:]):
            current[0].set_buckets_to_forward_prefetch(following)
            # In backward, FSDP2 prefetches a module's groups in reverse.
            following[0].set_buckets_to_backward_prefetch(current[::-1])
    logger.info("Applied FlexShard to the model")


def _apply_fsdp2(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    embedding: _FSDPGroup | None,
    head: _FSDPGroup | None,
    blocks: list[_FSDPGroup],
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool,
    ep_degree: int,
    edp_mesh: DeviceMesh | None,
    dp_mesh_dims: "DataParallelMeshDims | None",
    edp_mesh_dims: "DataParallelMeshDims | None",
    symm_mem_scope: FSDPSymmMemScope,
    unshard_in_current_stream: bool,
) -> None:
    """Shard ``model`` with FSDP2's ``fully_shard``: each of the decoder's
    groups, then the root for any leftover parameters. ``apply_fsdp_to_decoder``
    documents the arguments."""
    mp_policy = mixed_precision_policy(
        model,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    if embedding is not None:
        fully_shard(
            embedding.modules,
            **fsdp_config,
            reshard_after_forward=embedding.reshard_after_forward,
        )
        # Embedding backward needs only grad_output and the token indices,
        # not the weight, so skip the backward all-gather. The flag covers
        # the whole FSDP unit, so only set it when the unit holds nothing
        # but embedding weights (e.g. a learned norm weight inside
        # tok_embeddings would be read after being freed).
        if not model.enable_weight_tying and _fsdp_unit_holds_only_embedding_weights(
            cast(FSDPModule, model.tok_embeddings), model
        ):
            model.tok_embeddings.set_unshard_in_backward(False)
    if head is not None:
        fully_shard(
            head.modules,
            **fsdp_config,
            reshard_after_forward=head.reshard_after_forward,
        )

    for block in blocks:
        if ep_degree == 1 or block.routed_experts is None:
            # dict.get returns None for parameters that use the default
            # Shard(0), matching shard_placement_fn's contract.
            fully_shard(
                block.modules,
                **fsdp_config,
                reshard_after_forward=block.reshard_after_forward,
                shard_placement_fn=block.placements.get,
            )
        else:
            # With EP > 1, routed-expert parameters use edp_mesh, while other
            # parameters use dp_mesh: a mesh per parameter.
            from torch.distributed.fsdp._fully_shard._fsdp_common import (
                FSDPMeshInfo,
                ShardPlacementResult,
            )
            from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info

            assert edp_mesh is not None

            # Delegate to FSDP2's mesh-info builder. When mesh_dims is set
            # it extracts and FLATTENS the DP submesh from the full SPMD
            # mesh.
            edp_mesh_info = _get_mesh_info(edp_mesh, edp_mesh_dims)
            dp_mesh_info = _get_mesh_info(dp_mesh, dp_mesh_dims)
            # _get_mesh_info is typed to the DataParallelMeshInfo base; with
            # a shard dim it always yields FSDPMeshInfo/HSDPMeshInfo.
            assert isinstance(edp_mesh_info, FSDPMeshInfo)
            assert isinstance(dp_mesh_info, FSDPMeshInfo)

            expert_params = set(block.routed_experts.parameters())

            def _shard_placement_fn(
                param: nn.Parameter,
                _placements: dict[nn.Parameter, Shard] = block.placements,
                _expert_params: set[nn.Parameter] = expert_params,
                _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
            ) -> ShardPlacementResult:
                return ShardPlacementResult(
                    placement=_placements.get(param, Shard(0)),
                    mesh_info=(
                        _edp_mesh_info if param in _expert_params else _dp_mesh_info
                    ),
                )

            fully_shard(
                block.modules,
                **fsdp_config,
                reshard_after_forward=block.reshard_after_forward,
                shard_placement_fn=_shard_placement_fn,
            )

    fully_shard(model, **fsdp_config)

    enable_fsdp_symm_mem(model, symm_mem_scope)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # HSDP when the data-parallel mesh carries a replicate axis, else pure FSDP.
    if "dp_replicate" in (dp_mesh.mesh_dim_names or ()):
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")
    if cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP, and when
    # unsharding in the current stream, as all-gathers issued on the compute
    # stream only overlap with compute if they are issued ahead of time
    if ep_degree == 1 and not unshard_in_current_stream:
        return

    fsdp_modules = [cast(FSDPModule, m) for m in _decoder_prefetch_order(model)]
    for module, next_module in zip(fsdp_modules, fsdp_modules[1:]):
        module.set_modules_to_forward_prefetch([next_module])
        next_module.set_modules_to_backward_prefetch([module])

    if unshard_in_current_stream:
        assert isinstance(model, FSDPModule)
        model.set_unshard_in_current_stream(True)


def apply_fsdp_to_decoder(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
    edp_mesh_dims: "DataParallelMeshDims | None" = None,
    symm_mem_scope: FSDPSymmMemScope = None,
    *,
    unshard_in_current_stream: bool = False,
    fsdp_backend: FSDP2Config | FlexShardConfig | None = None,
):
    """
    Apply data parallelism (via FSDP2, or FlexShard) to a decoder-style
    transformer model.

    Shared by all dense and MoE decoders (llama3, qwen3, deepseek_v3,
    gpt_oss, qwen3_vl, ...). The MoE handling is a strict superset of the dense
    case: a dense model leaves ``ep_degree=1`` / ``edp_mesh=None`` and has no
    MoE blocks, so every transformer block is sharded as a single
    FSDP unit and the explicit prefetching below is skipped unless
    ``unshard_in_current_stream`` is set.

    Args:
        model (Decoder): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reductions.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to
            CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for
            resharding after the forward pass. Defaults to "default". Other
            options: "never", "always".
            - "default" applies default resharding behavior, implementing
              "smart defaults" for known optimal scenarios.
            - "always" enables ``reshard_after_forward`` for all forward passes.
            - "never" disables ``reshard_after_forward`` for all forward passes.
        ep_degree (int, optional): Expert-parallel degree. Defaults to 1 (no EP),
            in which case the MoE-specific sharding and prefetching are no-ops.
        edp_mesh (DeviceMesh | None, optional): The FSDP mesh for routed-expert
            parameters when EP > 1. Required (non-None) iff ``ep_degree > 1``.
        dp_mesh_dims: Under spmd_types, ``fully_shard`` must flatten
            ``dp_shard`` and ``cp`` into a single FSDP shard dim, so it
            needs to know which axes of the multi-dimensional SPMD mesh are
            data-parallel. We pass this explicitly via ``dp_mesh_dims``
            rather than letting FSDP infer it from mesh axis names: the
            naming contract between ``fully_shard`` and torchtitan is not
            strong enough to infer safely, and an explicit declaration
            avoids silent miscategorization when new mesh axes appear.
        edp_mesh_dims: Sibling of ``dp_mesh_dims`` for the sparse SPMD mesh
            used by routed experts.
        symm_mem_scope: Which FSDP modules use symmetric-memory communication.
        unshard_in_current_stream: Whether to issue pre-forward/pre-backward
            all-gathers on the current (compute) stream via
            ``FSDPModule.set_unshard_in_current_stream``.
        fsdp_backend: ``FlexShardConfig()`` shards each FSDP2 group with
            FlexShard instead; ``validate_flex_shard_training`` rejects the
            arguments it doesn't support. FSDP2 otherwise.
    """
    if unshard_in_current_stream and getattr(model, "mtp_layers", None) is not None:
        # tok_embeddings runs once per MTP depth, which the explicit
        # prefetching chain below does not model.
        raise ValueError(
            "parallelism.fsdp_unshard_in_current_stream is not supported for models "
            "with MTP layers."
        )
    embedding, head, blocks = _decoder_fsdp_groups(
        model,
        dp_mesh,
        reshard_after_forward_policy=reshard_after_forward_policy,
        pp_enabled=pp_enabled,
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
    )
    if isinstance(fsdp_backend, FlexShardConfig):
        _apply_flex_shard(
            model,
            dp_mesh,
            [group for group in (embedding, head) if group is not None] + blocks,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh=edp_mesh if ep_degree > 1 else None,
        )
    else:
        _apply_fsdp2(
            model,
            dp_mesh,
            embedding,
            head,
            blocks,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cpu_offload=cpu_offload,
            ep_degree=ep_degree,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=symm_mem_scope,
            unshard_in_current_stream=unshard_in_current_stream,
        )
