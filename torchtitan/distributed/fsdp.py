# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, cast, NamedTuple, TYPE_CHECKING

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

from torchtitan.config.parallelism import FSDPSymmMemScope
from torchtitan.distributed.flex_shard_fsdp import is_flex_shard_module
from torchtitan.distributed.fsdp_lifecycle import complete_fsdp_grouped_forward
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import MoE

__all__ = [
    "apply_fsdp_to_decoder",
    "apply_fsdp_to_multimodal_encoder",
    "decoder_ep_prefetch_order",
    "decoder_fsdp_groups",
    "disable_fsdp_gradient_division",
    "enable_fsdp_symm_mem",
    "FSDPGroup",
    "post_validate_fsdp_for_accelerator_graph",
    "get_fsdp_reshard_after_forward_policy",
    "linear_param_shard_placements",
    "mixed_precision_policy",
    "resolve_fsdp_mesh",
    "resolve_sparse_fsdp_mesh",
    "set_is_last_accumulation_step",
]

logger = logging.getLogger(__name__)


def post_validate_fsdp_for_accelerator_graph(model_parts: list[nn.Module]) -> None:
    """Finish grouped forwards and release full parameters for pool reuse.

    Eager and captured validation use the same cleanup. With ChunkedLossWrapper,
    the model skips lm_head, then the loss calls it after the root forward.
    A group containing norm and lm_head can still have norm pending. Validation
    has no backward callback to clear this state. FlexShard's ``reshard()``
    closes the same forward of norm and lm_head's bucket and reshards all
    buckets.
    """
    for model in model_parts:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                complete_fsdp_grouped_forward(module, None)
                module.reshard()
            elif is_flex_shard_module(module):
                # pyrefly: ignore [not-callable]
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


class FSDPGroup(NamedTuple):
    """Modules that ``apply_fsdp_to_decoder`` passes to one ``fully_shard`` call.

    ``placements`` holds every parameter whose placement isn't ``Shard(0)``,
    plus all routed-expert parameters. ``routed_experts`` is the block's routed
    experts, which expert parallelism shards over a separate mesh.
    """

    modules: list[nn.Module]
    reshard_after_forward: bool
    placements: dict[nn.Parameter, Shard]
    routed_experts: nn.Module | None = None


def decoder_fsdp_groups(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    *,
    reshard_after_forward_policy: str,
    pp_enabled: bool,
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
) -> tuple[FSDPGroup | None, FSDPGroup | None, list[FSDPGroup]]:
    """Return a decoder's FSDP groups as ``(embedding, head, blocks)``.

    The embedding group also holds the norm and output projection when weights
    are tied, and there is no head group then. Each transformer block is a
    group. The root group, ``fully_shard(model)``, takes the parameters left
    over and doesn't reshard after forward.

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
        embedding = FSDPGroup(
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
            embedding = FSDPGroup([model.tok_embeddings], reshard_after_forward, {})
        # As an optimization, do not reshard_after_forward the last layers
        # by default since FSDP would prefetch them immediately.
        if model.norm is not None and model.lm_head is not None:
            head = FSDPGroup(
                [model.norm, model.lm_head],
                reshard_after_forward_policy == "always",
                {},
            )

    blocks = []
    for transformer_block in model.layers.values():
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
            FSDPGroup(
                [transformer_block],
                reshard_after_forward,
                placements,
                None if moe is None else moe.routed_experts,
            )
        )
    return embedding, head, blocks


def decoder_ep_prefetch_order(model: "Decoder") -> list[nn.Module]:
    """Return the embedding, each transformer block and the output projection,
    in forward order, each standing for its FSDP group.

    With expert parallelism, each group explicitly prefetches the next one in
    forward and the previous one in backward: EP's device-to-host syncs keep the
    CPU from issuing the next all-gathers early, which implicit prefetching
    relies on.
    """
    order: list[nn.Module] = list(model.layers.values())
    if not order:
        return []
    if model.tok_embeddings is not None:
        order.insert(0, model.tok_embeddings)
    if model.norm is not None and model.lm_head is not None:
        order.append(model.lm_head)
    return order


def set_is_last_accumulation_step(
    model_part: nn.Module, is_last_accumulation_step: bool
) -> None:
    """Set whether the next backward is the last of gradient accumulation.

    Only the last backward reduces gradients and reshards parameters after
    backward. Earlier ones keep the parameters unsharded, and the next backward
    accumulates into their grads.
    """
    if isinstance(model_part, FSDPModule):
        # FlexShard has no equivalent: every backward waits for its gradient
        # reductions and releases its prefetch.
        model_part.set_is_last_backward(is_last_accumulation_step)
    # pyrefly: ignore [not-callable]
    model_part.set_reshard_after_backward(is_last_accumulation_step)
    # pyrefly: ignore [not-callable]
    model_part.set_requires_gradient_sync(is_last_accumulation_step)


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
):
    """
    Apply data parallelism (via FSDP2) to a decoder-style transformer model.

    Shared by all dense and MoE decoders (llama3, qwen3, deepseek_v3,
    gpt_oss, qwen3_vl, ...). The MoE handling is a strict superset of the dense
    case: a dense model leaves ``ep_degree=1`` / ``edp_mesh=None`` and has no
    MoE blocks, so every transformer block is sharded as a single
    FSDP unit and the expert-parallel prefetching below is skipped.

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
    """
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

    embedding, head, blocks = decoder_fsdp_groups(
        model,
        dp_mesh,
        reshard_after_forward_policy=reshard_after_forward_policy,
        pp_enabled=pp_enabled,
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
    )
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
            continue
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
    # in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return
    fsdp_modules = [cast(FSDPModule, m) for m in decoder_ep_prefetch_order(model)]
    for module, next_module in zip(fsdp_modules, fsdp_modules[1:]):
        module.set_modules_to_forward_prefetch([next_module])
        next_module.set_modules_to_backward_prefetch([module])
