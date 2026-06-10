# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, TYPE_CHECKING

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

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.models.common.decoder import Decoder


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


def enable_fsdp_symm_mem(model: nn.Module) -> None:
    """
    Enable symmetric-memory communication optimizations for all FSDP modules.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
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
    enable_symm_mem: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to a decoder-style transformer model.

    Shared by all dense and MoE decoders (llama3, qwen3, deepseek_v3,
    gpt_oss, qwen3_vl, ...). The MoE handling is a strict superset of the dense
    case: a dense model leaves ``ep_degree=1`` / ``edp_mesh=None`` and has no
    ``moe_enabled`` blocks, so every transformer block is sharded as a single
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
        edp_mesh (DeviceMesh | None, optional): The FSDP mesh for routed experts
            when EP > 1. Required (non-None) iff ``ep_degree > 1``.
        dp_mesh_dims: Under full_dtensor, ``fully_shard`` must flatten
            ``dp_shard`` and ``cp`` into a single FSDP shard dim, so it
            needs to know which axes of the multi-D SPMD mesh are
            data-parallel. We pass this explicitly via ``dp_mesh_dims``
            rather than letting FSDP infer it from mesh axis names: the
            naming contract between ``fully_shard`` and torchtitan is not
            strong enough to infer safely, and an explicit declaration
            avoids silent miscategorization when new mesh axes appear.
        edp_mesh_dims: Sibling of ``dp_mesh_dims`` for the sparse SPMD mesh
            used by routed experts. ``None`` outside full_dtensor.
        enable_symm_mem (bool): Whether to enable symmetric-memory FSDP
            communication.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.enable_weight_tying:
        # When weights are tied, tok_embeddings and output share the same parameter.
        # Group them together in one FSDP unit to avoid duplicate all-gathers.
        modules = [
            m
            for m in (model.tok_embeddings, model.norm, model.lm_head)
            if m is not None
        ]
        fully_shard(
            modules,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    else:
        if model.tok_embeddings is not None:
            fully_shard(
                model.tok_embeddings,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        # As an optimization, do not reshard_after_forward the last layers
        # by default since FSDP would prefetch them immediately.
        if model.norm is not None and model.lm_head is not None:
            fully_shard(
                [model.norm, model.lm_head],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )

    for layer_id, transformer_block in model.layers.items():
        # NOTE: In an MoE layer, we use shard_placement_fn to apply different
        # FSDP mesh and shard placement to different parameters:
        # - When EP > 1: routed experts use edp_mesh, other params use dp_mesh
        # - When EP = 1: all params use the same FSDP mesh, but experts may
        #   use Shard(1) when FSDP degree > num_experts to avoid padding
        # Dense blocks (no ``moe_enabled``) fall through to a plain fully_shard.
        if getattr(transformer_block, "moe_enabled", False):
            assert hasattr(transformer_block, "moe")
            # pyrefly: ignore [missing-attribute]
            experts = transformer_block.moe.experts
            expert_params = set(experts.parameters())
            num_experts = experts.num_experts

            if ep_degree > 1:
                assert edp_mesh is not None
                efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
            else:
                efsdp_ep_size = fsdp_config["mesh"].size()

            if efsdp_ep_size > num_experts:
                expert_shard_placement = Shard(1)
            else:
                expert_shard_placement = Shard(0)

            # When ep_degree == 1 and no Shard(1) override needed, skip
            # shard_placement_fn entirely for simplicity
            if ep_degree == 1 and expert_shard_placement == Shard(0):
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            elif ep_degree == 1:
                # ep_degree == 1 but need Shard(1) for experts to avoid padding
                def _experts_shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                ) -> Shard | None:
                    if param in _expert_params:
                        return Shard(1)
                    return None

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_experts_shard_placement_fn,
                )
            else:
                # ep_degree > 1: per-param mesh
                from torch.distributed.fsdp._fully_shard._fsdp_common import (
                    FSDPMeshInfo,
                    ShardPlacementResult,
                )
                from torch.distributed.fsdp._fully_shard._fsdp_init import (
                    _get_mesh_info,
                )

                assert edp_mesh is not None

                # Delegate to FSDP2's mesh-info builder. Under full_dtensor
                # (mesh_dims set) it extracts and FLATTENS the DP submesh from
                # the full SPMD mesh.
                edp_mesh_info = _get_mesh_info(edp_mesh, edp_mesh_dims)
                dp_mesh_info = _get_mesh_info(dp_mesh, dp_mesh_dims)
                # _get_mesh_info is typed to the DataParallelMeshInfo base; with
                # a shard dim it always yields FSDPMeshInfo/HSDPMeshInfo.
                assert isinstance(edp_mesh_info, FSDPMeshInfo)
                assert isinstance(dp_mesh_info, FSDPMeshInfo)

                def _shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                    _expert_placement: Shard = expert_shard_placement,
                    _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                    _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
                ) -> ShardPlacementResult:
                    if param in _expert_params:
                        return ShardPlacementResult(
                            placement=_expert_placement, mesh_info=_edp_mesh_info
                        )
                    else:
                        return ShardPlacementResult(
                            placement=Shard(0), mesh_info=_dp_mesh_info
                        )

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_shard_placement_fn,
                )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

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

    # set up explicit prefetching when EP is enabled for forward
    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and len(model.layers) > 0:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            # pyrefly: ignore [not-callable]
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif model.norm is not None and model.lm_head is not None:
            # pyrefly: ignore [not-callable]
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.lm_head]
            )

    # set up explicit prefetching when EP is enabled for backward
    # pyrefly: ignore [no-matching-overload]
    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if model.norm is not None and model.lm_head is not None and len(model.layers) > 0:
        model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif model.tok_embeddings is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])
