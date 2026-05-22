# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization utilities for Qwen3-VL.

This module applies PT-D parallelisms and various training techniques
(activation checkpointing, compile, FSDP) to the Qwen3-VL model.
"""

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.models.qwen3_vl.model import Qwen3VLModel
from torchtitan.tools.logging import logger


def _apply_fsdp_to_vision_encoder(
    vision_encoder: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str = "default",
    pp_enabled: bool = False,
    dp_mesh_dims: DataParallelMeshDims | None = None,
):
    """
    Apply FSDP to the vision encoder as a single unit.

    Wraps the entire vision encoder with one fully_shard call so all parameters
    are gathered in a single AllGather. The vision encoder's compute is small
    relative to the decoder, so per-layer sharding would launch many small
    AllGather kernels whose total overhead exceeds a single AllGather followed
    by computing all layers in one shot — even without overlap.

    Must be called before apply_fsdp on the decoder so the vision encoder is
    already sharded when the final fully_shard(model) is applied.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled=pp_enabled
    )

    fully_shard(
        vision_encoder,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )


def parallelize_qwen3_vl(
    model: Qwen3VLModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Qwen3-VL model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if parallel_dims.cp_enabled:
        raise NotImplementedError("Context Parallel is not yet supported for Qwen3-VL.")

    model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    # Apply activation checkpointing
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )
        if model.vision_encoder is not None:
            apply_ac(
                model.vision_encoder,
                ac_config,
                model_compile_enabled=model_compile_enabled,
                base_folder=dump_folder,
            )

    # Apply torch.compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)
        if model.vision_encoder is not None:
            apply_compile(model.vision_encoder, compile_config)

    # Apply FSDP / HSDP unconditionally (fully_shard handles dp_shard=1)
    dp_mesh_names: list[str] = []
    if parallel_dims.dp_replicate_enabled:
        dp_mesh_names.append("dp_replicate")
    dp_mesh_names.append("fsdp")
    if parallel_dims.tp_enabled:
        dp_mesh_names.append("tp")
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    dense_spmd_mesh = parallel_dims.get_activated_mesh(["dp", "cp", "tp"])
    dp_mesh_dims = (
        DataParallelMeshDims(
            shard="fsdp",
            replicate="dp_replicate" if parallel_dims.dp_replicate_enabled else None,
        )
        if dense_spmd_mesh is not None
        else None
    )

    # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
    edp_mesh = None
    sparse_spmd_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        sparse_spmd_mesh = parallel_dims.get_activated_mesh(
            ["dp_replicate", "efsdp", "ep"]
        )

    # FSDP the vision encoder as a single unit (see _apply_fsdp_to_vision_encoder)
    if model.vision_encoder is not None:
        _apply_fsdp_to_vision_encoder(
            model.vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallel_dims.pp_enabled,
            dp_mesh_dims=dp_mesh_dims,
        )

    # FSDP the decoder with MoE-aware sharding
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        sparse_spmd_mesh=sparse_spmd_mesh,
        dp_mesh_dims=dp_mesh_dims,
    )

    logger.info("Applied fully_shard to the Qwen3-VL model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the Qwen3-VL model")

    return model
