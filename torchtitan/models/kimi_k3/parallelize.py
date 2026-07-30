# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 parallelization for the eager Kimi K3 reference model."""

import torch.nn as nn

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_vision_encoder,
)


def parallelize_kimi_k3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
    """Apply FSDP2 while keeping the model's eager reference forward path."""
    del dump_folder

    unsupported_parallelisms = [
        name
        for name, enabled in (
            ("hybrid sharded data parallel", parallel_dims.dp_replicate_enabled),
            ("tensor parallel", parallel_dims.tp_enabled),
            ("pipeline parallel", parallel_dims.pp_enabled),
            ("context parallel", parallel_dims.cp_enabled),
            ("expert parallel", parallel_dims.ep_enabled),
        )
        if enabled
    ]
    if unsupported_parallelisms:
        raise NotImplementedError(
            "Kimi K3 eager reference currently supports FSDP2 data parallelism "
            f"only; disable {', '.join(unsupported_parallelisms)}."
        )
    if parallelism.spmd_backend != "default":
        raise NotImplementedError(
            "Kimi K3 eager FSDP2 currently supports the default SPMD backend only."
        )
    if compile_config.enable:
        raise NotImplementedError(
            "Kimi K3 eager reference does not support torch.compile."
        )
    if ac_config is not None:
        raise NotImplementedError(
            "Kimi K3 eager FSDP2 does not support activation checkpointing yet."
        )
    if training.enable_cpu_offload:
        raise NotImplementedError(
            "Kimi K3 eager FSDP2 does not support parameter CPU offload yet."
        )

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is not None:
        apply_fsdp_to_vision_encoder(
            vision_encoder,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )

    apply_fsdp_to_decoder(
        model,  # pyrefly: ignore [bad-argument-type]
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=1,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model
