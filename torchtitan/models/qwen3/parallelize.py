# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 parallelization scaffold.

This file is intentionally strategy-free. Autoresearch is expected to replace
``parallelize_qwen3`` with an implementation specialized to the exact train
command, model flavor, and cluster/system it is optimizing for.
"""

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed import ParallelDims
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.tools.logging import logger


def parallelize_qwen3(
    model: Qwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    """Generated machine-specific Qwen3 parallelization entry point.

    A valid implementation may make narrow assumptions about the train command,
    mesh shape, hardware topology, memory budget, model flavor, and enabled
    TorchTitan features. It does not need to be a universal implementation.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support TP.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support CP.")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support PP.")
    if parallel_dims.ep_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support EP.")

    if skip_dp or not parallel_dims.dp_enabled:
        return model

    if parallel_dims.dp_replicate_enabled:
        raise NotImplementedError("Qwen3 bootstrap FSDP does not support HSDP.")

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=compile_config.enable,
            base_folder=dump_folder,
        )

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)

    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, training.mixed_precision_param),
        reduce_dtype=getattr(torch, training.mixed_precision_reduce),
    )
    fsdp_config = {
        "mesh": fsdp_mesh,
        "mp_policy": mp_policy,
    }
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    if parallelism.fsdp_reshard_after_forward != "default":
        fsdp_config["reshard_after_forward"] = (
            parallelism.fsdp_reshard_after_forward == "always"
        )

    for layer in model.layers.values():
        fully_shard(layer, **fsdp_config)
    fully_shard(model, **fsdp_config)
    logger.info("Applied FSDP to the Qwen3 model")

    return model
