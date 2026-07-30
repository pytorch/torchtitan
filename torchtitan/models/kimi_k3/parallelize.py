# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization boundary for the first Kimi K3 implementation."""

import torch.nn as nn

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig


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
    """Validate the v1 single-device contract and return the model unchanged."""
    del dump_folder

    enabled_parallelisms = [
        name
        for name, enabled in (
            ("data parallel", parallel_dims.dp_enabled),
            ("tensor parallel", parallel_dims.tp_enabled),
            ("pipeline parallel", parallel_dims.pp_enabled),
            ("context parallel", parallel_dims.cp_enabled),
            ("expert parallel", parallel_dims.ep_enabled),
        )
        if enabled
    ]
    if enabled_parallelisms:
        raise NotImplementedError(
            "Kimi K3 v1 supports single-device execution only; disable "
            f"{', '.join(enabled_parallelisms)}."
        )
    if parallelism.spmd_backend != "default":
        raise NotImplementedError(
            "Kimi K3 v1 only supports the default local tensor backend."
        )
    if compile_config.enable:
        raise NotImplementedError("Kimi K3 v1 does not support torch.compile.")
    if ac_config is not None:
        raise NotImplementedError(
            "Kimi K3 v1 does not support activation checkpointing."
        )
    if training.enable_cpu_offload:
        raise NotImplementedError("Kimi K3 v1 does not support parameter CPU offload.")
    return model
