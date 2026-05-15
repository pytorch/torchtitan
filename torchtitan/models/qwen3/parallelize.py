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

from torch.distributed._composable.replicate import replicate

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
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

    dp_mesh = parallel_dims.get_mesh("batch")
    replicate(model, device_mesh=dp_mesh, static_graph=True)
    logger.info("Applied replicated DDP to the Qwen3 model")

    return model
