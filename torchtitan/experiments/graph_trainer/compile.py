# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configure GraphTrainer's non-strict whole-step tracing path."""

import logging
import warnings

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


logger = logging.getLogger(__name__)


def _maybe_enable_async_tp(
    compile_config: GraphTrainerCompileConfig,
    tp_mesh: DeviceMesh | None,
) -> None:
    """Configure Inductor's async TP pass for the provided TP mesh."""
    if not compile_config.enable_async_tensor_parallel or tp_mesh is None:
        return

    group_name = tp_mesh.get_group().group_name
    # TODO: Remove this call once PyTorch automatically registers symmetric
    # memory for process groups used by async TP:
    # https://github.com/pytorch/pytorch/issues/193027
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,  # pyrefly: ignore [deprecated]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        enable_symm_mem_for_group(group_name)  # pyrefly: ignore [deprecated]

    torch._inductor.config._micro_pipeline_tp = True
    logger.info("Async TP is enabled")


def apply_compile(
    model: nn.Module,
    *,
    compile_config: GraphTrainerCompileConfig,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """Configure tracing and leave whole-step capture to ``GraphRuntime``."""
    _maybe_enable_async_tp(
        compile_config,
        parallel_dims.get_dense_tp_mesh() if parallel_dims.tp_enabled else None,
    )
    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True
    if compile_config.precompile_artifact_dir:
        logger.info(
            "Precompiled graph artifact will be loaded from %s",
            compile_config.precompile_artifact_dir,
        )
    else:
        logger.info("Graph capture will happen at training time")
    return model
