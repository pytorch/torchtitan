# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configure GraphTrainer's non-strict whole-step tracing path."""

import logging

import torch
import torch.nn as nn

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import _maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


logger = logging.getLogger(__name__)


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
