# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
aot_fx_trace compilation dispatcher for graph_trainer training.

Supports `--compile.mode aot_fx_trace`: non-strict tracing of fwd+loss+bwd
via make_fx.

Additionally supports pre-compile via --compile.precompile_artifact_dir:
- When set during training, loads a precompiled artifact and skips compilation
  entirely
- Generate artifacts with precompile_main.py
"""

import torch
import torch.nn as nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.tools.logging import logger


def apply_compile(
    model: nn.Module,
    *,
    compile_config: GraphTrainerCompileConfig,
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims,
    dump_folder: str,
) -> nn.Module:
    """
    Apply compilation to the model based on the configured mode.

    Args:
        model: The model to compile
        compile_config: Compilation configuration with mode and passes
        parallelism: Parallelism configuration
        parallel_dims: Parallel dimensions
        dump_folder: Folder for dumping debug graphs
    """
    if not compile_config.enable:
        return model

    mode = compile_config.mode
    if mode is None:
        logger.info("No compile mode set, skipping compilation")
        return model

    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True

    if mode == "aot_fx_trace":
        # aot_fx_trace traces fwd+loss+bwd together inside forward_backward_step,
        # so no model-level wrapping is needed here. If precompile_artifact_dir
        # is set, the precompiled artifact will be loaded lazily in
        # GraphTrainer._make_fx_forward_backward_step.
        if compile_config.precompile_artifact_dir:
            logger.info(
                "aot_fx_trace compile mode: precompiled artifact will be loaded "
                f"from {compile_config.precompile_artifact_dir}"
            )
        else:
            logger.info(
                "aot_fx_trace compile mode: graph capture will happen at training time"
            )
        return model
    else:
        raise ValueError(
            f"Unknown compile mode: {mode}. Must be 'aot_fx_trace'."
        )
