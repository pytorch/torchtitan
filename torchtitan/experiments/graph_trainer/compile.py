# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
JIT/aot_fx_trace compilation dispatcher for graph_trainer training.

Supports two compilation modes via --compile.mode:
- JIT: standard torch.compile() with custom backend (deprecated)
- aot_fx_trace: non-strict tracing of fwd+loss+bwd via make_fx (default)

The standalone precompile utility still produces legacy monolithic AOT
artifacts. GraphPipelineRuntime requires a stage-graph artifact format before
GraphTrainer training can load them.
"""

import warnings

import torch
import torch.nn as nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import _maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import (
    get_transformer_block_buckets,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.jit_backend import (
    get_compile_backend_with_passes,
)
from torchtitan.tools.logging import logger


def _apply_jit_compile(
    model: nn.Module,
    compile_config: GraphTrainerCompileConfig,
) -> nn.Module:
    """Apply JIT compilation (torch.compile with custom backend)."""
    transformer_block_buckets = get_transformer_block_buckets(model)
    backend = get_compile_backend_with_passes(
        compile_config,
        transformer_block_buckets,
    )
    model.compile(
        backend=backend,
        fullgraph=True,
    )
    logger.info("Applied JIT compilation (torch.compile) to the model")
    return model


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

    _maybe_enable_async_tp(
        compile_config,
        parallel_dims.get_dense_tp_mesh() if parallel_dims.tp_enabled else None,
    )

    mode = compile_config.mode
    if mode is None:
        logger.info("No compile mode set, skipping compilation")
        return model

    if mode == "jit":
        warnings.warn(
            "compile.mode='jit' is deprecated and will be removed in the "
            "future. Please use --compile.mode='aot_fx_trace' instead.",
            FutureWarning,
            stacklevel=2,
        )

    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True

    if mode == "jit":
        if "model" not in compile_config.components:
            return model
        return _apply_jit_compile(
            model,
            compile_config,
        )
    elif mode == "aot_fx_trace":
        # GraphPipelineRuntime traces stage graphs lazily on its first step, so
        # no model-level wrapping is needed here. precompile_main drives its
        # standalone monolithic tracing path separately.
        if compile_config.precompile_artifact_dir:
            logger.info(
                "aot_fx_trace compile mode: precompile artifact path is "
                f"{compile_config.precompile_artifact_dir}"
            )
        else:
            logger.info(
                "aot_fx_trace compile mode: graph capture will happen at training time"
            )
        return model
    else:
        raise ValueError(
            f"Unknown compile mode: {mode}. Must be 'jit' or 'aot_fx_trace'."
        )
