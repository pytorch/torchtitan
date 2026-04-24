# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RL-specific parallelize function for the qwen3 model.
# Applies tensor parallelism via config-based sharding. Sharding specs are
# filled on the config before build (via direct ``set_qwen3_sharding_spec``
# calls in the RL trainer and vllm_wrapper). This file then just walks the
# built model and dispatches ``Module.parallelize(tp_mesh)``.

import logging

import torch.nn as nn

from torchtitan.config import ParallelismConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import apply_compile

logger = logging.getLogger(__name__)


def parallelize_qwen3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig | None = None,
):
    """Apply tensor parallelism to the Qwen3 dense model for RL training/inference.

    Expects ``set_qwen3_sharding_spec`` to have already been called on the
    model's config *before* build, so that ``model.parallelize(tp_mesh)``
    sees the right ``ShardingConfig`` on every sub-module.

    ``compile_config``: if enabled, applies per-layer ``torch.compile`` after TP.
    """
    if parallel_dims.tp_enabled:
        # MoE with vLLM inference is not supported yet.
        first_block = next(iter(model.layers.values()))
        if getattr(first_block, "moe_enabled", False):
            raise ValueError(
                "Running vLLM inference with torchtitan Qwen3 MoE model is "
                "not supported yet."
            )

        tp_mesh = parallel_dims.get_mesh("tp")
        model.parallelize(tp_mesh)

    if (
        compile_config is not None
        and compile_config.enable
        and "model" in compile_config.components
    ):
        apply_compile(model, compile_config)

    return model
