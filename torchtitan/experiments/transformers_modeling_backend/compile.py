# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from torchtitan.config import CompileConfig
from torchtitan.models.common.moe import MoE
from torchtitan.tools.logging import logger


def apply_compile_sparse(model: nn.Module, compile_config: CompileConfig):
    """Compile transformer blocks with native MoE awareness.

    Non-MoE layers: compiles the whole block.
    MoE layers: compiles ``GroupedExperts._experts_forward`` (the grouped_mm
    hot path). The rest of the MoE (router, dispatch/combine, shared experts)
    stays eager to avoid graph breaks from communication ops.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    for layer_id, transformer_block in model.layers.named_children():
        if getattr(transformer_block, "moe_enabled", False):
            # Unwrap CheckpointWrapper to access the actual decoder layer
            if isinstance(transformer_block, CheckpointWrapper):
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            moe_block = block.mlp
            if isinstance(moe_block, MoE):
                moe_block.experts._experts_forward = torch.compile(
                    moe_block.experts._experts_forward,
                    backend=compile_config.backend,
                    fullgraph=True,
                )
        else:
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")
