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
from torchtitan.tools.logging import logger


def _has_hooks(module: nn.Module) -> bool:
    """Check if a module or any of its children have forward hooks."""
    for mod in module.modules():
        if mod._forward_hooks or mod._forward_pre_hooks:
            return True
    return False


def apply_compile_sparse(model: nn.Module, compile_config: CompileConfig):
    """HF variant of torchtitan's ``apply_compile_sparse``.

    Cannot reuse the core version because:
    1. Core detects MoE via ``isinstance(submod, moe_module.MoE)``; HF uses
       a different module hierarchy (``SparseMoeBlock`` / ``Experts``).
    2. Core compiles self_attn/layernorms individually for MoE blocks; HF
       can't — our hook-based to_local creates AsyncCollectiveTensor
       boundary crossings that crash in backward with TP.
    3. Core compiles ``_run_experts_grouped_mm``; HF uses the HF-native
       ``grouped_mm`` via ``config._experts_implementation``.

    For non-MoE layers, compiles the whole block. For MoE layers, only
    compiles MLP sub-modules that have no TP/EP hooks (on themselves or
    children). The experts module is always skipped (FSDP hooks cause
    graph breaks). Gate/router and shared experts are skipped when they
    have TP hooks — compiling them individually creates DTensor boundary
    crossings that crash in backward when combined with compiled loss.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    for layer_id, transformer_block in model.layers.named_children():
        if getattr(transformer_block, "moe_enabled", False):
            # MoE layer: compile gate only, skip experts and other
            # sub-modules — see NOTE below.
            # Unwrap CheckpointWrapper (added by apply_ac) to access
            # the actual decoder layer's children.
            if isinstance(transformer_block, CheckpointWrapper):
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                if attr_name == "mlp":
                    for mlp_attr, mlp_submod in submod.named_children():
                        if mlp_attr == "experts":
                            # Skip experts — FSDP hooks cause graph breaks,
                            # and the HF for-loop runs eagerly.
                            continue
                        if _has_hooks(mlp_submod):
                            # Skip sub-modules with TP hooks (on the module
                            # itself or its children). Compiling these
                            # individually creates DTensor boundary crossings
                            # that crash in backward when combined with
                            # compiled loss.
                            continue
                        setattr(
                            submod,
                            mlp_attr,
                            torch.compile(
                                mlp_submod,
                                backend=compile_config.backend,
                                fullgraph=True,
                            ),
                        )
                # NOTE: Don't compile other sub-modules (self_attn,
                # layernorms) individually for MoE layers. When TP is
                # enabled, SequenceParallel and RowwiseParallel produce
                # AsyncCollectiveTensor outputs via async redistribute.
                # These cross into the non-compiled MoE block forward
                # which materializes them via to_local(). In backward,
                # the compiled module expects AsyncCollectiveTensor
                # tangents but gets plain tensors — causing a metadata
                # mismatch crash.
                # See: https://github.com/pytorch/pytorch/issues/172556
        else:
            # Non-MoE layer: compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")
