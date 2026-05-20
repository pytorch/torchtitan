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
    """Conservative proxy for TP/EP-wrapped submodules.

    HF TP/EP parallelization inserts forward hooks/pre-hooks (for example via
    ``NoParallel``) on sub-modules that participate in distributed boundary
    conversions. We use their presence to avoid individually compiling those
    MLP children in MoE layers.
    """
    for mod in module.modules():
        if mod._forward_hooks or mod._forward_pre_hooks:
            return True
    return False


def apply_compile_sparse(model: nn.Module, compile_config: CompileConfig):
    """HF variant of torchtitan's ``apply_compile_sparse``.

    Cannot reuse the core version because:
    1. Core detects MoE via ``isinstance(submod, moe_module.MoE)``; HF uses
       a different module hierarchy (``SparseMoeBlock`` / ``Experts``).
    2. HF stays conservative around MoE sub-modules that straddle TP/EP
       boundary conversions; compiling across the MoE input TP boundary hits
       the compiler issue described below.
    3. Core compiles ``_run_experts_grouped_mm``; HF uses the HF-native
       ``grouped_mm`` via ``config._experts_implementation``.

    For non-MoE layers, compiles the whole block. For MoE layers, only
    compiles MLP children that do not carry TP/EP hooks. The experts module
    is always skipped (FSDP hooks cause graph breaks). TP/EP-hooked MLP
    children and non-MLP MoE producers stay eager so we do not separately
    compile both sides of the MoE input TP boundary.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    for layer_id, transformer_block in model.layers.named_children():
        if getattr(transformer_block, "moe_enabled", False):
            # MoE layer: only individually compile MLP children that do not
            # participate in TP/EP boundary conversions; keep the rest eager.
            # See NOTE below.
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
                            # Skip TP/EP-wrapped MLP children. ``_has_hooks``
                            # is our conservative proxy for the MLP side of
                            # the MoE input TP boundary described in the NOTE
                            # below (for example ``mlp.gate`` / ``mlp.router``).
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
                # NOTE: Keep eager mode for sub-modules on both sides of the MoE input
                # TP boundary when TP-enabled sparse-MoE is used:
                #
                #   1. Upstream TP producers outside `mlp` — e.g. `input_layernorm` and
                #      `post_attention_layernorm` wrapped with `SequenceParallel()`.
                #   2. TP/EP-wrapped MLP children detected via `_has_hooks` — e.g.
                #      `mlp.gate` / `mlp.router` wrapped with
                #      `NoParallel(use_local_output=True)`.
                #
                # This boundary converts TP DTensors into local `to_local(Partial)`
                # inputs before MoE routing. Force-compiling the layernorms together
                # with `mlp.gate` triggers an AOTAutograd partitioning failure:
                #
                #   AssertionError: Node mm_5 was invalid, but is output
                #
                # Reproduced on HF `Qwen3MoeSparseMoeBlock` with 8 GPUs (dp=2 x tp=4,
                # no EP/PP) on torch==2.13.0.dev20260421+cu130.
                #
                # TODO: Once https://github.com/pytorch/torchtitan/issues/3345 is
                # fixed, this policy can be relaxed to compile these
                # TP-boundary-crossing sub-modules individually.

        else:
            # Non-MoE layer: compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")
