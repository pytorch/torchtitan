# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchtitan.config.job_config import ActivationCheckpoint as ACConfig
from torchtitan.tools.logging import logger


def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    save_list: set[torch._ops.OpOverload] | None = None,
) -> nn.Module:
    valid_ac_modes = ("full", "selective")
    save_list = save_list or set()
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(
            module, preserve_rng_state=False, early_stop=ac_config.early_stop
        )

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        mm_recompute_shapes = set()
        if len(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns) > 0:
            for module_fqn, submod in module.named_modules():
                fqn = module_fqn
                if base_fqn is not None:
                    fqn = f"{base_fqn}.{module_fqn}"
                if not any(
                    filter_fqn in fqn
                    for filter_fqn in ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
                ):
                    continue
                if not isinstance(submod, nn.Linear):
                    raise ValueError(
                        "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                        f"a nn.Linear, but got: {submod}"
                    )
                out_f, in_f = submod.weight.shape
                mm_recompute_shapes.add((in_f, out_f))
            logger.debug(
                f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
            )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                if (
                    func == torch.ops.aten._to_copy.default
                    and "cuda" in str(args[0].device)
                    and "device" in kwargs
                    and str(kwargs["device"]) == "cpu"
                ):
                    return CheckpointPolicy.MUST_SAVE
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    if args[1].shape in mm_recompute_shapes:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
            early_stop=ac_config.early_stop,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(
                module, preserve_rng_state=False, early_stop=ac_config.early_stop
            )
        else:
            return module


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    use_flex_attn: bool = False,
    save_list: set[torch._ops.OpOverload] | None = None,
) -> None:
    """Apply activation checkpointing to the model.

    Note that SAC, Flex Attention and model compilation have some conflicts.
    We explicitly ask the user to pass these configs to warn if there are conflicts.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ActivationCheckpoint): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.
        use_flex_attn (bool): Whether flex attention is enabled for the model.
        save_list (set[torch._ops.OpOverload]): The list of ops to save when selective
            activation checkpointing is used.
    Returns:
        None
    """

    if use_flex_attn and not model_compile_enabled:
        logger.warning(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "POTENTIAL PERFORMANCE ISSUE DETECTED:\n"
            "Flex attention requires compilation for optimal performance and will be\n"
            "compiled automatically regardless of config.compile settings. However,\n"
            "Selective Activation Checkpointing (SAC) requires compilation to be applied\n"
            "at the outermost level (e.g., compile(SAC(model))). Othewise the compilation\n"
            "will be ignored."
            "\n"
            "Without enabling config.compile, the apply order will be:\n"
            "SAC(compile(flex_attention)). The compilation of flex_attention will be\n"
            "skipped, which results in poor performance.\n"
            "\n"
            "For best results, enable config.compile to ensure proper compilation order:\n"
            "compile(SAC(compile(flex_attention)))\n"
            "\n"
            "The innermost torch.compile will be ignored, but the outermost will take\n"
            "effect and provide optimal performance.\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(
            transformer_block,
            ac_config,
            base_fqn=f"layers.{layer_id}",
            save_list=save_list,
        )
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
