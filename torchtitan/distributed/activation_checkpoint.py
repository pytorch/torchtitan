# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import os
from collections import defaultdict

import torch
import torch._functorch.config
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.tools.logging import logger


_layer_sac_count = 0


def _apply_layer_sac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """Apply layer selective activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply layer selective activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.

    Returns:
        nn.Module: The module with layer selective activation checkpointing applied.
    """
    global _layer_sac_count
    _layer_sac_count += 1
    ac_freq = int(ac_config.selective_ac_option)
    if not ac_freq or _layer_sac_count % ac_freq == 0:
        return ptd_checkpoint_wrapper(
            module,
            preserve_rng_state=ac_config.preserve_rng_state,
            determinism_check=ac_config.determinism_check,
            early_stop=ac_config.early_stop,
            debug=ac_config.debug,
        )
    else:
        return module


def _apply_op_sac(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    op_sac_save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    """Apply selective activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply selective activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        base_fqn (str, optional): The base fqn of the module. Defaults to None.
        op_sac_save_list (set[torch._ops.OpOverload]): The list of ops to save instead
            of recomputing.

    Returns:
        nn.Module: The module with selective activation checkpointing applied.
    """
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
            to_save = func in op_sac_save_list and not (
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
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """Apply full activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply full activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.

    Returns:
        nn.Module: The module with full activation checkpointing applied.
    """
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
) -> nn.Module:
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )

    if use_op_sac:
        op_sac_save_list = op_sac_save_list or set()
        return _apply_op_sac(
            module, ac_config, base_fqn=base_fqn, op_sac_save_list=op_sac_save_list
        )

    return _apply_layer_sac(module, ac_config)


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.
        op_sac_save_list (set[torch._ops.OpOverload]): The list of ops to save instead
            of recomputing.
    Returns:
        None
    """
    # Disable dynamo LRU cache to workaround an interaction between SAC, PP, and Flex:
    #
    # When forward runs with a second PP microbatch, it triggers recompilation with dynamic
    # shapes enabled. Now there are two valid compiled graphs. By default, dynamo selects
    # the latest one (the dynamic shapes version), so the runtime wrapper expects an extra
    # symint output. When SAC caches the inductor HOP output from the static graph for
    # batch_idx=0, it would miss that symint and cause an assertion failure. The workaround
    # here is to disable the LRU cache, and select graphs in insertion order instead.
    #
    # Also see: https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    if ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget mode requires model to be compiled"
        if ac_config.visualize_memory_budget_pareto:
            pareto_dir = os.path.join(base_folder, "memory_budget_pareto")
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir, exist_ok=True)
            torch._functorch.config.memory_budget_pareto_dir = pareto_dir
            torch._functorch.config.visualize_memory_budget_pareto = True

        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        logger.info(f"Selected {ac_config.memory_budget} budget option")
    else:
        layers = model.get_submodule("layers")
        for layer_id, transformer_block in layers.named_children():
            transformer_block = _apply_ac_to_transformer_block(
                transformer_block,
                ac_config,
                base_fqn=f"layers.{layer_id}",
                model_compile_enabled=model_compile_enabled,
                op_sac_save_list=op_sac_save_list,
            )
            layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
