# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation checkpointing module.

This module provides utilities to apply activation checkpointing to the model.

Key design patterns:
1. Policy factory functions are decorated with @lru_cache() to avoid dynamo recompilations
2. Ops are categorized into compute_intensive_ops and communication_intensive_ops
3. Policy functions have signature: (ctx, op, *args, **kwargs) -> CheckpointPolicy
4. Policy factories have a `cache_hash` attribute for dynamo cache management
"""

import os
from functools import lru_cache, partial
from typing import Callable

import torch
import torch._functorch.config
import torch.nn as nn
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.config.job_config import ActivationCheckpoint as ACConfig
from torchtitan.tools.logging import logger


# Type alias for policy functions
_PolicyFn = Callable[..., CheckpointPolicy]

_layer_sac_count = 0


def _sac_policy_fn(
    ctx,
    op,
    *args,
    compute_intensive_ops: dict,
    communication_intensive_ops: dict,
    **kwargs,
) -> CheckpointPolicy:
    # Save compute-intensive ops (mm, attention, conv, flex_attention, etc.)
    if op in compute_intensive_ops:
        return CheckpointPolicy.MUST_SAVE

    # Save communication-intensive ops (reduce_scatter, all_to_all, etc.)
    if op in communication_intensive_ops:
        return CheckpointPolicy.MUST_SAVE

    # Default: recompute everything else
    return CheckpointPolicy.PREFER_RECOMPUTE


@lru_cache()
def default_activation_checkpoint_policy() -> _PolicyFn:
    """Returns a checkpointing policy function that saves results of compute-intensive ops.

    The policy saves compute-intensive and communication-intensive ops while
    recomputing everything else. Uses dicts (not sets) to workaround dynamo
    guarding issues (https://github.com/pytorch/pytorch/issues/168163).

    Returns:
        A policy function that can be used with checkpoint contexts.

    Note:
        This function is cached with @lru_cache() to avoid dynamo recompilations.
        The cache_hash attribute is used by dynamo for cache management.
    """
    aten_op_types = get_default_op_list()
    compute_intensive_ops = {
        op.default: CheckpointPolicy.MUST_SAVE  # pyrefly: ignore [missing-attribute]
        for op in aten_op_types.compute_intensive_ops
    }

    compute_intensive_ops[
        torch.ops.aten._scaled_dot_product_cudnn_attention.default
    ] = CheckpointPolicy.MUST_SAVE
    compute_intensive_ops[
        torch.ops.aten._scaled_dot_product_attention_math.default
    ] = CheckpointPolicy.MUST_SAVE
    compute_intensive_ops[
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    ] = CheckpointPolicy.MUST_SAVE

    compute_intensive_ops[
        torch.ops.higher_order.flex_attention
    ] = CheckpointPolicy.MUST_SAVE
    compute_intensive_ops[
        torch._higher_order_ops.flex_attention
    ] = CheckpointPolicy.MUST_SAVE
    if hasattr(torch._higher_order_ops, "inductor_compiled_code"):
        compute_intensive_ops[
            torch._higher_order_ops.inductor_compiled_code
        ] = CheckpointPolicy.MUST_SAVE

    compute_intensive_ops[torch.ops.aten.max.default] = CheckpointPolicy.MUST_SAVE

    if hasattr(torch.ops, "torch_attn") and hasattr(
        torch.ops.torch_attn, "_varlen_attn"
    ):
        compute_intensive_ops[
            torch.ops.torch_attn._varlen_attn.default
        ] = CheckpointPolicy.MUST_SAVE

    communication_intensive_ops = {
        torch.ops._c10d_functional.reduce_scatter_tensor.default: CheckpointPolicy.MUST_SAVE,
        torch.ops._c10d_functional.all_to_all_single.default: CheckpointPolicy.MUST_SAVE,
    }

    # DeepEP ops for MoE expert parallelism
    # Try to import deepep module to register custom ops, then check if they exist
    try:
        import torchtitan.distributed.deepep  # noqa: F401 - registers torch.ops.deepep

        if hasattr(torch.ops, "deepep"):
            if hasattr(torch.ops.deepep, "dispatch"):
                communication_intensive_ops[
                    torch.ops.deepep.dispatch.default
                ] = CheckpointPolicy.MUST_SAVE
            if hasattr(torch.ops.deepep, "combine"):
                communication_intensive_ops[
                    torch.ops.deepep.combine.default
                ] = CheckpointPolicy.MUST_SAVE
    except ImportError:
        pass  # DeepEP not available

    policy_fn = partial(
        _sac_policy_fn,
        compute_intensive_ops=compute_intensive_ops,
        communication_intensive_ops=communication_intensive_ops,
    )
    # pyrefly: ignore [missing-attribute]
    policy_fn.cache_hash = "default_activation_checkpoint_policy"
    # pyrefly: ignore [bad-return]
    return policy_fn


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


def _get_mm_recompute_shapes(
    module: nn.Module,
    ac_config: ACConfig,
    base_fqn: str | None = None,
) -> set[tuple[int, int]]:
    """Extract mm shapes that should be force-recomputed based on FQN matching.

    Args:
        module: The module to analyze.
        ac_config: The activation checkpointing config.
        base_fqn: The base FQN of the module.

    Returns:
        Set of (in_features, out_features) shapes to force recompute.
    """
    mm_recompute_shapes: set[tuple[int, int]] = set()
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
    return mm_recompute_shapes


def _apply_op_sac(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
) -> nn.Module:
    """Apply selective activation checkpointing to the module.

    This function uses the policy-based approach. The policy is obtained from
    `default_activation_checkpoint_policy()` which returns a policy function that decides which
    ops to save vs recompute.

    Args:
        module (nn.Module): The module to apply selective activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        base_fqn (str, optional): The base fqn of the module. Defaults to None.

    Returns:
        nn.Module: The module with selective activation checkpointing applied.
    """
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    # Get mm shapes to force recompute based on FQN matching
    mm_recompute_shapes = _get_mm_recompute_shapes(module, ac_config, base_fqn)

    # Get the policy from default_activation_checkpoint_policy
    # This returns a policy function directly (via functools.partial)
    base_policy = default_activation_checkpoint_policy()

    def _create_wrapped_policy():
        """Create a policy that wraps the base policy with additional logic.

        This wrapper handles:
        1. Force recompute for specific mm shapes (per_op_sac_force_recompute_mm_shapes_by_fqns)
        2. CUDA->CPU tensor copies that must be saved
        """

        def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
            # Special case: CUDA->CPU tensor copies must be saved
            # This prevents issues with CPU offloading during recomputation
            if (
                func == torch.ops.aten._to_copy.default
                and len(args) > 0
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            # Special case: Force recompute for specific mm shapes
            # This is used for things like router gates in MoE models
            if func == torch.ops.aten.mm.default and len(args) > 1:
                if args[1].shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE

            # Delegate to the base policy for all other decisions
            return base_policy(ctx, func, *args, **kwargs)

        return wrapped_policy

    def selective_checkpointing_context_fn():
        """Context function that creates checkpoint contexts with the wrapped policy."""
        return create_selective_checkpoint_contexts(_create_wrapped_policy())

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
        return _apply_op_sac(module, ac_config, base_fqn=base_fqn)

    return _apply_layer_sac(module, ac_config)


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.
        base_folder (str): The base folder for saving memory budget pareto visualization.

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
            )
            layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
