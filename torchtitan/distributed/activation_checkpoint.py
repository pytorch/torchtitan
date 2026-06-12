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
    ActivationWrapper,
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.distributed._region_checkpoint import (
    checkpoint as region_checkpoint,
    unit,
)
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

    # Collect weight shapes to force-recompute, stored as mm RHS shape
    # (in_f, out_f). For aten.linear we transpose args[1].shape at lookup
    # time to match, since linear's weight is (out_f, in_f).
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

    # Some backends (e.g. PrivateUse1) register aten.linear as a leaf op
    # instead of decomposing it into aten.mm, so we must handle both.
    mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)

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
            if func in mm_ops:
                weight_shape = args[1].shape
                # linear weight is (out, in); normalize to (in, out) to match mm
                if func == torch.ops.aten.linear.default:
                    weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
                if weight_shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[mm_count_key] += 1
            # Saves output of all compute ops, except every second mm/linear
            to_save = func in op_sac_save_list and not (
                func in mm_ops and meta[mm_count_key] % 2 == 0
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


class _UnitWrapper(ActivationWrapper):
    """Routes a submodule's forward through unit() so a region policy can save it
    by name. FQN/state_dict transparent (ActivationWrapper), so wrapping a
    submodule does not change parameter names or the model's state_dict.
    """

    def __init__(self, mod: nn.Module, region_name: str):
        super().__init__(mod)
        self._region_name = region_name

    def forward(self, *args, **kwargs):
        return unit(
            self._checkpoint_wrapped_module, *args, name=self._region_name, **kwargs
        )


class _RegionCheckpointWrapper(ActivationWrapper):
    """Checkpoint a block, saving the regions named in ``policy`` and recomputing
    the rest. Unlike op-level SAC (a context_fn / policy_fn over aten ops), this
    drives the region checkpoint from torchtitan.distributed._region_checkpoint:
    the submodules wrapped by _UnitWrapper are the named regions. Subclasses
    ActivationWrapper so parameter FQNs and state_dict stay identical to
    ptd_checkpoint_wrapper.
    """

    def __init__(self, mod: nn.Module, policy: dict, ac_config: ACConfig):
        super().__init__(mod)
        self._region_policy = policy
        self._preserve_rng_state = ac_config.preserve_rng_state
        self._determinism_check = ac_config.determinism_check
        self._early_stop = ac_config.early_stop
        self._debug = ac_config.debug

    def forward(self, *args, **kwargs):
        return region_checkpoint(
            self._checkpoint_wrapped_module,
            *args,
            policy=self._region_policy,
            preserve_rng_state=self._preserve_rng_state,
            determinism_check=self._determinism_check,
            early_stop=self._early_stop,
            debug=self._debug,
            **kwargs,
        )


def _install_region_units(module: nn.Module, region_save_list: list[str]) -> dict:
    """Wrap each saved submodule of ``module`` in a _UnitWrapper, naming the region
    by the submodule's FQN within the block. Returns the name->policy dict for the
    region checkpoint. Each entry in ``region_save_list`` is the exact FQN of a
    submodule within the block, e.g. "attention.wq", "feed_forward.w1".

    Regions must be disjoint: a saved region must not contain another saved region.
    Nesting silently corrupts recompute (the outer region is skipped on the
    recompute pass and never re-runs the inner one, so a downstream recomputed op
    sees a stale value), so overlap is rejected rather than wrapped.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    save = set(region_save_list)
    matched = [
        (fqn, submod) for fqn, submod in module.named_modules() if fqn in save
    ]
    fqns = [fqn for fqn, _ in matched]
    for a in fqns:
        for b in fqns:
            if a != b and b.startswith(a + "."):
                raise ValueError(
                    f"Region SAC units must be disjoint, but region '{a}' contains "
                    f"region '{b}'. Remove one from the save list."
                )
    policy = {}
    for fqn, submod in matched:
        parent = (
            module if "." not in fqn else module.get_submodule(fqn.rsplit(".", 1)[0])
        )
        setattr(parent, fqn.rsplit(".", 1)[-1], _UnitWrapper(submod, fqn))
        policy[fqn] = CheckpointPolicy.MUST_SAVE
    missing = save - set(fqns)
    if missing:
        logger.warning(f"Region SAC save list entries matched no submodule: {missing}")
    return policy


def _apply_region_sac(
    module: nn.Module, ac_config: ACConfig, *, region_save_list: list[str]
) -> nn.Module:
    """Apply name-based region selective activation checkpointing to a block.

    Wraps the saved submodules as named regions (no edits to the model forward
    needed), then checkpoints the whole block, saving those regions.

    Args:
        module (nn.Module): The transformer block to checkpoint.
        ac_config (ACConfig): The activation checkpointing config.
        region_save_list (list[str]): Submodules to save (the rest recompute),
            named by their exact FQN within the block, e.g. "attention.wq",
            "feed_forward.w1".
    """
    policy = _install_region_units(module, region_save_list)
    return _RegionCheckpointWrapper(module, policy, ac_config)


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
    region_save_list: list[str] | None = None,
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
        # Name-based region SAC (this stack's alternative to counter-based op SAC):
        # selected when the model passes a region_save_list.
        if region_save_list is not None:
            return _apply_region_sac(
                module, ac_config, region_save_list=region_save_list
            )
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
    region_save_list: list[str] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.
        op_sac_save_list (set[torch._ops.OpOverload]): The list of ops to save instead
            of recomputing.
        region_save_list (list[str]): For op-level selective AC, exact FQNs of
            block submodules to save instead of recompute (apply_ac wraps each as a
            named region). When provided, name-based region SAC is used in place of
            counter-based op SAC.
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
                region_save_list=region_save_list,
            )
            layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
