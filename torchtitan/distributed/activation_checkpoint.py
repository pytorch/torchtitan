# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.

import os
from contextlib import contextmanager

import torch
import torch._functorch.config
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.module_tracker import ModuleTracker

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.tools.logging import logger


# Ops representing matrix multiplications from linear layers.
# Some backends (e.g. PrivateUse1) register aten.linear as a leaf op instead of
# decomposing it into aten.mm, so we handle both.
_MM_OPS = frozenset(
    {torch.ops.aten.mm.default, torch.ops.aten.linear.default}
)


_layer_sac_count = 0


def _apply_layer_sac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
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
    always_save_ops: set[torch._ops.OpOverload],
    save_mm_modules: set[str],
) -> nn.Module:
    """Apply op-level selective activation checkpointing using ModuleTracker.

    Instead of the legacy "save every other mm" counter, this uses ModuleTracker
    to identify which nn.Module produced each mm/linear op and makes explicit
    save/recompute decisions based on module names.

    Args:
        module: The transformer block to wrap.
        ac_config: Activation checkpointing config.
        always_save_ops: Non-mm ops whose outputs are always saved (e.g. SDPA,
            collectives, flex_attention).
        save_mm_modules: Set of leaf module names whose mm/linear outputs should
            be saved. E.g. {"wq", "wv", "w1", "w2"} for Llama3.
    """
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    # Interpret per_op_sac_force_recompute_mm_shapes_by_fqns as direct FQN
    # patterns for force-recomputing mm ops from specific modules (e.g.
    # "moe.router.gate"). Unlike the legacy shape-based matching, this matches
    # the module FQN component directly, avoiding collisions from same-shape
    # linears in different parts of the model.
    force_recompute_fqns = set(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns)

    tracker = ModuleTracker()

    def _get_leaf_module_name() -> str:
        parents = tracker.parents - {"Global"}
        if not parents:
            return ""
        fqn = max(parents, key=len)
        return fqn.rsplit(".", 1)[-1]

    def _fqn_matches_force_recompute() -> bool:
        if not force_recompute_fqns:
            return False
        parents = tracker.parents - {"Global"}
        if not parents:
            return False
        fqn = max(parents, key=len)
        return any(pattern in fqn for pattern in force_recompute_fqns)

    # During the recompute pass, the SAC infrastructure retrieves cached
    # tensors by (func, index) pair. The policy return value only matters
    # for non-cached ops during recompute (must be PREFER_RECOMPUTE to
    # avoid a RuntimeError). We use separate counters for forward and
    # recompute to maintain correct alignment with the old behavior.
    mm_fwd_decisions: list[CheckpointPolicy] = []
    mm_recompute_idx = [0]

    def _custom_policy(ctx, func, *args, **kwargs):
        if ctx.is_recompute:
            if func in _MM_OPS:
                if mm_recompute_idx[0] < len(mm_fwd_decisions):
                    decision = mm_fwd_decisions[mm_recompute_idx[0]]
                    mm_recompute_idx[0] += 1
                    return decision
            return CheckpointPolicy.PREFER_RECOMPUTE

        # CPU-to-device copies: always save (for offloading scenarios)
        if (
            func == torch.ops.aten._to_copy.default
            and "cuda" in str(args[0].device)
            and "device" in kwargs
            and str(kwargs["device"]) == "cpu"
        ):
            return CheckpointPolicy.MUST_SAVE

        # mm/linear ops: decide based on which module produced them
        if func in _MM_OPS:
            if _fqn_matches_force_recompute():
                mm_fwd_decisions.append(CheckpointPolicy.PREFER_RECOMPUTE)
                return CheckpointPolicy.PREFER_RECOMPUTE
            leaf = _get_leaf_module_name()
            if leaf in save_mm_modules:
                mm_fwd_decisions.append(CheckpointPolicy.MUST_SAVE)
                return CheckpointPolicy.MUST_SAVE
            mm_fwd_decisions.append(CheckpointPolicy.PREFER_RECOMPUTE)
            return CheckpointPolicy.PREFER_RECOMPUTE

        # Non-mm ops in always_save_ops
        if func in always_save_ops:
            return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_checkpointing_context_fn():
        mm_fwd_decisions.clear()
        mm_recompute_idx[0] = 0
        fwd_ctx, bwd_ctx = create_selective_checkpoint_contexts(_custom_policy)

        @contextmanager
        def combined_fwd():
            with tracker, fwd_ctx:
                yield

        return combined_fwd(), bwd_ctx

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
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
    model_compile_enabled: bool = False,
    always_save_ops: set[torch._ops.OpOverload] | None = None,
    save_mm_modules: set[str] | None = None,
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
        return _apply_op_sac(
            module,
            ac_config,
            always_save_ops=always_save_ops or set(),
            save_mm_modules=save_mm_modules or set(),
        )

    return _apply_layer_sac(module, ac_config)


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    always_save_ops: set[torch._ops.OpOverload] | None = None,
    save_mm_modules: set[str] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model: The model to apply activation checkpointing to.
        ac_config: Activation checkpointing config.
        model_compile_enabled: Whether torch.compile is enabled for the model.
        always_save_ops: Non-mm ops whose outputs are always saved during
            op-level SAC (e.g. SDPA variants, collectives).
        save_mm_modules: Leaf module names whose mm/linear outputs should be
            saved during op-level SAC. E.g. {"wq", "wv", "w1", "w2"}.
        base_folder: Dump folder for memory budget pareto visualization.
    """
    # Disable dynamo LRU cache to workaround an interaction between SAC, PP,
    # and Flex. See: https://github.com/pytorch/pytorch/issues/166926
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
                model_compile_enabled=model_compile_enabled,
                always_save_ops=always_save_ops,
                save_mm_modules=save_mm_modules,
            )
            layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
