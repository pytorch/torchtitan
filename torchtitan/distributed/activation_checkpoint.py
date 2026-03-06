# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.

import os
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch._functorch.config
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.module_tracker import ModuleTracker
from torch.utils.weak import WeakTensorKeyDictionary

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.tools.logging import logger


class _AutoNamingMode(TorchDispatchMode):
    """Names output tensors with two FQN-like identifiers.

    Each tensor gets two names so save lists can use either convention:

    1. **Module FQN** – the leaf nn.Module that produced the op, e.g.
       ``"attention.wq"``.  Readable when the module structure is known.

    2. **Op counter** – parent module + op + counter, e.g.
       ``"attention.mm_0_0"`` (0th output of the 0th ``mm`` inside
       ``attention``).  Mechanical, doesn't require knowing sub-module
       names.

    Push this mode **before** the SAC CachingTorchDispatchMode so it runs
    as an inner mode.  The SAC policy then inspects ``ctx.op_output`` and
    looks up the tensor's name here.
    """

    def __init__(self):
        self._tracker = ModuleTracker()
        self._parent_op_counter: dict = defaultdict(int)
        self.names: WeakTensorKeyDictionary = WeakTensorKeyDictionary()

    def __enter__(self):
        self._tracker.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        self._tracker.__exit__(*args)
        return super().__exit__(*args)

    @staticmethod
    def _clean_fqn(fqn: str) -> str:
        """Strip checkpoint wrapper prefix and root class name."""
        fqn = fqn.replace("._checkpoint_wrapped_module", "")
        # "TransformerBlock.attention.wq" → "attention.wq"
        return fqn.split(".", 1)[1] if "." in fqn else fqn

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        parents = self._tracker.parents - {"Global"}
        if not parents:
            return out

        raw_fqn = max(parents, key=len)
        module_fqn = self._clean_fqn(raw_fqn)

        parent_fqn = module_fqn.rsplit(".", 1)[0] if "." in module_fqn else ""
        op_name = (
            func.__name__.split(".")[0] if hasattr(func, "__name__") else str(func)
        )
        counter_key = (parent_fqn, op_name)
        count = self._parent_op_counter[counter_key]
        self._parent_op_counter[counter_key] += 1

        def _assign(tensor, output_idx):
            if parent_fqn:
                counter_name = f"{parent_fqn}.{op_name}_{count}_{output_idx}"
            else:
                counter_name = f"{op_name}_{count}_{output_idx}"
            self.names[tensor] = (module_fqn, counter_name)

        if isinstance(out, torch.Tensor):
            _assign(out, 0)
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    _assign(o, i)
        return out


def _fqn_match(fqn: str, pattern: str) -> bool:
    """Match pattern against fqn on component boundaries.

    ``"attention.wq"`` matches ``"attention.wq"`` and
    ``"layers.0.attention.wq"`` but NOT ``"xattention.wq"``.
    """
    return fqn == pattern or fqn.endswith("." + pattern)


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
    sac_save_list: list[str],
) -> nn.Module:
    """Apply op-level selective activation checkpointing.

    Uses ``_AutoNamingMode`` to name every tensor by its module FQN and by
    an automatic op counter.  The SAC policy inspects ``ctx.op_output`` to
    look up the tensor name and matches it against ``sac_save_list``.

    The save list can use either naming convention (suffix-matched):

    * **Module FQN**: ``"attention.wq"`` — matches the leaf module.
    * **Op counter**: ``"attention.mm_0_0"`` — matches the 0th mm inside
      ``attention``, 0th output.

    Args:
        module: Transformer block to wrap.
        ac_config: Activation checkpointing config.
        always_save_ops: Op-type-based saves (SDPA, collectives, …).
        sac_save_list: FQN patterns whose matching tensors are saved.
    """
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    force_recompute_fqns = set(ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns)

    naming = _AutoNamingMode()

    def _lookup(ctx):
        """Return (module_fqn, counter_name) for ctx.op_output, or None."""
        out = ctx.op_output
        if isinstance(out, torch.Tensor):
            return naming.names.get(out)
        if isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    info = naming.names.get(o)
                    if info is not None:
                        return info
        return None

    def _custom_policy(ctx, func, *args, **kwargs):
        if ctx.is_recompute:
            return CheckpointPolicy.PREFER_RECOMPUTE

        # CPU-to-device copies: always save (for offloading)
        if (
            func == torch.ops.aten._to_copy.default
            and "cuda" in str(args[0].device)
            and "device" in kwargs
            and str(kwargs["device"]) == "cpu"
        ):
            return CheckpointPolicy.MUST_SAVE

        # Non-mm ops in always_save_ops
        if func in always_save_ops:
            return CheckpointPolicy.MUST_SAVE

        # Look up the tensor's name and match against save list
        info = _lookup(ctx)
        if info is not None:
            module_fqn, counter_name = info

            # Force-recompute takes priority
            if any(_fqn_match(module_fqn, p) for p in force_recompute_fqns):
                return CheckpointPolicy.PREFER_RECOMPUTE

            # Check save list against both naming conventions
            for pattern in sac_save_list:
                if _fqn_match(module_fqn, pattern) or _fqn_match(counter_name, pattern):
                    return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_checkpointing_context_fn():
        # Reset per-invocation state
        naming._parent_op_counter.clear()

        fwd_ctx, bwd_ctx = create_selective_checkpoint_contexts(_custom_policy)

        @contextmanager
        def combined_fwd():
            with naming, fwd_ctx:
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
    sac_save_list: list[str] | None = None,
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
            sac_save_list=sac_save_list or [],
        )

    return _apply_layer_sac(module, ac_config)


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    always_save_ops: set[torch._ops.OpOverload] | None = None,
    sac_save_list: list[str] | None = None,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model: The model to apply activation checkpointing to.
        ac_config: Activation checkpointing config.
        model_compile_enabled: Whether torch.compile is enabled.
        always_save_ops: Non-mm ops whose outputs are always saved during
            op-level SAC (SDPA variants, collectives, …).
        sac_save_list: FQN patterns for tensors to save during op-level SAC.
            Supports two naming conventions (suffix-matched):

            * Module FQN: ``["attention.wq", "feed_forward.w1"]``
            * Op counter: ``["attention.mm_0_0", "feed_forward.mm_0_0"]``
        base_folder: Dump folder for memory budget pareto.
    """
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
                sac_save_list=sac_save_list,
            )
            layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
