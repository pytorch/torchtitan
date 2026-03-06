# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.

import os
from collections import defaultdict
from fnmatch import fnmatch

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
    """Names output tensors as ``"{scope}.{op}_{count}_{output_idx}"``.

    Must be activated at the **model** level (not per-layer) so that the
    full module hierarchy is visible.  For example::

        layers.0.attention.mm_0_0   # 1st mm inside layers.0.attention
        layers.0.attention.mm_1_0   # 2nd mm (e.g. wk)
        layers.0.feed_forward.mm_0_0
        layers.1.attention.mm_0_0   # same position, next layer

    The *scope* is the parent of the deepest active ``nn.Module``.
    Counters are per ``(scope, op)`` and reset each forward pass.
    """

    def __init__(self):
        self._tracker = ModuleTracker()
        self._scope_op_counter: dict = defaultdict(int)
        self.names: WeakTensorKeyDictionary = WeakTensorKeyDictionary()

    def __enter__(self):
        self._tracker.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        self._tracker.__exit__(*args)
        return super().__exit__(*args)

    @staticmethod
    def _clean_fqn(fqn: str) -> str:
        """Strip checkpoint wrapper internals and root class name."""
        fqn = fqn.replace("._checkpoint_wrapped_module", "")
        return fqn.split(".", 1)[1] if "." in fqn else fqn

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        parents = self._tracker.parents - {"Global"}
        if not parents:
            return out

        raw_fqn = max(parents, key=len)
        module_fqn = self._clean_fqn(raw_fqn)

        scope = module_fqn.rsplit(".", 1)[0] if "." in module_fqn else ""
        op_name = (
            func.__name__.split(".")[0] if hasattr(func, "__name__") else str(func)
        )
        key = (scope, op_name)
        count = self._scope_op_counter[key]
        self._scope_op_counter[key] += 1

        def _assign(tensor, output_idx):
            if scope:
                name = f"{scope}.{op_name}_{count}_{output_idx}"
            else:
                name = f"{op_name}_{count}_{output_idx}"
            self.names[tensor] = name

        if isinstance(out, torch.Tensor):
            _assign(out, 0)
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    _assign(o, i)
        return out


def _name_matches(name: str, pattern: str) -> bool:
    """Match *name* against *pattern* with ``*`` wildcard support.

    Matching is done on dot-separated components so ``*`` matches a single
    path segment::

        _name_matches("layers.0.attention.mm_0_0", "layers.*.attention.mm_0_0")  # True
        _name_matches("layers.0.attention.mm_0_0", "attention.mm_0_0")           # True (suffix)
        _name_matches("layers.0.attention.mm_0_0", "layers.*.mm_0_0")            # False
    """
    if name == pattern:
        return True
    # Suffix match: pattern matches the tail of name on component boundaries
    if "*" not in pattern:
        return name.endswith("." + pattern)
    return fnmatch(name, pattern)


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
    naming: _AutoNamingMode,
    always_save_ops: set[torch._ops.OpOverload],
    sac_save_list: list[str],
) -> nn.Module:
    """Apply op-level SAC to a single transformer block.

    The *naming* mode is shared across all layers and entered at the model
    level (not here).  This function only sets up the per-layer SAC context.
    """
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    force_recompute_patterns = set(
        ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns
    )

    def _lookup_name(ctx):
        out = ctx.op_output
        if isinstance(out, torch.Tensor):
            return naming.names.get(out)
        if isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    n = naming.names.get(o)
                    if n is not None:
                        return n
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

        # Explicit naming via checkpoint_name()
        if hasattr(ctx, "tensor_name") and ctx.tensor_name is not None:
            for pattern in sac_save_list:
                if ctx.tensor_name == pattern:
                    return CheckpointPolicy.MUST_SAVE

        # Automatic naming via _AutoNamingMode
        name = _lookup_name(ctx)
        if name is not None:
            if any(_name_matches(name, p) for p in force_recompute_patterns):
                return CheckpointPolicy.PREFER_RECOMPUTE
            for pattern in sac_save_list:
                if _name_matches(name, pattern):
                    return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_checkpointing_context_fn():
        # The naming mode is already active (entered at model level).
        # Only enter the SAC caching context here.
        return create_selective_checkpoint_contexts(_custom_policy)

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
    naming: _AutoNamingMode | None = None,
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
        assert naming is not None
        return _apply_op_sac(
            module,
            ac_config,
            naming=naming,
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

    For op-level SAC, an ``_AutoNamingMode`` is installed at the **model**
    level so tensor names include the full hierarchy (with layer indices).

    Args:
        model: The model to apply activation checkpointing to.
        ac_config: Activation checkpointing config.
        model_compile_enabled: Whether torch.compile is enabled.
        always_save_ops: Non-mm ops whose outputs are always saved during
            op-level SAC (SDPA variants, collectives, …).
        sac_save_list: Tensors to save during op-level SAC.  Names come
            from two sources:

            * **Automatic** (``_AutoNamingMode``): full-path counter names
              like ``"layers.0.attention.mm_0_0"``.  Use ``*`` for layer
              wildcards: ``"layers.*.attention.mm_0_0"``.
            * **Explicit** (``checkpoint_name()``): arbitrary strings set
              in model code via ``checkpoint_name(tensor, name)``.
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
        # For op-level SAC, install a model-level naming mode so tensor
        # names include the full path (layers.N.attention.mm_K_I).
        naming = None
        if ac_config.mode == "selective" and ac_config.selective_ac_option == "op":
            naming = _AutoNamingMode()

            def _enter_naming(module, args):
                naming._scope_op_counter.clear()
                naming.__enter__()

            def _exit_naming(module, args, output):
                naming.__exit__(None, None, None)
                return output

            model.register_forward_pre_hook(_enter_naming)
            model.register_forward_hook(_exit_naming)

        layers = model.get_submodule("layers")
        for layer_id, transformer_block in layers.named_children():
            transformer_block = _apply_ac_to_transformer_block(
                transformer_block,
                ac_config,
                naming=naming,
                model_compile_enabled=model_compile_enabled,
                always_save_ops=always_save_ops,
                sac_save_list=sac_save_list,
            )
            layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
