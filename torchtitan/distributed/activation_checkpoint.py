# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import os
from dataclasses import dataclass, field
from typing import Annotated, cast

import torch
import torch._functorch.config
import torch.nn as nn
import tyro
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.config import Configurable
from torchtitan.tools.logging import logger


def _get_default_save_ops() -> set:
    """Returns the default set of ops whose activations should be saved
    (compute + comm).

    Each op spec is either an op object (always included) or a tuple
    (root, dotted_path) for conditionally available ops — resolved via
    getattr and silently skipped if not registered.
    """
    # Ops whose outputs are expensive to recompute (matmuls, attention, etc.)
    compute_ops = [
        # SDPA variants
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        # For low precision training, always save the absolute maximum used
        # to compute the scaling factor for quantization.
        torch.ops.aten.max.default,
        # FlexAttention (torch.ops.higher_order.flex_attention is the same object)
        torch._higher_order_ops.flex_attention,
        torch.ops.aten.linear.default,
        # topk can be non-deterministic; save to keep MoE expert assignments
        # stable between forward and recompute.
        torch.ops.aten.topk.default,
        # Inductor compiled code (available when torch.compile is used)
        (torch._higher_order_ops, "inductor_compiled_code"),
        # torch_attn custom backend
        (torch.ops, "torch_attn._varlen_attn.default"),
    ]

    # Communication ops whose outputs should be saved to avoid re-communication.
    comm_ops = [
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        # DeepEP (available when deepep is installed)
        (torch.ops, "deepep.dispatch.default"),
        (torch.ops, "deepep.combine.default"),
        # HybridEP (available when hybridep is installed)
        (torch.ops, "hybridep.dispatch.default"),
        (torch.ops, "hybridep.combine.default"),
    ]

    def _resolve_ops(op_specs: list) -> dict:
        ops = {}
        for spec in op_specs:
            if isinstance(spec, tuple):
                obj, path = spec
                try:
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    ops[obj] = CheckpointPolicy.MUST_SAVE
                except AttributeError:
                    pass
            else:
                ops[spec] = CheckpointPolicy.MUST_SAVE
        return ops

    aten_op_types = get_default_op_list()
    save_ops = {
        op.default  # pyrefly: ignore [missing-attribute]
        for op in aten_op_types.compute_intensive_ops
    }
    save_ops.update(_resolve_ops(compute_ops))
    save_ops.update(_resolve_ops(comm_ops))
    return save_ops


def _disable_dynamo_lru_cache() -> None:
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


class ActivationCheckpointing(Configurable):
    """Base class for activation checkpointing policies.

    A policy is selected via the Trainer config (see ``ActivationCheckpointingConfig``)
    and applied to a model with ``policy.apply(model)``. To customize the per-op SAC
    save set, subclass ``SelectiveAC`` and override ``get_save_ops``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        preserve_rng_state: bool = True
        """
        If deterministic output compared to non-checkpointed passes is required, set
        to true. Results in stashing and restoring the RNG state during each checkpoint,
        may be slower. See https://docs.pytorch.org/docs/stable/checkpoint.html
        for details.
        """

        determinism_check: str = "default"
        """
        A string specifying the determinism function. See
        https://docs.pytorch.org/docs/stable/checkpoint.html for details.
        """

        debug: bool = False
        """
        Capture ac debug information. Will be slower. See
        https://docs.pytorch.org/docs/stable/checkpoint.html for details.
        """

    def __init__(self, config: "ActivationCheckpointing.Config", dump_folder: str = ""):
        self.config = config
        self.dump_folder = dump_folder

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        """Wrap a single transformer block with this policy's checkpointing."""
        raise NotImplementedError

    def apply(self, model: nn.Module) -> None:
        """Apply activation checkpointing to every transformer block of the model."""
        _disable_dynamo_lru_cache()
        layers = model.get_submodule("layers")
        for layer_id, transformer_block in layers.named_children():
            transformer_block = self._wrap_block(
                transformer_block, base_fqn=f"layers.{layer_id}"
            )
            layers.register_module(layer_id, transformer_block)
        logger.info(
            f"Applied {type(self).__name__} activation checkpointing to the model"
        )


class FullAC(ActivationCheckpointing):
    """Recompute the entire transformer block during the backward pass."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        pass

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        return ptd_checkpoint_wrapper(
            module,
            preserve_rng_state=self.config.preserve_rng_state,
            determinism_check=self.config.determinism_check,
            early_stop=False,
            debug=self.config.debug,
        )


class SelectiveAC(ActivationCheckpointing):
    """Per-op selective activation checkpointing.

    Saves the outputs of compute/communication ops that are expensive to
    recompute (see ``get_save_ops``) while recomputing the rest, and recomputes
    every second matmul to balance memory and compute. Override ``get_save_ops``
    in a subclass to tune which ops are saved.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        force_recompute_mm_shapes_by_fqns: list[str] = field(
            default_factory=lambda: ["moe.router.gate"]
        )
        """
        This list of fully qualified names is used to determine which mm shapes to
        force recompute, rather than being considered by rest of the sac policy,
        e.g save every other mm. Only nn.Linear modules are supported today.

        Note: this config applies to mms not limited to those matching the specified
        fqns, e.g. if "moe.router.gate", corresponding to Linear(in, out), is specified,
        ANY mm with shape matching (*, in) x (in, out) will be force recomputed.
        """

    def get_save_ops(self) -> set:
        """Returns the set of ops whose activations should be saved. Override
        to customize the save set."""
        return _get_default_save_ops()

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        config = cast("SelectiveAC.Config", self.config)
        save_ops = self.get_save_ops()

        # Collect weight shapes to force-recompute, stored as mm RHS shape
        # (in_f, out_f). For aten.linear we transpose args[1].shape at lookup
        # time to match, since linear's weight is (out_f, in_f).
        mm_recompute_shapes = set()
        mm_recompute_fqns = config.force_recompute_mm_shapes_by_fqns

        if mm_recompute_fqns:
            for module_fqn, submod in module.named_modules():
                fqn = f"{base_fqn}.{module_fqn}" if base_fqn else module_fqn
                if not any(f in fqn for f in mm_recompute_fqns):
                    continue
                if not isinstance(submod, nn.Linear):
                    raise ValueError(
                        "force_recompute_mm_shapes_by_fqns expected to "
                        f"match a nn.Linear, but got: {submod}"
                    )
                out_f, in_f = submod.weight.shape
                mm_recompute_shapes.add((in_f, out_f))

        # Some backends (e.g. PrivateUse1) register aten.linear as a leaf op
        # instead of decomposing it into aten.mm, so we must handle both.
        mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)

        def _get_custom_policy():
            meta = {"forward_mm_count": 0, "recompute_mm_count": 0}

            def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
                # Always save CUDA→CPU results to avoid recomputing them
                # (e.g. MoE D2H sync for all-to-all metadata).
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

                # Save all compute/comm ops, except every second mm/linear.
                if func in save_ops:
                    if func in mm_ops and meta[mm_count_key] % 2 == 0:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            return wrapped_policy

        return ptd_checkpoint_wrapper(
            module,
            context_fn=lambda: create_selective_checkpoint_contexts(
                _get_custom_policy()
            ),
            preserve_rng_state=config.preserve_rng_state,
            determinism_check=config.determinism_check,
            early_stop=False,
            debug=config.debug,
        )


class MemoryBudgetAC(ActivationCheckpointing):
    """Let the compiler partitioner trade compute for memory via a memory budget.

    Requires the model to be compiled (validated in ``Trainer.Config``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        memory_budget: float = 0.5
        """
        This value determines how much partitioner in the compiler should trade off
        compute for memory. 0.0 corresponds to the activation memory from applying
        activation checkpointing to the full compiled region, and 1.0 corresponds to
        the activation memory from the default runtime-optimized strategy. Read here:
        https://pytorch.org/blog/activation-checkpointing-techniques/
        """

        visualize_memory_budget_pareto: bool = False
        """
        This dumps out a SVG visualization of the expected runtime vs. activation
        memory tradeoffs for all memory budget values from 0 to 1 in increments of
        0.05 in {--dump_folder}/memory_budget_pareto folder. See an example here:
        https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
        """

    def apply(self, model: nn.Module) -> None:
        _disable_dynamo_lru_cache()
        config = cast("MemoryBudgetAC.Config", self.config)
        if config.visualize_memory_budget_pareto:
            pareto_dir = os.path.join(self.dump_folder, "memory_budget_pareto")
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir, exist_ok=True)
            torch._functorch.config.memory_budget_pareto_dir = pareto_dir
            torch._functorch.config.visualize_memory_budget_pareto = True

        torch._functorch.config.activation_memory_budget = config.memory_budget
        logger.info(f"Selected {config.memory_budget} budget option")


# Trainer config field type: select a policy via tyro subcommand, or ``None`` to
# disable activation checkpointing. Explicit subcommand names are required because
# every nested Config class is named "Config" and would otherwise collide.
ActivationCheckpointingConfig = (
    Annotated[SelectiveAC.Config, tyro.conf.subcommand("selective")]
    | Annotated[FullAC.Config, tyro.conf.subcommand("full")]
    | Annotated[MemoryBudgetAC.Config, tyro.conf.subcommand("memory-budget")]
    | Annotated[None, tyro.conf.subcommand("none")]
)
