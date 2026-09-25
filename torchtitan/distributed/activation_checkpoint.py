# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import logging
import os
from dataclasses import dataclass
from typing import Annotated, cast

import torch
import torch._functorch.config
import torch.nn as nn
import torch_remat as remat
import tyro

from torchtitan.config import Configurable
from torchtitan.protocols.module import Module


logger = logging.getLogger(__name__)


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
    and applied to a model with ``policy.apply(model)``.
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

    def apply(
        self,
        model: nn.Module,
        *,
        block_container_fqns: tuple[str, ...] = ("layers",),
    ) -> None:
        """Apply activation checkpointing to every transformer block of the model."""
        _disable_dynamo_lru_cache()
        for container_fqn in block_container_fqns:
            blocks = model.get_submodule(container_fqn)
            for layer_id, transformer_block in blocks.named_children():
                transformer_block = self._wrap_block(
                    transformer_block,
                    base_fqn=f"{container_fqn}.{layer_id}",
                )
                blocks.register_module(layer_id, transformer_block)
        logger.info(
            f"Applied {type(self).__name__} activation checkpointing to the model"
        )


class _RematAC(ActivationCheckpointing):
    """Shared ``torch_remat`` implementation for block checkpointing policies."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        preserve_rng_state: bool = False
        """
        Must remain false. torch_remat requires explicit RecomputeStateHooks for
        random state that can advance inside retained regions.
        """

        def __post_init__(self) -> None:
            if self.preserve_rng_state:
                raise ValueError(
                    "torch_remat activation checkpointing does not support "
                    "preserve_rng_state=True. Register a "
                    "torch_remat RecomputeStateHook for random state used in retained "
                    "regions."
                )
            if self.debug:
                raise ValueError(
                    "torch_remat activation checkpointing does not support the debug "
                    "option."
                )

    def _get_save_patterns(self) -> list[str]:
        raise NotImplementedError

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        config = cast("_RematAC.Config", self.config)
        checkpoint_region_name = base_fqn or type(module).__name__
        checkpointed_forward = remat.checkpoint(
            region_name=checkpoint_region_name,
            determinism_check=config.determinism_check,
            preserve_rng_state=False,
        )(module.forward)
        module.forward = checkpointed_forward
        return module

    def apply(
        self,
        model: nn.Module,
        *,
        block_container_fqns: tuple[str, ...] = ("layers",),
    ) -> None:
        config = cast("_RematAC.Config", self.config)
        save_patterns = self._get_save_patterns()
        transformer_blocks = [
            (f"{container_fqn}.{layer_id}", transformer_block)
            for container_fqn in block_container_fqns
            for layer_id, transformer_block in model.get_submodule(
                container_fqn
            ).named_children()
        ]
        if not transformer_blocks:
            logger.info(
                "%s found no transformer blocks in this model part",
                type(self).__name__,
            )
            return

        # TODO: Validate unmatched patterns once validation can account for save
        # regions across all pipeline stages instead of only this model part.
        for block_fqn, transformer_block in transformer_blocks:
            assert isinstance(transformer_block, Module)
            transformer_block.configure_remat_regions(save_patterns)
            self._wrap_block(transformer_block, base_fqn=block_fqn)
        logger.info(
            "Applied %s to %d transformer blocks. Save patterns: %s",
            type(self).__name__,
            len(transformer_blocks),
            save_patterns or "none",
        )


class FullAC(_RematAC):
    """Recompute each transformer block except correctness-critical regions."""

    @dataclass(kw_only=True, slots=True)
    class Config(_RematAC.Config):
        pass

    def _get_save_patterns(self) -> list[str]:
        return []


class SelectiveAC(_RematAC):
    """Retain model-declared expensive regions and recompute everything else."""

    @dataclass(kw_only=True, slots=True)
    class Config(_RematAC.Config):
        pass

    def _get_save_patterns(self) -> list[str]:
        return ["*"]


class RegionAC(_RematAC):
    """Retain selected model-declared regions and recompute everything else."""

    @dataclass(kw_only=True, slots=True)
    class Config(_RematAC.Config):
        save_regions: list[str]
        """
        Qualified save-region glob patterns, relative to a transformer block.
        Region names are defined at the corresponding ``torch_remat.region``
        call sites in model code. Everything outside a retained region is
        recomputed.

        NB: Save-region names are relative to a transformer block, so the same
        policy applies to every transformer block. Per-block remat policies are
        not currently supported.
        """

    def _get_save_patterns(self) -> list[str]:
        return cast("RegionAC.Config", self.config).save_regions


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

        def __post_init__(self) -> None:
            if not 0 <= self.memory_budget <= 1:
                raise ValueError("memory_budget must be finite and between 0 and 1.")

    def apply(
        self,
        model: nn.Module,
        *,
        block_container_fqns: tuple[str, ...] = ("layers",),
    ) -> None:
        del block_container_fqns
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
    | Annotated[RegionAC.Config, tyro.conf.subcommand("region")]
    | Annotated[FullAC.Config, tyro.conf.subcommand("full")]
    | Annotated[MemoryBudgetAC.Config, tyro.conf.subcommand("memory-budget")]
    | Annotated[None, tyro.conf.subcommand("none")]
)
