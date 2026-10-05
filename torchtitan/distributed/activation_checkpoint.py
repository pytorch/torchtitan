# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch_remat as remat
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.config import Configurable
from torchtitan.protocols.module import Module


logger = logging.getLogger(__name__)


def _full_ac_policy(
    _ctx: object, _op: object, *_args: object, **_kwargs: object
) -> CheckpointPolicy:
    """Recompute pure operations while PyTorch preserves registered effects."""
    return CheckpointPolicy.PREFER_RECOMPUTE


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
    """Recompute pure block operations while preserving registered effects."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
        pass

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        return ptd_checkpoint_wrapper(
            module,
            context_fn=lambda: create_selective_checkpoint_contexts(_full_ac_policy),
            preserve_rng_state=self.config.preserve_rng_state,
            determinism_check=self.config.determinism_check,
            early_stop=True,
            debug=self.config.debug,
        )


# TODO: Rename RegionAC to SelectiveAC, and give this preset a name that
# describes its default policy.
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

    def _region_policy(self) -> dict[str, list[str]]:
        """Keyword arguments for ``Module.configure_remat_regions``."""
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

    def apply(self, model: nn.Module) -> None:
        region_policy = self._region_policy()
        layers = model.get_submodule("layers")
        transformer_blocks = list(layers.named_children())
        if not transformer_blocks:
            logger.info(
                "%s found no transformer blocks in this model part",
                type(self).__name__,
            )
            return

        # TODO: Validate unmatched patterns once validation can account for save
        # regions across all pipeline stages instead of only this model part.
        for layer_id, transformer_block in transformer_blocks:
            assert isinstance(transformer_block, Module)
            transformer_block.configure_remat_regions(**region_policy)
            self._wrap_block(transformer_block, base_fqn=f"layers.{layer_id}")
        logger.info(
            "Applied %s to %d transformer blocks: %s",
            type(self).__name__,
            len(transformer_blocks),
            region_policy,
        )


class SelectiveAC(_RematAC):
    """A fixed ``RegionAC`` policy chosen to stay close to the former
    operator-level SelectiveAC default.

    Equivalent to saving every model-declared region except the routed-expert
    ``w13`` grouped projection, whose saved activations scale with top-k and
    dominate MoE activation memory. ``w2`` stays saved: its saved input is the
    activation output, which replay rebuilds anyway, so recomputing ``w2`` would
    cost time without freeing memory. Other regions under
    ``routed_experts`` (e.g. the EP token-dispatcher all-to-alls) are retained,
    so recomputation never replays EP communication. Code outside any
    model-declared region is always recomputed, so a model that declares no
    regions gets full recomputation. Use ``RegionAC`` for a different policy.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(_RematAC.Config):
        pass

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(
            "SelectiveAC is a fixed policy and cannot be subclassed. Use "
            "RegionAC with explicit save_regions to customize the policy."
        )

    def _region_policy(self) -> dict[str, list[str]]:
        return {"save_all_except": ["*routed_experts.w13.*"]}


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

    def _region_policy(self) -> dict[str, list[str]]:
        return {"save_patterns": cast("RegionAC.Config", self.config).save_regions}


ActivationCheckpointingConfig = (
    SelectiveAC.Config | RegionAC.Config | FullAC.Config | None
)
