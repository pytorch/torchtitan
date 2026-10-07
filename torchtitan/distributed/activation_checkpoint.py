# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import logging
from dataclasses import dataclass, field
from typing import cast

import torch.nn as nn
import torch_remat as remat

from torchtitan.config import Configurable
from torchtitan.protocols.module import Module


logger = logging.getLogger(__name__)


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

    def apply(self, model: nn.Module) -> None:
        """Apply activation checkpointing to every transformer block of the model."""
        raise NotImplementedError


class RegionAC(ActivationCheckpointing):
    """Retain selected model-declared regions and recompute everything else."""

    @dataclass(kw_only=True, slots=True)
    class Config(ActivationCheckpointing.Config):
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

        recompute_regions: list[str] = field(default_factory=list)
        """
        Region glob patterns to recompute. ``recompute_regions`` takes
        precedence over ``save_regions``: a region that matches both is
        recomputed. For example, with ``save_regions=["*"]``, this lists the
        few regions to recompute, which suits starting from no AC and
        recomputing just enough to fit a memory budget.
        """

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

    def _wrap_block(
        self, module: nn.Module, *, base_fqn: str | None = None
    ) -> nn.Module:
        config = cast("RegionAC.Config", self.config)
        checkpoint_region_name = base_fqn or type(module).__name__
        checkpointed_forward = remat.checkpoint(
            region_name=checkpoint_region_name,
            determinism_check=config.determinism_check,
            preserve_rng_state=False,
        )(module.forward)
        module.forward = checkpointed_forward
        return module

    def apply(self, model: nn.Module) -> None:
        config = cast("RegionAC.Config", self.config)
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
            transformer_block.configure_remat_regions(
                config.save_regions, config.recompute_regions
            )
            self._wrap_block(transformer_block, base_fqn=f"layers.{layer_id}")
        logger.info(
            "Applied %s to %d transformer blocks. Save patterns: %s, "
            "recompute patterns: %s",
            type(self).__name__,
            len(transformer_blocks),
            config.save_regions or "none",
            config.recompute_regions or "none",
        )


def _check_fixed_policy(
    config: "RegionAC.Config",
    policy_name: str,
    save_regions: list[str],
    recompute_regions: list[str],
) -> None:
    if (
        config.save_regions != save_regions
        or config.recompute_regions != recompute_regions
    ):
        raise ValueError(
            f"{policy_name} is a fixed policy. Use RegionAC with explicit "
            "save_regions and recompute_regions to customize it."
        )


class FullAC(RegionAC):
    """A fixed ``RegionAC`` policy that saves no policy-controlled region.

    Every operation in a transformer block is recomputed except the regions
    model code always retains with ``recompute=False``, such as MoE routing
    decisions, auxiliary-loss accumulation, and trailing residual adds.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RegionAC.Config):
        save_regions: list[str] = field(default_factory=list)

        def __post_init__(self) -> None:
            super(FullAC.Config, self).__post_init__()
            _check_fixed_policy(self, "FullAC", [], [])


_SELECTIVE_AC_SAVE_REGIONS = ["*"]
_SELECTIVE_AC_RECOMPUTE_REGIONS = ["*routed_experts.w13.*"]


# TODO: Give this preset a name that describes its policy.
class SelectiveAC(RegionAC):
    """A fixed ``RegionAC`` policy chosen to stay close to the former
    operator-level SelectiveAC default.

    Saves every model-declared region except the routed-expert ``w13`` grouped
    projection, whose saved activations scale with top-k and dominate MoE
    activation memory. ``w2`` stays saved: its saved input is the activation
    output, which replay rebuilds anyway, so recomputing ``w2`` would cost time
    without freeing memory. Other regions under ``routed_experts`` (e.g. the EP
    token-dispatcher all-to-alls) are retained, so recomputation never replays
    EP communication. Code outside any model-declared region is always
    recomputed, so a model that declares no regions gets full recomputation.
    Use ``RegionAC`` for a different policy.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RegionAC.Config):
        save_regions: list[str] = field(
            default_factory=lambda: list(_SELECTIVE_AC_SAVE_REGIONS)
        )
        recompute_regions: list[str] = field(
            default_factory=lambda: list(_SELECTIVE_AC_RECOMPUTE_REGIONS)
        )

        def __post_init__(self) -> None:
            super(SelectiveAC.Config, self).__post_init__()
            _check_fixed_policy(
                self,
                "SelectiveAC",
                _SELECTIVE_AC_SAVE_REGIONS,
                _SELECTIVE_AC_RECOMPUTE_REGIONS,
            )


ActivationCheckpointingConfig = (
    SelectiveAC.Config | RegionAC.Config | FullAC.Config | None
)
