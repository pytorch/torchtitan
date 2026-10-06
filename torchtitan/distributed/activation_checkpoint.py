# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import logging
from dataclasses import dataclass, field

import torch.nn as nn
import torch_remat as remat

from torchtitan.protocols.module import Module


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ActivationRematConfig:
    """Retain selected model-declared regions and recompute everything else.

    Each transformer block is checkpointed with ``torch_remat``. Regions that
    model code always retains with ``recompute=False``, such as MoE routing
    decisions, are saved under every policy. Apply it with
    ``apply_activation_remat``.
    """

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
    Region glob patterns to recompute even when they match ``save_regions``.
    With ``save_regions=["*"]``, this lists the few regions to recompute,
    which suits starting from no AC and recomputing just enough to fit a
    memory budget.
    """

    preserve_rng_state: bool = False
    """
    Must remain false. torch_remat requires explicit RecomputeStateHooks for
    random state that can advance inside retained regions.
    """

    determinism_check: str = "default"
    """
    A string specifying the determinism function. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """

    debug: bool = False
    """Not supported by torch_remat; must remain false."""

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


def _check_fixed_policy(
    config: ActivationRematConfig,
    save_regions: list[str],
    recompute_regions: list[str],
) -> None:
    if (
        config.save_regions != save_regions
        or config.recompute_regions != recompute_regions
    ):
        raise ValueError(
            f"{type(config).__name__} is a fixed policy. Use "
            "ActivationRematConfig with explicit save_regions and "
            "recompute_regions to customize it."
        )


@dataclass(kw_only=True, slots=True)
class FullActivationRematConfig(ActivationRematConfig):
    """A fixed policy that saves no policy-controlled region.

    Every operation in a transformer block is recomputed except the regions
    model code always retains with ``recompute=False``, such as MoE routing
    decisions, auxiliary-loss accumulation, and trailing residual adds.
    """

    save_regions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super(FullActivationRematConfig, self).__post_init__()
        _check_fixed_policy(self, [], [])


_SELECTIVE_SAVE_REGIONS = ["*"]
_SELECTIVE_RECOMPUTE_REGIONS = ["*routed_experts.w13.*"]


@dataclass(kw_only=True, slots=True)
class SelectiveActivationRematConfig(ActivationRematConfig):
    """A fixed policy chosen to stay close to the former operator-level
    SelectiveAC default.

    Saves every model-declared region except the routed-expert ``w13`` grouped
    projection, whose saved activations scale with top-k and dominate MoE
    activation memory. ``w2`` stays saved: its saved input is the activation
    output, which replay rebuilds anyway, so recomputing ``w2`` would cost time
    without freeing memory. Other regions under ``routed_experts`` (e.g. the EP
    token-dispatcher all-to-alls) are retained, so recomputation never replays
    EP communication. Code outside any model-declared region is always
    recomputed, so a model that declares no regions gets full recomputation.
    Use ``ActivationRematConfig`` for a different policy.
    """

    save_regions: list[str] = field(
        default_factory=lambda: list(_SELECTIVE_SAVE_REGIONS)
    )
    recompute_regions: list[str] = field(
        default_factory=lambda: list(_SELECTIVE_RECOMPUTE_REGIONS)
    )

    def __post_init__(self) -> None:
        super(SelectiveActivationRematConfig, self).__post_init__()
        _check_fixed_policy(self, _SELECTIVE_SAVE_REGIONS, _SELECTIVE_RECOMPUTE_REGIONS)


def apply_activation_remat(model: nn.Module, config: ActivationRematConfig) -> None:
    """Checkpoint every transformer block of ``model`` with ``torch_remat``."""
    layers = model.get_submodule("layers")
    transformer_blocks = list(layers.named_children())
    if not transformer_blocks:
        logger.info(
            "%s found no transformer blocks in this model part",
            type(config).__name__,
        )
        return

    # TODO: Validate unmatched patterns once validation can account for save
    # regions across all pipeline stages instead of only this model part.
    for layer_id, transformer_block in transformer_blocks:
        assert isinstance(transformer_block, Module)
        transformer_block.configure_remat_regions(
            config.save_regions, config.recompute_regions
        )
        transformer_block.forward = remat.checkpoint(
            region_name=f"layers.{layer_id}",
            determinism_check=config.determinism_check,
            preserve_rng_state=False,
        )(transformer_block.forward)
    logger.info(
        "Applied %s to %d transformer blocks. Save patterns: %s, "
        "recompute patterns: %s",
        type(config).__name__,
        len(transformer_blocks),
        config.save_regions or "none",
        config.recompute_regions or "none",
    )


ActivationCheckpointingConfig = (
    SelectiveActivationRematConfig
    | ActivationRematConfig
    | FullActivationRematConfig
    | None
)
