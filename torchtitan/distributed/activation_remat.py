# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the configs that apply activation remat (activation
# checkpointing) to the model. Technically, this is not a part of distributed,
# but distributed module is the best place to put it.

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import torch
import torch_remat as remat

from torchtitan.distributed.activation_offload import ActivationOffloader
from torchtitan.models.base.module import Module


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass(kw_only=True, slots=True)
class BlockRematOverride:
    """Remat policy fields for selected transformer blocks.

    A field left ``None`` inherits the base policy's value; a field that is
    set replaces it for these blocks.
    """

    blocks: list[str]
    """
    Exact FQNs of the blocks in the original model: ``layers.<id>`` for
    decoder blocks, ``mtp_layers.<id>`` for multi-token prediction blocks and
    ``<encoder>.layers.<id>`` for encoder blocks, e.g.
    ``vision_encoder.layers.3``; each model lists its block containers in
    ``BaseModel._remat_block_container_fqns``. Pipeline splitting keeps
    original layer ids, so a name follows its block to whichever stage holds
    it. Names are not checked against the model, since each pipeline stage
    holds only some blocks.
    """

    save_regions: list[str] | None = None
    """
    Replaces the base ``save_regions`` for these blocks; ``None`` inherits it.
    ``recompute_regions`` still takes precedence: a region that also matches
    the blocks' effective ``recompute_regions`` is recomputed. So to retain a
    region the base policy recomputes, override ``recompute_regions`` too.
    """

    recompute_regions: list[str] | None = None
    """Replaces the base ``recompute_regions`` for these blocks; ``None``
    inherits it."""

    offload_block_input: bool | None = None
    """Replaces the base ``offload_block_input`` for these blocks; ``None``
    inherits it."""

    offload_regions: list[str] | None = None
    """Replaces the base ``offload_regions`` for these blocks; ``None``
    inherits it."""

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("BlockRematOverride.blocks must not be empty.")
        if (
            self.save_regions is None
            and self.recompute_regions is None
            and self.offload_block_input is None
            and self.offload_regions is None
        ):
            raise ValueError(f"BlockRematOverride for {self.blocks} sets no field.")


@dataclass(frozen=True, slots=True)
class BlockRematPolicy:
    """The remat policy of one transformer block, after overrides."""

    save_regions: list[str]
    recompute_regions: list[str]
    offload_block_input: bool
    offload_regions: list[str]

    @property
    def offloads_activations(self) -> bool:
        return self.offload_block_input or bool(self.offload_regions)


@dataclass(kw_only=True, slots=True)
class ActivationRematConfig:
    """Retain selected model-declared regions and recompute everything else.

    Activation rematerialization, also known as activation checkpointing:
    ``apply`` checkpoints each transformer block with ``torch_remat``. Regions
    that model code always retains with ``recompute=False``, such as MoE
    routing decisions, are saved under every policy.
    """

    save_regions: list[str]
    """
    Qualified save-region glob patterns, relative to a transformer block.
    Region names are defined at the corresponding ``torch_remat.region``
    call sites in model code. Everything outside a retained region is
    recomputed.

    Save-region names are relative to a transformer block, and the policy
    applies to every block; ``block_overrides`` changes it for selected
    blocks.
    """

    recompute_regions: list[str] = field(default_factory=list)
    """
    Region glob patterns to recompute. ``recompute_regions`` takes precedence
    over ``save_regions``: a region that matches both is recomputed. For
    example, with ``save_regions=["*"]``, this lists the few regions to
    recompute, which suits starting from no remat and recomputing just enough
    to fit a memory budget.
    """

    offload_block_input: bool = False
    """
    Offload each transformer block's inputs that require grad (its input
    hidden state) to CPU: under full remat, the whole stash. Inputs that do
    not require grad (positions, masks) stay on GPU, as does one sharing
    storage with the model part's own input (e.g. a pipeline stage's input,
    which the stage retains). Setting any offload field makes ``apply``
    install the CPU offloader; see torchtitan/distributed/ACTIVATION_OFFLOAD.md.
    """

    offload_regions: list[str] = field(default_factory=list)
    """
    Glob patterns of save regions, relative to a transformer block, whose
    retained tensors (what the region's ops save for backward, and its output
    when recomputation reads it) go to CPU. ``["*"]`` offloads everything
    retained; a pattern matching only recomputed regions offloads nothing.
    Offloading is per storage: a save of an already-offloaded storage is
    offloaded whatever its region, and one of a storage another save keeps
    on GPU stays there, since offloading it would free nothing.
    """

    block_overrides: list[BlockRematOverride] = field(default_factory=list)
    """
    Per-block changes to the fields above, e.g. recomputing an expensive
    region in only a few blocks, or retaining it in only a few. Each block
    may be named by at most one override.
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
        overridden: set[str] = set()
        for override in self.block_overrides:
            for block_fqn in override.blocks:
                if block_fqn in overridden:
                    raise ValueError(
                        f"Block {block_fqn!r} is named by more than one "
                        "BlockRematOverride."
                    )
                overridden.add(block_fqn)

    def block_policy(self, block_fqn: str) -> BlockRematPolicy:
        """The policy of the block with original FQN ``block_fqn``."""
        policy = BlockRematPolicy(
            save_regions=self.save_regions,
            recompute_regions=self.recompute_regions,
            offload_block_input=self.offload_block_input,
            offload_regions=self.offload_regions,
        )
        for override in self.block_overrides:
            if block_fqn in override.blocks:
                return BlockRematPolicy(
                    save_regions=_override(override.save_regions, policy.save_regions),
                    recompute_regions=_override(
                        override.recompute_regions, policy.recompute_regions
                    ),
                    offload_block_input=_override(
                        override.offload_block_input, policy.offload_block_input
                    ),
                    offload_regions=_override(
                        override.offload_regions, policy.offload_regions
                    ),
                )
        return policy

    @property
    def offloads_activations(self) -> bool:
        """Whether this policy offloads any retained tensor to CPU, in any
        block."""
        return (
            self.offload_block_input
            or bool(self.offload_regions)
            or any(
                override.offload_block_input or override.offload_regions
                for override in self.block_overrides
            )
        )

    def _check_fixed_regions(
        self, save_regions: list[str], recompute_regions: list[str]
    ) -> None:
        if (
            self.save_regions != save_regions
            or self.recompute_regions != recompute_regions
            or any(
                override.save_regions is not None
                or override.recompute_regions is not None
                for override in self.block_overrides
            )
        ):
            raise ValueError(
                f"{type(self).__name__} is a fixed policy. Use "
                "ActivationRematConfig with explicit save_regions and "
                "recompute_regions to customize it (offload_block_input and "
                "offload_regions can be set on any policy, also per block)."
            )

    def apply(
        self,
        model_part: Module,
        *,
        block_container_fqns: Sequence[str] = ("layers",),
    ) -> None:
        """Checkpoint every transformer block of ``model_part`` with
        ``torch_remat``, and install the CPU offloader on ``model_part`` if the
        policy offloads anything. Offloading needs no step after
        ``fully_shard``: FSDP2 prepends its pre-forward hook, so it runs before
        the offloader's.

        Call once per model part (each pipeline stage is one), naming every
        block container in it, so that the model part gets one offloader.

        Args:
            model_part: Root module of the model part.
            block_container_fqns: FQNs within ``model_part`` of the submodules
                whose children are the transformer blocks, e.g.
                ``("vision_encoder.layers", "layers", "mtp_layers")``. A
                block's FQN, which ``block_overrides`` match against, is
                ``<container fqn>.<child name>``.

        The offloader needs no handle: the forward hooks it registers on the
        model part keep it alive, and it logs its first forward's stats.
        """
        transformer_blocks = [
            (f"{container_fqn}.{name}", block)
            for container_fqn in block_container_fqns
            for name, block in model_part.get_submodule(container_fqn).named_children()
        ]
        if not transformer_blocks:
            logger.info(
                "%s found no transformer blocks in this model part",
                type(self).__name__,
            )
            return

        accelerator = torch.accelerator.current_accelerator()
        if self.offloads_activations and (
            accelerator is None or accelerator.type != "cuda"
        ):
            # TODO: other accelerators need a graph-capture query for the
            # pinned buffer pool's reuse rule; streams and events would port.
            raise ValueError(
                "Activation offload requires CUDA, but the accelerator is "
                f"{accelerator}. Keep offload_block_input and offload_regions off."
            )

        # TODO: Validate unmatched patterns and block_overrides names once
        # validation can account for all pipeline stages instead of only this
        # model part.
        overridden = []
        for block_fqn, transformer_block in transformer_blocks:
            assert isinstance(transformer_block, Module)
            policy = self.block_policy(block_fqn)
            if any(block_fqn in override.blocks for override in self.block_overrides):
                overridden.append(block_fqn)
            transformer_block.configure_remat_regions(
                policy.save_regions, policy.recompute_regions
            )
            transformer_block._remat_block_fqn = block_fqn
            transformer_block.forward = remat.checkpoint(
                region_name=block_fqn,
                determinism_check=self.determinism_check,
                preserve_rng_state=False,
            )(transformer_block.forward)
        logger.info(
            "Applied %s to %d transformer blocks. Save patterns: %s, "
            "recompute patterns: %s",
            type(self).__name__,
            len(transformer_blocks),
            self.save_regions or "none",
            self.recompute_regions or "none",
        )
        if overridden:
            logger.info("Block overrides apply to: %s", ", ".join(overridden))
        if self.offloads_activations:
            # At its first forward, the offloader hooks each block marked
            # with _remat_block_fqn above and chains them in run order.
            ActivationOffloader(model_part, remat_config=self)


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
        self._check_fixed_regions([], [])


_DEFAULT_SAVE_REGIONS = ["*"]
_DEFAULT_RECOMPUTE_REGIONS = ["*routed_experts.w13.*"]


@dataclass(kw_only=True, slots=True)
class DefaultActivationRematConfig(ActivationRematConfig):
    """The default remat policy and the trainer's default.

    Saves every model-declared region except the routed-expert ``w13`` grouped
    projection, whose saved activations scale with top-k and dominate MoE
    activation memory. It balances step time and memory for current MoE
    models and equals saving every region for dense models. Its regions may
    change as measurements improve; use ``ActivationRematConfig`` with
    explicit regions for a fixed policy. ``w2`` stays saved: its saved input is the activation
    output, which replay rebuilds anyway, so recomputing ``w2`` would cost time
    without freeing memory. Other regions under ``routed_experts`` (e.g. the EP
    token-dispatcher all-to-alls) are retained, so recomputation never replays
    EP communication. Code outside any model-declared region is always
    recomputed, so a model that declares no regions gets full recomputation.
    """

    save_regions: list[str] = field(default_factory=lambda: list(_DEFAULT_SAVE_REGIONS))
    recompute_regions: list[str] = field(
        default_factory=lambda: list(_DEFAULT_RECOMPUTE_REGIONS)
    )

    def __post_init__(self) -> None:
        super(DefaultActivationRematConfig, self).__post_init__()
        self._check_fixed_regions(_DEFAULT_SAVE_REGIONS, _DEFAULT_RECOMPUTE_REGIONS)


def _override(value: _T | None, default: _T) -> _T:
    return default if value is None else value
