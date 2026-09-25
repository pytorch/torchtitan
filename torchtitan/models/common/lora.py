# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA linear modules.

Shape suffixes: ``X`` arbitrary leading dimensions, ``R`` routed rows,
``E`` experts, ``I`` input features, ``L`` LoRA rank, ``O`` output features.
"""

import functools
import math
from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.nn as nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

__all__ = ["get_lora_grouped_linear", "get_lora_linear"]


# TODO: Support checkpoint interoperability for LoRA adapter state. Native DCP
# base checkpoints cannot initialize LoRA models because lora_a/lora_b keys are
# absent, and Hugging Face state-dict adapters omit those keys during export.
class _LoRAMixin:
    _lora_scaling: float

    def __init__(self, config) -> None:
        super().__init__(config)  # type: ignore[misc]
        for param in nn.Module.parameters(self):  # type: ignore[arg-type]
            param.requires_grad_(False)
        self._lora_scaling = config.alpha / config.rank


class _LoRALinearMixin(_LoRAMixin):
    """Add a LoRA update to a Linear's local computation."""

    num_linears: int
    lora_a: Linear
    lora_b: Linear

    def __init__(self, config) -> None:
        super().__init__(config)
        if config.num_linears > 1:
            # A stacked base projection shares one A matrix across its logical
            # linears and stacks their B matrices along the same output axis as
            # the base weight. The adapters inherit only parameter sharding;
            # the base projection remains responsible for TP collectives.
            replicated_weight = ShardingConfig(
                state_shardings={"weight": dense_param_placement(tp=spmd.R)},
            )
            lora_a_sharding = (
                replicated_weight if config.sharding_config is not None else None
            )
            lora_b_sharding = (
                ShardingConfig(
                    state_shardings=dict(config.sharding_config.state_shardings),
                )
                if config.sharding_config is not None
                else None
            )
        else:
            lora_a_sharding, lora_b_sharding = self._adapter_sharding(
                config.sharding_config
            )
        self.lora_a = Linear.Config(
            in_features=config.in_features,
            out_features=config.rank,
            bias=False,
            sharding_config=lora_a_sharding,
            param_init={
                "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
            },
        ).build()
        self.lora_b = Linear.Config(
            in_features=config.rank,
            out_features=config.out_features,
            num_linears=config.num_linears,
            bias=False,
            sharding_config=lora_b_sharding,
            param_init={"weight": nn.init.zeros_},
        ).build()

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        base_out_XO = super()._linear(input, weight, bias)  # type: ignore[misc]
        lora_out_XO = self.lora_b(self.lora_a(input))
        if self.num_linears > 1:
            lora_out_XO = lora_out_XO.flatten(-2)
        return base_out_XO + self._lora_scaling * lora_out_XO

    @staticmethod
    def _adapter_sharding(
        base_sharding: ShardingConfig | None,
    ) -> tuple[ShardingConfig | None, ShardingConfig | None]:
        """Derive adapter sharding from the base linear's TP sharding."""
        base_weight_sharding = (
            base_sharding.state_shardings.get("weight") if base_sharding else None
        )
        if base_weight_sharding is None:
            return None, None

        replicated_weight = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        )
        if base_weight_sharding == dense_param_placement(tp=spmd.S(0)):
            lora_b_sharding = ShardingConfig(
                state_shardings={"weight": base_weight_sharding},
            )
            return replicated_weight, lora_b_sharding

        assert base_weight_sharding == dense_param_placement(tp=spmd.S(1))
        lora_a_sharding = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
        )
        return lora_a_sharding, replicated_weight


class _LoRAGroupedLinearMixin(_LoRAMixin):
    """Add an expert-specific LoRA update to a GroupedLinear."""

    num_linears: int
    lora_a: GroupedLinear
    lora_b: GroupedLinear

    def __init__(self, config) -> None:
        super().__init__(config)
        lora_a_sharding, lora_b_sharding = self._adapter_sharding(
            config.sharding_config
        )
        self.lora_a = GroupedLinear.Config(
            group_size=config.group_size,
            in_features=config.in_features,
            out_features=config.rank,
            sharding_config=lora_a_sharding,
            param_init={
                "weight": lambda w: nn.init.kaiming_uniform_(
                    w.flatten(0, -2), a=math.sqrt(5)
                ),
            },
        ).build()
        self.lora_b = GroupedLinear.Config(
            group_size=config.group_size,
            in_features=config.rank,
            out_features=config.out_features,
            num_linears=config.num_linears,
            sharding_config=lora_b_sharding,
            param_init={"weight": nn.init.zeros_},
        ).build()

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        base_out_RO = super()._grouped_mm(  # type: ignore[misc]
            input_RI=input_RI,
            weight_EOI=weight_EOI,
            offsets_E=offsets_E,
        )
        lora_hidden_RL = self.lora_a(input_RI, offsets_E)
        lora_out_RO = self.lora_b(lora_hidden_RL, offsets_E)
        if self.num_linears > 1:
            lora_out_RO = lora_out_RO.flatten(-2)
        return base_out_RO + self._lora_scaling * lora_out_RO

    @staticmethod
    def _adapter_sharding(
        base_sharding: ShardingConfig | None,
    ) -> tuple[ShardingConfig | None, ShardingConfig | None]:
        """Apply the base projection's expert placement to both adapters."""
        base_weight_sharding = (
            base_sharding.state_shardings.get("weight") if base_sharding else None
        )
        if base_weight_sharding is None:
            return None, None
        if base_weight_sharding != expert_param_placement_sparse():
            raise ValueError(
                "Grouped LoRA supports only expert-axis parameter sharding, got "
                f"{base_weight_sharding}."
            )

        return (
            ShardingConfig(state_shardings={"weight": base_weight_sharding}),
            ShardingConfig(state_shardings={"weight": base_weight_sharding}),
        )


def _create_lora_class(
    parent_cls: type[Module],
    mixin_cls: type[_LoRAMixin],
) -> type[Module]:
    parent_config_cls = parent_cls.Config

    class LoRAProjection(mixin_cls, parent_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float

    LoRAProjection.__name__ = f"LoRA{parent_cls.__name__}"
    LoRAProjection.__qualname__ = f"LoRA{parent_cls.__name__}"
    return LoRAProjection


@functools.cache
def get_lora_linear(parent_cls: type[Module]) -> type[Module]:
    """Get a cached LoRA version of a linear module class."""
    return _create_lora_class(parent_cls, _LoRALinearMixin)


@functools.cache
def get_lora_grouped_linear(parent_cls: type[Module]) -> type[Module]:
    """Get a cached LoRA version of a grouped-linear module class."""
    return _create_lora_class(parent_cls, _LoRAGroupedLinearMixin)
