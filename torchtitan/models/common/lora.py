# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA linear modules.

Shape suffixes: ``X`` arbitrary leading dimensions, ``I`` input features,
``O`` output features.
"""

import functools
import math
from dataclasses import dataclass

import spmd_types as spmd

import torch
import torch.nn as nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

__all__ = ["get_lora_linear"]


class _LoRALinearMixin:
    """Add a LoRA update to a Linear's local computation."""

    _lora_scaling: float
    num_linears: int
    lora_a: Linear
    lora_b: Linear

    def __init__(self, config) -> None:
        super().__init__(config)  # type: ignore[misc]
        for param in nn.Module.parameters(self):  # type: ignore[arg-type]
            param.requires_grad_(False)
        self._lora_scaling = config.alpha / config.rank
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


@functools.cache
def get_lora_linear(parent_cls: type[Module]) -> type[Module]:
    """Get a cached LoRA version of a linear module class."""
    parent_config_cls = parent_cls.Config

    class LoRALinear(_LoRALinearMixin, parent_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    return LoRALinear
