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
from typing import Any

import spmd_types as spmd

import torch
import torch.nn as nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
    compose_parallel_linear_cls,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

__all__ = [
    "LoRAColumnParallelLinear",
    "LoRALinear",
    "LoRARowParallelLinear",
    "specialize_lora_linear",
]


class LoRALinear(Linear):
    """Linear with a LoRA update on its local computation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        rank: int
        alpha: float

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._init_adapters(self, config)

    @staticmethod
    def _init_adapters(module, config: Any) -> None:
        """Freeze the base projection and construct its trainable adapters."""
        for param in nn.Module.parameters(module):
            param.requires_grad_(False)
        module._lora_scaling = config.alpha / config.rank
        lora_a_sharding, lora_b_sharding = LoRALinear._adapter_sharding(
            config.sharding_config
        )
        module.lora_a = Linear.Config(
            in_features=config.in_features,
            out_features=config.rank,
            bias=False,
            sharding_config=lora_a_sharding,
            param_init={
                "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
            },
        ).build()
        module.lora_b = Linear.Config(
            in_features=config.rank,
            out_features=config.out_features,
            bias=False,
            sharding_config=lora_b_sharding,
            param_init={"weight": nn.init.zeros_},
        ).build()

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the base projection and add the LoRA adapter output."""
        base_out_XO = super()._linear(input)
        return self._add_adapter_output(self, input, base_out_XO)

    @staticmethod
    def _add_adapter_output(
        module, input: torch.Tensor, base_out_XO: torch.Tensor
    ) -> torch.Tensor:
        """Add the adapter update to an already-computed base projection."""
        lora_out_XO = module.lora_b(module.lora_a(input))
        return base_out_XO + module._lora_scaling * lora_out_XO

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
def specialize_lora_linear(parent_cls: type[Module]) -> type[Module]:
    """Add LoRA local compute without changing a Linear's TP role."""
    if parent_cls is Linear:
        return LoRALinear

    if parent_cls in (ColumnParallelLinear, RowParallelLinear):
        return compose_parallel_linear_cls(LoRALinear, parent_cls)

    # Quantization runs before LoRA and may add fields to the parent config.
    # Retain that exact implementation while inserting the adapter into its
    # local compute.
    parent_config_cls = parent_cls.Config

    class SpecializedLoRALinear(parent_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float

        def __init__(self, config: Config):
            parent_cls.__init__(self, config)
            LoRALinear._init_adapters(self, config)

        def _linear(self, input: torch.Tensor) -> torch.Tensor:
            base_out_XO = parent_cls._linear(self, input)  # type: ignore[attr-defined]
            return LoRALinear._add_adapter_output(self, input, base_out_XO)

    SpecializedLoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    SpecializedLoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    return SpecializedLoRALinear


LoRAColumnParallelLinear = compose_parallel_linear_cls(LoRALinear, ColumnParallelLinear)
LoRARowParallelLinear = compose_parallel_linear_cls(LoRALinear, RowParallelLinear)
