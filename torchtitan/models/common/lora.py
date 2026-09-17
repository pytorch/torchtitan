# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA decoration for linear modules.

Shape suffixes: ``X`` arbitrary leading dimensions, ``I`` input features,
``O`` output features.
"""

import math
from dataclasses import dataclass

import spmd_types as spmd

import torch.nn as nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

__all__ = ["LinearLoRADecorator"]


@dataclass(frozen=True, slots=True)
class LinearLoRADecorator:
    """Attach LoRA state and computation to a built linear module."""

    rank: int
    alpha: float

    def apply(self, module: Module, config: Module.Config) -> None:
        """Attach adapters and add their result through a forward hook."""
        assert isinstance(config, Linear.Config)
        lora_a_sharding, lora_b_sharding = _adapter_sharding(config.sharding_config)
        lora_a = Linear.Config(
            in_features=config.in_features,
            out_features=self.rank,
            bias=False,
            sharding_config=lora_a_sharding,
            param_init={
                "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
            },
        ).build()
        lora_b = Linear.Config(
            in_features=self.rank,
            out_features=config.out_features,
            bias=False,
            sharding_config=lora_b_sharding,
            param_init={"weight": nn.init.zeros_},
        ).build()
        module.add_module("lora_a", lora_a)
        module.add_module("lora_b", lora_b)
        scaling = self.alpha / self.rank

        def add_lora_output(_module, args, kwargs, base_out_XO):
            input_XI = args[0] if args else kwargs["input"]
            lora_out_XO = lora_b(lora_a(input_XI))
            return base_out_XO + scaling * lora_out_XO

        # Run before other output hooks so later module-boundary processing
        # sees the combined base and adapter result. Any input pre-hooks have
        # already updated the arguments passed here.
        module.register_forward_hook(
            add_lora_output,
            prepend=True,
            with_kwargs=True,
        )


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
    if base_weight_sharding == dense_param_placement(tp=spmd.S(1)):
        lora_a_sharding = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
        )
        return lora_a_sharding, replicated_weight

    assert base_weight_sharding == dense_param_placement(tp=spmd.S(0))
    lora_b_sharding = ShardingConfig(
        state_shardings={"weight": base_weight_sharding},
    )
    return replicated_weight, lora_b_sharding
