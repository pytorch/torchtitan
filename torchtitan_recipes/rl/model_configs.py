# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config helpers shared by the RL recipes."""

from collections.abc import Callable
from typing import Any, cast

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    FreezeMoERoutingTransform,
    LMHeadFP32OutputConverter,
    ModelConfigTransform,
    transform_model_config_,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig


def build_rl_model_configs(
    build_model_config: Callable[..., Decoder.Config],
    flavor: str,
    *,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    fp32_lm_head: bool = True,
    # TODO: consider enabling router training, but with a lower lr
    freeze_moe_routing: bool = True,
    **kwargs: Any,
) -> tuple[Decoder.Config, Decoder.Config]:
    """Build the ``(trainer, generator)`` model configs of an RL recipe.

    Both come from the same ``build_model_config(flavor, **kwargs)`` call, so
    they share the architecture; each is sharded for its own role's
    parallelism. Changes made to one afterwards must be made to the other.

    RL defaults, applied to both (pass False to skip one):
        fp32_lm_head        lm_head -> HiMidLoLinear (fp32 logits)
        freeze_moe_routing  frozen router gate and expert bias, no aux loss
                            (FreezeMoERoutingTransform), so trainer and generator
                            expert choices don't drift apart
    Per-role transforms go through ``apply_transforms(config.generator, ...)``.
    """
    if fp32_lm_head:
        kwargs["converters"] = [
            *(kwargs.get("converters") or []),
            LMHeadFP32OutputConverter.Config(),
        ]
    model_configs = (
        build_model_config(flavor, parallelism=trainer_parallelism, **kwargs),
        build_model_config(
            flavor, parallelism=generator_parallelism.to_training(), **kwargs
        ),
    )
    transforms: list[ModelConfigTransform] = []
    if freeze_moe_routing:
        transforms.append(FreezeMoERoutingTransform())
    # A transform may return a new root (e.g. LoRA), so keep what it returns.
    trainer_model, generator_model = (
        cast(Decoder.Config, transform_model_config_(model_config, transforms))
        for model_config in model_configs
    )
    return trainer_model, generator_model
