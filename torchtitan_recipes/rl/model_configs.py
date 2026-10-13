# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config helpers shared by the RL recipes."""

from collections.abc import Callable, Sequence
from typing import Any, cast

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    FreezeMoERoutingTransform,
    LMHeadFP32OutputConverter,
    ModelConfigTransform,
    transform_model_config_,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig


def build_rl_model_configs(
    build_model_config: Callable[..., Decoder.Config],
    flavor: str,
    *,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    fp32_lm_head: bool = True,
    # TODO: consider enabling router training, but with a lower lr
    shared_transforms: Sequence[ModelConfigTransform] = (FreezeMoERoutingTransform(),),
    **kwargs: Any,
) -> tuple[Decoder.Config, Decoder.Config]:
    """Build the ``(trainer, generator)`` model configs of an RL recipe.

    Both come from the same ``build_model_config(flavor, **kwargs)`` call, so
    they share the architecture; each is sharded for its own role's
    parallelism. Changes made to one afterwards must be made to the other.

    RL defaults, applied to both:
        fp32_lm_head       lm_head -> HiMidLoLinear (fp32 logits); False skips it
        shared_transforms  FreezeMoERoutingTransform, so trainer and generator expert
                           choices don't drift apart; () trains the router
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
    # A transform may return a new root (e.g. LoRA), so keep what it returns.
    trainer_model, generator_model = (
        cast(
            Decoder.Config,
            transform_model_config_(model_config, list(shared_transforms)),
        )
        for model_config in model_configs
    )
    return trainer_model, generator_model
