# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config helpers shared by the RL recipes."""

from collections.abc import Callable
from typing import Any

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadFP32OutputConverter
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig


def build_rl_model_configs(
    build_model_config: Callable[..., Decoder.Config],
    flavor: str,
    *,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    fp32_lm_head: bool = True,
    freeze_moe_routing: bool = True,
    **kwargs: Any,
) -> tuple[Decoder.Config, Decoder.Config]:
    """Build the ``(trainer, generator)`` model configs of an RL recipe.

    Both come from the same ``build_model_config(flavor, **kwargs)`` call, so
    they share the architecture; each is sharded for its own role's
    parallelism. Changes made to one afterwards must be made to the other.

    RL defaults, applied to both (pass False to skip one):
        fp32_lm_head        lm_head -> HiMidLoLinear (fp32 logits)
        freeze_moe_routing  frozen expert bias and router gate, no aux loss,
                            so the trainer and generator pick the same experts
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
    # TODO: consider enabling router training, but with a lower lr
    if freeze_moe_routing:
        for model_config in model_configs:
            for _fqn, moe_config, _parent, _attr in model_config.traverse(MoE.Config):
                moe_config.freeze_expert_bias = True
                moe_config.router.freeze_gate = True
                moe_config.router.aux_loss = None
    return model_configs
