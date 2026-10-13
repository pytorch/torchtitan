# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model config helpers shared by the RL recipes."""

import copy
from collections.abc import Callable, Sequence
from typing import Any, cast, Protocol, TypeVar

from torchtitan.config import Configurable
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    apply_transforms,
    FreezeMoERoutingTransform,
    LMHeadFP32OutputTransform,
    ModelConfigTransform,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig


class _RLConfig(Protocol):
    trainer: Configurable.Config
    generator: Configurable.Config

    def __post_init__(self) -> None:
        ...


_ConfigT = TypeVar("_ConfigT", bound=Configurable.Config)


def build_rl_model_configs(
    build_model_config: Callable[..., Decoder.Config],
    flavor: str,
    *,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    **kwargs: Any,
) -> tuple[Decoder.Config, Decoder.Config]:
    """Build independent base model configs for the two RL roles.

    Both use the same architecture arguments and their own parallelism. After
    constructing the complete controller config, apply RL defaults and all
    shared and role-specific model transforms together with
    ``apply_rl_transforms``.
    """
    return (
        build_model_config(flavor, parallelism=trainer_parallelism, **kwargs),
        build_model_config(
            flavor, parallelism=generator_parallelism.to_training(), **kwargs
        ),
    )


def apply_rl_transforms(
    config: _ConfigT,
    *,
    shared_transforms: Sequence[ModelConfigTransform] = (),
    trainer_transforms: Sequence[ModelConfigTransform] = (),
    generator_transforms: Sequence[ModelConfigTransform] = (),
    fp32_lm_head: bool = True,
    freeze_moe_routing: bool = True,
) -> _ConfigT:
    """Apply complete transform lists to both roles of a copied RL config.

    Set the role settings, including companion runtime configs and token
    budgets, before calling this once. Each role orders RL defaults, shared
    transforms, and its own transforms together through ``apply_transforms``.
    Both role configs and the resulting controller config are validated. The
    input config is unchanged, including when transformation or validation fails.
    """
    defaults: list[ModelConfigTransform] = []
    if fp32_lm_head:
        defaults.append(LMHeadFP32OutputTransform())
    if freeze_moe_routing:
        defaults.append(FreezeMoERoutingTransform())
    shared = [*defaults, *shared_transforms]
    working = copy.copy(config)
    rl_config = cast(_RLConfig, working)
    rl_config.trainer = apply_transforms(
        rl_config.trainer, [*shared, *trainer_transforms]
    )
    rl_config.generator = apply_transforms(
        rl_config.generator, [*shared, *generator_transforms]
    )
    # The roles are already independent copies. Reuse them while copying the
    # rest of the controller, including when their input models were shared.
    working = copy.deepcopy(
        working,
        memo={
            id(rl_config.trainer): rl_config.trainer,
            id(rl_config.generator): rl_config.generator,
        },
    )
    cast(_RLConfig, working).__post_init__()
    return working
