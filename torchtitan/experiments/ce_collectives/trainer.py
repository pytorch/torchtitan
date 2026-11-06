# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in TorchTitan trainer; the default trainer remains unchanged."""

import logging
from dataclasses import dataclass, field, fields
from typing import cast

from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine

from .configs import CECollectivesConfig
from .fsdp import enable_fsdp_ce_collectives


logger = logging.getLogger(__name__)


class CETrainingEngine(TrainingEngine):
    def _initialize_forward_backward(self) -> None:
        config = cast(CETrainer.Config, self.config)
        num_installed = sum(
            enable_fsdp_ce_collectives(model, config.ce_collectives)
            for model in self.model_parts
        )
        if not num_installed:
            raise ValueError(
                "CE collectives require selected FSDP modules with a shard group larger than one."
            )
        logger.info("Installed CE collectives on %d FSDP modules", num_installed)
        super()._initialize_forward_backward()


class CETrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        ce_collectives: CECollectivesConfig = field(default_factory=CECollectivesConfig)

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.parallelism.fsdp_symm_mem_scope is not None:
                raise ValueError(
                    "CE collectives replace fsdp_symm_mem_scope; configure only one."
                )
            if self.parallelism.fsdp_enable_cpu_offloading:
                raise ValueError(
                    "CE collectives require CUDA-resident FSDP staging buffers."
                )

    engine_cls = CETrainingEngine


def to_ce_trainer_config(
    base: Trainer.Config, **collective_options
) -> CETrainer.Config:
    """Convert a standard TorchTitan recipe to the CE trainer."""
    if type(base) is not Trainer.Config:
        raise ValueError(
            "Convert a standard Trainer.Config; other experimental trainers need their own integration."
        )
    return CETrainer.Config(
        **{item.name: getattr(base, item.name) for item in fields(base)},
        ce_collectives=CECollectivesConfig(**collective_options),
    )
