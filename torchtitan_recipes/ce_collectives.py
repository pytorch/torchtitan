# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small C4 training recipes for validating the FSDP communication adapters."""

from torchtitan.experiments.ce_collectives.trainer import (
    CETrainer,
    to_ce_trainer_config,
)
from torchtitan.observability.metrics import TensorBoardLogger

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def nccl_debug() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=128, attn_backend="flex")
    config.training.steps = 12
    config.training.num_tokens_per_microbatch_per_dp_rank = 128
    config.debug.seed = 42
    config.debug.deterministic = True
    config.metrics.loggers = [TensorBoardLogger.Config()]
    return config


def ce_debug() -> CETrainer.Config:
    return to_ce_trainer_config(nccl_debug())


def ce_two_domain_debug() -> CETrainer.Config:
    # Four ranks laid out as two logical two-rank domains.
    return to_ce_trainer_config(nccl_debug(), all_gather=False, ranks_per_domain=2)


def graph_nccl_debug():
    from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
    from torchtitan.experiments.graph_trainer.llama3 import GraphTrainerLlama3Model

    config = to_graph_trainer_config(nccl_debug(), GraphTrainerLlama3Model.Config)
    config.compile.numerics_changing_optim = True
    return config


def graph_ce_debug():
    from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig

    config = graph_nccl_debug()
    config.compile.ce_collectives = CECollectivesConfig()
    return config


def graph_ce_all_gather_debug():
    from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig

    config = graph_nccl_debug()
    config.compile.ce_collectives = CECollectivesConfig(reduce_scatter=False)
    return config
