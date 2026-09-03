# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.muse_glimmer.config_registry import muse_glimmer_debugmodel

from . import model_registry


def graph_trainer_muse_glimmer_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        muse_glimmer_debugmodel(seq_len=2048), model_registry
    )
    config.compile = GraphTrainerCompileConfig(enable=True)
    return config
