# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.models.llama3.config_registry import llama3_8b, llama3_debugmodel

from . import model_registry
from .trainer import FlexShardGraphTrainer


def _to_flex_shard_config(base_config) -> FlexShardGraphTrainer.Config:
    config = to_graph_trainer_config(base_config, model_registry)
    values = {f.name: getattr(config, f.name) for f in fields(config)}
    values["compile"] = GraphTrainerCompileConfig(enable=False, mode=None)
    return FlexShardGraphTrainer.Config(**values)


def graph_trainer_flex_shard_llama3_debugmodel() -> FlexShardGraphTrainer.Config:
    return _to_flex_shard_config(llama3_debugmodel())


def graph_trainer_flex_shard_llama3_8b() -> FlexShardGraphTrainer.Config:
    return _to_flex_shard_config(llama3_8b())
