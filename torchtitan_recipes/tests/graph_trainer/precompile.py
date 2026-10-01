# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer precompile integration-test configurations."""

import os

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
)
from torchtitan_recipes.tests.graph_trainer.llama3 import (
    graph_trainer_llama3_debugmodel_sdpa,
)


def llama3_precompile_fsdp_tp() -> GraphTrainer.Config:
    config = graph_trainer_llama3_debugmodel_sdpa()
    config.compile.precompile_artifact_dir = os.environ[
        "TORCHTITAN_PRECOMPILE_ARTIFACT_DIR"
    ]
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 4
    return config


def deepseek_v3_precompile_fsdp_tp_ep() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_debugmodel()
    config.compile.precompile_artifact_dir = os.environ[
        "TORCHTITAN_PRECOMPILE_ARTIFACT_DIR"
    ]
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    return config
