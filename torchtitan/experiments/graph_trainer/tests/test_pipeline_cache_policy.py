# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel

from torchtitan.experiments.graph_trainer.configs import to_graph_trainer_config
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)


def test_graph_pipeline_rejects_ignored_startup_cache_policy() -> None:
    """GraphPipelineStage does not execute core's Python backward hook."""
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel(), GraphTrainerDeepSeekV3Model.Config
    )
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_empty_cache_before_first_backward = True
    with pytest.raises(
        ValueError, match="does not support pipeline_parallel_empty_cache"
    ):
        config.__post_init__()
