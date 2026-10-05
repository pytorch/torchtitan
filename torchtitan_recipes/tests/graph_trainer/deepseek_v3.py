# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V3 GraphTrainer configurations used by tests."""

from torchtitan.config.transform import (
    apply_transforms,
    MXFP8GroupedLinearConverter,
    TokenDispatcherTransform,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)

from torchtitan.experiments.graph_trainer.deepseek_v3 import build_model_config
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)

from torchtitan_recipes.graph_trainer.deepseek_v3 import _dist_moe_graph_config
from torchtitan_recipes.models.deepseek_v3 import (
    deepseek_v3_mxfp8_linear_converter_config,
)

from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_16b_dist_moe_bf16,
    deepseek_v3_16b_dist_moe_mxfp8,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_dist_moe_bf16,
    deepseek_v3_debugmodel_dist_moe_mxfp8,
)


def graph_trainer_deepseek_v3_debugmodel() -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        deepseek_v3_debugmodel(),
        GraphTrainerDeepSeekV3Model.Config,
    )
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_debugmodel_mxfp8() -> GraphTrainer.Config:
    base = deepseek_v3_debugmodel()
    # Quantize dense and moe gemms to mxfp8
    base.model = build_deepseek_v3_model_config(
        "debugmodel",
        seq_len=base.training.max_context_length,
        converters=[
            deepseek_v3_mxfp8_linear_converter_config(),
            MXFP8GroupedLinearConverter.Config(
                pad_multiple=128,
            ),
        ],
    )
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config


def graph_trainer_deepseek_v3_debugmodel_hybridep() -> GraphTrainer.Config:
    base = deepseek_v3_debugmodel()
    base.model = build_model_config(
        "debugmodel",
        seq_len=base.training.max_context_length,
    )
    config = to_graph_trainer_config(base, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    config.parallelism.expert_parallel_degree = 2
    return apply_transforms(
        config,
        [
            TokenDispatcherTransform(
                dispatcher=HybridEPTokenDispatcher,
                kwargs={"non_blocking_capacity_factor": 1.0},
            )
        ],
    )


def graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> GraphTrainer.Config:
    """Build the debug BF16 Dist-MoE GraphTrainer test recipe."""
    return _dist_moe_graph_config(deepseek_v3_debugmodel_dist_moe_bf16(seq_len=seq_len))


def graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> GraphTrainer.Config:
    """Build the debug MXFP8 Dist-MoE GraphTrainer test recipe."""
    return _dist_moe_graph_config(
        deepseek_v3_debugmodel_dist_moe_mxfp8(seq_len=seq_len)
    )


def graph_trainer_deepseek_v3_16b_dist_moe_bf16() -> GraphTrainer.Config:
    """Build the 16B BF16 Dist-MoE GraphTrainer test recipe."""
    return _dist_moe_graph_config(deepseek_v3_16b_dist_moe_bf16(seq_len=4096))


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8() -> GraphTrainer.Config:
    """Build the 16B MXFP8 Dist-MoE GraphTrainer test recipe."""
    return _dist_moe_graph_config(deepseek_v3_16b_dist_moe_mxfp8(seq_len=4096))
