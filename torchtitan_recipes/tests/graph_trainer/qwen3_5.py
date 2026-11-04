# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 GraphTrainer configurations used by tests."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.qwen3_5.model import GraphTrainerQwen35Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel_moe


def qwen35_debugmodel_moe_text(parallelism: ParallelismConfig) -> Trainer.Config:
    """Text-only Qwen3.5 MoE debugmodel trained on packed C4.

    GraphTrainer traces one static-shape step, which a vision encoder's
    per-batch image grids cannot provide, so this uses the text-only flavor.
    """
    config = qwen35_debugmodel_moe(parallelism=parallelism)
    config.model = build_model_config(
        "debugmodel_moe_text",
        seq_len=config.model.max_context_length,
        attn_backend="flex",
        parallelism=parallelism,
    )
    config.tokenizer = HuggingFaceTokenizer.Config()
    config.dataloader = GrainDataLoader.Config(
        max_num_documents=c4_document_capacity(
            config.training.num_tokens_per_microbatch_per_dp_rank
        ),
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
    )
    return config


def graph_trainer_qwen35_debugmodel_moe(
    parallelism: ParallelismConfig,
) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        qwen35_debugmodel_moe_text(parallelism), GraphTrainerQwen35Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    return config
