# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 GraphTrainer configurations used by tests."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.kimi_k3.model import GraphTrainerKimiK3Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def kimi_k3_debugmodel_text(
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Text-only Kimi K3 debugmodel trained on packed C4 with AdamW.

    GraphTrainer traces one static-shape step, which a vision encoder's
    per-batch image grids cannot provide, so this uses the text-only flavor.
    SimpleFSDP parameters do not carry the FlexShard layouts DistMuon reads, so
    AdamW trains every parameter.

    Parameters stay in FP32 with BF16 compute. With BF16 parameters,
    SimpleFSDP rounds each loss chunk's lm_head gradient to BF16 before
    accumulating it, while FSDP2 accumulates the chunks in the FP32 reduce
    dtype, so the two would not match bitwise.
    """
    config = kimi_k3_debugmodel(parallelism=parallelism)
    config.model = build_model_config(
        "debugmodel_text",
        seq_len=config.model.max_context_length,
        parallelism=config.parallelism,
    )
    config.tokenizer = HuggingFaceTokenizer.Config()
    config.dataloader = GrainDataLoader.Config(
        max_num_documents=c4_document_capacity(
            config.training.num_tokens_per_microbatch_per_dp_rank
        ),
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
    )
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    config.precision.dtype = "float32"
    return config


def graph_trainer_kimi_k3_debugmodel(
    parallelism: ParallelismConfig | None = None,
) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        kimi_k3_debugmodel_text(parallelism=parallelism),
        GraphTrainerKimiK3Model.Config,
    )
    config.compile = GraphTrainerCompileConfig()
    return config
