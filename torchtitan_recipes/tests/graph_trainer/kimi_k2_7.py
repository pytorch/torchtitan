# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K2.7 GraphTrainer configurations used by tests."""

from dataclasses import fields

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.kimi_k2_7 import (
    GraphTrainerKimiK25Model,
    without_qk_clip,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.kimi_k2_7 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.kimi_k2_7 import kimi_k2_5_debugmodel


def kimi_k2_7_debugmodel_text(
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Text-only Kimi K2.7 debugmodel without QK clipping, on packed C4 with AdamW.

    GraphTrainer traces one static-shape step, which a vision encoder's
    per-batch image grids cannot provide, and does not replay QK clipping's
    logit recording, so this uses the text-only flavor with plain flex
    attention. SimpleFSDP parameters do not carry the FlexShard layouts
    DistMuon reads, so AdamW trains every parameter, and the config drops the
    Kimi trainer config's DistMuon-only validation.

    Parameters stay in FP32 with BF16 compute. With BF16 parameters,
    SimpleFSDP rounds each loss chunk's lm_head gradient to BF16 before
    accumulating it, while FSDP2 accumulates the chunks in the FP32 reduce
    dtype, so the two would not match bitwise.
    """
    config = kimi_k2_5_debugmodel(parallelism=parallelism)
    config.model = without_qk_clip(
        build_model_config(
            "debugmodel_text",
            seq_len=config.model.max_context_length,
            attn_backend="flex",
            parallelism=config.parallelism,
        )
    )
    config.tokenizer = HuggingFaceTokenizer.Config()
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
    )
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )
    config.precision.dtype = "float32"
    return Trainer.Config(**{f.name: getattr(config, f.name) for f in fields(config)})


def graph_trainer_kimi_k2_7_debugmodel(
    parallelism: ParallelismConfig | None = None,
) -> GraphTrainer.Config:
    config = to_graph_trainer_config(
        kimi_k2_7_debugmodel_text(parallelism=parallelism),
        GraphTrainerKimiK25Model.Config,
    )
    config.compile = GraphTrainerCompileConfig()
    return config
