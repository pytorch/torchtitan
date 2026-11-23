# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GS1 native decision-model fine-tuning on TorchTitan."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.trainer import Trainer
from torchtitan_recipes.decision.base import build_candidate_distribution_config


def gs1_decision(
    model_size: str = "0.8B",
    *,
    parallelism: ParallelismConfig | None = None,
    brier_weight: float = 1.0,
) -> Trainer.Config:
    """Build a GS1 data recipe with distribution NLL plus expected Brier."""
    parallelism = parallelism or ParallelismConfig()
    model_config = build_model_config(model_size, seq_len=8192, parallelism=parallelism)
    return build_candidate_distribution_config(
        model_config=model_config,
        hf_assets_path=f"./assets/hf/GS1-{model_size}",
        data_path="./assets/hf/gs1-distribution/train.jsonl",
        parallelism=parallelism,
        brier_weight=brier_weight,
        max_context_length=model_config.max_context_length,
    )
