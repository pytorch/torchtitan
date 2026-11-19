# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GS1-format decision fine-tuning on TorchTitan-supported backbones."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.trainer import Trainer
from torchtitan_recipes.decision.base import DecisionModel


class GS1DecisionModel(DecisionModel):
    """Use GS1 decision data with a TorchTitan Qwen3.5 backbone."""

    @property
    def hf_assets_path(self) -> str:
        return f"./assets/hf/Qwen3.5-{self.model_size}"

    def build_backbone_config(self, *, seq_len: int) -> tuple[Qwen35Model.Config, int]:
        model_config = build_model_config(
            self.model_size, seq_len=seq_len, parallelism=self.parallelism
        )
        return model_config, model_config.max_context_length


def gs1_decision(
    model_size: str = "0.8B",
    *,
    parallelism: ParallelismConfig | None = None,
    brier_weight: float = 1.0,
) -> Trainer.Config:
    """Build a GS1 data recipe using TorchTitan's selected backbone family."""
    return GS1DecisionModel(
        model_size, parallelism=parallelism, brier_weight=brier_weight
    ).build_trainer_config(data_path="./assets/hf/gs1-distribution/train.jsonl")
