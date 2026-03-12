# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.models.llama3 import model_registry as llama3_registry
from torchtitan.models.qwen3 import model_registry as qwen3_registry

from .configs import SFTTrainerConfig
from .dataset import SFTDataLoader


def sft_debugmodel() -> SFTTrainerConfig:
    """SFT debug training with Llama3 and small question/answer dataset."""

    def process_sample(sample: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    return SFTTrainerConfig(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=llama3_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=SFTDataLoader.Config(
            dataset_path="json",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
                "split": "train",
            },
            sample_processor=process_sample,
            pack_sequences=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def sft_qwen3_8b_math() -> SFTTrainerConfig:
    """Qwen3-8B SFT on GSM8K math dataset."""

    def process_sample(sample: dict[str, Any]) -> list[dict[str, str]]:
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]

    model_spec = qwen3_registry("8B")
    model_spec.model.layer.attention.attn_backend = "flex"
    model_spec.model.layer.attention.attn_mask_type = "block_causal"
    return SFTTrainerConfig(
        hf_assets_path="./assets/hf/Qwen3-8B",
        model_spec=model_spec,
        optimizer=OptimizersContainer.Config(lr=2e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=15,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=2048,
            steps=180,  # Trains for ~2 epochs following OLMo 3
        ),
        dataloader=SFTDataLoader.Config(
            dataset_path="openai/gsm8k",
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_sample,
            pack_sequences=True,
        ),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
        validator=Validator.Config(
            enable=True,
            freq=50,
            steps=5,  # Large enough to get a solid signal, but won't run out of data and hang under DP
            dataloader=SFTDataLoader.Config(
                dataset_path="openai/gsm8k",
                load_dataset_kwargs={"name": "main", "split": "test"},
                sample_processor=process_sample,
                infinite=False,
                pack_sequences=True,
            ),
        ),
    )
