# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fine-tune Inkling 2 Mini's internal attention projections on one DGX Spark."""

from dataclasses import replace
from pathlib import Path

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import TextProcessor
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import (
    CallableFunction,
    DebugConfig,
    PrecisionConfig,
    TrainingConfig,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, LoRATransform
from torchtitan.experiments.inkling import build_model_config
from torchtitan.experiments.inkling.tokenizer import InklingTokenizer
from torchtitan.experiments.inkling.trainer import InklingTrainer
from torchtitan.observability.metrics import MetricsProcessor


def inkling_2_mini_lora(
    *,
    checkpoint: Path,
    text: Path,
    seq_len: int = 16,
    steps: int = 10,
) -> InklingTrainer.Config:
    """Build the single-Spark config for a local snapshot and line-based corpus."""
    model = build_model_config("2-mini")
    if not 1 <= seq_len <= model.max_context_length:
        raise ValueError(f"seq_len must be between 1 and {model.max_context_length}")
    dataset = SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="text",
            split="train",
            load_dataset_kwargs={"data_files": str(text.resolve())},
        ),
        processor=TextProcessor.Config(),
    )
    lora = LoRATransform(rank=8, alpha=8.0, target_modules=["attn.wq_du"])
    hf_assets_path = str(checkpoint.resolve())
    config = InklingTrainer.Config(
        model=replace(model, max_context_length=seq_len),
        hf_assets_path=hf_assets_path,
        tokenizer=InklingTokenizer.Config(),
        loss=CrossEntropyLoss.Config(),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1, enable_sequence_parallel=False
        ),
        precision=PrecisionConfig(dtype="bfloat16", param_dtype="bfloat16"),
        training=TrainingConfig(
            steps=steps,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            disable_accelerator_graphs=True,
        ),
        activation_remat=None,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
            # Short conv and loop messages share context within one row.
            # Never put a second document after the first one in that context.
            max_num_documents=1,
            shuffle=False,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r"\.lora_[ab]\.weight$", lr=1e-3)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0, min_lr_factor=1.0
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpointer=CheckpointManager.Config(
            initial_load_path=hf_assets_path,
            initial_load_in_hf=True,
            model_param_checkpoint_type_map=CallableFunction.Config(
                fn=lora.param_checkpoint_type
            ),
            last_save_model_only=False,
        ),
        debug=DebugConfig(seed=42),
    )
    return apply_transforms(config, [lora])
