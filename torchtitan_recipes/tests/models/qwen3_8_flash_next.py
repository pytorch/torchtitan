# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reduced text and multimodal fixtures for Flash-Next training."""

from dataclasses import replace

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.qwen3_8 import (
    QWEN3_8_SPECIAL_TOKENS,
    QWEN3_8_VISION_PROCESSOR_KWARGS,
)
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss
from torchtitan.observability.metrics import MetricsProcessor, TensorBoardLogger
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.models.qwen3_8_flash_next import (
    FlashNextTrainerConfig,
    make_dist_muon_optimizer,
)


def qwen38_flash_next_debugmodel(
    *,
    seq_len: int = 64,
    parallelism: ParallelismConfig | None = None,
    optimizer: OptimizersContainer.Config | None = None,
) -> Trainer.Config:
    """Text debug model; ``optimizer=None`` uses the report's DistMuon policy."""
    parallelism = parallelism or ParallelismConfig()
    model_config = replace(
        build_model_config(
            "debugmodel",
            seq_len=seq_len,
            parallelism=parallelism,
            indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
        ),
        local_compile=LocalCompileConfig(regions=[]),
    )
    optimizer = optimizer or make_dist_muon_optimizer(
        model_config, parallelism=parallelism, lr=8e-4
    )
    return FlashNextTrainerConfig(
        model=model_config,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(8 * seq_len),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        optim=Optim.Config(
            optimizer=optimizer,
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1, loggers=[TensorBoardLogger.Config()]
        ),
        parallelism=parallelism,
        activation_remat=None,
        debug=DebugConfig(seed=42, deterministic=True),
    )


def qwen38_flash_next_debugmodel_mm() -> Trainer.Config:
    parallelism = ParallelismConfig(expert_parallel_degree=2)
    model_config = replace(
        build_model_config(
            "debugmodel_mm",
            seq_len=1024,
            parallelism=parallelism,
            indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
        ),
        local_compile=LocalCompileConfig(regions=[]),
    )
    return FlashNextTrainerConfig(
        model=model_config,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_8_SPECIAL_TOKENS),
        vision_processor=VisionProcessor.Config(**QWEN3_8_VISION_PROCESSOR_KWARGS),
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m-test"],
            collator=MultiModalCollator.Config(build_mrope_positions=True),
            streaming_shuffle_buffer_size=128,
        ),
        optim=Optim.Config(
            optimizer=make_dist_muon_optimizer(
                model_config, parallelism=parallelism, lr=5e-3
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1024,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1, loggers=[TensorBoardLogger.Config()]
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=None,
        debug=DebugConfig(seed=42, deterministic=True, spmd_typechecking=True),
    )


def qwen38_flash_next_debugmodel_tp2() -> Trainer.Config:
    """Text debug model with TP2 + SP and EP2 on two ranks, typechecked.

    AdamW replaces the DistMuon policy until DistMuon supports TP layouts
    (https://github.com/pytorch/torchtitan/pull/5678).
    """
    config = qwen38_flash_next_debugmodel(
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=2,
            enable_sequence_parallel=True,
        ),
        optimizer=OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
        ),
    )
    config.debug.spmd_typechecking = True
    return config


def qwen38_flash_next_debugmodel_fsdp2_tp2_ep4_pp2() -> Trainer.Config:
    """FSDP2 + TP2 + SP + EP4 + PP2 with Interleaved1F1B and full remat.

    PLE (``layers.1``) lands in the second local model part of PP rank 0, so
    the first local part prepares routing for a table it does not own. AdamW
    replaces the DistMuon policy, which supports neither TP
    (https://github.com/pytorch/torchtitan/pull/5678) nor PP.
    """
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
        pipeline_parallel_degree=2,
        num_pp_microbatches=4,
        pipeline_parallel_schedule="Interleaved1F1B",
        enable_sequence_parallel=True,
        pipeline_parallel_module_fqns_per_model_part=[
            ["tok_embeddings"],
            ["layers.0"],
            ["layers.1", "layers.2", "layers.3"],
            [f"layers.{i}" for i in range(4, 8)]
            + ["norm", "lm_head", "hyper_connection_mixer"],
        ],
    )
    config = qwen38_flash_next_debugmodel(
        parallelism=parallelism,
        optimizer=OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
        ),
    )
    config.activation_remat = FullActivationRematConfig()
    return config
