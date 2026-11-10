# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model configurations used by tests."""

from dataclasses import replace
from typing import cast

import torch

from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
from torchtitan.components.data.multimodal.vision_processor import (
    NaViTPatchGridResize,
    VisionProcessor,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer, Optim
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.distributed.flex_shard import BlockShard, ComputeLayout, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k3 import (
    KIMI_K3_SPECIAL_TOKENS,
    KIMI_K3_VISION_PROCESSOR_KWARGS,
)
from torchtitan.models.kimi_k3_1 import build_model_config
from torchtitan.models.kimi_k3_1.dsa import KimiDSAAttention
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer,
    _kimi_k3_multimodal_dataloader,
    _KimiK3TrainerConfig,
)


def kimi_k3_1_debugmodel(
    seq_len: int | None = 256,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel", seq_len=seq_len, parallelism=parallelism
    )
    indexer = cast(KimiDSAAttention.Config, model_config.first_full_attention).indexer
    # Same storage axes as the Kimi K3 layouts in ``_dist_muon_optimizer``.
    storage_axes = (
        MeshAxisName.DP_SHARD.value,
        f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}",
    )
    owned = ComputeLayout(
        shardings_by_mesh_axis={axis: Owned() for axis in storage_axes},
        compute_dtype=torch.bfloat16,
    )
    per_index_head = ComputeLayout(
        shardings_by_mesh_axis={
            axis: BlockShard(dim=0, block_sizes=(indexer.index_head_dim,))
            for axis in storage_axes
        },
        compute_dtype=torch.bfloat16,
    )
    return _KimiK3TrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        vision_processor=replace(
            VisionProcessor.Config(**KIMI_K3_VISION_PROCESSOR_KWARGS),
            resize=NaViTPatchGridResize.Config(
                max_patches=256, max_patches_per_side=16
            ),
        ),
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                parallelism=parallelism,
                extra_full_attention_shardings={
                    "indexer.wq_b": per_index_head,
                    "indexer.wk": owned,
                    "indexer.weights_proj": owned,
                },
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        precision=PrecisionConfig(dtype="bfloat16"),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )
