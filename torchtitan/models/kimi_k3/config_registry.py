# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.hf_datasets.multimodal.utils.image import resize_to_patch_budget
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import KIMI_K3_SPECIAL_TOKENS, model_registry


def kimi_k3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=MMDataLoader.Config(
            dataset="cc12m-test",
            max_images_per_batch=8,
            patch_size=14,
            temporal_patch_size=1,
            spatial_merge_size=2,
            patch_order="raster",
            resize_fn=resize_to_patch_budget,
            min_pixels=56 * 56,
            max_pixels=224 * 224,
            max_patches=256,
            max_patches_per_side=16,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        # TODO: Kimi K3 has no spmd_types annotations yet.
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor"),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=256,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=None,
    )
