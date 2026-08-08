# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.pretokenized import PreTokenizedTextDataLoader
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import model_registry
from .loss import ZLossCrossEntropyLoss
from .lr_scheduler import Olmo3CosWithWarmup
from .optimizer import olmo3_pretrain_adamw


def olmo3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4),
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
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def olmo3_debugmodel_varlen_attn() -> Trainer.Config:
    config = olmo3_debugmodel()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    return config


def olmo3_7b() -> Trainer.Config:
    # OLMo-core's 7B pretraining recipe uses flash attention with the same
    # sliding-window pattern. VarlenAttention is TorchTitan's closest backend
    # for this setup and supports per-layer window_size through the model config.
    model_spec = model_registry("7B", attn_backend="varlen")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=ZLossCrossEntropyLoss.Config(),
        ),
        hf_assets_path="./assets/hf/Olmo-3-1025-7B",
        profiler=Profiler.Config(
            enable_profiling=True,
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            enable_tensorboard=True,
        ),
        model_spec=model_spec,
        optimizer=olmo3_pretrain_adamw(lr=3e-4),
        lr_scheduler=Olmo3CosWithWarmup.Config(
            warmup_steps=2000,
            total_steps=1_192_092,
            alpha_f=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            global_batch_size=512,
            seq_len=8192,
            steps=1_192_092,
        ),
        dataloader=PreTokenizedTextDataLoader.Config(
            dataset="dolma3_common_crawl_religion_0016",
            dataset_path=(
                "/home/ruisizhang123/ruisizhang123_data/tree/"
                "dolma3_mix-6T-1025-7B/pre-tokenize-data/common_crawl-religion-0016"
            ),
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=-1,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
    )
