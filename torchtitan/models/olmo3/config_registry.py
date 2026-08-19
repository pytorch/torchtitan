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
            profile_freq=1000,
        ),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
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
            dataset="dolma3_mix_6t_1025_7b",
            dataset_path=(
                "/home/ruisizhang123/ruisizhang123_data/"
                "dolma3_mix-6T-1025-7B/pre-tokenize-data"
            ),
            shuffle=True,
            shuffle_seed=34521,
            shuffle_strategy="global",
            shuffle_block_size=1024,
            # The mix has 906 token files. A global shuffle reaches them in
            # random order, so anything below that thrashes the LRU fd cache
            # and makes most reads pay a FUSE open on top of the pread.
            max_open_files=1024,
            # num_workers matches OLMo-core's 7B pretrain recipe. Unlike
            # OLMo-core's Weka/NVMe backends, the token files here are served
            # over a network FUSE mount where one instance read costs ~400ms,
            # so reader threads are enabled on top of the worker processes.
            num_workers=8,
            num_threads=4,
            read_ahead=32,
            prefetch_factor=8,
            persistent_workers=True,
            pin_memory=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=-1,
        ),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(interval=1000),
        activation_checkpoint=SelectiveAC.Config(),
    )
