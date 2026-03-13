# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from . import model_registry


def _configure_local_hf_cache() -> None:
    # Keep default behavior when users explicitly configure cache env vars.
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/huggingface"
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = os.path.join(
            os.environ["HF_HOME"], "datasets"
        )
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


def nemotron3_debugmodel() -> Trainer.Config:
    """Python config equivalent of train_configs/debug_model.toml."""
    _configure_local_hf_cache()
    return Trainer.Config(
        hf_assets_path="./assets/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        debug=DebugConfig(print_config=False),
        profiling=ProfilingConfig(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=10,
            enable_memory_snapshot=False,
            save_memory_snapshot_folder="memory_snapshot",
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            disable_color_printing=False,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=8e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=20,
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="default",
            tensor_parallel_degree=1,
            enable_async_tensor_parallel=False,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            folder="checkpoint",
            interval=10,
            last_save_model_only=False,
            export_dtype="float32",
            async_mode="disabled",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        validator=Validator.Config(
            enable=False,
            dataloader=HuggingFaceTextDataLoader.Config(
                dataset="c4_validation",
                infinite=False,
            ),
            freq=5,
            steps=10,
        ),
    )


def nemotron3_nano_30b() -> Trainer.Config:
    """Python config equivalent of train_configs/nemotron3-nano-30B.toml."""
    _configure_local_hf_cache()
    config = Trainer.Config(
        hf_assets_path="./assets/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        profiling=ProfilingConfig(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=100,
            profiler_warmup=3,
            profiler_active=1,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=True,
            save_tb_folder="tb",
            enable_wandb=True,
        ),
        model_spec=model_registry("nano-30B"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=3e-5,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=500),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            max_norm=1.0,
            steps=300,
            dtype="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="default",
            tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint",
            interval=500,
            last_save_model_only=False,
            export_dtype="bfloat16",
            async_mode="disabled",
            initial_load_path="./assets/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            initial_load_in_hf=True,
            initial_load_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
            selective_ac_option="op",
        ),
        compile=CompileConfig(enable=False, components=["model", "loss"]),
        validator=Validator.Config(
            enable=False,
            dataloader=HuggingFaceTextDataLoader.Config(
                dataset="c4_validation",
                infinite=False,
            ),
            freq=500,
            steps=1200,
        ),
    )
    # Keep fake_backend/local_tensor dry-runs lightweight and deterministic.
    if os.environ.get("COMM_MODE", "") in ("fake_backend", "local_tensor"):
        config.model_spec = model_registry("debugmodel")
        config.training.local_batch_size = 1
        config.training.seq_len = 256
        config.training.steps = 1
        config.metrics.enable_wandb = False
        config.checkpoint.enable = False
        config.checkpoint.initial_load_path = None
        config.checkpoint.initial_load_in_hf = False
    return config
