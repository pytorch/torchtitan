# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC

from .dataset_config import (
    _dataloader_config,
    BASE_DIR_GT_10M,
    DEFAULT_10M_TRAIN_LIST,
    IMAGE_SIZE,
)
from .loss import WorldModelLoss
from .model_config import (
    _blocks_only_float8,
    COMPRESSOR_MODEL,
    LATENT_CHANNELS,
    LATENT_SIZE,
    model_registry,
    WORLD_MODEL_FLOAT8_FILTER_FQNS,
)
from .tokenizer import WorldModelTokenizer
from .torchpackage_checkpoint import WorldModelTorchPackageCheckpointManager
from .trainer import WorldModelTrainer, WorldModelValidator


__all__ = [
    "BASE_DIR_GT_10M",
    "COMPRESSOR_MODEL",
    "DEFAULT_10M_TRAIN_LIST",
    "IMAGE_SIZE",
    "LATENT_CHANNELS",
    "LATENT_SIZE",
    "WORLD_MODEL_FLOAT8_FILTER_FQNS",
    "_blocks_only_float8",
    "_dataloader_config",
    "model_registry",
    "worldmodel",
]


def worldmodel() -> WorldModelTrainer.Config:
    local_batch_size = 16
    validation_freq = 512
    steps = validation_freq * 30
    validation_steps = 8
    compile_config = CompileConfig(enable=True, components=["model", "loss"])
    optimizer = default_adamw(lr=2e-4, weight_decay=1e-2)
    optimizer.implementation = "fused_opt_states_bf16"
    local_world_size, world_size, num_nodes = _world_sizes()
    checkpoint_folder = _reporterv2_checkpoint_folder()

    return WorldModelTrainer.Config(
        hf_assets_path=".",
        loss=WorldModelLoss.Config(plan_loss_weight=0.1),
        tokenizer=WorldModelTokenizer.Config(
            compressor_model=COMPRESSOR_MODEL,
            compressor_in_channels="auto",
        ),
        model_spec=model_registry("base"),
        dataloader=_dataloader_config(split="train"),
        optimizer=optimizer,
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2 * validation_freq,
            total_steps=steps,
            decay_ratio=0.1,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            global_batch_size=local_batch_size * world_size * 2,  # 2 grad acc
            seq_len=1,
            steps=steps,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        activation_checkpoint=FullAC.Config(),
        compile=compile_config,
        metrics=MetricsProcessor.Config(
            log_freq=16,
            enable_reporterv2=True,
            save_freq=validation_freq,
        ),
        checkpoint=WorldModelTorchPackageCheckpointManager.Config(
            enable=True,
            folder=checkpoint_folder,
            interval=validation_freq * 5,
            async_mode="async",
            keep_latest_k=0,
            enable_first_step_checkpoint=True,
            last_save_model_only=False,
            checkpoint_id_format="",
            exclude_from_loading=[
                "optimizer",
                "lr_scheduler",
                "dataloader",
                "train_state",
            ],
            export_torch_package=True,
        ),
        validator=WorldModelValidator.Config(
            enable=True,
            freq=validation_freq,
            steps=validation_steps,
            dataloader=_dataloader_config(split="val", fill_once=True),
            pose_dropout=0.0,
            noise_scheduler_steps=10,
            no_noise_prefill_frames_prob=0.0,
            fake_timesteps_prob=0.0,
        ),
        pose_dropout=0.1,
        noise_scheduler_steps=10,
        no_noise_prefill_frames_prob=0.5,
        fake_timesteps_prob=0.5,
        debug=DebugConfig(seed=0),
    )


def _world_sizes() -> tuple[int, int, int]:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(
        os.environ.get(
            "GROUP_WORLD_SIZE", str(max(1, world_size // max(1, local_world_size)))
        )
    )
    return local_world_size, world_size, num_nodes


def _reporterv2_checkpoint_folder() -> str:
    host = os.getenv("REPORTERV2_HOST")
    folder = os.getenv("REPORTERV2_TRAINING_ID") or "checkpoint"
    return f"{host.rstrip('/')}/checkpoint/{folder}" if host else folder


def main() -> None:
    spec = model_registry("debugmodel")
    model = spec.model.build()
    config = model.config
    nparams = sum(param.numel() for param in model.parameters())
    head_dim = config.transformer.n_embd // config.transformer.n_head
    print(
        {
            "flavor": spec.flavor,
            "input_size": config.input_size,
            "num_patches": config.num_patches,
            "hidden": config.transformer.n_embd,
            "heads": config.transformer.n_head,
            "head_dim": head_dim,
            "layers": config.transformer.n_layer,
            "parameters": nparams,
        }
    )


if __name__ == "__main__":
    main()
