# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    ActivationCheckpoint,
    Checkpoint,
    JobConfig,
    LRScheduler,
    Metrics,
    Model,
    Optimizer,
    Parallelism,
    Training,
    Validation,
)

default_config = JobConfig(
    model=Model(
        name="llama3",
        flavor="debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
    ),
    optimizer=Optimizer(lr=8e-4),
    lr_scheduler=LRScheduler(
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
    ),
    training=Training(
        local_batch_size=8,
        seq_len=2048,
        steps=10,
        dataset="c4_test",
    ),
    metrics=Metrics(log_freq=1),
    parallelism=Parallelism(pipeline_parallel_schedule="Interleaved1F1B"),
    checkpoint=Checkpoint(
        interval=10,
        last_save_model_only=False,
    ),
    activation_checkpoint=ActivationCheckpoint(
        mode="selective",
        selective_ac_option="2",
    ),
    validation=Validation(
        dataset="c4_validation",
        freq=5,
        steps=10,
    ),
)
