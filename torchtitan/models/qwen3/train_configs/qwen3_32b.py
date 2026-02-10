# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 8 H100 GPUs (with 96GiB memory).

from torchtitan.config import (
    ActivationCheckpoint,
    Checkpoint,
    Job,
    JobConfig,
    LRScheduler,
    Metrics,
    Model,
    Optimizer,
    Training,
)

default_config = JobConfig(
    job=Job(description="Qwen3 32B training"),
    model=Model(
        name="qwen3",
        flavor="32B",
        hf_assets_path="./assets/hf/Qwen3-32B",
    ),
    optimizer=Optimizer(lr=8e-4),
    lr_scheduler=LRScheduler(warmup_steps=600),
    training=Training(
        local_batch_size=2,
        seq_len=4096,
        steps=3000,
        dataset="c4",
    ),
    checkpoint=Checkpoint(
        interval=500,
        last_save_model_only=False,
        export_dtype="float16",
    ),
    activation_checkpoint=ActivationCheckpoint(
        mode="full",
        selective_ac_option="op",
    ),
)
