# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    job=Job(description="Qwen 3 1.7B training"),
    model=Model(
        name="qwen3",
        flavor="1.7B",
        hf_assets_path="./assets/hf/Qwen3-1.7B",
    ),
    optimizer=Optimizer(lr=8e-4),
    lr_scheduler=LRScheduler(warmup_steps=20),
    training=Training(
        local_batch_size=4,
        seq_len=4096,
        steps=100,
        dataset="c4",
    ),
    checkpoint=Checkpoint(
        interval=50,
        last_save_model_only=False,
        export_dtype="float16",
    ),
    activation_checkpoint=ActivationCheckpoint(
        mode="selective",
        selective_ac_option="op",
    ),
)
