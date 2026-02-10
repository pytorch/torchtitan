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
    Parallelism,
    Training,
)

default_config = JobConfig(
    job=Job(description="Qwen 3 MoE debug model training"),
    metrics=Metrics(log_freq=1),
    model=Model(
        name="qwen3",
        flavor="debugmodel_moe",
        hf_assets_path="./tests/assets/tokenizer",
    ),
    optimizer=Optimizer(lr=3e-4),
    lr_scheduler=LRScheduler(warmup_steps=2),
    training=Training(
        local_batch_size=4,
        seq_len=4096,
        steps=10,
        dataset="c4_test",
    ),
    parallelism=Parallelism(
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=1,
    ),
    checkpoint=Checkpoint(
        interval=10,
        last_save_model_only=False,
        export_dtype="float16",
    ),
    activation_checkpoint=ActivationCheckpoint(
        mode="selective",
        selective_ac_option="op",
    ),
)
