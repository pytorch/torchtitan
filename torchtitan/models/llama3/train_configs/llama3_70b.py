# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 64 A100 GPUs.

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
    Profiling,
    Training,
    Validation,
)

default_config = JobConfig(
    job=Job(
        description="Llama 3 70B training",
    ),
    profiling=Profiling(
        enable_profiling=True,
        profile_freq=100,
    ),
    metrics=Metrics(
        enable_tensorboard=True,
    ),
    model=Model(
        name="llama3",
        flavor="70B",
        hf_assets_path="./assets/hf/Llama-3.1-70B",
    ),
    optimizer=Optimizer(lr=1.5e-4),
    training=Training(
        local_batch_size=8,
        seq_len=8192,
        steps=1000,
        dataset="c4",
    ),
    parallelism=Parallelism(
        tensor_parallel_degree=8,
    ),
    checkpoint=Checkpoint(interval=500),
    activation_checkpoint=ActivationCheckpoint(mode="full"),
    validation=Validation(
        dataset="c4_validation",
        freq=500,
        steps=1200,
    ),
)
