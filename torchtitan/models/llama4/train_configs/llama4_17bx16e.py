# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 64 H100 GPUs.

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
    Quantize,
    Training,
)
from torchtitan.config.job_config import (
    Float8Linear,
    MXLinear,
    QuantizedLinear,
)

default_config = JobConfig(
    job=Job(description="Llama 4 Scout 17Bx16E training"),
    model=Model(
        name="llama4",
        flavor="17bx16e",
        hf_assets_path="./assets/hf/Llama-4-Scout-17B-16E",
    ),
    optimizer=Optimizer(lr=4e-3, eps=1e-15),
    lr_scheduler=LRScheduler(
        warmup_steps=600,
        min_lr_factor=0.1,
    ),
    training=Training(
        local_batch_size=8,
        seq_len=8192,
        steps=3000,
        dataset="c4",
    ),
    parallelism=Parallelism(
        tensor_parallel_degree=8,
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=8,
    ),
    checkpoint=Checkpoint(interval=500),
    activation_checkpoint=ActivationCheckpoint(mode="full"),
    quantize=Quantize(
        linear=QuantizedLinear(
            float8=Float8Linear(filter_fqns=["output", "router.gate"]),
            mx=MXLinear(filter_fqns=["output", "router.gate"]),
        ),
    ),
)
