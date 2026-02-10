# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: this config is still under development

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
    job=Job(description="Llama 4 Maverick 17Bx128E training"),
    model=Model(
        name="llama4",
        flavor="17bx128e",
        hf_assets_path="./assets/hf/Llama-4-Maverick-17B-128E",
    ),
    optimizer=Optimizer(lr=4e-3, eps=1e-15),
    lr_scheduler=LRScheduler(
        warmup_steps=600,
        min_lr_factor=0.1,
    ),
    training=Training(
        local_batch_size=1,
        seq_len=8192,
        steps=3000,
        dataset="c4",
    ),
    parallelism=Parallelism(
        tensor_parallel_degree=8,
        pipeline_parallel_degree=4,
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
