# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this config is a preset for 128 H100 GPUs.

from torchtitan.config import (
    ActivationCheckpoint,
    Checkpoint,
    Compile,
    Job,
    JobConfig,
    LRScheduler,
    Metrics,
    Model,
    Optimizer,
    Parallelism,
    Profiling,
    Quantize,
    Training,
    Validation,
)
from torchtitan.config.job_config import Float8Linear, QuantizedLinear

default_config = JobConfig(
    job=Job(
        description="Llama 3 405B training",
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
        flavor="405B",
        hf_assets_path="./assets/hf/Llama-3.1-405B",
        converters=["float8"],
    ),
    optimizer=Optimizer(lr=8e-5),
    lr_scheduler=LRScheduler(warmup_steps=600),
    training=Training(
        local_batch_size=2,
        seq_len=8192,
        steps=3000,
        dataset="c4",
    ),
    parallelism=Parallelism(
        tensor_parallel_degree=8,
        enable_async_tensor_parallel=True,
    ),
    checkpoint=Checkpoint(interval=500),
    activation_checkpoint=ActivationCheckpoint(mode="full"),
    compile=Compile(enable=True),
    quantize=Quantize(
        linear=QuantizedLinear(
            float8=Float8Linear(
                enable_fsdp_float8_all_gather=True,
                precompute_float8_dynamic_scale_for_fsdp=True,
                filter_fqns=["output"],
            ),
        ),
    ),
    validation=Validation(
        dataset="c4_validation",
        freq=500,
        steps=1200,
    ),
)
