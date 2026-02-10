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
    Quantize,
    Training,
    Validation,
)
from torchtitan.config.job_config import (
    Float8GroupedMM,
    Float8Linear,
    MXGroupedMM,
    MXLinear,
    QuantizedGroupedMM,
    QuantizedLinear,
)

default_config = JobConfig(
    job=Job(description="Llama 4 debug training"),
    metrics=Metrics(log_freq=1),
    model=Model(
        name="llama4",
        flavor="debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
    ),
    optimizer=Optimizer(lr=4e-3, eps=1e-15),
    lr_scheduler=LRScheduler(
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.1,
    ),
    training=Training(
        local_batch_size=8,
        seq_len=2048,
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
    ),
    activation_checkpoint=ActivationCheckpoint(
        mode="selective",
        selective_ac_option="op",
    ),
    quantize=Quantize(
        linear=QuantizedLinear(
            float8=Float8Linear(filter_fqns=["output", "router.gate"]),
            mx=MXLinear(filter_fqns=["output", "router.gate"]),
        ),
        grouped_mm=QuantizedGroupedMM(
            float8=Float8GroupedMM(fqns=["experts"]),
            mx=MXGroupedMM(fqns=["experts"]),
        ),
    ),
    validation=Validation(
        dataset="c4_validation",
        freq=5,
        steps=10,
    ),
)
