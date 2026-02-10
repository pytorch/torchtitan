# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    Quantize,
    Training,
)
from torchtitan.config.job_config import (
    Float8GroupedMM,
    Float8Linear,
    QuantizedGroupedMM,
    QuantizedLinear,
)

default_config = JobConfig(
    job=Job(description="DeepSeek-V3 16B model training"),
    model=Model(
        name="deepseek_v3",
        flavor="16B",
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
    ),
    optimizer=Optimizer(lr=2.2e-4),
    lr_scheduler=LRScheduler(
        decay_ratio=0.8,
        decay_type="cosine",
        min_lr_factor=0.1,
    ),
    training=Training(
        local_batch_size=4,
        seq_len=4096,
        steps=1000,
        dataset="c4",
    ),
    parallelism=Parallelism(
        pipeline_parallel_schedule="Interleaved1F1B",
        expert_parallel_degree=8,
        expert_tensor_parallel_degree=1,
    ),
    checkpoint=Checkpoint(interval=10),
    activation_checkpoint=ActivationCheckpoint(
        mode="selective",
        selective_ac_option="op",
    ),
    compile=Compile(enable=True, components=["loss"]),
    quantize=Quantize(
        linear=QuantizedLinear(
            float8=Float8Linear(filter_fqns=["output", "router.gate"]),
        ),
        grouped_mm=QuantizedGroupedMM(
            float8=Float8GroupedMM(fqns=["experts"]),
        ),
    ),
)
