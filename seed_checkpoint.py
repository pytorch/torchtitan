# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os

import torch.distributed.checkpoint as DCP

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import create_tokenizer
from torchtitan.float8_linear import build_fp8_linear
from torchtitan.logging_utils import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

_is_local_logging = True
if "SLURM_JOB_ID" in os.environ:
    _is_local_logging = False


def main(job_config: JobConfig):
    init_logger()

    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = create_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    model_config.vocab_size = tokenizer.n_words
    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    model = model_cls.from_model_args(model_config)

    # apply fp8 linear module swap
    if job_config.training.fp8_linear:
        build_fp8_linear(model, job_config)

    model.init_weights()

    checkpoint_id = os.path.join(job_config.checkpoint.folder, "step-0")
    logger.info(f"Creating seed (step-0) checkpoint in {checkpoint_id}")
    DCP.save(
        state_dict={
            "model": model.state_dict(),
        },
        checkpoint_id=checkpoint_id,
    )


"""
1. how do i serialize enough info about the model config to ensure i don't try to load an incompatible checkpoint later?
 - maybe skip this. users responsible to manage their checkpoints, and we can partially help by managing their 'dump folder'?

2. would i apply fp8 before creating the seed or not?  I think probably before
3. can i skip optimizer in seed file? i think so. optimizer can later create its states from the model post-sharding
"""
if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
