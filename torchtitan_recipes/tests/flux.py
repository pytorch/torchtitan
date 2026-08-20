# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``flux`` integration test suite."""

from torchtitan.trainer import Trainer


def flux_debugmodel_test() -> Trainer.Config:
    """Flux debug model pointed at the offline test encoders and tokenizers."""
    from torchtitan.models.flux.config_registry import flux_debugmodel

    config = flux_debugmodel()
    config.hf_assets_path = "tests/assets/tokenizer"
    config.tokenizer.test_mode = True
    config.tokenizer.t5_tokenizer_path = "tests/assets/tokenizer"
    config.tokenizer.clip_tokenizer_path = "tests/assets/tokenizer"
    config.encoder.random_init = True
    config.encoder.clip_encoder = (
        "tests/assets/flux_test_encoders/clip-vit-large-patch14/"
    )
    config.encoder.t5_encoder = "tests/assets/flux_test_encoders/t5-v1_1-xxl/"
    return config


def flux_debugmodel_hsdp2x2_cp2_validation() -> Trainer.Config:
    config = flux_debugmodel_test()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.validator.enable = True
    config.validator.steps = 5
    config.checkpoint.enable = True
    config.training.disable_cuda_graphs = True
    return config


def flux_debugmodel_compile() -> Trainer.Config:
    config = flux_debugmodel_test()
    config.compile.enable = True
    config.training.disable_cuda_graphs = True
    return config
