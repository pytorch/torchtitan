# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``flux`` integration test suite."""

from dataclasses import replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader
from torchtitan.config import CompileConfig
from torchtitan.models.flux.configs import SamplingConfig
from torchtitan.models.flux.flux_datasets import (
    DATASETS,
    FluxCollator,
    FluxSampleProcessor,
    FluxValidationDatasetConfig,
)
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.models.flux.validate import FluxValidator


def _use_offline_test_assets(config: FluxTrainer.Config) -> FluxTrainer.Config:
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


def flux_debugmodel_test() -> FluxTrainer.Config:
    """Flux debug model pointed at the offline test encoders and tokenizers."""
    from torchtitan.models.flux.config_registry import flux_debugmodel

    return _use_offline_test_assets(flux_debugmodel())


def flux_debugmodel_inference_test() -> FluxTrainer.Config:
    """Flux inference config pointed at the offline test assets."""
    from torchtitan.models.flux.config_registry import flux_debugmodel_inference

    return _use_offline_test_assets(flux_debugmodel_inference())


def flux_debugmodel_hsdp2x2_cp2_validation() -> FluxTrainer.Config:
    config = flux_debugmodel_test()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    validation_dataset = DATASETS["cc12m-test-validation"]
    validation_processor = validation_dataset.processor
    assert isinstance(validation_processor, FluxSampleProcessor.Config)
    validation_dataset = replace(
        validation_dataset,
        processor=replace(validation_processor, img_size=256),
    )
    config.validator = FluxValidator.Config(
        freq=5,
        steps=5,
        sampling=SamplingConfig(
            enable_classifier_free_guidance=True,
            classifier_free_guidance_scale=5.0,
            denoising_steps=4,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FluxValidationDatasetConfig(dataset=validation_dataset),
            collator=FluxCollator.Config(),
            streaming_shuffle_buffer_size=128,
        ),
        save_img_count=1,
        save_img_folder="img",
    )
    config.checkpointer = CheckpointManager.Config()
    config.training.disable_cuda_graphs = True
    return config


def flux_debugmodel_compile() -> FluxTrainer.Config:
    config = flux_debugmodel_test()
    config.compile = CompileConfig()
    config.training.disable_cuda_graphs = True
    return config
