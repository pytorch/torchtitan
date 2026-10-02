# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flux model configurations used by tests."""

from dataclasses import replace

from torchtitan.components.checkpointer import (
    CheckpointManager,
    DATALOADER,
    LR_SCHEDULER,
    OPTIMIZER,
    TRAIN_STATE,
)
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import MSELoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC

from torchtitan.models.flux import build_model_config
from torchtitan.models.flux.configs import FluxEncoderConfig, Inference, SamplingConfig
from torchtitan.models.flux.flux_datasets import (
    DATASETS,
    FluxCollator,
    FluxSampleProcessor,
    FluxValidationDatasetConfig,
)
from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.models.flux.utils import (
    IMAGE_LATENT_SIZE_RATIO,
    PATCH_HEIGHT,
    PATCH_WIDTH,
)
from torchtitan.models.flux.validate import FluxValidator
from torchtitan.observability.metrics import MetricsProcessor

# NOTE: Flux needs `img_size` in both `dataset.processor` and to define the `seq_len` of the model
# There two utils are created to take the img_size defined once in the config.
def _flux_dataset(dataset_name: str, *, img_size: int) -> SingleDatasetConfig:
    dataset = DATASETS[dataset_name]
    processor = dataset.processor
    if not isinstance(processor, FluxSampleProcessor.Config):
        raise ValueError(
            f"Flux dataset {dataset_name!r} must use FluxSampleProcessor.Config"
        )
    return replace(dataset, processor=replace(processor, img_size=img_size))


def _flux_seq_len(img_size: int, max_t5_encoding_len: int) -> int:
    latent_width = img_size // IMAGE_LATENT_SIZE_RATIO // PATCH_WIDTH
    latent_height = img_size // IMAGE_LATENT_SIZE_RATIO // PATCH_HEIGHT
    return latent_width * latent_height + max_t5_encoding_len


def flux_debugmodel() -> FluxTrainer.Config:
    hf_assets_path = "tests/assets/tokenizer"
    img_size = 256
    max_t5_encoding_len = 256
    training_dataset = _flux_dataset("cc12m-test", img_size=img_size)
    return FluxTrainer.Config(
        hf_assets_path=hf_assets_path,
        loss=MSELoss.Config(),
        tokenizer=FluxTokenizerContainer.Config(
            t5_tokenizer_path="google/t5-v1_1-xxl",
            clip_tokenizer_path="openai/clip-vit-large-patch14",
            max_t5_encoding_len=max_t5_encoding_len,
        ),
        encoder=FluxEncoderConfig(
            autoencoder_path="assets/hf/FLUX.1-dev/ae.safetensors",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=build_model_config("flux-debug"),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=1,
                decay_ratio=0.0,
            ),
            max_norm=2.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2048,
            max_context_length=_flux_seq_len(img_size, max_t5_encoding_len),
            steps=10,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=training_dataset,
            collator=FluxCollator.Config(),
            streaming_shuffle_buffer_size=128,
        ),
        parallelism=ParallelismConfig(
            context_parallel_degree=1,
        ),
        activation_checkpoint=FullAC.Config(),
        checkpointer=None,
        validator=None,
        inference=Inference(
            save_img_folder="inference_results",
            prompts_path="./torchtitan/models/flux/inference/prompts.txt",
            local_batch_size=2,
        ),
    )


def flux_debugmodel_inference() -> FluxTrainer.Config:
    config = flux_debugmodel()
    config.checkpointer = CheckpointManager.Config(
        exclude_from_loading=[DATALOADER, LR_SCHEDULER, OPTIMIZER, TRAIN_STATE],
    )
    return config


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
    return _use_offline_test_assets(flux_debugmodel())


def flux_debugmodel_inference_test() -> FluxTrainer.Config:
    """Flux inference config pointed at the offline test assets."""
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
    config.training.disable_cuda_graphs = True
    return config
