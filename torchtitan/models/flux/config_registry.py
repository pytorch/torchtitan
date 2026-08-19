# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import MSELoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.quantization import MXFP8LinearConverter
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
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

from . import model_registry

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
    validation_dataset = _flux_dataset("cc12m-test-validation", img_size=img_size)
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
        model_spec=model_registry("flux-debug"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=_flux_seq_len(img_size, max_t5_encoding_len),
            max_norm=2.0,
            steps=10,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=training_dataset,
            collator=FluxCollator.Config(),
            streaming_shuffle_buffer_size=128,
        ),
        parallelism=ParallelismConfig(context_parallel_degree=1),
        activation_checkpoint=FullAC.Config(),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        validator=FluxValidator.Config(
            freq=5,
            steps=48,
            sampling=SamplingConfig(
                enable_classifier_free_guidance=True,
                classifier_free_guidance_scale=5.0,
                denoising_steps=4,
            ),
            # Validate on the local cc12m-test asset (no HF download) so CI
            # does not flake on the network. Production flux_dev/flux_schnell
            # still validate on the real coco-validation set.
            dataloader=GrainDataLoader.Config(
                dataset=FluxValidationDatasetConfig(
                    dataset=validation_dataset,
                ),
                collator=FluxCollator.Config(),
                streaming_shuffle_buffer_size=128,
            ),
            save_img_count=1,
            save_img_folder="img",
            all_timesteps=False,
        ),
        inference=Inference(
            save_img_folder="inference_results",
            prompts_path="./torchtitan/models/flux/inference/prompts.txt",
            local_batch_size=2,
        ),
    )


def flux_dev() -> FluxTrainer.Config:
    img_size = 256
    max_t5_encoding_len = 512
    training_dataset = _flux_dataset("cc12m-wds", img_size=img_size)
    validation_dataset = _flux_dataset("coco-validation", img_size=img_size)
    return FluxTrainer.Config(
        loss=MSELoss.Config(),
        tokenizer=FluxTokenizerContainer.Config(
            t5_tokenizer_path="google/t5-v1_1-xxl",
            clip_tokenizer_path="openai/clip-vit-large-patch14",
            max_t5_encoding_len=max_t5_encoding_len,
        ),
        encoder=FluxEncoderConfig(
            autoencoder_path="assets/hf/FLUX.1-dev/ae.safetensors",
        ),
        metrics=MetricsProcessor.Config(log_freq=100),
        model_spec=model_registry("flux-dev"),
        optimizer=default_adamw(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=32,
            seq_len=_flux_seq_len(img_size, max_t5_encoding_len),
            steps=30000,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=training_dataset,
            collator=FluxCollator.Config(),
            streaming_shuffle_buffer_size=128,
        ),
        activation_checkpoint=FullAC.Config(),
        checkpoint=CheckpointManager.Config(interval=1000),
        validator=FluxValidator.Config(
            freq=1000,
            steps=12,
            sampling=SamplingConfig(
                enable_classifier_free_guidance=True,
                classifier_free_guidance_scale=5.0,
                denoising_steps=50,
            ),
            dataloader=GrainDataLoader.Config(
                dataset=FluxValidationDatasetConfig(
                    dataset=validation_dataset,
                ),
                collator=FluxCollator.Config(),
                streaming_shuffle_buffer_size=128,
            ),
            save_img_count=50,
            save_img_folder="img",
            all_timesteps=False,
        ),
    )


def flux_schnell() -> FluxTrainer.Config:
    img_size = 256
    max_t5_encoding_len = 256
    training_dataset = _flux_dataset("cc12m-wds", img_size=img_size)
    validation_dataset = _flux_dataset("coco-validation", img_size=img_size)
    return FluxTrainer.Config(
        loss=MSELoss.Config(),
        tokenizer=FluxTokenizerContainer.Config(
            t5_tokenizer_path="google/t5-v1_1-xxl",
            clip_tokenizer_path="openai/clip-vit-large-patch14",
            max_t5_encoding_len=max_t5_encoding_len,
        ),
        encoder=FluxEncoderConfig(
            autoencoder_path="assets/hf/FLUX.1-dev/ae.safetensors",
        ),
        metrics=MetricsProcessor.Config(log_freq=100),
        model_spec=model_registry("flux-schnell"),
        optimizer=default_adamw(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=3000,
            decay_ratio=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=64,
            seq_len=_flux_seq_len(img_size, max_t5_encoding_len),
            steps=30000,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=training_dataset,
            collator=FluxCollator.Config(),
            streaming_shuffle_buffer_size=128,
        ),
        activation_checkpoint=FullAC.Config(),
        checkpoint=CheckpointManager.Config(interval=1000),
        validator=FluxValidator.Config(
            freq=1000,
            steps=6,
            sampling=SamplingConfig(
                enable_classifier_free_guidance=True,
                classifier_free_guidance_scale=5.0,
                denoising_steps=50,
            ),
            dataloader=GrainDataLoader.Config(
                dataset=FluxValidationDatasetConfig(
                    dataset=validation_dataset,
                ),
                collator=FluxCollator.Config(),
                streaming_shuffle_buffer_size=128,
            ),
            save_img_count=50,
            save_img_folder="img",
            all_timesteps=False,
        ),
    )


def flux_schnell_mxfp8() -> FluxTrainer.Config:
    """Flux schnell with MXFP8 quantization and torch.compile.
    Requires SM100+ (B200/B100) and torchao nightly."""
    config = flux_schnell()
    config.compile = CompileConfig(enable=True)
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "flux-schnell",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=[
                    "double_blocks",
                    "single_blocks",
                    "img_in",
                    "txt_in",
                    "time_in",
                    "vector_in",
                    "final_layer",
                ],
            ),
        ],
    )
    return config


def flux_dev_mxfp8() -> FluxTrainer.Config:
    """Flux dev with MXFP8 quantization and torch.compile.
    Requires SM100+ (B200/B100) and torchao nightly."""
    config = flux_dev()
    config.compile = CompileConfig(enable=True)
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "flux-dev",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=[
                    "double_blocks",
                    "single_blocks",
                    "img_in",
                    "txt_in",
                    "time_in",
                    "vector_in",
                    "final_layer",
                ],
            ),
        ],
    )
    return config
