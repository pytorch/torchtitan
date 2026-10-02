# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified Flux model recipes."""

from dataclasses import replace

from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import MSELoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.transform import MXFP8LinearConverter
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.flux import build_model_config
from torchtitan.models.flux.configs import FluxEncoderConfig
from torchtitan.models.flux.flux_datasets import (
    DATASETS,
    FluxCollator,
    FluxSampleProcessor,
)
from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.models.flux.utils import (
    IMAGE_LATENT_SIZE_RATIO,
    PATCH_HEIGHT,
    PATCH_WIDTH,
)
from torchtitan.observability.metrics import MetricsProcessor


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


def _flux_recipe(flavor: str, *, max_t5_encoding_len: int) -> FluxTrainer.Config:
    img_size = 256
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
        model=build_model_config(flavor),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=3000,
                decay_ratio=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=(
                24576 if flavor == "flux-dev" else 32768
            ),
            max_context_length=_flux_seq_len(img_size, max_t5_encoding_len),
            steps=30000,
            disable_cuda_graphs=True,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=_flux_dataset("cc12m-wds", img_size=img_size),
            collator=FluxCollator.Config(),
            streaming_shuffle_buffer_size=128,
        ),
        activation_checkpoint=FullAC.Config(),
        checkpointer=None,
        validator=None,
    )


def flux_dev() -> FluxTrainer.Config:
    return _flux_recipe("flux-dev", max_t5_encoding_len=512)


def flux_schnell() -> FluxTrainer.Config:
    return _flux_recipe("flux-schnell", max_t5_encoding_len=256)


def _enable_mxfp8(config: FluxTrainer.Config, *, flavor: str) -> FluxTrainer.Config:
    config.compile = LocalCompileConfig()
    config.model = build_model_config(
        flavor,
        converters=[
            MXFP8LinearConverter.Config(
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


def flux_schnell_mxfp8() -> FluxTrainer.Config:
    """Flux schnell with MXFP8 quantization and torch.compile on SM100+."""
    return _enable_mxfp8(flux_schnell(), flavor="flux-schnell")


def flux_dev_mxfp8() -> FluxTrainer.Config:
    """Flux dev with MXFP8 quantization and torch.compile on SM100+."""
    return _enable_mxfp8(flux_dev(), flavor="flux-dev")
