# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class Encoder:
    model_name: str = "flux.2-dev"
    """FLUX.2 model name used to load text encoder and autoencoder weights."""

    max_length: int = 512
    """Maximum prompt length for dummy encoder (test mode)."""

    text_encoder_device: str = "cpu"
    """Device to place the text encoder on. Default is CPU to avoid GPU OOM."""

    text_encoder_cache_dir: str | None = None
    """Directory to store or read precomputed text encoder outputs."""

    text_encoder_cache_mode: str = "off"
    """Cache mode: 'off' or 'read'."""

    text_encoder_fsdp_offload: bool = False
    """If True, use FSDP CPU param offload during precompute to reduce GPU peak."""

    text_encoder_fsdp_sharded_load: bool = True
    """If True, use meta-init + sharded load for Mistral when using fully_shard."""

    guidance: float | None = None
    """Guidance value used when the model expects a guidance embedding.
    If None, a model-specific default is used when available."""

    test_mode: bool = False
    """Whether to use random-initialized encoders for test purposes."""


@dataclass(kw_only=True, slots=True)
class Validation:
    enable_classifier_free_guidance: bool = False
    """Whether to use classifier-free guidance during sampling."""

    classifier_free_guidance_scale: float = 5.0
    """Classifier-free guidance scale when sampling."""

    denoising_steps: int = 50
    """How many denoising steps to sample when generating an image."""

    eval_freq: int = 100
    """Frequency of evaluation/sampling during training."""


@dataclass(kw_only=True, slots=True)
class Inference:
    save_img_folder: str = "inference_results"
    """Path to save the inference results."""

    prompts_path: str = "./torchtitan/experiments/flux2/inference/prompts.txt"
    """Path to file with newline separated prompts to generate images for."""

    local_batch_size: int = 2
    """Batch size for inference."""

    img_size: int = 256
    """Image size for inference."""
