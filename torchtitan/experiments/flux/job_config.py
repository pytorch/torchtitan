# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Training:
    classifer_free_guidance_prob: float = 0.0
    """Classifier-free guidance with probability p to dropout the text conditioning"""
    img_size: int = 256
    """Image width to sample"""


@dataclass
class Encoder:
    t5_encoder: str = "google/t5-v1_1-small"
    """T5 encoder to use, HuggingFace model name. This field could be either a local folder path,
        or a Huggingface repo name."""
    clip_encoder: str = "openai/clip-vit-large-patch14"
    """Clip encoder to use, HuggingFace model name. This field could be either a local folder path,
        or a Huggingface repo name."""
    autoencoder_path: str = (
        "torchtitan/experiments/flux/assets/autoencoder/ae.safetensors"
    )
    """Autoencoder checkpoint path to load. This should be a local path referring to a safetensors file."""
    max_t5_encoding_len: int = 512
    """Maximum length of the T5 encoding."""


@dataclass
class Eval:
    enable_classifer_free_guidance: bool = False
    """Whether to use classifier-free guidance during sampling"""
    classifier_free_guidance_scale: float = 5.0
    """Classifier-free guidance scale when sampling"""
    denoising_steps: int = 50
    """How many denoising steps to sample when generating an image"""
    eval_freq: int = 100
    """Frequency of evaluation/sampling during training"""
    save_img_folder: str = "img"
    """Directory to save image generated/sampled from the model"""


@dataclass
class JobConfig:
    """
    Extend the tyro parser with custom config classe for Flux model.
    """

    training: Training = field(default_factory=Training)
    encoder: Encoder = field(default_factory=Encoder)
    eval: Eval = field(default_factory=Eval)
