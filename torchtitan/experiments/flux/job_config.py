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
    test_mode: bool = False
    """Whether to use intergration test mode, which will randomly initialize the encoder and use a dummy tokenizer"""

    steps: int = 0
    """Number of training steps"""


@dataclass
class Encoder:
    t5_encoder: str | None = "google/t5-v1_1-small"
    """T5 encoder to use, HuggingFace model name. This field could be either a local folder path, \
    or a Huggingface repo name. If none, it is not loaded (assumes preprocessed data)"""
    clip_encoder: str | None = "openai/clip-vit-large-patch14"
    """Clip encoder to use, HuggingFace model name. This field could be either a local folder path, \
    or a Huggingface repo name. If none, it is not loaded (assumes preprocessed data)"""

    autoencoder_path: str | None = "/models/autoencoder/ae.safetensors"
    """Autoencoder checkpoint path to load. This should be a local path referring to a safetensors file. 
    If none, it is not loaded (assumes preprocessed data), but shift and scale must be provided"""

    autoencoder_shift: float | None = None
    """Shift of the autoencoder. If None, will read from autoencoder_path"""

    autoencoder_scale: float | None = None
    """Scale of the autoencoder. If None, will read from autoencoder_path"""

    max_t5_encoding_len: int = 256
    """Maximum length of the T5 encoding."""

    context_in_dim: int = 1024
    """Context input dimension."""

    empty_encodings_path: str | None = None
    """Path to the empty encodings. If None, will generate empty encodings during training."""


@dataclass
class Eval:
    enable_classifer_free_guidance: bool = False
    """Enable classifier-free guidance during evaluation"""

    classifier_free_guidance_scale: float = 5.0
    """Classifier-free guidance scale when sampling"""

    denoising_steps: int = 50
    """How many denoising steps to sample when generating an image"""

    eval_freq: int = 614400
    """Frequency of evaluation/sampling during training in samples"""

    save_img_folder: str = "eval_images"
    """Directory to save image generated/sampled from the model"""

    inception_ckpt: str = (
        "/checkpoint/inception_ckpt/pt_inception-2015-12-05-6726825d.pth"
    )
    """Inception checkpoint path to load. This should be a local path referring to a pth file."""

    clip_ckpt: str = "/checkpoint/clip_ckpt"
    """Clip checkpoint path to load. This should be a local path referring to a pt file."""

    coco_stats: str = "/checkpoint/coco_stats/val2014_512x512_30k_stats.npz"
    """Coco stats path to load. This should be a local path referring to a npz file."""

    dataset: str | None = None
    """Dataset to use for validation."""

    dataset_path: str | None = None
    """
    Path to the dataset in the file system.
    """
    batch_size: int = 16
    """Batch size for validation."""
    prompts_csv: str = "prompts.tsv"
    """Path to the prompts csv file."""
    num_prompts: int | None = None
    """Number of prompts to use for validation."""
    target_eval_loss: float = 0.596
    """Target loss for validation. If the validation loss is less than this value, the training will be considered successful."""


@dataclass
class Inference:
    """Inference configuration"""

    save_path: str = "inference_results"
    """Path to save the inference results"""
    prompts_path: str = "prompts.txt"
    """Path to file with newline separated prompts to generate images for"""
    batch_size: int = 16
    """Batch size for inference"""


@dataclass
class JobConfig:
    """
    Extend the tyro parser with custom config classe for Flux model.
    """

    training: Training = field(default_factory=Training)
    encoder: Encoder = field(default_factory=Encoder)
    eval: Eval = field(default_factory=Eval)
    inference: Inference = field(default_factory=Inference)
