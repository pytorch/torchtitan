# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from typing import Callable

import torch
from einops import rearrange
from PIL import ExifTags, Image
from torch import Tensor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.flux.model.autoencoder import AutoEncoder
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.model.model import FluxModel
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    generate_noise_latent,
    pack_latents,
    preprocess_data,
    unpack_latents,
)
from torchtitan.tools.logging import logger

from ..trainer import FluxTrainer


# ----------------------------------------
#       Util functions for Sampling
# ----------------------------------------


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


# ----------------------------------------
#       Sampling functions
# ----------------------------------------


def generate_image(
    device: torch.device,
    dtype: torch.dtype,
    job_config: FluxTrainer.Config,
    model: FluxModel,
    prompt: str | list[str],
    autoencoder: AutoEncoder,
    t5_tokenizer: BaseTokenizer,
    clip_tokenizer: BaseTokenizer,
    t5_encoder: FluxEmbedder,
    clip_encoder: FluxEmbedder,
) -> torch.Tensor:
    """
    Sampling and save a single images from noise using a given prompt.
    For randomized noise generation, the random seend should already be set at the beginning of training.
    Since we will always use the local random seed on this rank, we don't need to pass in the seed again.
    """

    if isinstance(prompt, str):
        prompt = [prompt]

    # allow for packing and conversion to latent space. Use the same resolution as training time.
    # pyrefly: ignore [missing-attribute]
    img_height = 16 * (job_config.validator.dataloader.img_size // 16)
    # pyrefly: ignore [missing-attribute]
    img_width = 16 * (job_config.validator.dataloader.img_size // 16)

    enable_classifier_free_guidance = (
        # pyrefly: ignore [missing-attribute]
        job_config.validation.enable_classifier_free_guidance
    )

    # Tokenize the prompt. Unsqueeze to add a batch dimension.
    clip_tokens = clip_tokenizer.encode(prompt)
    t5_tokens = t5_tokenizer.encode(prompt)
    if len(prompt) == 1:
        # pyrefly: ignore [missing-attribute]
        clip_tokens = clip_tokens.unsqueeze(0)
        # pyrefly: ignore [missing-attribute]
        t5_tokens = t5_tokens.unsqueeze(0)

    batch = preprocess_data(
        device=device,
        dtype=dtype,
        autoencoder=None,
        clip_encoder=clip_encoder,
        t5_encoder=t5_encoder,
        # pyrefly: ignore [bad-argument-type]
        batch={
            "clip_tokens": clip_tokens,
            "t5_tokens": t5_tokens,
        },
    )

    if enable_classifier_free_guidance:
        num_images = len(prompt)

        empty_clip_tokens = clip_tokenizer.encode("")
        empty_t5_tokens = t5_tokenizer.encode("")
        # pyrefly: ignore [missing-attribute]
        empty_clip_tokens = empty_clip_tokens.repeat(num_images, 1)
        # pyrefly: ignore [missing-attribute]
        empty_t5_tokens = empty_t5_tokens.repeat(num_images, 1)

        empty_batch = preprocess_data(
            device=device,
            dtype=dtype,
            autoencoder=None,
            clip_encoder=clip_encoder,
            t5_encoder=t5_encoder,
            batch={
                "clip_tokens": empty_clip_tokens,
                "t5_tokens": empty_t5_tokens,
            },
        )

    img = denoise(
        device=device,
        dtype=dtype,
        model=model,
        img_width=img_width,
        img_height=img_height,
        # pyrefly: ignore [missing-attribute]
        denoising_steps=job_config.validation.denoising_steps,
        clip_encodings=batch["clip_encodings"],
        t5_encodings=batch["t5_encodings"],
        enable_classifier_free_guidance=enable_classifier_free_guidance,
        empty_t5_encodings=(
            # pyrefly: ignore [unbound-name]
            empty_batch["t5_encodings"]
            if enable_classifier_free_guidance
            else None
        ),
        empty_clip_encodings=(
            # pyrefly: ignore [unbound-name]
            empty_batch["clip_encodings"]
            if enable_classifier_free_guidance
            else None
        ),
        # pyrefly: ignore [missing-attribute]
        classifier_free_guidance_scale=job_config.validation.classifier_free_guidance_scale,
    )

    img = autoencoder.decode(img)
    return img


def denoise(
    device: torch.device,
    dtype: torch.dtype,
    model: FluxModel,
    img_width: int,
    img_height: int,
    denoising_steps: int,
    clip_encodings: torch.Tensor,
    t5_encodings: torch.Tensor,
    enable_classifier_free_guidance: bool = False,
    empty_t5_encodings: torch.Tensor | None = None,
    empty_clip_encodings: torch.Tensor | None = None,
    classifier_free_guidance_scale: float | None = None,
) -> torch.Tensor:
    """
    Sampling images from noise using a given prompt, by running inference with trained Flux model.
    Save the generated images to the given output path.
    """
    bsz = clip_encodings.shape[0]
    latents = generate_noise_latent(bsz, img_height, img_width, device, dtype)
    _, latent_channels, latent_height, latent_width = latents.shape

    # create denoising schedule
    timesteps = get_schedule(denoising_steps, latent_height * latent_width, shift=True)

    if enable_classifier_free_guidance:
        # Double batch size for CFG: [unconditional, conditional]
        latents = torch.cat([latents, latents], dim=0)
        # pyrefly: ignore [no-matching-overload]
        t5_encodings = torch.cat([empty_t5_encodings, t5_encodings], dim=0)
        # pyrefly: ignore [no-matching-overload]
        clip_encodings = torch.cat([empty_clip_encodings, clip_encodings], dim=0)
        bsz *= 2

    # create positional encodings
    POSITION_DIM = 3
    latent_pos_enc = create_position_encoding_for_latents(
        bsz, latent_height, latent_width, POSITION_DIM
    ).to(latents)
    text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM).to(latents)

    # convert img-like latents into sequences of patches
    latents = pack_latents(latents)

    # this is ignored for schnell
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((bsz,), t_curr, dtype=dtype, device=device)
        pred = model(
            img=latents,
            img_ids=latent_pos_enc,
            txt=t5_encodings,
            txt_ids=text_pos_enc,
            y=clip_encodings,
            timesteps=t_vec,
        )
        if enable_classifier_free_guidance:
            pred_u, pred_c = pred.chunk(2)
            pred = pred_u + classifier_free_guidance_scale * (pred_c - pred_u)

            # repeat along batch dimension to update both unconditional and conditional latents
            pred = pred.repeat(2, 1, 1)

        latents = latents + (t_prev - t_curr) * pred

    # take the conditional latents for the final result
    if enable_classifier_free_guidance:
        latents = latents.chunk(2)[1]

    # convert sequences of patches into img-like latents
    latents = unpack_latents(latents, latent_height, latent_width)

    return latents


def save_image(
    name: str,
    output_dir: str,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
):
    logger.info(f"Saving image to {output_dir}/{name}")
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, name)

    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    exif_data[ExifTags.Base.Model] = name
    if add_sampling_metadata:
        exif_data[ExifTags.Base.ImageDescription] = prompt
    img.save(output_name, exif=exif_data, quality=95, subsampling=0)
