# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import Tensor

from torchtitan.experiments.flux.model.autoencoder import AutoEncoder
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder


def preprocess_data(
    # arguments from the recipe
    device: torch.device,
    dtype: torch.dtype,
    *,
    # arguments from the config
    autoencoder: Optional[AutoEncoder],
    clip_encoder: FluxEmbedder,
    t5_encoder: FluxEmbedder,
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    """
    Take a batch of inputs and encoder as input and return a batch of preprocessed data.

    Args:
        device (torch.device): device to do preprocessing on
        dtype (torch.dtype): data type to do preprocessing in
        autoencoer(AutoEncoder): autoencoder to use for preprocessing
        clip_encoder (HFEmbedder): CLIPTextModel to use for preprocessing
        t5_encoder (HFEmbedder): T5EncoderModel to use for preprocessing
        batch (dict[str, Tensor]): batch of data to preprocess. Tensor shape: [bsz, ...]

    Returns:
        dict[str, Tensor]: batch of preprocessed data
    """

    clip_tokens = batch["clip_tokens"].squeeze(1).to(device=device, dtype=torch.int)
    t5_tokens = batch["t5_tokens"].squeeze(1).to(device=device, dtype=torch.int)

    clip_text_encodings = clip_encoder(clip_tokens)
    t5_text_encodings = t5_encoder(t5_tokens)

    if autoencoder is not None:
        images = batch["image"].to(device=device, dtype=dtype)
        img_encodings = autoencoder.encode(images)
        batch["img_encodings"] = img_encodings.to(device=device, dtype=dtype)

    batch["clip_encodings"] = clip_text_encodings.to(dtype)
    batch["t5_encodings"] = t5_text_encodings.to(dtype)

    return batch


def generate_noise_latent(
    bsz: int,
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> Tensor:
    """Generate noise latents for the Flux flow model. The random seed will be set at the begining of training.

    Args:
        bsz (int): batch_size.
        height (int): The height of the image.
        width (int): The width of the image.
        device (str | torch.device): The device to use.
        dtype (torch.dtype): The dtype to use.

    Returns:
        Tensor: The noise latents.
            Shape: [num_samples, LATENT_CHANNELS, height // IMG_LATENT_SIZE_RATIO, width // IMG_LATENT_SIZE_RATIO]

    """
    LATENT_CHANNELS, IMAGE_LATENT_SIZE_RATIO = 16, 8
    return torch.randn(
        bsz,
        LATENT_CHANNELS,
        height // IMAGE_LATENT_SIZE_RATIO,
        width // IMAGE_LATENT_SIZE_RATIO,
        dtype=dtype,
    ).to(device)


def create_position_encoding_for_latents(
    bsz: int, latent_height: int, latent_width: int, position_dim: int = 3
) -> Tensor:
    """
    Create the packed latents' position encodings for the Flux flow model.

    Args:
        bsz (int): The batch size.
        latent_height (int): The height of the latent.
        latent_width (int): The width of the latent.

    Returns:
        Tensor: The position encodings.
            Shape: [bsz, (latent_height // PATCH_HEIGHT) * (latent_width // PATCH_WIDTH), POSITION_DIM)
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2

    height = latent_height // PATCH_HEIGHT
    width = latent_width // PATCH_WIDTH

    position_encoding = torch.zeros(height, width, position_dim)

    row_indices = torch.arange(height)
    position_encoding[:, :, 1] = row_indices.unsqueeze(1)

    col_indices = torch.arange(width)
    position_encoding[:, :, 2] = col_indices.unsqueeze(0)

    # Flatten and repeat for the full batch
    # [height, width, 3] -> [bsz, height * width, 3]
    position_encoding = position_encoding.view(1, height * width, position_dim)
    position_encoding = position_encoding.repeat(bsz, 1, 1)

    return position_encoding


def pack_latents(x: Tensor) -> Tensor:
    """
    Rearrange latents from an image-like format into a sequence of patches.
    Equivalent to `einops.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`.

    Args:
        x (Tensor): The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]

    Returns:
        Tensor: The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2

    b, c, latent_height, latent_width = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH

    # [b, c, h*ph, w*ph] -> [b, c, h, w, ph, pw]
    x = x.unfold(2, PATCH_HEIGHT, PATCH_HEIGHT).unfold(3, PATCH_WIDTH, PATCH_WIDTH)

    # [b, c, h, w, ph, PW] -> [b, h, w, c, ph, PW]
    x = x.permute(0, 2, 3, 1, 4, 5)

    # [b, h, w, c, ph, PW] -> [b, h*w, c*ph*PW]
    return x.reshape(b, h * w, c * PATCH_HEIGHT * PATCH_WIDTH)


def unpack_latents(x: Tensor, latent_height: int, latent_width: int) -> Tensor:
    """
    Rearrange latents from a sequence of patches into an image-like format.
    Equivalent to `einops.rearrange("b (h w) (c ph pw) -> b c (h ph) (w pw)")`.

    Args:
        x (Tensor): The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
        latent_height (int): The height of the unpacked latents.
        latent_width (int): The width of the unpacked latents.

    Returns:
        Tensor: The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2

    b, _, c_ph_pw = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH
    c = c_ph_pw // (PATCH_HEIGHT * PATCH_WIDTH)

    # [b, h*w, c*ph*pw] -> [b, h, w, c, ph, pw]
    x = x.reshape(b, h, w, c, PATCH_HEIGHT, PATCH_WIDTH)

    # [b, h, w, c, ph, pw] -> [b, c, h, ph, w, pw]
    x = x.permute(0, 3, 1, 4, 2, 5)

    # [b, c, h, ph, w, pw] -> [b, c, h*ph, w*pw]
    return x.reshape(b, c, h * PATCH_HEIGHT, w * PATCH_WIDTH)
