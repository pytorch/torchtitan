import math
from typing import Callable

import torch
from einops import rearrange
from PIL import ExifTags, Image

from torch import Tensor

from torchtitan.experiments.flux.model.autoencoder import AutoEncoder

from torchtitan.experiments.flux.model.model import FluxModel
from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    generate_noise_latent,
    pack_latents,
    unpack_latents,
)


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


def generate_images(
    device: torch.device,
    dtype: torch.dtype,
    model: FluxModel,
    decoder: AutoEncoder,
    # image params:
    img_width: int,
    img_height: int,
    # sampling params:
    denoising_steps: int,
    seed: int,
    clip_encodings: torch.Tensor,
    t5_encodings: torch.Tensor,
    enable_classifer_free_guidance: bool = False,
    empty_t5_encodings: torch.Tensor | None = None,
    empty_clip_encodings: torch.Tensor | None = None,
    classifier_free_guidance_scale: float | None = None,
):
    """
    Sampling images from noise using a given prompt, by running inference with trained Flux model.
    Save the generated images to the given output path.
    """
    bsz = clip_encodings.shape[0]
    latents = generate_noise_latent(bsz, img_height, img_width, device, dtype, seed)
    _, latent_channels, latent_height, latent_width = latents.shape

    # create denoising schedule
    timesteps = get_schedule(denoising_steps, latent_channels, shift=True)

    # create positional encodings
    POSITION_DIM = 3
    latent_pos_enc = create_position_encoding_for_latents(
        bsz, latent_height, latent_width, POSITION_DIM
    ).to(latents)
    text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM).to(latents)

    if enable_classifer_free_guidance:
        latents = torch.cat([latents, latents], dim=0)
        t5_encodings = torch.cat([empty_t5_encodings, t5_encodings], dim=0)
        clip_encodings = torch.cat([empty_clip_encodings, clip_encodings], dim=0)

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
        if enable_classifer_free_guidance:
            pred_u, pred_c = pred.chunk(2)
            pred = pred_u + classifier_free_guidance_scale * (pred_c - pred_u)

        latents = latents + (t_prev - t_curr) * pred

    # convert sequences of patches into img-like latents
    latents = unpack_latents(latents, latent_height, latent_width)

    img = decoder.decode(latents)
    return img


def save_image(
    name: str,
    output_name: str,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
):
    print(f"Saving {output_name}")
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
