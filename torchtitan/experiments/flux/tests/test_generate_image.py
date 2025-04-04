# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from typing import Callable

import torch
from einops import rearrange

from PIL import ExifTags, Image

from torch import Tensor

from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer

from torchtitan.experiments.flux.model.autoencoder import (
    AutoEncoder,
    AutoEncoderParams,
    load_ae,
)
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder

from torchtitan.experiments.flux.model.model import FluxModel, FluxModelArgs
from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    generate_noise_latent,
    pack_latents,
    preprocess_flux_data,
    unpack_latents,
)


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


class TestGenerateImage:
    def test_generate_image(self):
        """
        Run a forward pass of flux model to generate an image.
        """
        name = "flux-dev"
        img_width = 512
        img_height = 512
        seed = None
        prompt = (
            "a photo of a forest with mist swirling around the tree trunks. The word "
            '"FLUX" is painted over it in big, red brush strokes with visible texture'
        )
        device = "cuda"
        num_steps = None
        loop = False
        guidance = 3.5
        output_dir = "output"
        add_sampling_metadata = True

        prompt = prompt.split("|")
        if len(prompt) == 1:
            prompt = prompt[0]
            additional_prompts = None
        else:
            additional_prompts = prompt[1:]
            prompt = prompt[0]

        assert not (
            (additional_prompts is not None) and loop
        ), "Do not provide additional prompts and set loop to True"

        torch_device = torch.device(device)
        if num_steps is None:
            num_steps = 30

        # allow for packing and conversion to latent space
        img_height = 16 * (img_height // 16)
        img_width = 16 * (img_width // 16)

        # init all components
        model = FluxModel(FluxModelArgs()).to(device=torch_device, dtype=torch.bfloat16)

        ae = load_ae(
            ckpt_path="assets/autoencoder/ae.safetensors",
            autoencoder_params=AutoEncoderParams(),
            device=torch_device,
            dtype=torch.bfloat16,
        )
        clip_tokenizer = FluxTokenizer(
            model_path="openai/clip-vit-large-patch14", max_length=77
        )
        t5_tokenizer = FluxTokenizer(model_path="google/t5-v1_1-small", max_length=512)
        clip_encoder = FluxEmbedder(version="openai/clip-vit-large-patch14").to(
            torch_device, dtype=torch.bfloat16
        )
        t5_encoder = FluxEmbedder(version="google/t5-v1_1-small").to(
            torch_device, dtype=torch.bfloat16
        )

        rng = torch.Generator(device="cpu")

        if seed is None:
            seed = rng.seed()
        print(f"Generating with seed {seed}:\n{prompt}")
        t0 = time.perf_counter()
        output_name = os.path.join(output_dir, f"img_{seed}.jpg")

        # Tokenize the prompt, on CPU
        clip_tokens = clip_tokenizer.encode(prompt)
        t5_tokens = t5_tokenizer.encode(prompt)

        batch = preprocess_flux_data(
            device=torch_device,
            dtype=torch.bfloat16,
            autoencoder=None,
            clip_encoder=clip_encoder,
            t5_encoder=t5_encoder,
            batch={
                "clip_tokens": clip_tokens,
                "t5_tokens": t5_tokens,
            },
        )

        img = self._generate_images(
            device=torch_device,
            dtype=torch.bfloat16,
            model=model,
            decoder=ae,
            img_width=img_width,
            img_height=img_height,
            denoising_steps=num_steps,
            seed=seed,
            clip_encodings=batch["clip_encodings"],
            t5_encodings=batch["t5_encodings"],
            guidance=guidance,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")

        self._save_image(name, output_name, img, add_sampling_metadata, prompt)

    def _generate_images(
        self,
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
        guidance: float = 4.0,
    ):

        bsz = clip_encodings.shape[0]
        latents = generate_noise_latent(bsz, img_height, img_width, device, dtype, seed)
        _, latent_channels, latent_height, latent_width = latents.shape

        # create denoising schedule
        timesteps = get_schedule(denoising_steps, latent_channels, shift=True)

        # create positional encodings
        POSITION_DIM = 3  # constant for Flux flow model
        latent_pos_enc = create_position_encoding_for_latents(
            bsz, latent_height, latent_width, POSITION_DIM
        ).to(latents)
        text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM).to(latents)

        # convert img-like latents into sequences of patches
        latents = pack_latents(latents)

        # this is ignored for schnell
        guidance_vec = torch.full((bsz,), guidance, device=device, dtype=dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((bsz,), t_curr, dtype=dtype, device=device)
            pred = model(
                img=latents,
                img_ids=latent_pos_enc,
                txt=t5_encodings,
                txt_ids=text_pos_enc,
                y=clip_encodings,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            latents = latents + (t_prev - t_curr) * pred

        # convert sequences of patches into img-like latents
        latents = unpack_latents(latents, latent_height, latent_width)

        img = decoder.decode(latents)
        return img

    def _save_image(
        self,
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
