# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from dataclasses import dataclass

import torch

from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer

from torchtitan.experiments.flux.model.autoencoder import AutoEncoderParams, load_ae
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder

from torchtitan.experiments.flux.model.model import FluxModel, FluxModelArgs
from torchtitan.experiments.flux.utils import (
    generate_images,
    preprocess_flux_data,
    save_image,
)


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


@torch.inference_mode()
def test_generate_image(
    name: str = "flux-dev",
    img_width: int = 512,
    img_height: int = 512,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
):
    """
    Run a forward pass of flux model to generate an image.

    Args:
        name: Name of the model to load
        img_height: height of the sample in pixels (should be a multiple of 16)
        img_width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """

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
    opts = SamplingOptions(
        prompt=prompt,
        width=img_width,
        height=img_height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if opts.seed is None:
        opts.seed = rng.seed()
    print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
    t0 = time.perf_counter()
    output_name = os.path.join(output_dir, f"img_{opts.seed}.jpg")

    # Tokenize the prompt, on CPU
    clip_tokens = clip_tokenizer.encode(opts.prompt)
    t5_tokens = t5_tokenizer.encode(opts.prompt)

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

    img = generate_images(
        device=torch_device,
        dtype=torch.bfloat16,
        model=model,
        decoder=ae,
        img_width=opts.width,
        img_height=opts.height,
        denoising_steps=opts.num_steps,
        seed=opts.seed,
        clip_encodings=batch["clip_encodings"],
        t5_encodings=batch["t5_encodings"],
        guidance=opts.guidance,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f"Done in {t1 - t0:.1f}s.")

    save_image(name, output_name, img, add_sampling_metadata, prompt)


if __name__ == "__main__":
    test_generate_image()
