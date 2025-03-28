import os
import re
import time
from dataclasses import dataclass
from glob import iglob
from random import sample

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.datasets.flux_dataset import build_flux_dataloader
from torchtitan.experiments.flux.model_builder import (
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

from torchtitan.experiments.flux.sampling import get_schedule
from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    denoise,
    generate_noise_latent,
    pack_latents,
    save_image,
    unpack_latent,
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
def generate_image(
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
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
):
    """
    Run Forward pass of flux model to generate an image.

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

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    img_height = 16 * (img_height // 16)
    img_width = 16 * (img_width // 16)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [
            fn
            for fn in iglob(output_name.format(idx="*"))
            if re.search(r"img_[0-9]+\.jpg$", fn)
        ]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # init all components
    t5 = load_t5(name, torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(name, torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=img_width,
        height=img_height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        # TODO(jianiw): Replace this dummy JobConfig with a real one
        config = JobConfig()
        config.parse_args(
            [
                "--training.dataset",
                "cc12m",
                "--training.batch_size",
                "16",
                "--training.seed",
                str(opts.seed),
            ]
        )

        dataloader = build_flux_dataloader(
            dp_world_size=1,  # TODO(jianiw): Change world size
            dp_rank=0,  # TODO(jianiw): Change rank
            t5_encoder=t5,
            clip_encoder=clip,
            job_config=config,
            infinite=False,
        )

        # TODO(jianiw): Remove this hack to continue loading the next batch
        sample_data = next(iter(dataloader))

        bsz = sample_data["clip_encodings"].shape[0]
        latents = generate_noise_latent(
            bsz,
            img_height,
            img_width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        _, latent_channels, latent_height, latent_width = latents.shape

        latent_pos_enc = create_position_encoding_for_latents(
            bsz, latent_height, latent_width
        ).to(latents)
        text_pos_enc = torch.zeros(bsz, sample_data["t5_encodings"].shape[1], 3).to(
            latents
        )

        # Convert latent into a sequence of patches
        latents = pack_latents(latents)

        timesteps = get_schedule(
            opts.num_steps, latent_channels, shift=(name != "flux-schnell")
        )

        # denoise initial noise
        x = denoise(
            model,
            img=latents,
            img_ids=latent_pos_enc.to(latents),
            txt=sample_data["t5_encodings"].to(latents),
            txt_ids=text_pos_enc.to(latents),
            vec=sample_data["clip_encodings"].to(latents),
            target=sample_data["image"].to(latents),
            timesteps=timesteps,
            guidance=opts.guidance,
        )

        # decode latents to pixel space
        x = unpack_latent(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")

        idx = save_image(name, output_name, idx, x, add_sampling_metadata, prompt)


if __name__ == "__main__":
    generate_image()
