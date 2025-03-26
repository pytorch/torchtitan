import os
from dataclasses import dataclass

import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import ExifTags, Image
from safetensors.torch import load_file as load_sft
from torchtitan.config_manager import JobConfig

from torchtitan.datasets.flux_datasets import build_flux_dataloader
from torchtitan.datasets.tokenizer.HFEmbedder import HFEmbedder

from torchtitan.experiments.flux.model import Flux, FluxParams
from torchtitan.experiments.flux.modules.autoencoder import (
    AutoEncoder,
    AutoEncoderParams,
)


def save_image(
    nsfw_classifier,
    name: str,
    output_name: str,
    idx: int,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
    nsfw_threshold: float = 0.85,
) -> int:
    fn = output_name.format(idx=idx)
    print(f"Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

    if nsfw_score < nsfw_threshold:
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        idx += 1
    else:
        print("Your generated image may contain NSFW content.")

    return idx


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_ae(
    name: str, device: str | torch.device = "cuda", hf_download: bool = False
) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    lora_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=None,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=512,  # ??????????????
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    )
}


def load_flow_model(
    name: str,
    device: str | torch.device = "cuda",
    hf_download: bool = False,
    verbose: bool = False,
) -> Flux:
    # Loading Flux from init status
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device(device):
        model = Flux(configs[name].params).to(torch.bfloat16)

    return model


def load_t5(device: str = "cpu", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder(
        "google/t5-v1_1-small", max_length=max_length, torch_dtype=torch.bfloat16
    ).to(device)


def load_clip(device: str = "cpu") -> HFEmbedder:
    # The max length is set to be 77
    return HFEmbedder(
        "openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16
    ).to(device)


def build_dataloader(
    dataset_name,
    t5,
    clip,
    batch_size,
    world_size,
    rank,
    seed,
    device="cpu",
):
    config = JobConfig()
    config.parse_args(
        [
            "--training.dataset",
            dataset_name,
            "--training.batch_size",
            str(batch_size),
            "--training.seed",
            str(seed),
        ]
    )

    return build_flux_dataloader(
        dp_world_size=world_size,
        dp_rank=rank,
        embedder=[t5, clip],
        job_config=config,
        infinite=False,
        device=device,
    )
