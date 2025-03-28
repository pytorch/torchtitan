import os
from dataclasses import dataclass

import torch

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

from torchtitan.experiments.flux.model import FluxModel, FluxParams
from torchtitan.experiments.flux.modules.autoencoder import (
    AutoEncoder,
    AutoEncoderParams,
)
from torchtitan.experiments.flux.modules.embedder import HFEmbedder


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
    t5_encoder: str | None
    clip_encoder: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=None,
        t5_encoder="google/t5-v1_1-small",
        clip_encoder="openai/clip-vit-large-patch14",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,  # This dimension should be the same as the CLIP embedding dimension
            context_in_dim=512,  # This dimension should be the same as the T5 embedding dimension
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


def load_t5(model_name, device: str = "cpu", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    assert configs[model_name].t5_encoder is not None
    return HFEmbedder(
        configs[model_name].t5_encoder,
        max_length=max_length,
        torch_dtype=torch.bfloat16,
    ).to(device)


def load_clip(model_name, device: str = "cpu") -> HFEmbedder:
    # The max length is set to be 77
    assert configs[model_name].clip_encoder is not None
    return HFEmbedder(
        configs[model_name].clip_encoder, max_length=77, torch_dtype=torch.bfloat16
    ).to(device)


def load_flow_model(
    name: str,
    device: str | torch.device = "cuda",
    hf_download: bool = False,
) -> FluxModel:
    """
    FLUX model loader from checkpoint or repo. Load model according to the config.

    Args:
        name: Name of the model to load
        device: Device to load the model to
        hf_download: Whether to download the model from huggingface

    Returns:
        FluxModel
    """
    # Loading Flux from init status
    print(f"Init FLUX model {name}")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device(device):
        model = FluxModel(configs[name].params).to(torch.bfloat16)

    return model


def load_ae(
    name: str, device: str | torch.device = "cuda", hf_download: bool = False
) -> AutoEncoder:
    """
    Load the autoencoder from the given model name.
    Args:
        name (str): The name of the autoencoder.
        device (str or torch.device): The device to load the autoencoder to.
    Returns:
        AutoEncoder: The loaded autoencoder.
    """
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
