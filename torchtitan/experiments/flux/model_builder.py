import os
from dataclasses import dataclass

import torch

from huggingface_hub import hf_hub_download

from torchtitan.experiments.flux.model import FluxModel, FluxParams
from torchtitan.experiments.flux.modules.autoencoder import AutoEncoderParams


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
        FluxFlow
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
