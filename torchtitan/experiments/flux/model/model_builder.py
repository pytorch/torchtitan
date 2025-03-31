import os
from dataclasses import dataclass
from typing import Optional

import torch

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

from torchtitan.experiments.flux.model.model import FluxModel, FluxModelArgs, FluxParams
from torchtitan.experiments.flux.model.modules.autoencoder import (
    AutoEncoder,
    AutoEncoderParams,
)


@dataclass
class ModelSpec:  # TODO(jianiw): Fit this class into BaseModelArgs, then pass this class to model.py
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
        params=FluxModelArgs(),
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


def load_flow_model_from_ckpt(
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
        model = FluxModel(model_args=configs[name].params).to(torch.bfloat16)

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
