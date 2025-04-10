# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from diffusers import AutoencoderKL


def load_ae(
    ckpt_path: str,
    device: str | torch.device = "cuda",
    dtype=torch.bfloat16,
) -> AutoencoderKL:
    """
    Load the autoencoder from the given model name.
    Args:
        name (str): The name of the autoencoder.
        device (str or torch.device): The device to load the autoencoder to.
    Returns:
        AutoEncoder: The loaded autoencoder.
    """
    # Loading the autoencoder

    if not os.path.exists(ckpt_path):
        raise ValueError(
            f"Autoencoder path {ckpt_path} does not exist. Please download it first."
        )

    print("Init AE")
    ae = AutoencoderKL.from_pretrained(ckpt_path, device=device, dtype=dtype)
    return ae
