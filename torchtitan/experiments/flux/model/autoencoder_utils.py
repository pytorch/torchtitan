# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import Tensor

from torchtitan.experiments.flux.model.autoencoder import AutoEncoder


def encode_with_mean_logvar(
    autoencoder: AutoEncoder, x: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Encode images and return the mean and logvar values.

    Args:
        autoencoder (AutoEncoder): The autoencoder model
        x (Tensor): Input images tensor [batch_size, channels, height, width]

    Returns:
        Tuple[Tensor, Tensor]: (mean, logvar)
            - mean: The mean values from the encoder
            - logvar: The log variance values from the encoder
    """
    # Get the raw encoder output
    z = autoencoder.encoder(x)

    # Split into mean and logvar
    mean, logvar = torch.chunk(z, 2, dim=1)

    return mean, logvar


def generate_latent_from_mean_logvar(
    autoencoder: AutoEncoder, mean: Tensor, logvar: Tensor
) -> Tensor:
    """
    Generate latent encoding from mean and logvar.

    Args:
        autoencoder (AutoEncoder): The autoencoder model
        mean (Tensor): Mean tensor from the encoder
        logvar (Tensor): Log variance tensor from the encoder

    Returns:
        Tensor: The final encoded values after sampling and scaling
    """
    # Sample from the distribution
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mean)
    z_sampled = mean + std * eps

    # Apply scaling and shifting (same as in AutoEncoder.encode)
    encoded = autoencoder.scale_factor * (z_sampled - autoencoder.shift_factor)

    return encoded
