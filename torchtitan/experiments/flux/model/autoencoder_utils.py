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
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Encode images and return the mean, logvar, and final encoded values.

    Args:
        autoencoder (AutoEncoder): The autoencoder model
        x (Tensor): Input images tensor [batch_size, channels, height, width]

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (mean, logvar, encoded)
            - mean: The mean values from the encoder
            - logvar: The log variance values from the encoder
            - encoded: The final encoded values after sampling and scaling
    """
    # Get the raw encoder output
    z = autoencoder.encoder(x)

    # Split into mean and logvar
    mean, logvar = torch.chunk(z, 2, dim=1)

    # Sample from the distribution (same as in DiagonalGaussian)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mean)
    z_sampled = mean + std * eps

    # Apply scaling and shifting (same as in AutoEncoder.encode)
    encoded = autoencoder.scale_factor * (z_sampled - autoencoder.shift_factor)

    return mean, logvar, encoded


def decode_from_mean_logvar(
    autoencoder: AutoEncoder, mean: Tensor, logvar: Tensor, sample: bool = True
) -> Tensor:
    """
    Decode from mean and logvar values.

    Args:
        autoencoder (AutoEncoder): The autoencoder model
        mean (Tensor): Mean values from the encoder
        logvar (Tensor): Log variance values from the encoder
        sample (bool): Whether to sample from the distribution or use the mean directly

    Returns:
        Tensor: Decoded images
    """
    if sample:
        # Sample from the distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        z_sampled = mean + std * eps
    else:
        # Use mean directly
        z_sampled = mean

    # Apply scaling and shifting (same as in AutoEncoder.encode)
    encoded = autoencoder.scale_factor * (z_sampled - autoencoder.shift_factor)

    # Decode
    return autoencoder.decode(encoded)
