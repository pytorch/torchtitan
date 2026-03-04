# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
):
    """
    Fills the input tensor with values sampled from a truncated normal distribution.
    Values are drawn from a normal distribution with the given mean and standard
    deviation. Any sampled values outside the range defined by a and b are resampled
    until they fall within the bounds.

    To avoid numerical instability in torch.nn.init.trunc_normal_, the initialization
    is always performed using float32 precision. The result is then cast back to the
    original data type of the input tensor.

    Args:
        tensor: an n dimensional torch Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the lower bound for truncation
        b: the upper bound for truncation

    Returns:
        The input tensor filled with values from the truncated normal distribution.
    """

    tmp = tensor.float()
    nn.init.trunc_normal_(tmp, mean=mean, std=std, a=a, b=b)
    tensor.copy_(tmp)
