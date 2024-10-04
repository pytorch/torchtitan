# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import torch

from torch import nn


class TanhGate(nn.Module):
    """Implements a basic learnable gate to scale layer outputs"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to gate

        Returns:
            torch.Tensor: The output tensor after gating. Has the same shape as ``x``.
        """
        return x * self.scale.tanh()
