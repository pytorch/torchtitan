# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss return types and helpers for RL training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossOutput:
    """Return type of every RL loss.

    `token_mean_metric_sums` is a local SUM over valid tokens,
    meaning that it is already weighted by the number of valid
    tokens in the batch. To calculate metrics correctly, they
    need to be all-reduced and then divided by the global number
    of valid tokens.
    """

    loss: torch.Tensor
    token_mean_metric_sums: dict[str, torch.Tensor]
    num_local_valid_tokens: torch.Tensor


def sequence_scalar_token_weighted_sum(
    values: torch.Tensor,
    response_lens: torch.Tensor,
) -> torch.Tensor:
    """Sum each sample-level scalar weighted by the number of valid tokens it has."""
    return (values * response_lens.to(values.dtype)).sum().detach()
