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
    """Return type of every RL loss class.

    `loss` is the scalar backward target. `token_mean_metric_sums`
    carries every logged loss diagnostic (including `loss/total`) as
    **local numerators**. `num_valid_tokens` is the shared local
    denominator. The trainer reducer SUMs both across DP and divides
    once, so logging is bias-free under unequal sharding.

    Today (sample-level GRPOLoss) the numerators come from
    `sequence_scalar_token_weighted_sum`; tomorrow (loss-migration PR)
    they come from `token_mean_sum(per_token_value, loss_mask)`. The
    contract — and the trainer reducer — does not change.
    """

    loss: torch.Tensor
    token_mean_metric_sums: dict[str, torch.Tensor]
    num_valid_tokens: torch.Tensor


def sequence_scalar_token_weighted_sum(
    values: torch.Tensor,
    response_lens: torch.Tensor,
) -> torch.Tensor:
    """Bridge: treat each sample-level scalar as repeated over its
    response tokens.

    Transitional helper for today's sample-level GRPOLoss. The
    loss-migration PR replaces calls to this with true
    `token_mean_sum(per_token_value, loss_mask)` — the LossOutput
    contract stays the same.
    """
    return (values * response_lens.to(values.dtype)).sum().detach()
