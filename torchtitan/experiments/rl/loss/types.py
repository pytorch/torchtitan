# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss return types for RL training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True, slots=True)
class LossOutput:
    """Loss scalar and loss-owned metrics returned by an RL loss.

    Args:
        loss: Scalar tensor used for loss.backward().
        metrics: Per-rank metric tensors that the trainer SUM-reduces across
            the loss mesh. Producers must pre-normalize each value by the
            global token count so SUM-reducing reconstructs the global
            metric.

    Example:
        LossOutput(
            loss=pg_loss,
            metrics={
                "loss/total/mean": pg_loss.detach(),
                "loss/ratio/mean": ratio_share,
            },
        )
    """

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
