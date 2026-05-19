# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared types for the RL loss modules.

Mirrors the shape of ``forge/src/forge/rl/loss/types.py`` but drops
the pydantic + Metric dependencies; metrics here are plain
``dict[str, torch.Tensor]`` so the trainer can ``all_reduce`` them
directly with the same SUM/MAX pattern the existing pipeline already
uses (see :meth:`PolicyTrainer._reduce_forward_backward_metrics`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

__all__ = ["AggType", "LossOutput"]


AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Return value for every loss in this package.

    Each loss emits one scalar tensor (``loss`` — backproppable) and
    two dicts of per-step metrics keyed by reduce-op. The trainer
    ``all_reduce``s them across DP ranks with SUM and MAX respectively
    before logging, so:

    - ``sum_metrics`` should hold values that should be SUMMED across
      ranks (e.g. ``local_sum / global_N`` → SUM-reduce = global mean).
    - ``max_metrics`` should hold values that should be MAXed across
      ranks (e.g. per-rank max importance ratio).
    """

    loss: torch.Tensor
    sum_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    max_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
