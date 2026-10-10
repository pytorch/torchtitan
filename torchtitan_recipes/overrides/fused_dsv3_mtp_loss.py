# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSv3 MTP cross entropy with compact autograd state and a gated dispatch.

T = local tokens, V = the full vocabulary. MTP weighting, token-count
normalization, label alignment, and TP communication stay in the native loss.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import torch

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.deepseek_v3.mtp import MTPLoss

from ._dsv3_mtp_cross_entropy import ops


logger = logging.getLogger(__name__)


def _cross_entropy_loss(
    pred_TV: torch.Tensor,
    labels_T: torch.Tensor,
    *,
    global_vocab_size: int | None = None,
    reduction: Literal["sum", "none"] = "sum",
) -> torch.Tensor:
    if (
        ops.ACCEPTED
        and reduction == "sum"
        and global_vocab_size in (None, ops.VOCAB_SIZE)
        and spmd_mesh_size("tp") == 1
        and ops.supports(pred_TV, labels_T)
    ):
        return ops.cross_entropy_sum(pred_TV, labels_T)
    return cross_entropy_loss(
        pred_TV, labels_T, global_vocab_size=global_vocab_size, reduction=reduction
    )


class FusedDSv3MTPLoss(MTPLoss):
    """Reuse native MTP composition and specialize only its raw CE callback."""

    @dataclass(kw_only=True, slots=True)
    class Config(MTPLoss.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.fn = _cross_entropy_loss
        if not ops.ACCEPTED:
            logger.warning(
                "MTP cross-entropy fusion has not met all acceptance gates; "
                "model dispatch will use native cross entropy."
            )


@override(
    target=MTPLoss.Config,
    exact=True,
    description="Use the acceptance-gated DSv3 MTP cross-entropy autograd override.",
)
def fused_dsv3_mtp_loss(cfg: MTPLoss.Config) -> MTPLoss.Config:
    if cfg.global_vocab_size not in (None, ops.VOCAB_SIZE):
        logger.warning(
            "MTP cross-entropy fusion requires the full 129280 vocabulary; "
            "keeping the native loss configuration."
        )
        return cfg
    return derive(cfg, FusedDSv3MTPLoss.Config)
