# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard

from torchtitan.components.loss import BaseLoss, IGNORE_INDEX
from torchtitan.config import CompileConfig
from torchtitan.distributed.spmd_types import get_spmd_backend


def z_loss_cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    z_loss_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = pred.float()
    flat_logits = logits.flatten(0, 1)
    flat_labels = labels.flatten(0, 1)

    ce_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )

    mask = flat_labels != IGNORE_INDEX
    z_loss = z_loss_multiplier * (flat_logits.logsumexp(-1).pow(2) * mask).sum()
    return ce_loss, z_loss


class ZLossCrossEntropyLoss(BaseLoss):
    """Cross-entropy plus OLMo3's auxiliary softmax z-loss."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        z_loss_multiplier: float = 1e-5

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.z_loss_multiplier = config.z_loss_multiplier
        self.fn = z_loss_cross_entropy_loss
        self._maybe_compile(compile_config)

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(pred, DTensor) and any(
            isinstance(placement, Shard) and placement.dim == pred.ndim - 1
            for placement in pred.placements
        ):
            raise ValueError("ZLossCrossEntropyLoss does not support vocab-sharded logits")

        ce_loss, z_loss = self.fn(
            pred,
            labels,
            z_loss_multiplier=self.z_loss_multiplier,
        )
        loss = ce_loss + z_loss

        if global_valid_tokens is not None:
            with spmd.no_typecheck():
                loss = loss / global_valid_tokens
                ce_loss = ce_loss / global_valid_tokens
                z_loss = z_loss / global_valid_tokens
                if get_spmd_backend() == "spmd_types":
                    spmd.assert_type(
                        loss, {"dp": spmd.P, "cp": spmd.P, "tp": spmd.I}
                    )

        return loss, {
            "loss/ce/mean": ce_loss.detach(),
            "loss/z/mean": z_loss.detach(),
        }
