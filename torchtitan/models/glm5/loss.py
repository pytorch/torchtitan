# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.components.loss import Loss


class Glm5WarmupLoss(Loss[torch.Tensor]):
    """Consume the already globally normalized KL-only warmup objective."""

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        evaluation_loss: Loss.Config

    def __init__(self, config: Config):
        self.evaluation_loss = config.evaluation_loss.build()

    def __call__(self, pred, labels, global_loss_token_counts=None, **loss_inputs):
        # AuxLoss owns the routed-token normalization and per-layer KL metrics.
        # Labels remain in the dataloader/checkpoint contract but need no LM head.
        # Trainer evaluates through Validator.validate under no_grad. Do not
        # reinterpret malformed training outputs as a different objective.
        if torch.is_grad_enabled():
            if not isinstance(pred, torch.Tensor) or pred.ndim != 0:
                raise ValueError("GLM warmup training requires a scalar KL objective")
            return pred, {}
        if isinstance(pred, torch.Tensor) and pred.ndim == 0:
            raise ValueError("GLM warmup evaluation requires teacher logits, not KL")
        return self.evaluation_loss(pred, labels, global_loss_token_counts, **loss_inputs)
