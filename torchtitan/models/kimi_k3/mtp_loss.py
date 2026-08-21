# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-token-prediction loss (report sec 3.3), without a core change.

This was recorded as blocked on "the trainer's loss interface must carry more
than one head". It does not have to:

* The trainer calls ``loss_fn(pred, labels, global_valid_tokens)`` with a single
  ``pred``. Adding heads to that signature would be a core change.
* But MTP's targets are just ``labels`` shifted -- depth k predicts the token
  k+1 ahead -- so this loss needs no extra data from the trainer, only the extra
  logits.
* The model hands them to a rank-local holder during forward and this loss
  takes them, clearing as it goes. A holder is not elegant, but it needs nothing
  from core -- which is the constraint experiments are held to -- and the
  alternative, a hook, does not work: ``post_optimizer_build_fn`` receives
  optimizers, model parts and parallel dims, never ``loss_fn``.

  Clearing on read is what makes it safe under gradient accumulation and PP
  microbatching: forward and loss alternate per microbatch, so a value that is
  taken exactly once cannot be reused by a later microbatch whose forward
  produced none.

Weighting follows the family's formulation: the main next-token loss plus
``mtp_weight`` times the mean of the per-depth losses, so the main objective
keeps its scale as depths are added rather than being progressively drowned.
"""

from dataclasses import dataclass, field

import torch

from torchtitan.components.loss import BaseLoss, CrossEntropyLoss

# Rank-local hand-off from the model's forward to this loss. Written by
# KimiK3AttnResModel.forward, taken (and cleared) here.
_PENDING: list[torch.Tensor] | None = None


def put_mtp_logits(logits: list[torch.Tensor]) -> None:
    global _PENDING
    _PENDING = logits


def take_mtp_logits() -> list[torch.Tensor] | None:
    global _PENDING
    logits, _PENDING = _PENDING, None
    return logits


class KimiMTPLoss(BaseLoss):
    """Main next-token cross-entropy plus the MTP depths' cross-entropy.

    Reduces to exactly the inner loss when MTP is off, so a flavor can turn
    ``num_nextn_predict_layers`` on and off without changing the loss config and
    without a silent change in what is optimised.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        mtp_weight: float = 0.3
        """Weight on the mean per-depth MTP loss."""

        loss_fn: BaseLoss.Config = field(default_factory=CrossEntropyLoss.Config)
        """Loss applied to the main head and to each MTP depth."""

    def __init__(self, config: Config, *, compile_config=None):
        self.mtp_weight = config.mtp_weight
        self.inner = config.loss_fn.build(compile_config=compile_config)
        # BaseLoss.__call__ would use self.fn; this class overrides __call__ and
        # delegates to the inner loss instead, so self.fn stays the inner's.
        self.fn = self.inner.fn

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        main_loss, metrics = self.inner(pred, labels, global_valid_tokens)

        mtp_logits = take_mtp_logits()
        if not mtp_logits:
            return main_loss, metrics

        depth_losses = []
        for k, logits in enumerate(mtp_logits):
            shift = k + 1
            # Depth k's prediction at position t targets the token at t+shift, and
            # the model already dropped the positions with no target, so the
            # labels line up by taking the same shift off the front.
            target = labels[:, shift:]
            n = min(logits.size(1), target.size(1))
            if n <= 0:
                continue
            depth_loss, _ = self.inner(
                logits[:, :n], target[:, :n], global_valid_tokens
            )
            depth_losses.append(depth_loss)

        if not depth_losses:
            return main_loss, metrics

        mtp_mean = torch.stack(depth_losses).mean()
        metrics = {**metrics, "loss/mtp": mtp_mean.detach()}
        return main_loss + self.mtp_weight * mtp_mean, metrics
