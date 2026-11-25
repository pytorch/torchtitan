# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Score centering (https://arxiv.org/abs/2609.20807) for per-token policy-gradient losses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from torchtitan.components.loss import compute_topk_logprobs
from torchtitan.config import Configurable

# Floor of each tail mass (the probability outside the top-k): it can round to <= 0 in fp32.
_MIN_TAIL_MASS = 1e-6


class ScoreCentering(Configurable):
    """Cancels the drift that a trainer/generator mismatch adds to the policy gradient.

    The trainer scores tokens under its distribution ``p``; the generator sampled them from a
    slightly different ``q`` (kernels, quantization, MoE routing, stale weights). A loss whose
    gradient at a token is ``-A * w(p_y / q_y) * grad(log p_y)`` then has, at every prefix, the
    expected update ``V * sum_v q_v w_v grad(log p_v) + signal``: the first term carries no
    reward information (it distills ``p`` toward ``q``, scaled by the prefix's expected
    advantage ``V``) and compounds as weight syncs copy ``p`` into the generator. Score
    centering subtracts that term at every position by adding
    ``A * sum_v sg(c_v) * log p_v`` to the token's loss.

    The generator sends only its top-k (the head ``H``). On the tail the generator is assumed
    to be the trainer scaled to the right mass, ``q_v = rho * p_v``, so every tail token has the
    ratio ``1 / rho`` and the weight ``w(1 / rho)``; ``sum_v p_v grad(log p_v) = 0`` then folds
    the tail into the head:

        c_v = q_v * w(p_v / q_v) - alpha * p_v   for v in H,
        rho = (1 - sum_H q) / (1 - sum_H p),    alpha = rho * w(1 / rho)

    Example (k = 2, w = 1): p = [0.40, 0.30, 0.15, 0.10, 0.05],
    q = [0.50, 0.22, 0.14, 0.09, 0.05] -> rho = 0.28 / 0.30, c = [0.127, -0.060]. Averaged over
    the sampled token, the logit gradient at a constant advantage goes from q - p =
    [0.10, -0.08, -0.01, -0.01, 0] to [0, 0, 0, -0.003, 0.003], the error of the tail model.
    A generator that matches the trainer bitwise gives c = 0.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_topk_logprobs: int = 32
        """Generator logprobs per generated token: its top-k tokens. Costs 8 * k bytes per
        generated token in transport. k = 32 and k = 128 matched full-vocabulary centering in
        every setting of arXiv 2609.20807 (App. A.4)."""

        def __post_init__(self) -> None:
            if self.num_topk_logprobs < 1:
                raise ValueError(
                    "ScoreCentering.num_topk_logprobs must be >= 1, got "
                    f"{self.num_topk_logprobs}"
                )

    def __init__(self, config: Config) -> None:
        del config  # num_topk_logprobs configures the generator, via the controller.

    def __call__(
        self,
        logits: torch.Tensor,
        *,
        temperature: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        generator_topk_token_ids: torch.Tensor,
        generator_topk_logprobs: torch.Tensor,
        log_token_weight: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        vocab_parallel_group: dist.ProcessGroup | None,
        global_vocab_size: int | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Per-token correction with the gradient of ``sum_H sg(c_v) * log p_v`` (its value is
        0) and per-token metric values.

        Args:
            logits: [T, V] current-policy logits (a vocab shard under TP).
            temperature: [T] temperature each token was sampled at.
            advantages: [T] per-token advantages.
            loss_mask: [T] loss tokens with a finite generator logprob.
            generator_topk_token_ids: [T, k] the generator's top-k token ids (0 off the mask).
            generator_topk_logprobs: [T, k] the generator's logprobs of those ids.
            log_token_weight: the loss's ``log w(log_ratio, advantages)``; -inf where w = 0.
            vocab_parallel_group: TP group when ``logits`` hold a vocab shard, else None.
            global_vocab_size: full vocabulary size; required under TP.

        Returns:
            ([T] correction, the caller adds ``advantages * correction`` to the token loss;
            metrics, each a [T] value the caller averages over loss tokens).
        """
        trainer_topk_logprobs = compute_topk_logprobs(
            logits,
            generator_topk_token_ids,
            vocab_parallel_group=vocab_parallel_group,
            global_vocab_size=global_vocab_size,
            temperature=temperature,
        )
        # Off the loss mask the generator rows are zero placeholders or NaN, and NaN * 0 is
        # NaN, so mask the inputs, not only the coefficients.
        row_mask = loss_mask.unsqueeze(-1)
        with torch.no_grad():
            trainer_logprobs = torch.where(row_mask, trainer_topk_logprobs, 0.0)
            generator_logprobs = torch.where(row_mask, generator_topk_logprobs, 0.0)
            trainer_probs = torch.exp(trainer_logprobs)
            generator_probs = torch.exp(generator_logprobs)
            trainer_topk_mass = trainer_probs.sum(-1)
            generator_topk_mass = generator_probs.sum(-1)
            log_rho = torch.log(
                (1 - generator_topk_mass).clamp_min(_MIN_TAIL_MASS)
            ) - torch.log((1 - trainer_topk_mass).clamp_min(_MIN_TAIL_MASS))
            # q * w(p / q) in log space: inside a ratio band q * (p / q) is p, exactly and
            # without overflow when q underflows.
            weighted_generator_probs = torch.exp(
                generator_logprobs
                + log_token_weight(
                    trainer_logprobs - generator_logprobs, advantages.unsqueeze(-1)
                )
            )
            alpha = torch.exp(log_rho + log_token_weight(-log_rho, advantages))
            coefficients = torch.where(
                row_mask,
                weighted_generator_probs - alpha.unsqueeze(-1) * trainer_probs,
                0.0,
            )
            metrics = {
                # The paper sees > 99.9% generator mass in the top-128 (App. A.4).
                "score_centering/generator_topk_mass/mean": generator_topk_mass,
                "score_centering/trainer_topk_mass/mean": trainer_topk_mass,
                # 0 when the generator matches the trainer, or no weight caps or masks a ratio.
                "score_centering/coefficient_l1/mean": coefficients.abs().sum(-1),
            }
        # log p - sg(log p): the gradient of sum_H c_v * log p_v with a forward value of 0, so
        # loss/mean stays comparable with runs without score centering.
        # TODO: score the label and the top-k ids in one call; this separate call saves one
        # more fp32 [T, V_local] per chunk for backward.
        correction = (
            coefficients * (trainer_topk_logprobs - trainer_topk_logprobs.detach())
        ).sum(-1)
        return correction, metrics
