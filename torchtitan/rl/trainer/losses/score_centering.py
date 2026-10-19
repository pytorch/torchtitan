# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Score centering loss: REINFORCE plus a correction that cancels trainer/generator drift."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchtitan.components.loss import compute_logprobs, compute_topk_logprobs, Loss
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.rl.trainer.losses.dapo import _MAX_LOG_RATIO, _normalize

# Floor for the probability mass outside the top-k: `1 - sum_H` can round to <= 0 in fp32.
_MIN_TAIL_MASS = 1e-6


class ScoreCenteringLoss(Loss[torch.Tensor]):
    """REINFORCE loss with score centering (https://arxiv.org/abs/2609.20807).

    When the generator's distribution `q` differs from the trainer's `p` (quantization,
    kernels, staleness), the expected policy gradient at each position splits (Eq. 3):

        E_q[A * grad log p_y] = drift + signal
        drift  = E_q[A] * sum_v q_v grad log p_v    # pulls p toward q; no reward information
        signal = Cov_q(A, grad log p_y)

    Each weight sync feeds the drift back into the generator, so it compounds. Score
    centering subtracts the expected score `sum_v q_v grad log p_v` from each token's score,
    which cancels the drift. The sum runs over the generator's top-k tokens `H`; outside `H`,
    `q` is modeled as `p` rescaled to the generator's tail mass (Eq. 7-12). The token weight
    `w(r)` is one of:

        max_ratio=C            w(r) = min(r, C)                    truncated IS
        ratio_band=(lo, hi)    w(r) = r if lo <= r <= hi else 0    masked IS
        neither                w(r) = 1                            SC alone

    and enters as:

        rho        = (1 - sum_H q) / (1 - sum_H p)
        alpha      = rho * w(1 / rho)                    # weight of the modeled tail
        token_loss = -A * (detach(w(p_y / q_y)) * log p_y
                           - sum_H detach(q_v * w(p_v / q_v) - alpha * p_v) * log p_v)
        loss       = sum(token_loss) / global_loss_token_counts

    When `p == q`, every weight is 1 and the correction is 0: plain REINFORCE. With
    `ratio_band`, `q_v * w(p_v / q_v) = p_v` inside the band, so the correction is 0 unless a
    top-k entry or the tail leaves it: it cancels the bias of the masked probability mass.

    Example (one position, k=2, max_ratio=None):

        q_head = [0.5, 0.3],  p_head = [0.4, 0.4]   # tails 0.2 and 0.2, so rho = 1
        coefficients = q_head - rho * p_head = [0.1, -0.1]
        # At A = 1, sampling from q moves the head logits by +[0.1, -0.1] in expectation;
        # the correction moves them by -[0.1, -0.1], so the net drift is 0.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        max_ratio: float | None = 2.0
        """Cap on the detached ratio `p / q` that weights each token (truncated IS; the paper's
        TIS+SC). None weights every token by 1 (SC alone). The paper finds both alike under
        quantization, and the cap clearly better under large staleness."""

        ratio_band: tuple[float, float] | None = None
        """Masked IS: absolute bounds `(lo, hi)` on the ratio `r = p / q` (unlike
        `DAPOLoss.ratio_clip_*`, which are offsets from 1). A token with `lo <= r <= hi` is
        weighted by `r`, otherwise by 0; the same rule masks each top-k entry and the modeled
        tail, whose ratio is `(1 - sum_H p) / (1 - sum_H q)`. Requires `max_ratio=None`, and
        `lo <= 1 <= hi` so a generator that matches the trainer stays plain REINFORCE.
        Example: `(0.2, 5.0)`."""

        global_vocab_size: int | None = None
        """Full vocabulary size; required when TP shards the vocabulary."""

        def __post_init__(self) -> None:
            if self.ratio_band is None:
                return
            if self.max_ratio is not None:
                raise ValueError(
                    "ScoreCenteringLoss: set max_ratio=None to use ratio_band "
                    f"(got max_ratio={self.max_ratio}, ratio_band={self.ratio_band})."
                )
            low, high = self.ratio_band
            if not 0 < low <= 1 <= high:
                raise ValueError(
                    "ScoreCenteringLoss: ratio_band=(lo, hi) needs 0 < lo <= 1 <= hi, "
                    f"got {self.ratio_band}."
                )

    def __init__(self, config: Config) -> None:
        self.max_ratio = config.max_ratio
        self.log_ratio_band = (
            None
            if config.ratio_band is None
            else (math.log(config.ratio_band[0]), math.log(config.ratio_band[1]))
        )
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        *,
        generator_logprobs: torch.Tensor,
        temperature: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        generator_topk_token_ids: torch.Tensor,
        generator_topk_logprobs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-centered policy-gradient loss.

        Args:
            pred: [T, V] current-policy logits.
            labels: [T] pre-shifted target token ids.
            generator_logprobs: [T] logprobs from the sampling policy.
            temperature: [T] temperature each token was sampled at.
            loss_mask: [T] bool mask; True for response tokens.
            advantages: [T] per-token advantages (0.0 for prompt/padding).
            generator_topk_token_ids: [T, k] the generator's k most likely tokens per position.
            generator_topk_logprobs: [T, k] the generator's logprobs of those tokens.
            global_loss_token_counts: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; the loss denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        logits = pred / temperature.unsqueeze(-1)
        vocab_parallel_group = spmd_mesh_group("tp")
        trainer_logprobs, token_entropy = compute_logprobs(
            logits,
            labels,
            vocab_parallel_group=vocab_parallel_group,
            return_entropy=True,
            global_vocab_size=self.global_vocab_size,
        )
        # TODO: score the label and the top-k ids in one compute_topk_logprobs call. In eager,
        # two calls save two [T, V_local] fp32 buffers for backward where DAPOLoss saves one
        # (16 -> 32 MiB at T=512, V=8K; the "loss" local compile region brings it back to 16).
        # Kept apart so batch-invariant runs keep compute_logprobs' vocab gather.
        trainer_topk_logprobs = compute_topk_logprobs(
            logits,
            generator_topk_token_ids,
            vocab_parallel_group=vocab_parallel_group,
            global_vocab_size=self.global_vocab_size,
        )
        effective_loss_mask = loss_mask & torch.isfinite(generator_logprobs)

        # The ratio weights and the correction coefficients are constants in the gradient.
        with torch.no_grad():
            masked_log_ratio = torch.where(
                effective_loss_mask, trainer_logprobs - generator_logprobs, 0.0
            )
            ratio = torch.exp(
                torch.clamp(masked_log_ratio, -_MAX_LOG_RATIO, _MAX_LOG_RATIO)
            )
            trainer_topk_probs = torch.exp(trainer_topk_logprobs)
            generator_topk_probs = torch.exp(generator_topk_logprobs)
            trainer_head_mass = trainer_topk_probs.sum(-1)
            generator_head_mass = generator_topk_probs.sum(-1)
            rho = (1 - generator_head_mass).clamp_min(_MIN_TAIL_MASS) / (
                1 - trainer_head_mass
            ).clamp_min(_MIN_TAIL_MASS)
            if self.log_ratio_band is not None:
                log_low, log_high = self.log_ratio_band
                token_in_band = (masked_log_ratio >= log_low) & (
                    masked_log_ratio <= log_high
                )
                token_weight = torch.where(token_in_band, ratio, 0.0)
                # q * w(p / q) = p inside the band and 0 outside, without dividing by q.
                head_log_ratio = trainer_topk_logprobs - generator_topk_logprobs
                weighted_generator_topk_probs = torch.where(
                    (head_log_ratio >= log_low) & (head_log_ratio <= log_high),
                    trainer_topk_probs,
                    0.0,
                )
                # rho * w(1 / rho): 1 if the tail ratio p_tail / q_tail = 1 / rho is in the band.
                tail_log_ratio = -rho.log()
                alpha = ((tail_log_ratio >= log_low) & (tail_log_ratio <= log_high)).to(
                    rho.dtype
                )
            elif self.max_ratio is None:
                token_weight = torch.ones_like(ratio)
                weighted_generator_topk_probs = generator_topk_probs
                alpha = rho
            else:
                token_weight = torch.clamp(ratio, max=self.max_ratio)
                # q * min(p / q, C) = min(p, C * q), which stays finite when q underflows.
                weighted_generator_topk_probs = torch.minimum(
                    trainer_topk_probs, self.max_ratio * generator_topk_probs
                )
                # rho * min(1 / rho, C) = min(1, C * rho)
                alpha = torch.clamp(self.max_ratio * rho, max=1.0)
            coefficients = (
                weighted_generator_topk_probs - alpha.unsqueeze(-1) * trainer_topk_probs
            )
            # Off the loss mask the top-k rows are zero placeholders or NaN, and NaN * 0 is NaN.
            coefficients = torch.where(
                effective_loss_mask.unsqueeze(-1), coefficients, 0.0
            )

        correction = (coefficients * trainer_topk_logprobs).sum(-1)
        token_loss = -advantages * (token_weight * trainer_logprobs - correction)
        loss = _normalize(
            (token_loss * effective_loss_mask).sum(), global_loss_token_counts
        )

        with torch.no_grad():
            # Gradient of the correction w.r.t. the logits, before the advantage.
            coefficient_sum = coefficients.sum(-1)
            head_logit_grad = (
                coefficients - trainer_topk_probs * coefficient_sum[:, None]
            )
            tail_logit_grad = (1 - trainer_head_mass).clamp_min(0) * coefficient_sum
            logit_correction_l1 = head_logit_grad.abs().sum(-1) + tail_logit_grad.abs()
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": _normalize(
                    (ratio * effective_loss_mask).sum(), global_loss_token_counts
                ),
                # Mean of log p - log q over loss tokens: the k1 estimate of -KL(q || p).
                "bit_wise/logprob_diff/mean": _normalize(
                    masked_log_ratio.float().sum(), global_loss_token_counts
                ),
                "bit_wise/ratio_tokens_different/mean": _normalize(
                    (masked_log_ratio.abs() > 1e-6).float().sum(),
                    global_loss_token_counts,
                ),
                "bit_wise/logprob_diff/max": masked_log_ratio.abs().max(),
                # Mean entropy of softmax(logits / temperature) over tokens used by the loss.
                "trainer/entropy/mean": _normalize(
                    (token_entropy * effective_loss_mask).sum(),
                    global_loss_token_counts,
                ),
                # Share of q (and p) inside the generator's top-k; the paper sees > 99.9% at k=128.
                "score_centering/generator_head_mass/mean": _normalize(
                    torch.where(effective_loss_mask, generator_head_mass, 0.0).sum(),
                    global_loss_token_counts,
                ),
                "score_centering/trainer_head_mass/mean": _normalize(
                    torch.where(effective_loss_mask, trainer_head_mass, 0.0).sum(),
                    global_loss_token_counts,
                ),
                # sum_H |q * w - alpha * p|: 0 when trainer and generator agree, and with
                # max_ratio or ratio_band whenever no head or tail ratio is truncated or masked.
                "score_centering/coefficient_l1/mean": _normalize(
                    coefficients.abs().sum(-1).sum(), global_loss_token_counts
                ),
                # L1 norm of the correction's logit gradient: sum_H |c - p * sum_H c| +
                # (1 - sum_H p) * |sum_H c|. Unlike coefficient_l1, it stays ~0 when an
                # all-or-none tail flips at a near-deterministic position.
                "score_centering/logit_correction_l1/mean": _normalize(
                    logit_correction_l1.sum(), global_loss_token_counts
                ),
            }
            if self.log_ratio_band is not None:
                metrics["loss/ratio_masked_frac"] = _normalize(
                    (~token_in_band & effective_loss_mask).float().sum(),
                    global_loss_token_counts,
                )
                metrics["score_centering/tail_masked_frac"] = _normalize(
                    ((alpha == 0) & effective_loss_mask).float().sum(),
                    global_loss_token_counts,
                )
            if self.max_ratio is not None:
                metrics["loss/ratio_truncated_frac"] = _normalize(
                    ((ratio > self.max_ratio) & effective_loss_mask).float().sum(),
                    global_loss_token_counts,
                )

        return loss, metrics
