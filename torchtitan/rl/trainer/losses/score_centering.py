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
from torchtitan.rl.trainer.losses.dapo import MAX_LOG_RATIO, normalize

# Floor for each tail mass (the probability outside the top-k): it can round to <= 0 in fp32.
_MIN_TAIL_MASS = 1e-6


class ScoreCenteringLoss(Loss[torch.Tensor]):
    """REINFORCE loss with score centering (https://arxiv.org/abs/2609.20807).

    Score centering stops the generator's small errors from steering the trainer.

    The generator (e.g. vLLM) samples tokens, and the trainer learns from them. Their
    probabilities differ a little: quantization, kernels, stale weights. Give every token the
    same advantage, 1, so the reward says nothing. A correct loss then leaves the trainer as it
    is. Below: one position, a 3-token vocab, and the trainer after one SGD step on its logits
    (lr = 1) with the gradient averaged over the generator's samples. One # is 0.05.

                            token 0              token 1              token 2
        generator           ####           0.20  ############## 0.70  ## 0.10
        trainer, before     ############   0.60  ######         0.30  ## 0.10
        after one step:
          plain REINFORCE   ########       0.42  #########      0.47  ## 0.11
          truncated IS      ###########    0.57  #######        0.33  ## 0.10
          masked IS         #############  0.67  #####          0.23  ## 0.10
          this loss         ############   0.60  ######         0.30  ## 0.10

    - At advantage 1, plain REINFORCE moves the trainer toward the generator (a negative
      advantage pushes it away). The paper calls this "drift" (Eq. 3). Each weight sync feeds
      it back into the generator, so it compounds.
    - Importance sampling (IS) weights each sampled token by `ratio`, its trainer prob over its
      generator prob. On average, each token is then trained as if its probability were its
      generator prob times its weight. Exact IS gives the trainer's probs, so nothing moves.
    - Large ratios make exact IS noisy, so the weight is capped (`max_ratio=2`: token 0 gets 2
      for its ratio 3) or masked (`ratio_band=(0.5, 5)`: token 1 gets 0 for its ratio 0.43).
      Both still move the trainer: the cap trains token 0 as if its probability were 0.2 * 2 = 0.4.
    - This loss adds a `correction` that cancels the move, in both modes.

    The loss at one position, as code. The generator also sends its top-k: its k most likely
    tokens and their logprobs. The "tail" is every other token. `weight(r)` is
    `min(r, max_ratio)` (truncated IS, the default), or `r if lo <= r <= hi else 0` when
    `ratio_band=(lo, hi)` is set (masked IS).

        ratio        = exp(trainer_logprobs - generator_logprobs)       # the sampled token
        topk_ratio   = trainer_topk_probs / generator_topk_probs        # each top-k token
        rho          = (1 - generator_topk_mass) / (1 - trainer_topk_mass)   # tail mass ratio
        token_weight = weight(ratio)                                    # no gradient
        weighted_generator_topk_probs = generator_topk_probs * weight(topk_ratio)
        alpha        = rho * weight(1 / rho)                            # the tail's weight
        coefficients = weighted_generator_topk_probs - alpha * trainer_topk_probs  # no gradient
        correction   = sum(coefficients * trainer_topk_logprobs)
        token_loss   = -advantages * (token_weight * trainer_logprobs - correction)
        loss         = sum(token_loss) / global_loss_token_counts

    In the picture, the top-k is tokens 0 and 1, and both tails are 0.1, so rho = 1, alpha = 1:

        max_ratio=2:         coefficients = [0.2 * 2, 0.7 * (3/7)] - [0.6, 0.3] = [-0.2, 0.0]
        ratio_band=(0.5, 5): coefficients = [0.2 * 3, 0.7 * 0]     - [0.6, 0.3] = [0.0, -0.3]

    A capped tail, with max_ratio=2: trainer_topk_probs = [0.5, 0.3] (tail 0.2) and
    generator_topk_probs = [0.3, 0.68] (tail 0.02). The tail gets weight 2 for its ratio 10, so
    at advantage 1, IS alone lowers the tail's logit. The correction can only use the top-k
    logprobs, and lowering them raises the tail. So tokens 0 and 1 get nonzero coefficients.

        rho = 0.02 / 0.2 = 0.1,  alpha = 0.1 * weight(10) = 0.2
        weighted_generator_topk_probs = [0.5, 0.3]   # ratios 1.67 and 0.44: exact weights
        coefficients = [0.5, 0.3] - 0.2 * [0.5, 0.3] = [0.4, 0.24]

    Why it cancels (Eq. numbers as in the arXiv PDF). For token v, `trainer_probs[v]` and
    `generator_probs[v]` are the two models' probs, `ratio[v]` is trainer over generator, and
    `score[v]` is `grad(log trainer_probs[v])`. At one position, with every advantage 1, the
    weighted update averages over the generator's samples to

        expected_weighted_score = sum_v generator_probs[v] * weight(ratio[v]) * score[v]

    - The generator sends no tail probs. The tail model (Eq. 7) assumes its tail is a scaled
      copy of the trainer's: `generator_probs[v] = rho * trainer_probs[v]` outside the top-k.
      Then every tail ratio is `1 / rho`, and each tail term is
      `alpha * trainer_probs[v] * score[v]`.
    - `sum_v trainer_probs[v] * score[v]` is `grad(1) = 0`. So the tail terms sum to
      `-alpha * sum_{v in top-k} trainer_probs[v] * score[v]`, and `expected_weighted_score`
      is `grad(correction)` (Eq. 10).
    - So under the tail model, for any advantages, `-grad(token_loss)` at sampled token y averages
      to `covariance(advantages, token_weight * score[y])`, the reward's signal (Eq. 5, 12).
    - With no capped or masked top-k or tail ratio, `coefficients` is 0. So a generator that
      matches the trainer gives plain REINFORCE.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        max_ratio: float = 2.0
        """Truncated IS (the paper's TIS+SC): `weight(r) = min(r, max_ratio)`, where `r` is a
        token's trainer prob over its generator prob. Must be >= 1. 2.0 is the paper's TIS cap
        (App. A.3). Unused when `ratio_band` is set."""

        ratio_band: tuple[float, float] | None = None
        """Masked IS, used instead of `max_ratio` when set:
        `weight(r) = r if lo <= r <= hi else 0`. `(lo, hi)` are absolute bounds on the ratio
        (unlike `DAPOLoss.ratio_clip_*`, which are offsets from 1). The same rule applies to the
        sampled token's `ratio`, to each top-k token's `topk_ratio` and to the tail's ratio
        `1 / rho`. Requires `0 < lo <= 1 <= hi`. With `lo <= 1 <= hi`, a generator that matches
        the trainer stays plain REINFORCE.
        Example: `(0.5, 5.0)`, the paper's masked-IS band (App. A.3)."""

        global_vocab_size: int | None = None
        """Full vocabulary size; required when TP shards the vocabulary."""

        def __post_init__(self) -> None:
            # Bounds that include 1 keep a generator that matches the trainer plain REINFORCE.
            if self.max_ratio < 1:
                raise ValueError(
                    f"ScoreCenteringLoss: max_ratio must be >= 1, got {self.max_ratio}."
                )
            if self.ratio_band is not None and not (
                0 < self.ratio_band[0] <= 1 <= self.ratio_band[1]
            ):
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
        # Kept apart: under TP the label's logprob comes from _LossParallelCrossEntropy, whose
        # softplus form stays nonzero for confident tokens; the top-k path is logit - logsumexp.
        trainer_topk_logprobs = compute_topk_logprobs(
            logits,
            generator_topk_token_ids,
            vocab_parallel_group=vocab_parallel_group,
            global_vocab_size=self.global_vocab_size,
        )
        effective_loss_mask = loss_mask & torch.isfinite(generator_logprobs)

        # `token_weight` and `coefficients` are constants: no gradient flows through them.
        with torch.no_grad():
            masked_log_ratio = torch.where(
                effective_loss_mask, trainer_logprobs - generator_logprobs, 0.0
            )
            ratio = torch.exp(
                torch.clamp(masked_log_ratio, -MAX_LOG_RATIO, MAX_LOG_RATIO)
            )
            trainer_topk_probs = torch.exp(trainer_topk_logprobs)
            generator_topk_probs = torch.exp(generator_topk_logprobs)
            trainer_topk_mass = trainer_topk_probs.sum(-1)
            generator_topk_mass = generator_topk_probs.sum(-1)
            rho = (1 - generator_topk_mass).clamp_min(_MIN_TAIL_MASS) / (
                1 - trainer_topk_mass
            ).clamp_min(_MIN_TAIL_MASS)
            if self.log_ratio_band is not None:
                log_low, log_high = self.log_ratio_band
                token_in_band = (masked_log_ratio >= log_low) & (
                    masked_log_ratio <= log_high
                )
                token_weight = torch.where(token_in_band, ratio, 0.0)
                # generator_topk_probs * weight(topk_ratio), without dividing by a generator prob:
                # trainer_topk_probs inside the band, 0 outside.
                topk_log_ratio = trainer_topk_logprobs - generator_topk_logprobs
                weighted_generator_topk_probs = torch.where(
                    (topk_log_ratio >= log_low) & (topk_log_ratio <= log_high),
                    trainer_topk_probs,
                    0.0,
                )
                # alpha = rho * weight(1 / rho): 1 if the tail ratio 1 / rho is in the band, else 0.
                tail_log_ratio = -rho.log()
                alpha = ((tail_log_ratio >= log_low) & (tail_log_ratio <= log_high)).to(
                    rho.dtype
                )
            else:
                token_weight = torch.clamp(ratio, max=self.max_ratio)
                # generator_topk_probs * min(topk_ratio, max_ratio), without the division, so it
                # stays finite when a generator prob underflows to 0.
                weighted_generator_topk_probs = torch.minimum(
                    trainer_topk_probs, self.max_ratio * generator_topk_probs
                )
                # alpha = rho * min(1 / rho, max_ratio) = min(1, max_ratio * rho)
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
        loss = normalize(
            (token_loss * effective_loss_mask).sum(), global_loss_token_counts
        )

        with torch.no_grad():
            # Gradient of the correction w.r.t. the logits, before the advantage.
            coefficient_sum = coefficients.sum(-1)
            topk_logit_grad = (
                coefficients - trainer_topk_probs * coefficient_sum[:, None]
            )
            tail_logit_grad = (1 - trainer_topk_mass).clamp_min(0) * coefficient_sum
            logit_correction_l1 = topk_logit_grad.abs().sum(-1) + tail_logit_grad.abs()
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": normalize(
                    (ratio * effective_loss_mask).sum(), global_loss_token_counts
                ),
                # Mean of masked_log_ratio over loss tokens (not clamped, unlike `ratio`): the k1
                # estimate of -KL(generator || trainer).
                "bit_wise/logprob_diff/mean": normalize(
                    masked_log_ratio.float().sum(), global_loss_token_counts
                ),
                "bit_wise/ratio_tokens_different/mean": normalize(
                    (masked_log_ratio.abs() > 1e-6).float().sum(),
                    global_loss_token_counts,
                ),
                "bit_wise/logprob_diff/max": masked_log_ratio.abs().max(),
                # Mean entropy of softmax(logits / temperature) over tokens used by the loss.
                "trainer/entropy/mean": normalize(
                    (token_entropy * effective_loss_mask).sum(),
                    global_loss_token_counts,
                ),
                # Generator and trainer prob mass in the top-k. At k=128 the paper sees > 99.9%
                # for the generator in most settings (App. A.4).
                "score_centering/generator_topk_mass/mean": normalize(
                    torch.where(effective_loss_mask, generator_topk_mass, 0.0).sum(),
                    global_loss_token_counts,
                ),
                "score_centering/trainer_topk_mass/mean": normalize(
                    torch.where(effective_loss_mask, trainer_topk_mass, 0.0).sum(),
                    global_loss_token_counts,
                ),
                # sum(|coefficients|): 0 when no top-k ratio and no tail ratio is capped or
                # masked, e.g. when the generator matches the trainer.
                "score_centering/coefficient_l1/mean": normalize(
                    coefficients.abs().sum(-1).sum(), global_loss_token_counts
                ),
                # L1 norm of the correction's logit gradient (topk_logit_grad, tail_logit_grad).
                # Unlike coefficient_l1, it stays ~0 when the masked tail's alpha flips between
                # 0 and 1 at a position whose top-k holds almost all of the trainer's mass.
                "score_centering/logit_correction_l1/mean": normalize(
                    logit_correction_l1.sum(), global_loss_token_counts
                ),
            }
            if self.log_ratio_band is not None:
                metrics["loss/ratio_masked_frac"] = normalize(
                    (~token_in_band & effective_loss_mask).float().sum(),
                    global_loss_token_counts,
                )
                metrics["score_centering/tail_masked_frac"] = normalize(
                    ((alpha == 0) & effective_loss_mask).float().sum(),
                    global_loss_token_counts,
                )
            else:
                metrics["loss/ratio_truncated_frac"] = normalize(
                    ((ratio > self.max_ratio) & effective_loss_mask).float().sum(),
                    global_loss_token_counts,
                )

        return loss, metrics
