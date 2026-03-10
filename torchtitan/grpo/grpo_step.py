# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from torchtitan.config.job_config import JobConfig
from torchtitan.grpo.importance_sampling import compute_rollout_importance_weights
from torchtitan.grpo.utils import masked_mean
from torchtitan.tools.logging import logger


def create_loss_fn():
    """
    Create the loss function for GRPO training.

    Returns:
        loss_fn: Function that computes cross entropy loss
    """

    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1), reduction="none"
        ).reshape(labels.shape)

    return loss_fn


def compute_logp_from_model(
    model, input_ids, labels, loss_fn, temperature=1.0, **kwargs
):
    """
    Compute log probabilities for on-policy training.

    Args:
        model: The policy model
        input_ids: Input token IDs
        labels: Target labels
        loss_fn: Loss function that computes cross entropy

    Returns:
        pred: Model predictions
        logp: Log probabilities (negative of loss)
    """
    pred = model(input_ids, **kwargs)
    logger.debug(f"pred shape: {pred.shape}")
    if temperature != 1.0:
        # Apply temperature scaling to the logits
        pred = pred / temperature
    chunked_pred = torch.chunk(pred, 16, dim=1)
    chunked_labels = torch.chunk(labels, 16, dim=1)
    chunked_logp = [
        -loss_fn(chunk, label_chunk)
        for chunk, label_chunk in zip(chunked_pred, chunked_labels)
    ]
    logp = torch.cat(chunked_logp, dim=1)
    return pred, logp


def compute_entropy_loss(pred, entropy_loss_fn, entropy_weight, device):
    """
    Compute entropy loss for the predictions.

    Args:
        pred: Model predictions [batch_size, seq_len, vocab_size]
        entropy_loss_fn: Function to compute entropy
        entropy_weight: Weight for entropy loss
        device: Device to create tensors on

    Returns:
        entropy_loss: Computed entropy loss
    """
    if entropy_weight > 0:
        chunked_pred = torch.chunk(pred, 16, dim=1)
        chunked_entropy_loss = [entropy_loss_fn(chunk) for chunk in chunked_pred]
        entropy_loss = entropy_weight * torch.cat(chunked_entropy_loss, dim=1)
    else:
        entropy_loss = torch.tensor(0.0, device=device)

    return entropy_loss


def compute_policy_ratio(logp, old_logp=None, mask=None, policy_ratio_type="token"):
    """
    Compute the policy ratio for GRPO.

    Args:
        logp: Log probabilities from current policy
        old_logp: Old log probabilities (if None, uses on-policy with detach)

    Returns:
        ratio: exp(logp - reference) where reference is old_logp or logp.detach()
    """
    if old_logp is None:
        # On-policy: use current logp with detach
        log_ratio = logp - logp.detach()
    else:
        # Off-policy: use provided old logp
        log_ratio = logp - old_logp

    if policy_ratio_type == "sequence" and mask is not None:
        # Compute sequence-level importance weights (GSPO)
        # Average log ratios across valid tokens
        seq_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True) / mask.sum(
            dim=-1, keepdim=True
        ).clamp(min=1.0)
        # Broadcast back to token level
        return torch.exp((logp - logp.detach()) + seq_log_ratio)
    else:
        # Token-level importance weights (standard GRPO)
        return torch.exp(log_ratio)


def scale_rewards(reward, pos_scaler, neg_scaler):
    """
    Scale rewards separately for positive and negative values.

    Args:
        reward: Raw reward tensor
        pos_scaler: Scaling factor for positive rewards
        neg_scaler: Scaling factor for negative rewards

    Returns:
        scaled_reward: Rewards with separate scaling applied
    """
    return ((reward > 0).float() * reward * pos_scaler) + (
        (reward <= 0).float() * reward * neg_scaler
    )


def compute_logit_loss(pred, weight, device):
    """
    Compute logit regularization loss.

    Args:
        pred: Model predictions
        weight: Weight for logit loss
        device: Device to create tensors on

    Returns:
        logit_loss: L2 regularization on logits
    """
    if weight > 0:
        return torch.mean((weight * pred) ** 2, dim=-1)
    else:
        return torch.tensor(0.0, device=device)


def compose_grpo_loss(
    reward, ratio, kl_div_est=None, kl_beta=0.0, logit_loss=None, entropy_loss=None
):
    """
    Compose the final GRPO loss from components.

    Args:
        reward: Scaled rewards
        ratio: Policy ratio
        kl_div_est: KL divergence estimate (optional)
        kl_beta: KL penalty weight
        logit_loss: Logit regularization loss (optional)
        entropy_loss: Entropy bonus loss (optional)

    Returns:
        loss: Combined GRPO loss
    """
    loss = -reward * ratio

    if kl_div_est is not None and kl_beta > 0:
        loss = loss + (kl_div_est * kl_beta)

    if logit_loss is not None:
        loss = loss + logit_loss

    if entropy_loss is not None:
        loss = loss + entropy_loss

    return loss


def compute_grpo_loss_from_predictions(
    pred,
    labels,
    reward,
    mask,
    loss_fn,
    entropy_loss_fn,
    job_config: JobConfig,
    device,
    old_logp=None,
    old_ref_logp=None,
    ref_pred=None,
    ref_model=None,
    input_ids=None,
    use_ref_model=False,
    inf_logps=None,
):
    """
    Compute full GRPO loss from model predictions.

    Args:
        pred: Model predictions
        labels: Target labels
        reward: Rewards (already scaled)
        mask: Attention mask
        loss_fn: Cross entropy loss function
        entropy_loss_fn: Entropy computation function
        job_config: Configuration object
        device: Device for tensor creation
        old_logp: Old log probabilities (if None, on-policy)
        old_ref_logp: Old reference log probabilities (for off-policy with ref model)
        ref_pred: Reference model predictions (optional)
        ref_model: Reference model (needed if ref_pred not provided and use_ref_model=True)
        input_ids: Input IDs (needed if computing ref_pred)
        use_ref_model: Whether to use reference model
        inf_logps: Precomputed log probabilities from inference model

    Returns:
        loss: GRPO loss
        metrics: Dictionary of metrics
        logp: Log probabilities (for saving in off-policy buffer)
    """
    # Compute log probabilities
    chunked_pred = torch.chunk(pred, 16, dim=1)
    chunked_labels = torch.chunk(labels, 16, dim=1)
    chunked_logp = [
        -loss_fn(chunk, label_chunk)
        for chunk, label_chunk in zip(chunked_pred, chunked_labels)
    ]
    logp = torch.cat(chunked_logp, dim=1)

    # Apply logp threshold if configured
    with torch.no_grad():
        logp_threshold_mask = torch.ones_like(mask)
        if (job_config.grpo.onpolicy_logp_threshold != 0.0) and (
            job_config.grpo.onpolicy_logp_threshold != 1.0
        ):
            threshold = job_config.grpo.onpolicy_logp_threshold
            # Convert probability to log probability if needed
            if 0 < threshold <= 1:
                threshold = torch.log(torch.tensor(threshold))
            # Create mask for tokens that meet the threshold
            logp_threshold_mask = (logp >= threshold).float()
            # Update the effective mask
            orig_mask = mask.clone()
            mask = mask * logp_threshold_mask
        else:
            orig_mask = mask  # no need to clone

    # Compute ratio
    ratio = compute_policy_ratio(
        logp, old_logp, mask, job_config.grpo.policy_ratio_type
    )

    # For off-policy, apply clipping to ratio
    if old_logp is not None:
        logger.debug("Applying ratio clipping to off-policy loss")
        coef_1 = torch.clamp(ratio, None, 1 + job_config.grpo.clip_ratio_upper_bound)
        coef_2 = torch.clamp(ratio, 1 - job_config.grpo.clip_ratio_lower_bound, None)
        # Use different clipping for positive/negative rewards
        clipped_ratio = (coef_1 * (reward > 0).float()) + (
            coef_2 * (reward <= 0).float()
        )
    else:
        logger.debug("On-policy: no ratio clipping applied")
        clipped_ratio = ratio
    if job_config.grpo.rollout_is_threshold is not None and inf_logps is not None:
        imp_ratio_mask = (inf_logps <= 0).float() * mask
        imp_ratio_logp = old_logp if old_logp is not None else logp.detach()
        importance_ratio, is_metrics = compute_rollout_importance_weights(
            imp_ratio_logp,
            inf_logps,
            imp_ratio_mask,
            rollout_is_level=job_config.grpo.rollout_is_level,
            rollout_is_mode=job_config.grpo.rollout_is_mode,
            rollout_is_threshold=job_config.grpo.rollout_is_threshold,
            rollout_is_threshold_lower=job_config.grpo.rollout_is_threshold_lower,
            rollout_is_veto_threshold=job_config.grpo.rollout_is_veto_threshold,
        )
        clipped_ratio = clipped_ratio * importance_ratio
    else:
        importance_ratio = torch.ones_like(clipped_ratio)
        is_metrics = {}
    # Compute auxiliary losses
    logit_loss = compute_logit_loss(pred, job_config.grpo.logit_loss_weight, device)
    entropy_loss = compute_entropy_loss(
        pred, entropy_loss_fn, job_config.grpo.entropy_loss_weight, device
    )

    # Handle reference model
    if use_ref_model:
        # Use old ref logp if provided (off-policy case)
        if old_ref_logp is not None:
            ref_logp = old_ref_logp
        else:
            # Compute fresh ref logp
            if ref_pred is None and ref_model is not None and input_ids is not None:
                with torch.no_grad():
                    ref_pred = ref_model(input_ids)

            if ref_pred is not None:
                ref_logp = -loss_fn(ref_pred, labels).detach()
            else:
                ref_logp = torch.tensor(0.0, device=device)

        # KL divergence
        if job_config.grpo.kl_estimator_type == "k3":
            kl_div_est = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
            kl_div_est = torch.clamp(kl_div_est, 0.0)
        elif job_config.grpo.kl_estimator_type == "mse":
            kl_div_est = 0.5 * (logp - ref_logp) ** 2
        elif job_config.grpo.kl_estimator_type == "abs":
            kl_div_est = torch.abs(logp - ref_logp)
        else:
            kl_div_est = torch.tensor(0.0, device=device)
    else:
        kl_div_est = None
        ref_logp = None

    # Compose final loss
    loss = compose_grpo_loss(
        reward=reward,
        ratio=clipped_ratio,  # Use clipped ratio for loss
        kl_div_est=kl_div_est,
        kl_beta=job_config.grpo.kl_beta if use_ref_model else 0.0,
        logit_loss=logit_loss,
        entropy_loss=entropy_loss,
    )

    # Compute metrics
    with torch.no_grad():
        # Raw ratio for on-policy
        if old_logp is None:
            raw_ratio = ratio.clone().detach()
        else:
            raw_ratio = None

        # Logits mean
        logits_mean = masked_mean(torch.mean(pred, dim=-1), mask)

        # Positive/negative masks
        pos_mask = (reward > 0).float()
        neg_mask = (reward <= 0).float()

        total_pos = int(pos_mask.sum() > 1)
        total_neg = int(neg_mask.sum() > 1)

        pos_mask = pos_mask * mask
        neg_mask = neg_mask * mask

        # Compute margin and stats
        if use_ref_model and ref_logp is not None:
            margin = reward * (logp - ref_logp)

            if pos_mask.sum() > 1:
                pos_logp = (logp * pos_mask).sum() / pos_mask.sum()
                pos_ref_logp = (ref_logp * pos_mask).sum() / pos_mask.sum()
                pos_margin = (margin * pos_mask).sum() / pos_mask.sum()
            else:
                pos_logp = torch.tensor(0.0, device=device)
                pos_ref_logp = torch.tensor(0.0, device=device)
                pos_margin = torch.tensor(0.0, device=device)

            if neg_mask.sum() > 1:
                neg_logp = (logp * neg_mask).sum() / neg_mask.sum()
                neg_ref_logp = (ref_logp * neg_mask).sum() / neg_mask.sum()
                neg_margin = (margin * neg_mask).sum() / neg_mask.sum()
            else:
                neg_logp = torch.tensor(0.0, device=device)
                neg_ref_logp = torch.tensor(0.0, device=device)
                neg_margin = torch.tensor(0.0, device=device)

            margin = (margin * mask).sum() / mask.sum()
            if ref_pred is not None:
                logits_mean_ref = masked_mean(torch.mean(ref_pred, dim=-1), mask)
            else:
                logits_mean_ref = torch.tensor(0.0, device=device)
        else:
            # No ref model
            margin = torch.tensor(0.0, device=device)
            pos_margin = torch.tensor(0.0, device=device)
            neg_margin = torch.tensor(0.0, device=device)
            pos_ref_logp = torch.tensor(0.0, device=device)
            neg_ref_logp = torch.tensor(0.0, device=device)
            logits_mean_ref = torch.tensor(0.0, device=device)

            if pos_mask.sum() > 1:
                pos_logp = (logp * pos_mask).sum() / pos_mask.sum()
            else:
                pos_logp = torch.tensor(0.0, device=device)

            if neg_mask.sum() > 1:
                neg_logp = (logp * neg_mask).sum() / neg_mask.sum()
            else:
                neg_logp = torch.tensor(0.0, device=device)

        # Compute logp statistics for monitoring
        # Use masked operations to avoid indexing issues with DTensors
        num_valid = mask.sum()

        if num_valid > 0:
            logp_mean = masked_mean(logp, orig_mask)
            # For min/max, add large values to masked positions
            # This works with DTensor arithmetic operations
            logp_for_min = (
                logp + (1 - orig_mask) * 1e10
            )  # Add large value where mask is 0
            logp_for_max = (
                logp - (1 - orig_mask) * 1e10
            )  # Subtract large value where mask is 0
            logp_min = logp_for_min.min()
            logp_max = logp_for_max.max()
            if inf_logps is not None:
                inf_logp_mean = masked_mean(inf_logps, orig_mask)
                inf_logp_for_min = inf_logps + (1 - orig_mask) * 1e10
                inf_logp_for_max = inf_logps - (1 - orig_mask) * 1e10
                inf_logp_min = inf_logp_for_min.min()
                inf_logp_max = inf_logp_for_max.max()
            else:
                inf_logp_mean = torch.tensor(0.0, device=device)
                inf_logp_min = torch.tensor(0.0, device=device)
                inf_logp_max = torch.tensor(0.0, device=device)
        else:
            logp_mean = torch.tensor(0.0, device=device)
            logp_min = torch.tensor(0.0, device=device)
            logp_max = torch.tensor(0.0, device=device)
            inf_logp_mean = torch.tensor(0.0, device=device)
            inf_logp_min = torch.tensor(0.0, device=device)
            inf_logp_max = torch.tensor(0.0, device=device)
        threshold_filter_ratio = 1.0 - masked_mean(logp_threshold_mask, orig_mask)

        # Collect all metrics
        metrics = {
            "loss_metrics/global_logp": masked_mean(logp, orig_mask, per_seq=True),
            "loss_metrics/global_ratio": masked_mean(ratio, mask, per_seq=True),
            "loss_metrics/global_logit_loss": masked_mean(
                logit_loss, mask, per_seq=True
            ),
            "loss_metrics/global_entropy_loss": masked_mean(
                entropy_loss, mask, per_seq=True
            ),
            "loss_metrics/global_logits_mean": logits_mean,
            "loss_metrics/global_total_pos": total_pos,
            "loss_metrics/global_total_neg": total_neg,
            "loss_metrics/global_pos_logp": masked_mean(
                pos_logp, orig_mask, per_seq=True
            ),
            "loss_metrics/global_neg_logp": masked_mean(
                neg_logp, orig_mask, per_seq=True
            ),
            "loss_metrics/global_pos_reward": (reward > 0).float().mean(),
            "loss_metrics/global_neg_reward": (reward <= 0).float().mean(),
            "loss_metrics/global_advantages": reward.mean(),
            # Logp distribution metrics
            "logp_dist/mean": logp_mean,
            "logp_dist/min": logp_min,
            "logp_dist/max": logp_max,
            "logp_dist/threshold_filter_ratio": threshold_filter_ratio,
        }
        metrics.update(is_metrics)

        if inf_logps is not None:
            metrics.update(
                {
                    "inflogp_dist/mean": inf_logp_mean,
                    "inflogp_dist/min": inf_logp_min,
                    "inflogp_dist/max": inf_logp_max,
                    "inflogp_dist/importance_ratio": masked_mean(
                        importance_ratio, mask
                    ),
                }
            )

        if raw_ratio is not None:
            metrics["raw_ratio"] = masked_mean(raw_ratio, mask)

        if use_ref_model and ref_logp is not None:
            metrics.update(
                {
                    "loss_metrics/ref_logp": masked_mean(ref_logp, orig_mask),
                    "loss_metrics/kl_div_est": masked_mean(kl_div_est, mask),
                    "loss_metrics/margin": masked_mean(margin, mask),
                    "loss_metrics/pos_margin": pos_margin,
                    "loss_metrics/neg_margin": neg_margin,
                    "loss_metrics/pos_ref_logp": pos_ref_logp,
                    "loss_metrics/neg_ref_logp": neg_ref_logp,
                    "loss_metrics/logits_mean_ref": logits_mean_ref,
                }
            )

        # For off-policy, add clipping metrics
        if old_logp is not None:
            coef_x = ratio  # It's already the ratio exp(logp - old_logp)
            is_clipped_neg = (coef_x < 1 - job_config.grpo.clip_ratio_lower_bound) & (
                reward <= 0
            )
            is_clipped_pos = (coef_x > 1 + job_config.grpo.clip_ratio_upper_bound) & (
                reward > 0
            )
            is_clipped = is_clipped_neg | is_clipped_pos

            metrics.update(
                {
                    "loss_metrics/global_clip_ratio": masked_mean(is_clipped, mask),
                    "loss_metrics/global_clip_ratio_pos": masked_mean(
                        is_clipped_pos, mask
                    ),
                    "loss_metrics/global_clip_ratio_neg": masked_mean(
                        is_clipped_neg, mask
                    ),
                    "loss_metrics/global_offp_logp": masked_mean(logp, mask),
                    "loss_metrics/global_offp_ref_logp": masked_mean(ref_logp, mask)
                    if ref_logp is not None
                    else torch.tensor(0.0, device=device),
                }
            )

    return loss, metrics, logp


@dataclass
class GRPOPPLossContext:
    """Mutable state for the GRPO PP loss closure.

    Updated before each pp_schedule.step() call with the current nanobatch's
    auxiliary data.  The PP schedule splits the batch into ``n_microbatches``
    chunks and calls the loss closure once per chunk.  We pre-chunk the
    context tensors to match and use an internal counter to track which
    chunk we're on.

    Fields split into three groups:
      - Config/utilities (set once at init)
      - Per-nanobatch data (updated via ``update()`` before each step)
      - Internal chunk tracking
    """

    # ── Config & utilities (set once at init) ──────────────────────
    loss_fn: Callable = None
    entropy_loss_fn: Callable = None
    job_config: JobConfig = None
    device: torch.device = None
    use_ref_model: bool = False
    grpo_by_token: bool = False

    # ── Per-nanobatch scalars ──────────────────────────────────────
    dynamic_scale: float = 1.0
    dynamic_grad_accum_size: float = 1.0
    total_masked_tokens: int = 1

    # ── Chunk tracking (internal) ──────────────────────────────────
    _tensor_chunks: dict = field(default_factory=dict)
    _n_microbatches: int = 1
    _mb_idx: int = 0
    _chunk_metrics: list = field(default_factory=list)

    # ── Output (merged after pp_schedule.step) ─────────────────────
    metrics: dict = field(default_factory=dict)

    # Tensor field names that get chunked along the batch dimension
    _TENSOR_FIELDS = frozenset(
        {
            "mask",
            "reward",
            "old_logp",
            "old_ref_logp",
            "inf_logps",
            "input_ids",
        }
    )

    def update(self, n_microbatches: int = 1, **kwargs):
        """Update per-nanobatch fields, pre-chunking tensors for PP microbatches."""
        self._n_microbatches = n_microbatches
        self._mb_idx = 0
        self._chunk_metrics = []

        for k, v in kwargs.items():
            if k in self._TENSOR_FIELDS:
                if v is not None and isinstance(v, Tensor) and v.dim() >= 1:
                    self._tensor_chunks[k] = list(v.chunk(n_microbatches, dim=0))
                else:
                    self._tensor_chunks[k] = [v] * n_microbatches
            else:
                setattr(self, k, v)

    def _get(self, name: str):
        """Get the current microbatch chunk for a tensor field."""
        return self._tensor_chunks[name][self._mb_idx]

    def merge_chunk_metrics(self) -> dict:
        """Merge metrics across PP microbatch chunks.

        Simple averaging for normal metrics, min/max for extremes.
        """
        if not self._chunk_metrics:
            return {}
        if len(self._chunk_metrics) == 1:
            self.metrics = self._chunk_metrics[0]
            return self.metrics

        merged = {}
        all_keys = set()
        for m in self._chunk_metrics:
            all_keys.update(m.keys())

        for key in sorted(all_keys):
            values = [m[key] for m in self._chunk_metrics if key in m]
            if not values:
                continue
            if "min" in key:
                merged[key] = min(values)
            elif "max" in key:
                merged[key] = max(values)
            else:
                merged[key] = sum(values) / len(values)

        self.metrics = merged
        return merged


def create_grpo_pp_loss_fn(ctx: GRPOPPLossContext) -> Callable:
    """Create a closure-based loss_fn for the PP schedule.

    Returns a function with signature ``loss_fn(pred, labels) -> scalar_loss``
    that computes the full GRPO loss using data from *ctx*.  The PP schedule
    calls this on the **last stage only**, once per PP microbatch chunk.

    Scaling notes:
      - ``rescale_accumulated_loss`` wraps this and divides by ``n_microbatches``
      - Per-sequence: ``mean(chunk)`` is correct — averaging chunks then dividing
        by n_microbatches equals the global mean.
      - Per-token: we use ``total_masked_tokens / n_microbatches`` as denominator
        so that after the rescale division the effective denominator is
        ``total_masked_tokens``.
    """

    def grpo_pp_loss_fn(pred, labels):
        chunk_mask = ctx._get("mask")
        chunk_reward = ctx._get("reward")

        mb_loss, metrics, _ = compute_grpo_loss_from_predictions(
            pred=pred,
            labels=labels,
            reward=chunk_reward,
            mask=chunk_mask,
            loss_fn=ctx.loss_fn,
            entropy_loss_fn=ctx.entropy_loss_fn,
            job_config=ctx.job_config,
            device=ctx.device,
            old_logp=ctx._get("old_logp"),
            old_ref_logp=ctx._get("old_ref_logp"),
            ref_pred=None,
            ref_model=None,
            input_ids=ctx._get("input_ids"),
            use_ref_model=ctx.use_ref_model,
            inf_logps=ctx._get("inf_logps"),
        )

        # Apply GRPO-specific reduction
        if not ctx.grpo_by_token:
            # Per-sequence: mean across sequences in this chunk.
            # rescale_accumulated_loss divides by n_microbatches, and
            # sum_chunks(mean_chunk) / n_microbatches = global_mean. ✓
            scalar_loss = (mb_loss * chunk_mask).sum(-1) / chunk_mask.sum(-1)
            scalar_loss = scalar_loss.mean()
            scalar_loss = ctx.dynamic_scale * scalar_loss / ctx.dynamic_grad_accum_size
        else:
            # Per-token: use total / n_microbatches as denominator so that
            # after rescale division the effective denominator is total. ✓
            scalar_loss = (mb_loss * chunk_mask).sum() / (
                ctx.total_masked_tokens / ctx._n_microbatches
            )

        ctx._chunk_metrics.append(metrics)
        ctx._mb_idx += 1
        del pred  # free memory before backward
        return scalar_loss

    return grpo_pp_loss_fn
