# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torchtitan.config.job_config import JobConfig
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


def compute_logp_from_model(model, input_ids, labels, loss_fn, temperature=1.0):
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
    pred = model(input_ids)
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


def compute_policy_ratio(logp, old_logp=None):
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
        return torch.exp(logp - logp.detach())
    else:
        # Off-policy: use provided old logp
        return torch.exp(logp - old_logp)


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
    ratio = compute_policy_ratio(logp, old_logp)

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
        else:
            logp_mean = torch.tensor(0.0, device=device)
            logp_min = torch.tensor(0.0, device=device)
            logp_max = torch.tensor(0.0, device=device)
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
