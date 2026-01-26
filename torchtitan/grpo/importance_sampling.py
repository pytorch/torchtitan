# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Taken from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/mismatch_helper.py

Then modified to work with our stuff
"""

from typing import Any, Optional

import torch

from torchtitan.grpo.utils import masked_mean, masked_sum


def compute_rollout_importance_weights(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_level: str = "token",
    rollout_is_mode: str = "truncate",
    rollout_is_threshold: Optional[float] = None,
    rollout_is_threshold_lower: Optional[float] = None,
    rollout_is_veto_threshold: Optional[float] = 1e-4,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    """Compute importance sampling weights and metrics for rollout-training mismatch correction.

    This function handles the computation of importance sampling (IS) weights to correct
    for the distribution mismatch between rollout policy and training policy.

    Reference:
        When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

    Memory-efficient implementation that prevents CUDA OOM by:
    - Using log-space computation where possible
    - Applying safety bounds to prevent numerical overflow
    - Computing metrics without creating huge intermediate tensors

    Args:
        old_log_prob: Log probabilities from training policy (e.g., FSDP), shape (batch_size, seq_length)
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM), shape (batch_size, seq_length)
        response_mask: Mask for valid tokens, shape (batch_size, seq_length)
        rollout_is_level: Level of IS aggregation:
            - "token": Per-token ratios (biased)
            - "sequence": Product of ratios (unbiased)
            - "geometric": Geometric mean of ratios (experimental)
        rollout_is_mode: How to handle weights exceeding threshold:
            - "truncate": Cap weights at upper_threshold only (TIS)
            - "mask": Zero out weights outside [lower_threshold, upper_threshold] (MIS)
        rollout_is_threshold: Upper threshold for IS weights
        rollout_is_threshold_lower: Lower threshold for IS weights (mask mode only; if None, defaults to 1/upper)
        rollout_is_veto_threshold: Per-token veto threshold. If any token ratio < this, zero entire sequence.
            If None, veto mechanism is disabled.

    Returns:
        Tuple of (weights_proto, metrics) where:
            weights_proto: DataProto containing IS weights with key "rollout_is_weights",
                shape (batch_size, seq_length). Returns None if rollout_is_threshold is None.
            metrics: Dictionary of IS statistics and mismatch metrics (KL, PPL, etc.),
                all converted to scalars and prefixed with "mismatch/"
    """
    if rollout_is_threshold is None:
        return None, {}

    # Parse thresholds: if lower not specified, use 1/upper (reciprocal)
    upper_threshold = rollout_is_threshold
    if rollout_is_threshold_lower is not None:
        lower_threshold = rollout_is_threshold_lower
    else:
        # Default: lower = 1/upper (reciprocal)
        lower_threshold = 1.0 / upper_threshold

    # Step 1: Compute raw importance weights based on the specified level
    log_ratio = old_log_prob - rollout_log_prob

    # Pre-compute log thresholds
    device = old_log_prob.device
    log_threshold_upper = torch.log(torch.tensor(upper_threshold, device=device))
    log_threshold_lower = torch.log(torch.tensor(lower_threshold, device=device))
    metrics = {}

    # Safety bound to prevent numerical overflow (exp(20) ≈ 485M)
    SAFETY_BOUND = 20.0

    # Store unclamped values in log-space for accurate metrics
    if rollout_is_level == "token":
        # Token-level IS: π_train(a|s) / π_rollout(a|s) per token
        log_ratio_for_metrics = log_ratio

        # Apply safety bound to prevent overflow
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_safe)

    elif rollout_is_level == "sequence":
        # Sequence-level IS: π_train(y|x) / π_rollout(y|x) for entire sequence
        # Product of token ratios: exp(Σ log(π_train/π_rollout))
        log_ratio_sum = masked_sum(log_ratio, response_mask, per_seq=True).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_sum  # Store for metrics

        # Apply safety bound to prevent overflow
        log_ratio_sum_safe = torch.clamp(
            log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND
        )
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(old_log_prob)

    elif rollout_is_level == "geometric":
        # Geometric mean IS: (∏ π_train/π_rollout)^(1/T)
        # Equivalent to exp(mean(log(π_train/π_rollout)))
        log_ratio_mean = masked_mean(log_ratio, response_mask, per_seq=True).unsqueeze(
            -1
        )
        log_ratio_for_metrics = log_ratio_mean  # Store for metrics

        # Geometric mean rarely explodes due to averaging, but apply safety bound anyway
        log_ratio_mean_safe = torch.clamp(
            log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND
        )
        rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(old_log_prob)

    else:
        raise ValueError(
            f"Invalid rollout_is_level: {rollout_is_level}. Must be 'token', 'sequence', or 'geometric'."
        )

    # Step 1.5: Apply per-token veto check in log space (memory efficient)
    if rollout_is_veto_threshold is not None:
        log_veto_threshold = torch.log(
            torch.tensor(rollout_is_veto_threshold, device=device)
        )

        # Check if any token ratio is below veto threshold (in log space)
        # log(π_train/π_rollout) < log(veto_threshold) ⟺ π_train/π_rollout < veto_threshold
        catastrophic_tokens = (log_ratio < log_veto_threshold) & response_mask.bool()

        # For each sequence, check if it has any catastrophic token
        # Use broadcasting instead of expand_as to save memory
        has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)

        # Create veto mask: 0 if sequence has catastrophic token, 1 otherwise
        veto_mask = (~has_catastrophic).float()
    else:
        # No veto mechanism
        catastrophic_tokens = torch.zeros_like(response_mask, dtype=torch.bool)
        has_catastrophic = torch.zeros(
            (old_log_prob.size(0), 1), dtype=torch.bool, device=device
        )
        veto_mask = torch.ones(
            (old_log_prob.size(0), 1), dtype=torch.float32, device=device
        )

    # Step 3: Apply truncation or masking based on mode
    if rollout_is_mode == "truncate":
        # Truncated IS (TIS): only cap upper bound to prevent overweighting
        rollout_is_weights = rollout_is_weights.clamp(max=upper_threshold)

    elif rollout_is_mode == "mask":
        # Masked IS (MIS): zero out weights outside [lower_threshold, upper_threshold]
        mask = (rollout_is_weights >= lower_threshold) & (
            rollout_is_weights <= upper_threshold
        )
        mask = mask.float()

        # Track MIS-specific metrics
        metrics["rollout_is_masked_fraction"] = masked_mean(1 - mask, response_mask)

        # Sequence-level masking fraction
        if rollout_is_level in ["sequence", "geometric"]:
            # All tokens in a sequence have the same weight, so reuse mask
            metrics["rollout_is_seq_masked_fraction"] = (1 - mask[:, 0]).mean()
        else:
            # Check if any token in each sequence is masked
            seq_has_masked = masked_sum(1 - mask, response_mask, per_seq=True) > 0
            metrics["rollout_is_seq_masked_fraction"] = seq_has_masked.float().mean()

        rollout_is_weights = rollout_is_weights * mask

    else:
        raise ValueError(
            f"Invalid rollout_is_mode: {rollout_is_mode}. Must be 'truncate' or 'mask'."
        )
    metrics["inf_logratio"] = log_ratio_for_metrics.mean()
    # Apply veto mask AFTER all thresholding
    # This zeros out entire sequences that have any catastrophic token
    rollout_is_weights = rollout_is_weights * veto_mask

    # Apply response_mask to ensure weights are 0 where mask is 0
    rollout_is_weights = rollout_is_weights * response_mask

    # Wrap in DataProto for consistency with worker methods

    # Convert all tensor metrics to scalars for logging
    # Note: No need to detach since old_log_prob and rollout_log_prob are computed with torch.no_grad()
    metrics_scalar = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[f"mismatch/{key}"] = value.item()
        else:
            metrics_scalar[f"mismatch/{key}"] = value

    return rollout_is_weights, metrics_scalar
