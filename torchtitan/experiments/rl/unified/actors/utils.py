# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def compute_token_log_probs(
    model: torch.nn.Module,
    prompt_token_ids: list[int],
    gen_token_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for generated tokens.

    Args:
        model: The model to use for computing logits
        prompt_token_ids: Token IDs for the prompt
        gen_token_ids: Token IDs for the generated completion
        device: Device to run computation on

    Returns:
        Per-token log probabilities for the generated tokens
    """
    full_sequence = prompt_token_ids + gen_token_ids
    full_tensor = torch.tensor(
        full_sequence, dtype=torch.long, device=device
    ).unsqueeze(0)

    # Forward pass — trainer uses is_position_id=False so positions=None is fine
    logits = model(full_tensor, attention_masks=None)

    # Convert to float32 for numerical stability
    logits_f32 = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    target_tokens = full_tensor[:, 1:]

    # Extract log probs for generated tokens only
    prompt_len = len(prompt_token_ids)
    gen_start_idx = prompt_len - 1
    gen_end_idx = gen_start_idx + len(gen_token_ids)

    gen_token_logprobs = log_probs[0, gen_start_idx:gen_end_idx, :]
    gen_token_ids_tensor = target_tokens[0, gen_start_idx:gen_end_idx]
    token_lps = gen_token_logprobs.gather(
        1, gen_token_ids_tensor.unsqueeze(-1)
    ).squeeze(-1)

    return token_lps


def compute_policy_gradient_loss(
    model: torch.nn.Module,
    vllm_token_ids: list[list[int]],
    prompt_token_ids: list[list[int]],
    advantages: torch.Tensor,
    ref_token_log_probs: list[torch.Tensor],
    kl_coef: float = 0.1,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict, list[torch.Tensor]]:
    """
    Compute GRPO/PPO policy gradient loss with per-token KL divergence.

    Uses per-token log ratios (averaged across tokens) instead of per-sequence
    sums to prevent ratio explosion when sequences are long.

    Args:
        model: Current policy model
        vllm_token_ids: Generated token IDs for each completion
        prompt_token_ids: Prompt token IDs for each completion
        advantages: [batch] - Advantages for each sample
        ref_token_log_probs: Per-token log probs from reference model (frozen)
        kl_coef: KL divergence penalty coefficient
        ppo_clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient

    Returns:
        loss: Total loss (PG + entropy + KL)
        metrics: Training metrics dict
        batch_token_log_probs: List of per-token log probs for each sample (for verification)
    """
    device = next(model.parameters()).device
    advantages = advantages.to(device)

    # Compute per-token log probs under current policy (WITH GRADIENTS)
    batch_token_log_probs = []

    for prompt_toks, gen_toks in zip(prompt_token_ids, vllm_token_ids):
        token_lps = compute_token_log_probs(model, prompt_toks, gen_toks, device)
        batch_token_log_probs.append(token_lps)

    # Per-token log ratios and KL, averaged across tokens per sample
    per_sample_mean_log_ratio = []
    per_sample_mean_kl = []
    all_token_log_probs = []

    for policy_token_lps, ref_token_lps in zip(
        batch_token_log_probs, ref_token_log_probs
    ):
        # Per-token log ratio: log(pi/pi_ref) for each token
        token_log_ratio = policy_token_lps - ref_token_lps.detach()
        # Average across tokens in this sequence
        per_sample_mean_log_ratio.append(token_log_ratio.mean())
        # Per-token KL: E[ratio - 1 - log_ratio] (Schulman approx)
        token_ratio = torch.exp(token_log_ratio)
        token_kl = token_ratio - 1 - token_log_ratio
        per_sample_mean_kl.append(token_kl.mean())
        all_token_log_probs.append(policy_token_lps)

    mean_log_ratio = torch.stack(per_sample_mean_log_ratio)  # [batch]
    mean_kl = torch.stack(per_sample_mean_kl)  # [batch]

    # PPO clipped objective using per-token-averaged ratio
    ratio = torch.exp(mean_log_ratio)
    unclipped_loss = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
    clipped_loss = clipped_ratio * advantages
    pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

    # Entropy bonus (averaged across all tokens)
    all_token_lps = torch.cat(all_token_log_probs)
    entropy = -all_token_lps.mean()
    entropy_bonus = -entropy_coef * entropy

    # KL divergence penalty (averaged across samples)
    kl_div = mean_kl.mean()

    # Total loss
    total_loss = pg_loss + entropy_bonus + kl_coef * kl_div

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
        .float()
        .mean()
        .item(),
    }

    return total_loss, metrics, batch_token_log_probs


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
) -> dict:
    """
    Check if vLLM log probs and computed log probs are bit-wise identical,
    and compute the log ratio (train/generator) between them.

    Args:
        vllm_token_log_probs: Per-token log probs from vLLM (generator)
        batch_token_log_probs: Per-token log probs computed by the trainer model

    Returns:
        Verification result dict with identity status, delta info, and log ratio stats
    """
    result = {
        "logprob_bitwise_identical": True,
        "num_samples_checked": len(vllm_token_log_probs),
        "total_tokens_checked": 0,
        "num_tokens_different": 0,
        "logprob_max_delta": 0.0,
        "avg_delta": 0.0,
        "logprob_diff_mean": 0.0,
        "logprob_diff_max": 0.0,
    }

    all_deltas = []
    all_log_ratios = []

    for vllm_lps, titan_lps in zip(vllm_token_log_probs, batch_token_log_probs):
        # Convert vLLM log probs to tensor
        vllm_tensor = torch.tensor(vllm_lps, dtype=torch.float32)
        # Convert titan log probs to float32 for comparison
        titan_tensor = titan_lps.detach().cpu().float()

        num_tokens = len(vllm_lps)
        result["total_tokens_checked"] += num_tokens

        # Check bitwise identity
        bitwise_match = torch.equal(vllm_tensor, titan_tensor)

        if not bitwise_match:
            result["logprob_bitwise_identical"] = False
            num_different = (vllm_tensor != titan_tensor).sum().item()
            result["num_tokens_different"] += num_different
            deltas = (vllm_tensor - titan_tensor).abs()
            all_deltas.append(deltas)

        # Log ratio: log(pi_train / pi_generator) = logprob_train - logprob_generator
        # Should be 0 when weights are identical (ratio = 1)
        all_log_ratios.append(titan_tensor - vllm_tensor)

    # Compute aggregate delta stats
    if all_deltas:
        combined_deltas = torch.cat(all_deltas)
        result["logprob_max_delta"] = combined_deltas.max().item()
        result["avg_delta"] = combined_deltas.mean().item()

    # Compute log ratio stats
    if all_log_ratios:
        combined_log_ratios = torch.cat(all_log_ratios)
        result["logprob_diff_mean"] = combined_log_ratios.mean().item()
        result["logprob_diff_max"] = combined_log_ratios.abs().max().item()

    return result
