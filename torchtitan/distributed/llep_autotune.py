# Copyright (c) Nous Research and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LLEP Autotune: automatically finds optimal LLEP hyperparameters (α, m, λ)
before training starts.

Approach:
1. Run a few forward-only passes on real training data to collect routing stats
2. Use the real expert_counts to simulate LPT plans for different α values
3. Pick the (α, λ) that best balances GPU loads while minimizing P2P transfers
4. Set m analytically from model dimensions

The entire autotune runs in <30 seconds and uses no extra GPU memory beyond
a single forward pass (no backward, no optimizer states).

Reference: "Least-Loaded Expert Parallelism: Load Balancing An Imbalanced
Mixture-of-Experts" (Nguyen et al., Salesforce AI Research)
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


@dataclass
class LLEPAutotuneResult:
    """Result of LLEP autotuning."""

    max_tokens_factor: float  # α
    min_tokens_per_gemm: int  # m
    adaptive_threshold: float  # λ
    imbalance_p50: float
    imbalance_p90: float
    predicted_balance: float  # max/mean after LLEP with chosen α
    predicted_transfers: float  # avg transfers per layer
    predicted_memory_overhead_gb: float
    should_enable_llep: bool  # whether LLEP is worth enabling at all


def collect_routing_stats(
    model: torch.nn.Module,
    dataloader,
    num_samples: int = 3,
    device: str = "cuda",
) -> list[dict[str, torch.Tensor]]:
    """
    Run forward-only passes to collect per-layer expert routing counts.

    Hooks into every MoE module's router to capture expert_counts without
    modifying the model. Returns a list of dicts (one per sample), where
    each dict maps layer_name -> expert_counts tensor.
    """
    from torchtitan.models.moe.moe import MoE

    all_stats: list[dict[str, torch.Tensor]] = []

    # Find all MoE modules and their routers
    moe_modules: list[tuple[str, MoE]] = []
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            moe_modules.append((name, module))

    if not moe_modules:
        logger.info("[LLEP autotune] No MoE modules found, skipping")
        return []

    logger.info(
        f"[LLEP autotune] Found {len(moe_modules)} MoE layers, "
        f"collecting routing stats from {num_samples} samples..."
    )

    # Hook to capture expert counts from router output
    captured_counts: dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str, num_experts: int):
        def hook_fn(module, input, output):
            # Router returns (top_scores, selected_experts_indices, num_tokens_per_expert)
            if isinstance(output, tuple) and len(output) >= 3:
                num_tokens_per_expert = output[2]
                captured_counts[layer_name] = num_tokens_per_expert.detach().clone()

        return hook_fn

    data_iter = iter(dataloader)
    for sample_idx in range(num_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.warning(
                f"[LLEP autotune] Dataloader exhausted after {sample_idx} samples"
            )
            break

        input_dict, labels = batch
        # Move to device
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.to(device)

        # Register hooks on routers
        captured_counts.clear()
        hooks = []
        for layer_name, moe_module in moe_modules:
            h = moe_module.router.register_forward_hook(
                make_hook(layer_name, moe_module.experts.num_experts)
            )
            hooks.append(h)

        # Forward only, no grad
        with torch.no_grad():
            try:
                inputs = input_dict["input"]
                model(inputs)
            except Exception as e:
                logger.warning(f"[LLEP autotune] Forward pass failed: {e}")
                for h in hooks:
                    h.remove()
                break

        for h in hooks:
            h.remove()

        if captured_counts:
            all_stats.append(dict(captured_counts))
            logger.info(
                f"[LLEP autotune] Sample {sample_idx + 1}/{num_samples}: "
                f"collected counts for {len(captured_counts)} layers"
            )

    return all_stats


def _compute_imbalance_ratio(
    expert_counts: torch.Tensor,
    ep_size: int,
    num_local_experts: int,
) -> float:
    """Compute max_gpu_load / mean_gpu_load from expert counts."""
    effective = ep_size * num_local_experts
    counts = (
        expert_counts[:effective]
        if expert_counts.size(0) > effective
        else expert_counts
    )
    gpu_loads = counts.view(ep_size, num_local_experts).sum(dim=1).float()
    mean_load = gpu_loads.mean()
    if mean_load == 0:
        return 1.0
    return (gpu_loads.max() / mean_load).item()


def find_optimal_params(
    all_stats: list[dict[str, torch.Tensor]],
    ep_size: int,
    num_local_experts: int,
    num_experts: int,
    expert_weight_bytes: int,
    available_memory_gb: float,
    alpha_candidates: list[float] | None = None,
) -> LLEPAutotuneResult:
    """
    Simulate LPT plans for different α values using real routing stats.
    Pick the best (α, λ) that balances loads while minimizing transfers.

    Args:
        all_stats: Routing stats from collect_routing_stats
        ep_size: Number of EP ranks
        num_local_experts: Experts per GPU
        num_experts: Total experts
        expert_weight_bytes: Size of one expert's weights in bytes (w1+w2+w3)
        available_memory_gb: Free GPU memory in GB
        alpha_candidates: α values to try (default: [0.9, 1.0, 1.1, 1.2, 1.5])
    """
    from torchtitan.distributed.llep import compute_llep_lpt_plan

    if alpha_candidates is None:
        alpha_candidates = [0.9, 1.0, 1.1, 1.2, 1.5]

    # Step 1: Compute imbalance ratios across all layers and samples
    all_imbalance_ratios = []
    all_expert_counts = []  # flattened list of (layer, sample) expert_counts

    for step_stats in all_stats:
        for layer_name, counts in step_stats.items():
            ratio = _compute_imbalance_ratio(counts, ep_size, num_local_experts)
            all_imbalance_ratios.append(ratio)
            all_expert_counts.append(counts)

    if not all_imbalance_ratios:
        logger.info("[LLEP autotune] No routing stats collected, using defaults")
        return LLEPAutotuneResult(
            max_tokens_factor=1.1,
            min_tokens_per_gemm=1024,
            adaptive_threshold=0.0,
            imbalance_p50=1.0,
            imbalance_p90=1.0,
            predicted_balance=1.0,
            predicted_transfers=0,
            predicted_memory_overhead_gb=0,
            should_enable_llep=False,
        )

    imbalance_arr = np.array(all_imbalance_ratios)
    p50 = float(np.percentile(imbalance_arr, 50))
    p90 = float(np.percentile(imbalance_arr, 90))

    logger.info(
        f"[LLEP autotune] Imbalance stats across {len(all_imbalance_ratios)} "
        f"layer-samples: P50={p50:.2f}x, P90={p90:.2f}x, "
        f"max={imbalance_arr.max():.2f}x"
    )

    # Step 2: If routing is well-balanced, LLEP is not worth it
    if p90 < 1.1:
        logger.info(
            "[LLEP autotune] Routing is well-balanced (P90 < 1.1x), "
            "LLEP would only add overhead. Recommending LLEP disabled."
        )
        return LLEPAutotuneResult(
            max_tokens_factor=1.1,
            min_tokens_per_gemm=1024,
            adaptive_threshold=999.0,  # effectively disabled
            imbalance_p50=p50,
            imbalance_p90=p90,
            predicted_balance=p50,
            predicted_transfers=0,
            predicted_memory_overhead_gb=0,
            should_enable_llep=False,
        )

    # Step 3: Set λ from observed imbalance
    # Trigger LLEP when imbalance exceeds P50 * 0.9 (catch most imbalanced steps)
    recommended_lambda = max(1.0, p50 * 0.9)

    # Step 4: Simulate α candidates using worst-case-per-layer scoring
    #
    # Priority order:
    #   1. Memory safety (hard reject if OOM)
    #   2. Balance (minimize worst-case imbalance across layers)
    #   3. Communication (minimize transfers, as tiebreaker)
    #
    # We evaluate per-layer independently and use worst-case for selection,
    # because all layers share GPU memory and the slowest layer sets step time.

    expert_weight_gb = expert_weight_bytes / 1e9
    BALANCE_THRESHOLD = 1.1  # max/mean ≤ 1.1x is "well balanced"

    # Group expert_counts by layer (across samples)
    layer_names = list(all_stats[0].keys()) if all_stats else []
    counts_by_layer: dict[str, list[torch.Tensor]] = {}
    for step_stats in all_stats:
        for layer_name, counts in step_stats.items():
            if layer_name not in counts_by_layer:
                counts_by_layer[layer_name] = []
            counts_by_layer[layer_name].append(counts)

    best_alpha = alpha_candidates[-1]  # fallback to most conservative
    best_worst_balance = float("inf")
    best_total_transfers = float("inf")
    best_memory_gb = 0.0
    best_is_acceptable = False

    for alpha in alpha_candidates:
        worst_balance = 0.0  # worst across layers
        worst_foreign = 0  # worst foreign experts on any GPU across layers
        total_transfers = 0  # sum across all layer-samples

        for layer_name, layer_counts_list in counts_by_layer.items():
            for counts in layer_counts_list:
                ratio = _compute_imbalance_ratio(counts, ep_size, num_local_experts)
                if ratio < recommended_lambda:
                    # LLEP would skip — standard EP balance applies
                    worst_balance = max(worst_balance, ratio)
                    continue

                plan = compute_llep_lpt_plan(
                    global_expert_counts=counts,
                    ep_size=ep_size,
                    ep_rank=0,
                    num_local_experts=num_local_experts,
                    max_tokens_factor=alpha,
                    min_tokens_per_gemm=1024,
                )

                gpu_loads = plan.gpu_loads.float()
                mean_load = gpu_loads.mean()
                balance = (gpu_loads.max() / mean_load).item() if mean_load > 0 else 1.0
                worst_balance = max(worst_balance, balance)
                total_transfers += len(plan.weight_transfers)

                # Track worst-case foreign experts on any single GPU
                foreign_per_gpu: dict[int, int] = {}
                for wt in plan.weight_transfers:
                    foreign_per_gpu[wt.dst_rank] = (
                        foreign_per_gpu.get(wt.dst_rank, 0) + 1
                    )
                if foreign_per_gpu:
                    worst_foreign = max(worst_foreign, max(foreign_per_gpu.values()))

        memory_overhead_gb = worst_foreign * expert_weight_gb
        n_samples = sum(len(v) for v in counts_by_layer.values())
        avg_transfers = total_transfers / max(n_samples, 1)
        is_acceptable = worst_balance <= BALANCE_THRESHOLD

        # Priority 1: Memory safety (hard reject)
        if memory_overhead_gb > available_memory_gb * 0.8:
            logger.info(
                f"[LLEP autotune] α={alpha}: REJECTED "
                f"(memory +{memory_overhead_gb:.1f}GB > "
                f"{available_memory_gb * 0.8:.1f}GB budget)"
            )
            continue

        logger.info(
            f"[LLEP autotune] α={alpha}: "
            f"worst_balance={worst_balance:.3f}x, "
            f"avg_transfers={avg_transfers:.1f}/layer, "
            f"memory=+{memory_overhead_gb:.1f}GB, "
            f"{'✓ balanced' if is_acceptable else '✗ imbalanced'}"
        )

        # Priority 2 & 3: Balance first, then comm as tiebreaker
        should_replace = False
        if is_acceptable and not best_is_acceptable:
            # First acceptable candidate — always take it
            should_replace = True
        elif is_acceptable and best_is_acceptable:
            # Both acceptable — prefer fewer transfers (less comm)
            if total_transfers < best_total_transfers:
                should_replace = True
        elif not is_acceptable and not best_is_acceptable:
            # Neither acceptable — prefer better balance
            if worst_balance < best_worst_balance:
                should_replace = True
            elif (
                worst_balance == best_worst_balance
                and total_transfers < best_total_transfers
            ):
                should_replace = True

        if should_replace:
            best_alpha = alpha
            best_worst_balance = worst_balance
            best_total_transfers = total_transfers
            best_memory_gb = memory_overhead_gb
            best_is_acceptable = is_acceptable

    best_avg_transfers = best_total_transfers / max(
        sum(len(v) for v in counts_by_layer.values()), 1
    )

    return LLEPAutotuneResult(
        max_tokens_factor=best_alpha,
        min_tokens_per_gemm=1024,  # keep default, hardware-independent at scale
        adaptive_threshold=recommended_lambda,
        imbalance_p50=p50,
        imbalance_p90=p90,
        predicted_balance=best_worst_balance,
        predicted_transfers=best_avg_transfers,
        predicted_memory_overhead_gb=best_memory_gb,
        should_enable_llep=True,
    )


def autotune_llep(
    model: torch.nn.Module,
    dataloader,
    job_config,
    ep_group=None,
) -> LLEPAutotuneResult | None:
    """
    Top-level LLEP autotune entry point.

    Runs before training to find optimal LLEP hyperparameters.
    Applies the result to all MoE modules in the model.

    Args:
        model: The model (already parallelized)
        dataloader: Training dataloader
        job_config: Job configuration
        ep_group: Expert parallel process group (optional, auto-detected from model)
    """
    from torchtitan.models.moe.moe import MoE

    start_time = time.time()
    logger.info("[LLEP autotune] Starting LLEP autotuning...")

    num_samples = 3
    if hasattr(job_config, "llep") and hasattr(job_config.llep, "autotune_samples"):
        samples = job_config.llep.autotune_samples
        if samples is not None:
            num_samples = samples

    # Find MoE modules to get model info
    moe_modules: list[tuple[str, MoE]] = []
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            moe_modules.append((name, module))

    if not moe_modules:
        logger.info("[LLEP autotune] No MoE modules found, skipping")
        return None

    first_moe = moe_modules[0][1]
    num_experts = first_moe.experts.num_experts

    # Get EP info from the first MoE module
    if ep_group is None:
        ep_group = first_moe._ep_group

    if ep_group is None:
        logger.warning("[LLEP autotune] No EP group found, skipping")
        return None

    ep_size = dist.get_world_size(group=ep_group)
    num_local_experts = num_experts // ep_size

    # Expert weight size: w1 + w2 + w3 (SwiGLU)
    w1_shape = first_moe.experts.w1.shape  # may be DTensor
    if hasattr(w1_shape, "__len__") and len(w1_shape) >= 2:
        # Shape is (num_local_experts, hidden_dim, dim) or similar
        dim = w1_shape[-1] if len(w1_shape) == 3 else w1_shape[-1]
        hidden_dim = w1_shape[-2] if len(w1_shape) == 3 else w1_shape[-2]
    else:
        dim = 7168  # fallback
        hidden_dim = 2048
    dtype_bytes = 2  # bf16
    expert_weight_bytes = 3 * dim * hidden_dim * dtype_bytes

    # Available GPU memory
    free_mem, total_mem = torch.cuda.mem_get_info()
    available_memory_gb = free_mem / 1e9

    logger.info(
        f"[LLEP autotune] Model: {num_experts} experts, EP={ep_size}, "
        f"{num_local_experts} experts/GPU, "
        f"expert_weight={expert_weight_bytes / 1e6:.1f}MB, "
        f"available_memory={available_memory_gb:.1f}GB"
    )

    # Step 1: Collect routing stats
    stats = collect_routing_stats(model, dataloader, num_samples)

    if not stats:
        logger.warning("[LLEP autotune] No routing stats collected, using defaults")
        return None

    # Step 2: Find optimal params
    result = find_optimal_params(
        all_stats=stats,
        ep_size=ep_size,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        expert_weight_bytes=expert_weight_bytes,
        available_memory_gb=available_memory_gb,
    )

    elapsed = time.time() - start_time

    # Step 3: Apply to all MoE modules
    for name, moe_module in moe_modules:
        moe_module._llep_max_tokens_factor = result.max_tokens_factor
        moe_module._llep_min_tokens_per_gemm = result.min_tokens_per_gemm
        moe_module._llep_adaptive_threshold = result.adaptive_threshold
        if result.should_enable_llep and not moe_module._llep_enabled:
            if not moe_module.use_llep:
                logger.info(
                    f"[LLEP autotune] Recommending LLEP for {name} "
                    f"but use_llep=False in config. Set [llep] enabled=true to use."
                )

    # Compute speedup: before (standard EP) vs after (LLEP with tuned params)
    before_imbalance = result.imbalance_p90
    after_imbalance = result.predicted_balance
    speedup = before_imbalance / after_imbalance if after_imbalance > 0 else 1.0

    logger.info(
        f"[LLEP autotune] Completed in {elapsed:.1f}s\n"
        f"  Without LLEP: worst imbalance = {result.imbalance_p50:.2f}x (P50), "
        f"{result.imbalance_p90:.2f}x (P90)\n"
        f"  With LLEP (α={result.max_tokens_factor}, λ={result.adaptive_threshold:.2f}): "
        f"worst imbalance = {result.predicted_balance:.3f}x, "
        f"{result.predicted_transfers:.1f} transfers/layer, "
        f"+{result.predicted_memory_overhead_gb:.1f}GB memory\n"
        f"  Improvement: {before_imbalance:.2f}x → {after_imbalance:.3f}x imbalance "
        f"({speedup:.2f}x less straggling)\n"
        f"  Config: α={result.max_tokens_factor}, m={result.min_tokens_per_gemm}, "
        f"λ={result.adaptive_threshold:.2f}\n"
        f"  LLEP recommended: {result.should_enable_llep}"
    )

    return result
