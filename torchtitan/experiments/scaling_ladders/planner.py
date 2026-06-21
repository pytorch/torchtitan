# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turn a policy + compute spec into a resolved plan and a ``Trainer.Config``.

``resolve()`` does the pure arithmetic (no process groups, no GPU) and is shared
by ``plan()`` (dry-run dict) and ``trainer_config()`` so they can never disagree.
The period table is built once and drives BOTH the scheduler periods and the
checkpoint steps, so the pre-decay checkpoint lands exactly where decay begins.
"""

import dataclasses
from dataclasses import dataclass, field, replace

from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config.configs import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .checkpoint import LadderCheckpointManager
from .lr_scheduler import WSDSScheduler
from .model import count_ladder_params, count_total_params, model_registry, RUNGS
from .policy import WSDSChinchillaPolicy
from .validate import LadderValidator

# Untied input embedding gets weight_decay=0; lm_head keeps the policy decay.
_EMBEDDING_PATTERN = r"tok_embeddings"
# Warn when the rounded batch deviates from the scaling-law target by this much.
_BATCH_TOLERANCE = 0.10
# Warn when actual params deviate from the rung's nominal label by this much.
_PARAM_TOLERANCE = 0.05


@dataclass(kw_only=True)
class ComputeSpec:
    """Per-rung compute budget. ``world_size`` must match NGPU at launch."""

    world_size: int
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    local_batch_size: int = 1


# Per-parameter memory under bf16 mixed precision with AdamW: bf16 param (2) +
# grad (2) + fp32 master/exp_avg/exp_avg_sq (12). FSDP shards this by dp_shard.
_BYTES_PER_PARAM = 18
# Shard only enough that the model+optimizer+grad state takes at most this
# fraction of the memory budget, leaving the rest for activations. Below this,
# leftover ranks become data-parallel REPLICAS (DDP) rather than FSDP shards --
# cheap all-reduce instead of per-layer all-gather, which is the right choice
# whenever the model comfortably fits on one GPU.
_MODEL_MEM_FRAC = 0.30


def _divisors_ascending(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is <= ``cap`` (>= 1)."""
    cap = max(1, min(cap, n))
    return next(d for d in range(cap, 0, -1) if n % d == 0)


def auto_compute_spec(
    rung: str,
    policy: WSDSChinchillaPolicy,
    *,
    gpus: int = 8,
    gpu_mem_gib: float = 183.0,
    mem_frac: float = 0.85,
) -> ComputeSpec:
    """Derive a ComputeSpec from the rung's memory footprint -- no hardcoding.

    Picks the *least* sharding (``dp_shard``) whose model+optimizer state fits in
    ``_MODEL_MEM_FRAC`` of the budget, makes the leftover ranks data-parallel
    replicas (DDP for small rungs, which fit on one GPU), and aims for a single
    gradient-accumulation step (``local_batch_size = global_seqs / dp_degree``).
    The activation footprint is hard to predict analytically, so the launch-time
    OOM probe (see the runner) trims ``local_batch_size`` if this is optimistic.
    """
    model_gib = count_total_params(rung) * _BYTES_PER_PARAM / 2**30
    budget = gpu_mem_gib * mem_frac
    dp_shard = next(
        (
            s
            for s in _divisors_ascending(gpus)
            if model_gib / s <= budget * _MODEL_MEM_FRAC
        ),
        gpus,
    )
    dp_replicate = gpus // dp_shard
    dp_degree = dp_replicate * dp_shard
    target_seqs = round(
        policy.target_token_batch(count_ladder_params(rung)) / policy.seq_len
    )
    local_batch_size = max(1, round(target_seqs / dp_degree))
    return ComputeSpec(
        world_size=gpus,
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=dp_replicate,
            data_parallel_shard_degree=dp_shard,
        ),
        local_batch_size=local_batch_size,
    )


@dataclass(kw_only=True)
class ResolvedPlan:
    """All derived quantities for one run; serializes to the plan/JSON dict."""

    rung: str
    ladder_params: int
    dp_degree: int
    seq_len: int
    target_token_batch: int
    global_batch_size: int
    actual_token_batch: int
    local_batch_size: int
    grad_accum_steps: int
    steps: int
    warmup_steps: int
    peak_lr: float
    beta2: float
    weight_decay: float
    decay_fraction: float
    chinchilla_multiple: float
    chinchilla_periods: list[float]
    period_lengths: list[int]
    period_lr_multipliers: list[float] | None
    pre_decay_steps: list[int]
    post_decay_steps: list[int]
    checkpoint_steps: list[int]
    seed: int


def _audit_params(rung: str, N: int) -> None:
    nominal = RUNGS[rung].nominal_params
    if nominal is None:
        return
    diff = abs(N - nominal) / nominal
    if diff > _PARAM_TOLERANCE:
        logger.warning(
            f"Rung {rung}: ladder_params {N:,} differs from nominal "
            f"{nominal:,} by {diff:.1%} (> {_PARAM_TOLERANCE:.0%})."
        )


def resolve(
    rung: str, policy: WSDSChinchillaPolicy, compute: ComputeSpec, *, seed: int
) -> ResolvedPlan:
    N = count_ladder_params(rung)
    _audit_params(rung, N)

    # dp_degree is pure arithmetic (no process group; init_device_mesh is lazy).
    pd = ParallelDims.from_config(compute.parallelism, compute.world_size)
    dp_degree = pd.dp_replicate * pd.dp_shard

    seq_len = policy.seq_len
    target_token_batch = policy.target_token_batch(N)
    # The policy sets the global batch (in sequences), independent of lbs:
    # seqs_per_dp is the sequences each data-parallel rank sees per optimizer step.
    seqs_per_dp = max(1, round(target_token_batch / seq_len / dp_degree))
    global_batch_size = seqs_per_dp * dp_degree
    actual_token_batch = global_batch_size * seq_len
    # local_batch_size is a pure memory/throughput knob: it must divide seqs_per_dp
    # so gradient accumulation is integral. Snap a requested lbs down to the largest
    # such divisor, so the OOM probe can shrink it without moving the schedule.
    local_batch_size = _largest_divisor_leq(seqs_per_dp, compute.local_batch_size)
    grad_accum_steps = seqs_per_dp // local_batch_size
    batch_diff = abs(actual_token_batch - target_token_batch) / target_token_batch
    if batch_diff > _BATCH_TOLERANCE:
        logger.warning(
            f"Rung {rung}: actual token batch {actual_token_batch:,} differs from "
            f"target {target_token_batch:,} by {batch_diff:.1%} (> "
            f"{_BATCH_TOLERANCE:.0%}); consider a smaller local_batch_size."
        )

    peak_lr = policy.peak_lr(N)
    beta2 = policy.beta2(actual_token_batch)

    # Cumulative Chinchilla period ends in steps; period_lengths are the
    # incremental spans. Deriving steps from the same table keeps the scheduler
    # decay boundaries and the checkpoint steps aligned.
    periods = policy.chinchilla_periods()
    cum_end = [
        round(c * policy.tokens_per_param * N / actual_token_batch) for c in periods
    ]
    period_lengths = [cum_end[0]] + [
        cum_end[i] - cum_end[i - 1] for i in range(1, len(cum_end))
    ]
    if any(p <= 0 for p in period_lengths):
        raise ValueError(
            f"Rung {rung}: a Chinchilla period rounds to <= 0 steps "
            f"({period_lengths}); increase model size or batch granularity."
        )
    steps = cum_end[-1]
    warmup_steps = round(policy.warmup_tokens(N) / actual_token_batch)

    first_decay = round(policy.decay_fraction * period_lengths[0])
    if warmup_steps + first_decay > period_lengths[0]:
        raise ValueError(
            f"Rung {rung}: warmup ({warmup_steps}) + first-period decay "
            f"({first_decay}) exceeds first period length ({period_lengths[0]})."
        )

    pre_decay_steps = [
        cum_end[i] - round(policy.decay_fraction * period_lengths[i])
        for i in range(len(periods))
    ]
    post_decay_steps = list(cum_end)
    checkpoint_steps = sorted(set(pre_decay_steps + post_decay_steps))
    period_lr_multipliers = (
        [1.0 / (c**0.5) for c in periods] if policy.stepped_schedule else None
    )

    return ResolvedPlan(
        rung=rung,
        ladder_params=N,
        dp_degree=dp_degree,
        seq_len=seq_len,
        target_token_batch=target_token_batch,
        global_batch_size=global_batch_size,
        actual_token_batch=actual_token_batch,
        local_batch_size=local_batch_size,
        grad_accum_steps=grad_accum_steps,
        steps=steps,
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        beta2=beta2,
        weight_decay=policy.weight_decay,
        decay_fraction=policy.decay_fraction,
        chinchilla_multiple=policy.chinchilla_multiple,
        chinchilla_periods=periods,
        period_lengths=period_lengths,
        period_lr_multipliers=period_lr_multipliers,
        pre_decay_steps=pre_decay_steps,
        post_decay_steps=post_decay_steps,
        checkpoint_steps=checkpoint_steps,
        seed=seed,
    )


def to_plan_dict(plan: ResolvedPlan) -> dict:
    return dataclasses.asdict(plan)


def to_trainer_config(
    plan: ResolvedPlan,
    *,
    compute: ComputeSpec,
    dataset: str,
    val_dataset: str,
    hf_assets_path: str,
    dump_folder: str,
    enable_validation: bool,
    val_steps: int,
    log_freq: int,
    attn_backend: str,
    compile_enabled: bool = False,
    converters: list | None = None,
) -> Trainer.Config:
    optimizer = OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=_EMBEDDING_PATTERN,
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": plan.peak_lr,
                    "betas": (0.9, plan.beta2),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": plan.peak_lr,
                    "betas": (0.9, plan.beta2),
                    "eps": 1e-8,
                    "weight_decay": plan.weight_decay,
                },
            ),
        ]
    )
    return Trainer.Config(
        model_spec=model_registry(
            plan.rung, attn_backend=attn_backend, converters=converters
        ),
        loss=ChunkedCELoss.Config(),
        hf_assets_path=hf_assets_path,
        dump_folder=dump_folder,
        optimizer=optimizer,
        lr_scheduler=WSDSScheduler.Config(
            warmup_steps=plan.warmup_steps,
            period_lengths=plan.period_lengths,
            decay_fraction=plan.decay_fraction,
            period_lr_multipliers=plan.period_lr_multipliers,
        ),
        training=TrainingConfig(
            local_batch_size=plan.local_batch_size,
            global_batch_size=plan.global_batch_size,
            seq_len=plan.seq_len,
            steps=plan.steps,
        ),
        parallelism=replace(compute.parallelism),
        compile=CompileConfig(enable=compile_enabled),
        checkpoint=LadderCheckpointManager.Config(
            enable=True,
            checkpoint_steps=plan.checkpoint_steps,
            keep_latest_k=0,
        ),
        metrics=MetricsProcessor.Config(enable_tensorboard=True, log_freq=log_freq),
        dataloader=HuggingFaceTextDataLoader.Config(dataset=dataset),
        validator=LadderValidator.Config(
            enable=enable_validation,
            fixed_steps=plan.post_decay_steps,
            steps=val_steps,
            dataloader=HuggingFaceTextDataLoader.Config(
                dataset=val_dataset, infinite=False
            ),
        ),
        debug=DebugConfig(seed=plan.seed),
    )
