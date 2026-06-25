# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric helpers for the async RL loop to keep the loop functions free of metric computations."""

# TODO(async-rl): revisit this module's path/name — not observability/metrics (that is public API),
# but components/ may not be the right home either.

import contextlib
import time
from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.types import Rollout


class MetricsTimer:
    """Times named code spans; flush() drains them into Mean metrics, then resets.

    Example:
        metric_timer = MetricsTimer()
        with metric_timer.record("timing/step/total"):
            for _ in range(num_microbatches):
                with metric_timer.record("timing/step/forward_backward"):
                    ...
        time_metrics = metric_timer.flush()
        # -> [Metric("timing/step/total", Mean(...)),
        #     Metric("timing/step/forward_backward", Mean(...))]   # mean over num_microbatches
    """

    def __init__(self) -> None:
        self.durations: dict[str, list[float]] = defaultdict(list)

    @contextlib.contextmanager
    def record(self, key: str):
        # TODO(async-rl): consider asynccontextmanager if span entry/exit ever needs async work; this
        # sync manager still measures awaited blocks correctly.
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[key].append(time.perf_counter() - start)

    def flush(self) -> list[m.Metric]:
        """Return one Mean metric per recorded span, then reset so the timer can be reused.

        Callers that also need the raw durations (e.g. the perf panel) must snapshot
        `self.durations` BEFORE calling this, since it drains them.
        """
        durations = self.durations
        self.durations = defaultdict(list)
        return [
            m.Metric(key, m.Mean.from_list(values)) for key, values in durations.items()
        ]


def combine_microbatch_metrics(
    microbatch_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Combine per-microbatch loss metrics over the grad-accumulation: mean/frac keys are pre-normalized
    by num_global_valid_tokens so summing them reconstructs the global value. For keys ending in "max",
    the max value is taken.

    Example:
        # already normalized by num_global_valid_tokens
        combine_microbatch_metrics([{"loss/ratio_clipped_frac": 0.1, "x/max": 2.0},
                                    {"loss/ratio_clipped_frac": 0.2, "x/max": 5.0}])
        # output
        # -> {"loss/ratio_clipped_frac": 0.3, "x/max": 5.0}
    """
    combined: dict[str, float] = {}
    for microbatch in microbatch_metrics:
        for key, value in microbatch.items():
            if key not in combined:
                combined[key] = value
            elif key.endswith("/max"):
                combined[key] = max(combined[key], value)
            elif key.endswith(("/mean", "_mean", "/frac", "_frac")):
                combined[key] += value
    return combined


def compute_perf_ratio_metrics(
    *, num_global_valid_tokens: int, durations: dict[str, list[float]]
) -> list[m.Metric]:
    """Trainer-side timing metrics derived from one step's timings."""
    step_s = sum(durations.get("timing/step/total", [0.0]))
    wait_s = sum(durations.get("timing/step/wait_for_training_batch", []))
    forward_backward_s = sum(durations.get("timing/step/forward_backward", []))
    optim_s = sum(durations.get("timing/step/optim", []))
    sync_s = sum(durations.get("timing/step/weight_sync/push", [])) + sum(
        durations.get("timing/step/weight_sync/pull", [])
    )
    trainer_compute_s = forward_backward_s + optim_s
    accounted_s = wait_s + trainer_compute_s + sync_s
    timing_gap_ratio = (step_s - accounted_s) / step_s if step_s else 0.0
    ratios = {
        "perf/trainer/fwd_bwd_step_time_ratio": (
            trainer_compute_s / step_s if step_s else 0.0
        ),
        "perf/trainer/weight_sync_step_time_ratio": sync_s / step_s if step_s else 0.0,
        "perf/trainer/batch_step_time_ratio": wait_s / step_s if step_s else 0.0,
        "perf/trainer/unaccounted_step_time_ratio": timing_gap_ratio,
        "perf/trainer/tokens_per_second_full_step": (
            num_global_valid_tokens / step_s if step_s else 0.0
        ),
        "perf/trainer/tokens_per_second_fwd_bwd": (
            num_global_valid_tokens / trainer_compute_s if trainer_compute_s else 0.0
        ),
    }
    return [m.Metric(key, m.NoReduce(value)) for key, value in ratios.items()]


def compute_policy_age_metrics(
    *,
    trainer_policy_version: int,
    min_policy_versions: list[int],
    max_offpolicy_steps: int,
) -> list[m.Metric]:
    """Age of each packed training sample at the moment the trainer consumes the batch.

    Computed in the trainer loop (not at pack time) so the logged age is faithful to the version the
    batch actually trains against, and so the consume-time freshness invariant is checked here.

    Args:
        trainer_policy_version: Policy version that will consume this batch.
        min_policy_versions: Oldest sampled policy version for each packed training sample.
        max_offpolicy_steps: Maximum allowed consume-time trainer-version lag.

    Example:
        # trainer at v=10; training samples' oldest versions [8, 9] -> ages [2, 1]
        compute_policy_age_metrics(trainer_policy_version=10, min_policy_versions=[8, 9], max_offpolicy_steps=3)
        # -> train_batch/policy_age mean 1.5, train_batch/policy_age_max 2
    """
    policy_ages = [
        trainer_policy_version - min_policy_version
        for min_policy_version in min_policy_versions
    ]
    max_policy_age = max(policy_ages, default=0)
    if max_policy_age > max_offpolicy_steps:
        raise RuntimeError(
            "rollout backpressure admitted stale training data: "
            f"max_policy_age={max_policy_age}, "
            f"max_offpolicy_steps={max_offpolicy_steps}, "
            f"trainer_policy_version={trainer_policy_version}"
        )
    return [
        m.Metric("train_batch/policy_age", m.Mean.from_list(policy_ages)),
        m.Metric("train_batch/policy_age_max", m.NoReduce(float(max_policy_age))),
    ]


def compute_rollout_metrics(prefix: str, rollouts: list[Rollout]) -> list[m.Metric]:
    """Build rollout-derived metrics: lengths, truncation, reward breakdown, and each turn's
    per-generation metrics (the latter keep their own generator-side keys, unprefixed).

    Args:
        prefix: Metric namespace (e.g. `"rollout"` or `"validation"`).
        rollouts: Rollouts to compute metrics for.
    """
    # Lengths, truncation, reward
    completion_lens = [
        len(rollout_turn.completion_token_ids)
        for rollout in rollouts
        for rollout_turn in rollout.turns
    ]
    prompt_lens = [
        len(rollout.turns[0].prompt_token_ids) for rollout in rollouts if rollout.turns
    ]
    total_lens = [
        len(rollout.turns[-1].prompt_token_ids)
        + len(rollout.turns[-1].completion_token_ids)
        for rollout in rollouts
        if rollout.turns
    ]

    truncated = [float(rollout.status.is_truncated()) for rollout in rollouts]
    rewards = [rollout.reward for rollout in rollouts if rollout.reward is not None]
    num_turns = [float(len(rollout.turns)) for rollout in rollouts]

    out: list[m.Metric] = [
        m.Metric(f"{prefix}/output_tokens", m.Mean.from_list(completion_lens)),
        m.Metric(f"{prefix}/output_tokens", m.Std.from_list(completion_lens)),
        m.Metric(f"{prefix}/output_tokens", m.Max.from_list(completion_lens)),
        m.Metric(f"{prefix}/response_length", m.Mean.from_list(completion_lens)),
        m.Metric(f"{prefix}/response_length", m.Max.from_list(completion_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
        m.Metric(f"{prefix}/total_length", m.Mean.from_list(total_lens)),
        m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
        m.Metric(f"{prefix}/num_turns", m.Mean.from_list(num_turns)),
        m.Metric(f"{prefix}/num_turns", m.Max.from_list(num_turns)),
        m.Metric(f"{prefix}/truncation_rate", m.Mean.from_list(truncated)),
        m.Metric(f"{prefix}_reward", m.SummaryStats.from_list(rewards)),
    ]

    # Per-component reward breakdown
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for name, value in rollout.reward_breakdown.items():
            values_by_name[name].append(float(value))
    out.extend(
        m.Metric(f"{prefix}_reward/component/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    )

    # Per-generation turn metrics (latencies, output tokens) measured by the generator.
    # They carry their own keys (e.g. "generator/..."), so they ride through unprefixed.
    out.extend(
        metric
        for rollout in rollouts
        for rollout_turn in rollout.turns
        for metric in rollout_turn.metrics
    )
    return out
