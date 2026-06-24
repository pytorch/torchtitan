# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric helpers for the async RL controller: a generic duration timer, microbatch-metric
combination, the trainer goodput panel, trainer-consumed policy age, and rollout-derived metrics."""

import contextlib
import time
from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.types import Rollout


class MetricsTimer:
    """Collect named durations; the caller decides the log step + namespace and flushes them.

    Generic and loop-agnostic. The caller records spans and passes `metrics()` to
    `metrics_processor.log(step=...)` once per step (trainer) or per pass (validation). `metrics()`
    drains the timer, so snapshot `durations` first if a derived panel needs the raw timings. Name the
    variable for its lifetime (e.g. `step_timer`, `validation_timer`).

    Example:
        timer = MetricsTimer()
        with timer.record("timing/step/train"):
            ...
        metrics = timer.metrics()   # [Metric("timing/step/train", Mean(...)), ...]
    """

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    @contextlib.contextmanager
    def record(self, key: str):
        if key in self.durations:
            raise ValueError(f"duplicate timing key {key!r} in one timer")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[key] = time.perf_counter() - start

    def metrics(self) -> list[m.Metric]:
        """Return one Mean metric per recorded span, then reset so the timer can be reused.

        Callers that also need the raw durations (e.g. the goodput panel) must snapshot
        `self.durations` BEFORE calling this, since it drains them.
        """
        durations = self.durations
        self.durations = {}
        return [m.Metric(key, m.Mean(value)) for key, value in durations.items()]


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
    *, num_global_valid_tokens: int, durations: dict[str, float]
) -> list[m.Metric]:
    """Trainer-side goodput panel derived from one step's timings.

    The gap between goodput (`/ step total`) and active (`/ train`) IS the trainer-idle + sync bubble.

    Example:
        # step total=10, wait=8, train=1, weight_sync=0.5, num_global_valid_tokens=100
        # -> perf/trainer_idle_ratio=0.8, perf/weight_sync_overhead_ratio=0.05,
        #    perf/goodput_tokens_per_second=10, perf/active_tokens_per_second=100
    """
    step_s = sum(durations.values())
    wait_s = durations.get("timing/step/wait_for_training_batch", 0.0)
    train_s = durations.get("timing/step/train", 0.0)
    sync_s = durations.get("timing/step/weight_sync/push", 0.0) + durations.get(
        "timing/step/weight_sync/pull", 0.0
    )
    ratios = {
        "perf/trainer_idle_ratio": wait_s / step_s if step_s else 0.0,
        "perf/weight_sync_overhead_ratio": sync_s / step_s if step_s else 0.0,
        "perf/goodput_tokens_per_second": (
            num_global_valid_tokens / step_s if step_s else 0.0
        ),
        "perf/active_tokens_per_second": (
            num_global_valid_tokens / train_s if train_s else 0.0
        ),
    }
    return [m.Metric(key, m.NoReduce(value)) for key, value in ratios.items()]


def compute_policy_age_metrics(
    *, trainer_policy_version: int, min_policy_versions: list[int]
) -> list[m.Metric]:
    """Age (in optimizer steps) of each packed training_sample at the moment the trainer consumes the batch.

    Computed in the trainer loop (not at pack time) so the logged age is faithful to the version the
    batch actually trains against.

    Example:
        # trainer at v=10; training_samples' oldest versions [8, 9] -> ages [2, 1]
        # -> train_batch/policy_age (mean 1.5), train_batch/policy_age_max (2)
    """
    ages = [trainer_policy_version - version for version in min_policy_versions]
    return [
        m.Metric("train_batch/policy_age", m.Mean.from_list(ages)),
        m.Metric("train_batch/policy_age_max", m.NoReduce(float(max(ages, default=0)))),
    ]


def compute_rollout_metrics(prefix: str, rollouts: list[Rollout]) -> list[m.Metric]:
    """Build rollout-derived metrics: lengths, truncation, reward breakdown, and each turn's
    per-generation metrics (the latter keep their own generator-side keys, unprefixed).

    Args:
        prefix: Metric namespace (e.g. `"rollout"` or `"validation"`).
        rollouts: Rollouts to summarize.
    """
    # Lengths, truncation, reward
    # TODO: adapt for multi-turn rollouts
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
