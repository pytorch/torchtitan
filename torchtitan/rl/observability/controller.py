# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric helpers for the async RL loop to keep the loop functions free of metric computations."""

# TODO(async-rl): revisit this module's path/name — not observability/metrics (that is public API),
# but components/ may not be the right home either.

import contextlib
import logging
import time
from collections import defaultdict

from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Rollout

logger = logging.getLogger(__name__)


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
        """Return one Mean metric per recorded span, then reset so the timer can be reused."""
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
    *, num_global_valid_tokens: int, time_metrics: list[m.Metric]
) -> list[m.Metric]:
    """Trainer throughput, and each loop phase's share of `timing/step/total`. A phase not recorded
    this step gets no ratio, and then there is no `unaccounted` either (no fallback zeros).

    Example:
        # 100 tokens; total 10 s = wait_for_training_batch 2 + forward_backward 4 + optimizer 1
        #   + wait_for_push 1 + wait_for_pull 1 + 1 s in no phase
        # -> tokens_per_second_full_step 10, tokens_per_second_forward_backward 25,
        #    step_time_ratio/forward_backward 0.4, ..., step_time_ratio/unaccounted 0.1
    """
    # Each span is recorded once/step; Mean.from_list stores the summed seconds in `.value`.
    seconds = {
        metric.key: metric.value.value
        for metric in time_metrics
        if isinstance(metric.value, m.Mean)
    }
    step_s = seconds.get("timing/step/total")
    if not step_s:  # no step wall-clock -> no denominator to derive ratios from
        return []

    out: list[m.Metric] = []

    def _add_metric(key: str, value: float) -> None:
        out.append(m.Metric(key, m.NoReduce(value)))

    # Throughput over the whole step (includes the idle wait for the next batch).
    _add_metric(
        "perf/trainer/tokens_per_second_full_step", num_global_valid_tokens / step_s
    )
    fwd_bwd_s = seconds.get("timing/step/forward_backward")
    if fwd_bwd_s:
        _add_metric(
            "perf/trainer/tokens_per_second_forward_backward",
            num_global_valid_tokens / fwd_bwd_s,
        )

    # Each phase's share of the step (skip a phase that was not recorded).
    phase_seconds = {
        phase: seconds.get(f"timing/step/{phase}")
        for phase in (
            "wait_for_training_batch",
            "forward_backward",
            "wait_for_push",
            "optimizer",
            "wait_for_pull",
        )
    }
    for phase, phase_s in phase_seconds.items():
        if phase_s is not None:
            _add_metric(f"perf/trainer/step_time_ratio/{phase}", phase_s / step_s)

    # Step time the phases don't cover -- only when every phase is present, else it misleads.
    if None not in phase_seconds.values():
        accounted_s = sum(phase_seconds.values())
        _add_metric(
            "perf/trainer/step_time_ratio/unaccounted", (step_s - accounted_s) / step_s
        )

    return out


def compute_policy_age_metrics(
    *,
    trainer_policy_version: int,
    min_policy_versions: list[int],
    target_offpolicy_steps: int,
    max_offpolicy_steps: int | None,
) -> list[m.Metric]:
    """Age of each packed training sample at the moment the trainer consumes the batch.

    Computed in the trainer loop (not at pack time) so the logged age is faithful to the version the
    batch actually trains against. Without a bound (`max_offpolicy_steps=None`) samples older than the
    target are counted and warned about, never rejected. With a bound, exceeding it is an invariant
    failure: the window makes it impossible.

    Args:
        trainer_policy_version: Policy version that will consume this batch.
        min_policy_versions: Oldest sampled policy version for each packed training sample.
        target_offpolicy_steps: Target mean offpolicy steps used to size the active buffer.
        max_offpolicy_steps: Hard consume-time offpolicy step limit, `target + windowed_fifo_batches`;
            None when there is no window.

    Example:
        # trainer at v=10; training samples' oldest versions [8, 9, 5] -> ages [2, 1, 5]
        compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[8, 9, 5],
            target_offpolicy_steps=3,
            max_offpolicy_steps=None,
        )
        # -> train_batch/policy_age mean 2.67, train_batch/policy_age_max 5,
        #    train_batch/pct_samples_over_target_age 33.3, one logger.warning (uncapped, 5 > 3)
        # with max_offpolicy_steps=4 the same batch raises RuntimeError (5 > 4)
    """
    policy_ages = [
        trainer_policy_version - min_policy_version
        for min_policy_version in min_policy_versions
    ]
    max_policy_age = max(policy_ages, default=0)
    if max_offpolicy_steps is not None and max_policy_age > max_offpolicy_steps:
        raise RuntimeError(
            "rollout backpressure admitted stale training data: "
            f"max_policy_age={max_policy_age}, "
            f"target_offpolicy_steps={target_offpolicy_steps}, "
            f"max_offpolicy_steps={max_offpolicy_steps}, "
            f"trainer_policy_version={trainer_policy_version}"
        )
    num_samples_over_target_age = sum(
        policy_age > target_offpolicy_steps for policy_age in policy_ages
    )
    pct_samples_over_target_age = 100.0 * num_samples_over_target_age / len(policy_ages)
    if max_offpolicy_steps is None and num_samples_over_target_age:
        logger.warning(
            f"Training batch contains {num_samples_over_target_age} samples "
            f"({pct_samples_over_target_age:.1f}%) older than target_offpolicy_steps={target_offpolicy_steps} "
            f"(max_policy_age={max_policy_age}, trainer_policy_version={trainer_policy_version}). "
            "Oldest-ready consumption with no cap trains these instead of dropping them; "
            "frequent hits indicate a heavy generation tail."
        )
    return [
        m.Metric("train_batch/policy_age", m.Mean.from_list(policy_ages)),
        m.Metric("train_batch/policy_age_max", m.NoReduce(float(max_policy_age))),
        m.Metric(
            "train_batch/pct_samples_over_target_age",
            m.NoReduce(pct_samples_over_target_age),
        ),
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
