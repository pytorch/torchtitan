# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric helpers for the async RL controller: a generic duration timer, microbatch-metric
combination, the trainer goodput panel, and trainer-consumed policy age."""

import contextlib
import time

from torchtitan.experiments.rl.observability import metrics as m


class MetricsTimer:
    """Collect named durations; the caller decides the log step + namespace and flushes them.

    Generic and loop-agnostic — the timer does not flush itself. The caller records spans and then
    passes `metrics()` to `metrics_processor.log(step=...)` once per step (trainer) or per pass
    (validation). Name the variable for its lifetime (e.g. `step_timer`, `validation_timer`).

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
        return [m.Metric(key, m.Mean(value)) for key, value in self.durations.items()]


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


def perf_ratio_metrics(
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


def policy_age_metrics(
    *, trainer_policy_version: int, oldest_sampled_versions: list[int]
) -> list[m.Metric]:
    """Age (in optimizer steps) of each packed training_sample at the moment the trainer consumes the batch.

    Computed in the trainer loop (not at pack time) so the logged age is faithful to the version the
    batch actually trains against.

    Example:
        # trainer at v=10; training_samples sampled at [8, 9] -> ages [2, 1]
        # -> train_batch/policy_age (mean 1.5), train_batch/policy_age_max (2)
    """
    ages = [trainer_policy_version - version for version in oldest_sampled_versions]
    return [
        m.Metric("train_batch/policy_age", m.Mean.from_list(ages)),
        m.Metric("train_batch/policy_age_max", m.NoReduce(float(max(ages, default=0)))),
    ]
