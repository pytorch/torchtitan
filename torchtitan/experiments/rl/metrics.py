# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared metrics utilities for RL training loops.
"""

from dataclasses import dataclass


@dataclass
class CumulativeMetrics:
    """Accumulator for timing and memory metrics across training steps."""

    rollout_s: float = 0.0
    train_s: float = 0.0
    optimizer_s: float = 0.0
    weight_sync_s: float = 0.0
    total_s: float = 0.0
    peak_rollout_gib: float = 0.0
    peak_train_gib: float = 0.0
    peak_optimizer_gib: float = 0.0
    peak_weight_sync_gib: float = 0.0
    completed_steps: int = 0


@dataclass(frozen=True)
class PhaseMetrics:
    """Metrics for a single phase of a RL step."""

    name: str
    time_s: float
    peak_mem_gib: float = 0.0
    peak_mem_pct: float = 0.0
    note: str | None = None


def update_cumulative_metrics(
    cumulative: CumulativeMetrics,
    rollout_time_s: float,
    train_time_s: float,
    optimizer_time_s: float,
    weight_sync_time_s: float,
    total_time_s: float,
    rollout_peak_gib: float = 0.0,
    train_peak_gib: float = 0.0,
    optimizer_peak_gib: float = 0.0,
    weight_sync_peak_gib: float = 0.0,
) -> None:
    """Update cumulative metrics with data from a single RL step.

    Args:
        cumulative: The CumulativeMetrics instance to update.
        rollout_time_s: Time spent in rollout phase (seconds).
        train_time_s: Time spent in training phase (seconds).
        optimizer_time_s: Time spent in optimizer step (seconds).
        weight_sync_time_s: Time spent syncing weights (seconds).
        total_time_s: Total step time (seconds).
        rollout_peak_gib: Peak memory during rollout phase (GiB).
        train_peak_gib: Peak memory during train phase (GiB).
        optimizer_peak_gib: Peak memory during optimizer step (GiB).
        weight_sync_peak_gib: Peak memory during weight sync phase (GiB).
    """
    cumulative.rollout_s += rollout_time_s
    cumulative.train_s += train_time_s
    cumulative.optimizer_s += optimizer_time_s
    cumulative.weight_sync_s += weight_sync_time_s
    cumulative.total_s += total_time_s
    cumulative.completed_steps += 1
    cumulative.peak_rollout_gib = max(cumulative.peak_rollout_gib, rollout_peak_gib)
    cumulative.peak_train_gib = max(cumulative.peak_train_gib, train_peak_gib)
    cumulative.peak_optimizer_gib = max(
        cumulative.peak_optimizer_gib, optimizer_peak_gib
    )
    cumulative.peak_weight_sync_gib = max(
        cumulative.peak_weight_sync_gib, weight_sync_peak_gib
    )


def print_step_metrics(
    phases: list[PhaseMetrics],
    total_time_s: float,
    include_mem_pct: bool = False,
) -> None:
    """Print per-phase timing and memory breakdown for a training step.

    Args:
        phases: List of PhaseMetrics for each phase in the step.
        total_time_s: Total time for the step (seconds).
        include_mem_pct: If True, include memory percentage in output.
    """
    print(f"  {'Phase':<16s} {'Time':>8s}  {'Peak Mem':>10s}")
    print(f"  {'-' * 38}")

    for phase in phases:
        line = f"  {phase.name + ':':<16s} {phase.time_s:>7.2f}s"
        if phase.peak_mem_gib > 0:
            line += f",  {phase.peak_mem_gib:>5.2f} GiB"
            if include_mem_pct and phase.peak_mem_pct > 0:
                line += f"  ({phase.peak_mem_pct:.1f}%)"
            if phase.note:
                line += f"  {phase.note}"
        elif phase.note:
            line += f"  {phase.note}"
        print(line)

    print(f"  {'total:':<16s} {total_time_s:>7.2f}s")


def print_training_summary(
    cumulative: CumulativeMetrics,
    only_print_memory_if_nonzero: bool = False,
) -> None:
    """Print a formatted training summary.

    Args:
        cumulative: The CumulativeMetrics instance containing accumulated data.
        only_print_memory_if_nonzero: If True, only print memory stats when > 0.
            If False, always print memory stats.
    """
    if cumulative.completed_steps == 0:
        return

    print("\n" + "=" * 80)
    print(f"Training Summary ({cumulative.completed_steps} steps):")
    print(f"  {'Total wall-clock:':<28s} {cumulative.total_s:>9.2f}s")

    for label, val in [
        ("Cumul. rollout:", cumulative.rollout_s),
        ("Cumul. train:", cumulative.train_s),
        ("Cumul. optimizer:", cumulative.optimizer_s),
        ("Cumul. weight_sync:", cumulative.weight_sync_s),
    ]:
        pct = 100 * val / cumulative.total_s
        print(f"  {label:<28s} {val:>9.2f}s  ({pct:>5.1f}%)")

    should_print_rollout_mem = (
        not only_print_memory_if_nonzero or cumulative.peak_rollout_gib > 0
    )
    should_print_train_mem = (
        not only_print_memory_if_nonzero or cumulative.peak_train_gib > 0
    )
    should_print_optimizer_mem = (
        not only_print_memory_if_nonzero or cumulative.peak_optimizer_gib > 0
    )
    should_print_weight_sync_mem = (
        not only_print_memory_if_nonzero or cumulative.peak_weight_sync_gib > 0
    )

    if should_print_rollout_mem:
        print(
            f"  {'Peak mem (rollout):':<28s} "
            f"{cumulative.peak_rollout_gib:>8.2f} GiB"
        )
    if should_print_train_mem:
        print(f"  {'Peak mem (train):':<28s} " f"{cumulative.peak_train_gib:>8.2f} GiB")
    if should_print_optimizer_mem:
        print(
            f"  {'Peak mem (optimizer):':<28s} "
            f"{cumulative.peak_optimizer_gib:>8.2f} GiB"
        )
    if should_print_weight_sync_mem:
        print(
            f"  {'Peak mem (weight_sync):':<28s} "
            f"{cumulative.peak_weight_sync_gib:>8.2f} GiB"
        )

    print("=" * 80)
