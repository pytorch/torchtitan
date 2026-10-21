# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric helpers for the async RL loop to keep the loop functions free of metric computations."""

# TODO(async-rl): revisit this module's path/name — not observability/metrics (that is public API),
# but components/ may not be the right home either.

import contextlib
import functools
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from torchtitan.rl._env import RL_ROLE_ENV
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Rollout, RolloutStatus

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
    *,
    num_global_valid_tokens: int,
    num_global_tokens: int,
    time_metrics: list[m.Metric],
) -> list[m.Metric]:
    """Return trainer throughput and each loop phase's share of `timing/step/total`.

    `num_global_valid_tokens` counts the loss tokens; `num_global_tokens` counts every non-padding
    token, prompts included, so `all_tokens_per_second_*` compares across runs whatever their loss
    masks.

    Example: num_global_valid_tokens=100, num_global_tokens=400, timing/step/total = 20 s:

        timing/step/<phase>                        seconds   perf/trainer/step_time_ratio/<phase>
        metrics_log (the previous step's)          1         0.05
        sync_log_step                              1         0.05
        wait_for_training_batch                    4         0.2
        forward_backward                           8         0.4
        blocking_trainer_push_model_state_dict     1         0.05
        optimizer                                  1         0.05
        save_checkpoint                            2         0.1
        blocking_generator_pull_model_state_dict   1         0.05
        (in no phase)                              1         unaccounted: 0.05

        tokens_per_second_full_step             = 100 / 20 = 5
        tokens_per_second_forward_backward      = 100 / 8  = 12.5
        all_tokens_per_second_full_step         = 400 / 20 = 20
        all_tokens_per_second_forward_backward  = 400 / 8  = 50

    A phase not recorded this step gets no ratio, and `unaccounted` is skipped.
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
    _add_metric(
        "perf/trainer/all_tokens_per_second_full_step", num_global_tokens / step_s
    )

    # Throughput over forward/backward only (no optimizer step or checkpoint save).
    forward_backward_s = seconds.get("timing/step/forward_backward")
    if forward_backward_s:
        _add_metric(
            "perf/trainer/tokens_per_second_forward_backward",
            num_global_valid_tokens / forward_backward_s,
        )
        _add_metric(
            "perf/trainer/all_tokens_per_second_forward_backward",
            num_global_tokens / forward_backward_s,
        )

    # Each phase's share of the step (skip a phase that was not recorded).
    # blocking_*: how long the trainer waited for the overlapped push/pull, not how long it took.
    phase_seconds = {
        phase: seconds.get(f"timing/step/{phase}")
        for phase in (
            "metrics_log",
            "sync_log_step",
            "wait_for_training_batch",
            "forward_backward",
            "blocking_trainer_push_model_state_dict",
            "optimizer",
            "save_checkpoint",
            "blocking_generator_pull_model_state_dict",
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
        rollout.turns[0].num_prompt_tokens for rollout in rollouts if rollout.turns
    ]
    total_lens = [
        rollout.turns[-1].num_prompt_tokens
        + len(rollout.turns[-1].completion_token_ids)
        for rollout in rollouts
        if rollout.turns
    ]

    truncated = [float(rollout.status.is_truncated()) for rollout in rollouts]
    system_errors = [
        float(rollout.status is RolloutStatus.ERROR_SYSTEM) for rollout in rollouts
    ]
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
        m.Metric(f"{prefix}/system_error_rate", m.Mean.from_list(system_errors)),
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
    # Fold across turns: group metrics outlive the rollouts (until the step is logged),
    # and every full GC walks them.
    out.extend(
        m.fold(
            metric
            for rollout in rollouts
            for rollout_turn in rollout.turns
            for metric in rollout_turn.metrics
        )
    )
    return out


_PROCESS_ROLES = (
    "controller",
    "trainer",
    "generator",
    "batcher",
    "router",
    "rollout_worker",
    "env_server",
    "env_server_worker",
    "other",
)


def compute_host_memory_metrics() -> list[m.Metric]:
    """Anonymous memory per process role on this host, plus the job's cgroup total on this host and
    the node's `MemAvailable`.

    Anonymous memory leaves out shared libraries, which summed RSS counts once per process. It is
    read from `smaps_rollup` (a page-table walk, ~1 ms per 4 GiB): `status`'s `RssAnon` misses each
    CPU's unflushed count (up to ~18 MiB per CPU on 144-CPU hosts), so it reads about 2x low.
    Scans every process in the controller's Slurm job cgroup on this host, i.e. every job step
    (outside Slurm, the controller's own cgroup): trainer ranks run in another step than the
    controller, so its own cgroup misses them. A process's role is `TORCHTITAN_RL_ROLE` in its
    spawn env:

        controller                         this process
        trainer, generator                 Monarch procs and their children
        batcher, router, rollout_worker    the controller host's CPU procs (`spawn_cpu_procs`)
        env_server                         each Verifiers env server
        env_server_worker                  processes an env server spawns, and their children
        other                              the rest, e.g. Monarch host agents and the job's other
                                           steps (step_extern, sidecars such as leakwatch)

    Example:
        compute_host_memory_metrics()
        # -> host_memory/rss_anon_gib/env_server_worker 230.5,
        #    host_memory/num_processes/env_server_worker 611, ...,
        #    host_memory/cgroup_memory_current_gib 655.0, host_memory/node_mem_available_gib 300.2
    """
    cgroup_dir = _cgroup_dir()
    if cgroup_dir is None:
        return []
    rss_anon_kib = dict.fromkeys(_PROCESS_ROLES, 0)
    num_processes = dict.fromkeys(_PROCESS_ROLES, 0)
    for pid in _cgroup_pids(cgroup_dir):
        # A process may exit after the cgroup listing, or be unreadable (hidepid).
        try:
            rollup = Path(f"/proc/{pid}/smaps_rollup").read_text()
            role = "controller" if int(pid) == os.getpid() else _process_role(pid)
        except OSError:
            continue
        # "Anonymous:   123456 kB"; a zombie or kernel thread has no such line.
        for line in rollup.splitlines():
            if line.startswith("Anonymous:"):
                rss_anon_kib[role] += int(line.split()[1])
                num_processes[role] += 1
                break

    metrics = [
        m.Metric(f"host_memory/rss_anon_gib/{role}", m.NoReduce(kib / 2**20))
        for role, kib in rss_anon_kib.items()
    ]
    metrics += [
        m.Metric(f"host_memory/num_processes/{role}", m.NoReduce(count))
        for role, count in num_processes.items()
    ]
    # The root cgroup has no memory.current.
    with contextlib.suppress(FileNotFoundError):
        memory_current = int((cgroup_dir / "memory.current").read_text())
        metrics.append(
            m.Metric(
                "host_memory/cgroup_memory_current_gib",
                m.NoReduce(memory_current / 2**30),
            )
        )
    # "MemAvailable:   312345678 kB": what the node can still give before swapping or OOM.
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            metrics.append(
                m.Metric(
                    "host_memory/node_mem_available_gib",
                    m.NoReduce(int(line.split()[1]) / 2**20),
                )
            )
            break
    return metrics


# Slurm's per-job cgroup (cgroup/v2 plugin): job_<id>/step_<step>/user/task_<n>.
_SLURM_JOB_CGROUP = re.compile(r"job_\d+")


@functools.cache
def _cgroup_dir() -> Path | None:
    """The cgroup v2 directory whose processes the host memory metrics scan; cached, so a host
    without one warns once.

    Example:
        "0::/system.slice/slurmstepd.scope/job_7/step_batch/user/task_0"
            -> /sys/fs/cgroup/system.slice/slurmstepd.scope/job_7
        "0::/user.slice/session-3.scope" -> /sys/fs/cgroup/user.slice/session-3.scope
    """
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            cgroup_dir = _job_cgroup_dir(Path("/sys/fs/cgroup"), line[3:])
            if (cgroup_dir / "cgroup.procs").exists():
                return cgroup_dir
    logger.warning("No cgroup v2 found; host memory metrics are off")
    return None


def _job_cgroup_dir(cgroup_root: Path, cgroup_path: str) -> Path:
    """`cgroup_path`'s nearest ancestor that is a Slurm job cgroup, else `cgroup_path` itself.

    Example: (/sys/fs/cgroup, "/system.slice/slurmstepd.scope/job_7/step_0/user/task_0")
        -> /sys/fs/cgroup/system.slice/slurmstepd.scope/job_7
    """
    parts = Path(cgroup_path.lstrip("/")).parts
    for depth in range(len(parts), 0, -1):
        if _SLURM_JOB_CGROUP.fullmatch(parts[depth - 1]):
            return cgroup_root.joinpath(*parts[:depth])
    return cgroup_root.joinpath(*parts)


def _cgroup_pids(cgroup_dir: Path) -> list[str]:
    """Pids in `cgroup_dir`; for a Slurm job cgroup, in every cgroup below it (cgroup v2 keeps
    processes in leaf cgroups, here each step's task cgroups).

    Example: job_7/step_batch/user/task_0 holds [100], job_7/step_0/user/task_0 holds [200]
        -> _cgroup_pids(.../job_7) == ["100", "200"]
    """
    if not _SLURM_JOB_CGROUP.fullmatch(cgroup_dir.name):
        return (cgroup_dir / "cgroup.procs").read_text().split()
    pids = []
    for dirpath, _, filenames in os.walk(cgroup_dir):
        if "cgroup.procs" in filenames:
            # A step's cgroup can go away mid-walk.
            with contextlib.suppress(OSError):
                pids += (Path(dirpath) / "cgroup.procs").read_text().split()
    return pids


def _process_role(pid: str) -> str:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except PermissionError:  # another user's process
        return "other"
    prefix = f"{RL_ROLE_ENV}=".encode()
    for entry in environ.split(b"\0"):
        if entry.startswith(prefix):
            role = entry[len(prefix) :].decode(errors="replace")
            return role if role in _PROCESS_ROLES else "other"
    return "other"
