# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read TensorBoard scalars back into structured per-checkpoint records.

Reuses the ``EventAccumulator`` approach from ``scripts/loss_compare.py`` on
``{run_dir}/{save_tb_folder}/{timestamp}/`` and maps results onto the plan's
checkpoint steps. ``val_loss`` is only present at post-decay steps (where
``LadderValidator`` fires); pre-decay records carry train_loss / grad_norm only.

``val_loss`` is exact-at-step (the validator logs at the checkpoint step), but
``train_loss`` / ``grad_norm`` are the nearest *earlier* logged scalar: train
metrics are only written on ``should_log`` steps (every ``log_freq``), and the
arbitrary rounded checkpoint steps rarely coincide with them.
"""

import os
import statistics

from .planner import ResolvedPlan

_TRAIN_LOSS_TAG = "loss_metrics/global_avg_loss"
_GRAD_NORM_TAG = "grad_norm"
_VAL_LOSS_TAG = "validation_metrics/loss"
_THROUGHPUT_TAG = "throughput(tps)"


def _find_event_dir(run_dir: str, save_tb_folder: str) -> str | None:
    base = os.path.join(run_dir, save_tb_folder)
    if not os.path.isdir(base):
        return None
    subdirs = sorted(
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    )
    # Each run writes one timestamped subdir; pick the latest if a folder was
    # reused across runs (the timestamp is minute-resolution).
    return os.path.join(base, subdirs[-1]) if subdirs else None


def _read_scalars(event_dir: str, tag: str) -> dict[int, float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(event_dir)
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return {}
    return {s.step: s.value for s in acc.Scalars(tag)}


def _nearest(scalars: dict[int, float], step: int) -> float | None:
    if step in scalars:
        return scalars[step]
    earlier = [s for s in scalars if s <= step]
    return scalars[max(earlier)] if earlier else None


def read_run_metrics(
    run_dir: str, plan: ResolvedPlan, *, save_tb_folder: str = "tb"
) -> dict:
    event_dir = _find_event_dir(run_dir, save_tb_folder)
    train = _read_scalars(event_dir, _TRAIN_LOSS_TAG) if event_dir else {}
    grad = _read_scalars(event_dir, _GRAD_NORM_TAG) if event_dir else {}
    val = _read_scalars(event_dir, _VAL_LOSS_TAG) if event_dir else {}

    checkpoints = []
    for i, c in enumerate(plan.chinchilla_periods):
        for step, phase in (
            (plan.pre_decay_steps[i], "pre-decay"),
            (plan.post_decay_steps[i], "post-decay"),
        ):
            record = {
                "step": step,
                "tokens": step * plan.actual_token_batch,
                "chinchilla_multiple": c,
                "phase": phase,
                "train_loss": _nearest(train, step),
                "grad_norm": _nearest(grad, step),
            }
            if phase == "post-decay":
                record["val_loss"] = val.get(step)
            checkpoints.append(record)

    return {
        "rung": plan.rung,
        "ladder_params": plan.ladder_params,
        "checkpoints": checkpoints,
    }


def read_run_throughput(run_dir: str, *, save_tb_folder: str = "tb") -> float | None:
    """Median steady-state throughput (tokens/sec) for a run, from TensorBoard.

    The first logged step includes torch.compile + startup warmup, so it is
    dropped; the median over the remaining logged steps is robust to stragglers.
    Returns None if no throughput scalars are present.
    """
    event_dir = _find_event_dir(run_dir, save_tb_folder)
    if event_dir is None:
        return None
    scalars = _read_scalars(event_dir, _THROUGHPUT_TAG)
    steady = [v for step, v in scalars.items() if step > 1]
    return statistics.median(steady) if steady else None
