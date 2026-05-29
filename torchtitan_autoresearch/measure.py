"""Canonical, tamper-proof throughput measurement.

Post-mortem finding 1b: the search moved reported tps +2.9% by walking
``metrics.log_freq`` 5->2->1, which narrows the averaging window onto the
warmest steps — a pure measurement artifact that got baked into the final
command. The fix is to take the measurement away from the agent: the harness
pins ``--metrics.log_freq=1`` and computes steady-state tps itself over a
*fixed* step window, reporting mean/std/cv so variance is a first-class number
(post-mortem 1a) instead of something discovered via 93 ad-hoc reruns.

This module only parses; the driver is responsible for launching runs with the
pinned flags. It tolerates the ANSI-colored console format in TorchTitan's
``run.log``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STEP = re.compile(
    r"step:\s*([\d,]+)\s+"
    r"loss:\s*([-\d.eE+]+)\s+"
    r"grad_norm:\s*([-\d.eE+]+)\s+"
    r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+"
    r"tps:\s*([\d,]+)\s+"
    r"tflops:\s*([\d.]+)"
    r"(?:\s+mfu:\s*([\d.]+)%)?"
)
_PEAK_FLOPS = re.compile(r"Peak FLOPS used for computing MFU:\s*([\d.eE+]+)")
_CAPACITY = re.compile(r"CUDA capacity:\s*(.+?)\s+with\s+([\d.]+)GiB")


@dataclass
class Step:
    step: int
    loss: float
    grad_norm: float
    memory_gb: float
    memory_pct: float
    tps: float
    tflops: float
    mfu: float | None


@dataclass
class Measurement:
    """Steady-state summary over a fixed window. ``tps`` is the ranking score."""

    steps: list[Step] = field(default_factory=list)
    window: tuple[int, int] = (0, 0)
    tps_mean: float = 0.0
    tps_std: float = 0.0
    tps_cv: float = 0.0  # coefficient of variation within the window
    tps_min: float = 0.0
    tps_max: float = 0.0
    tflops_mean: float = 0.0
    mfu_mean: float | None = None
    peak_memory_gb: float = 0.0
    peak_memory_pct: float = 0.0
    loss_finite: bool = False
    loss_decreasing: bool = False
    peak_flops: float | None = None
    gpu: str = ""
    n_window: int = 0


def _f(x: str) -> float:
    return float(x.replace(",", ""))


def parse_steps(text: str) -> list[Step]:
    steps: list[Step] = []
    for raw in text.splitlines():
        line = _ANSI.sub("", raw)
        m = _STEP.search(line)
        if not m:
            continue
        steps.append(
            Step(
                step=int(m.group(1).replace(",", "")),
                loss=_f(m.group(2)),
                grad_norm=_f(m.group(3)),
                memory_gb=_f(m.group(4)),
                memory_pct=_f(m.group(5)),
                tps=_f(m.group(6)),
                tflops=_f(m.group(7)),
                mfu=_f(m.group(8)) if m.group(8) else None,
            )
        )
    return steps


def _stats(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)  # sample std
    return mean, math.sqrt(var)


def measure(text: str, window: tuple[int, int]) -> Measurement:
    """Summarize steady-state from a run log over the inclusive ``window``.

    The window is harness-defined (e.g. (11, 20)); step 1 is always excluded
    elsewhere because it carries first-iteration/compile overhead. Peak memory
    is taken over ALL steps, not just the window, since OOM risk is global.
    """
    steps = parse_steps(text)
    out = Measurement(steps=steps, window=window)
    if not steps:
        return out

    pf = _PEAK_FLOPS.search(text)
    out.peak_flops = float(pf.group(1)) if pf else None
    cap = _CAPACITY.search(text)
    if cap:
        out.gpu = cap.group(1)

    out.peak_memory_gb = max(s.memory_gb for s in steps)
    out.peak_memory_pct = max(s.memory_pct for s in steps)

    lo, hi = window
    win = [s for s in steps if lo <= s.step <= hi]
    if not win:
        # Fall back to "everything after step 1" if the window did not land.
        win = [s for s in steps if s.step > 1] or steps
    out.n_window = len(win)

    tps = [s.tps for s in win]
    out.tps_mean, out.tps_std = _stats(tps)
    out.tps_min, out.tps_max = min(tps), max(tps)
    out.tps_cv = out.tps_std / out.tps_mean if out.tps_mean else 0.0
    out.tflops_mean, _ = _stats([s.tflops for s in win])
    mfus = [s.mfu for s in win if s.mfu is not None]
    out.mfu_mean = (sum(mfus) / len(mfus)) if mfus else None

    losses = [s.loss for s in steps]
    out.loss_finite = all(math.isfinite(x) for x in losses)
    out.loss_decreasing = losses[-1] < losses[0] if len(losses) >= 2 else False
    return out
