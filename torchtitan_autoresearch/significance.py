"""Noise-aware promotion decisions.

Post-mortem finding 1a: run-to-run tps variance was ~2-3% on identical commits,
yet the search spent ~240 experiments crawling a band almost entirely inside
that noise, plus 93 ad-hoc reruns to "validate" sub-noise deltas. This module
turns "is this a real improvement?" into a pre-registered statistical decision
against a measured noise floor, and tells the driver when a rerun would actually
resolve the question vs. when it's wasted (operationalizing "do not rerun the
current best by habit").

Two paths:
  - single sample each: compare the delta to ``z`` * combined sigma derived from
    a measured coefficient of variation (the noise floor).
  - multiple samples each: Welch's t-test (unequal variance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NoiseModel:
    """Champion variance, measured by repeating the champion run.

    The empirical tps noise on this workload was heavy-tailed (median rel-dev
    ~0.8% but p90 ~4.5%), so a single run can land far in the tail. We therefore
    keep both a robust ``cv`` (from the median absolute deviation, not the
    outlier-sensitive std) and a high ``tail_pct`` quantile; single-sample
    promotion must clear the tail, not just 2σ. The driver recalibrates when the
    recipe changes a lot, since the noise floor itself shifts (MXFP8 added
    allocator jitter).
    """

    cv: float = 0.02
    tail_pct: float = 4.5  # high quantile of |rel deviation|, in percent
    n_calibration_runs: int = 0

    @classmethod
    def from_samples(cls, tps_samples: list[float], tail_q: float = 0.9) -> "NoiseModel":
        n = len(tps_samples)
        if n < 3:
            return cls(n_calibration_runs=n)
        med = sorted(tps_samples)[n // 2]
        # MAD-based robust std estimate (1.4826 * MAD ~= sigma for normal data).
        devs = sorted(abs(x - med) for x in tps_samples)
        mad = devs[n // 2]
        cv = (1.4826 * mad / med) if med else 0.02
        rel = sorted(abs(x - med) / med for x in tps_samples)
        tail = 100.0 * rel[min(int(tail_q * n), n - 1)]
        return cls(cv=cv, tail_pct=tail, n_calibration_runs=n)


@dataclass
class Verdict:
    recommend: str  # "promote" | "reject" | "rerun"
    significant: bool
    delta_pct: float
    threshold_pct: float
    detail: str


def _mean_std(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var), n


def decide(
    candidate_tps: list[float],
    champion_tps: list[float],
    noise: NoiseModel,
    z: float = 2.0,
    rerun_band: float = 1.0,
) -> Verdict:
    """Decide whether ``candidate`` beats ``champion`` beyond the noise floor.

    ``z`` is the significance threshold in sigmas (default ~2σ). The ``rerun``
    zone is the gap between ``rerun_band``*sigma and ``z``*sigma: a delta there
    is promising but unproven, so one more sample is worth it; anywhere else a
    rerun changes nothing and should be skipped.
    """
    c_mean, c_std, c_n = _mean_std(candidate_tps)
    b_mean, b_std, b_n = _mean_std(champion_tps)
    delta = c_mean - b_mean
    delta_pct = 100.0 * delta / b_mean if b_mean else 0.0

    if c_n >= 2 and b_n >= 2:
        # Welch's t-test standard error; report threshold as z-sigma equivalent.
        se = math.sqrt(c_std**2 / c_n + b_std**2 / b_n)
        threshold = z * se
        threshold_pct = 100.0 * threshold / b_mean if b_mean else 0.0
        method = "welch"
    else:
        # Single-sample path: sigma from the robust noise floor, but the promote
        # bar is the heavy tail, not 2σ — one run can land in the tail (this is
        # exactly how sub-noise "wins" survived in the original search).
        sigma = noise.cv * b_mean
        se = sigma * math.sqrt(1.0 / max(c_n, 1) + 1.0 / max(b_n, 1))
        threshold_pct = max(z * 100.0 * se / b_mean, noise.tail_pct)
        threshold = threshold_pct / 100.0 * b_mean
        method = f"single-sample cv={noise.cv:.3f} tail={noise.tail_pct:.1f}%"

    rerun_threshold = rerun_band * se

    if delta >= threshold:
        return Verdict("promote", True, delta_pct, threshold_pct,
                       f"+{delta_pct:.2f}% >= {threshold_pct:.2f}% ({method})")
    if delta >= rerun_threshold:
        return Verdict("rerun", False, delta_pct, threshold_pct,
                       f"+{delta_pct:.2f}% promising but < {threshold_pct:.2f}% bar; add a sample ({method})")
    return Verdict("reject", False, delta_pct, threshold_pct,
                   f"{delta_pct:+.2f}% within noise ({method})")
