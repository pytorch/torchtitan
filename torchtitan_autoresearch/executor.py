# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The GPU-execution boundary, behind an injectable interface.

The harness orchestration (gate, quality, api) only depends on this Protocol, so
the whole control flow is testable on CPU with ``FakeExecutor``.
``SubprocessExecutor`` is the real implementation that shells out to the
TorchTitan launcher.

v1 evaluation is FAITHFULNESS-ONLY. Throughput is the climbed axis; a candidate
is quality-preserving iff its short seed-pinned deterministic loss AND grad_norm
trajectory stays within the golden's own rounding noise -- no magnitude excursion
beyond the calibrated band, and no directional bias (non-trending). A change that
moves the math beyond that noise is "affecting" and is rejected; there is no
held-out eval in v1, so quality is preserved by construction.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Protocol

from torchtitan_autoresearch import crash_classify as cc, measure as M
from torchtitan_autoresearch.types import Candidate


@dataclass
class ThroughputResult:
    ok: bool
    tps_mean: float = 0.0
    tps_cv: float = 0.0
    peak_mem_gb: float = 0.0
    crash_text: str = ""  # populated when ok is False


@dataclass
class VerifyResult:
    faithful: bool
    detail: str = ""


@dataclass
class AxisCheck:
    """Faithfulness verdict for one trajectory axis (loss or grad_norm)."""

    ok: bool
    max_dev: float  # worst per-step relative deviation vs golden
    mean_dev: float  # signed mean relative deviation (detects directional bias)
    band: float
    detail: str


class Executor(Protocol):
    def run_throughput(
        self, c: Candidate, steps: int, window: tuple[int, int]
    ) -> ThroughputResult:
        ...

    def run_verify(self, c: Candidate) -> VerifyResult:
        ...


# ---------------------------------------------------------------------------
# Fake executor -- deterministic canned results, for CPU testing of the harness.
# ---------------------------------------------------------------------------
class FakeExecutor:
    """Drives any scenario from a per-candidate spec dict.

    spec fields: tps, cv, faithful, crash (text), peak_mem. Missing fields fall
    back to sensible defaults.
    """

    def __init__(self, specs: dict[str, dict]):
        self.specs = specs

    def _spec(self, c: Candidate) -> dict:
        for key in (c.commit, c.label, c.family):
            if key and key in self.specs:
                return self.specs[key]
        return {}

    def run_throughput(self, c, steps, window):
        s = self._spec(c)
        if s.get("crash"):
            return ThroughputResult(ok=False, crash_text=s["crash"])
        return ThroughputResult(
            ok=True,
            tps_mean=float(s.get("tps", 0.0)),
            tps_cv=float(s.get("cv", 0.0)),
            peak_mem_gb=float(s.get("peak_mem", 0.0)),
        )

    def run_verify(self, c):
        s = self._spec(c)
        return VerifyResult(
            faithful=bool(s.get("faithful", True)), detail=s.get("verify_detail", "")
        )


# ---------------------------------------------------------------------------
# Subprocess executor -- the real implementation (needs GPUs + TorchTitan).
# ---------------------------------------------------------------------------
class SubprocessExecutor:
    """Runs candidates through the real TorchTitan launcher on GPUs.

    Two run modes, each cached per (candidate, mode):
      - ``tps``    : fast timed run (no validation, non-deterministic) -> throughput.
      - ``verify`` : short seed-pinned deterministic run whose per-step loss AND
                     grad_norm trajectory is compared to the golden's. A candidate
                     is faithful iff BOTH axes stay within the golden's own rounding
                     band (magnitude) and show no directional bias (trend). Faithful
                     => quality preserved by construction, so it is kept on speed
                     alone; not faithful => affecting => rejected (v1).

    Measurement is harness-pinned: ``--metrics.log_freq`` and step caps come from
    the constitution, not the candidate. ``base_command`` carries fixed regime
    flags prepended to each candidate command. The golden anchors
    (``golden_det_losses``/``golden_det_grad_norms``) and the calibrated bands
    (``loss_band``/``grad_band``) are set by the driver after calibration.
    """

    def __init__(
        self,
        repo_root: str,
        log_freq: int,
        ngpu: int,
        *,
        module: str = "qwen3",
        config: str = "qwen3_14b",
        base_command: list[str] | None = None,
        verify_steps: int = 8,
        trend_factor: float = 0.5,
        band_headroom: float = 3.0,
        run_dir: str = "/tmp/ar_runs",
    ):
        self.repo_root = repo_root
        self.log_freq = log_freq
        self.ngpu = ngpu
        self.module = (
            module  # TorchTitan --module (model family), from the constitution
        )
        self.config = config  # TorchTitan --config (config_fn), from the constitution
        self.base_command = base_command or []
        self.verify_steps = verify_steps  # deterministic steps compared to golden
        self.trend_factor = trend_factor  # bias allowance as a fraction of the band
        self.band_headroom = band_headroom  # golden jitter x this -> the band
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        # Faithfulness anchors + bands, set by the driver after calibration:
        self.golden_det_losses: list[float] = []
        self.golden_det_grad_norms: list[float] = []
        self.loss_band: float = 5e-4
        self.grad_band: float = 5e-4
        self._log_cache: dict[tuple[str, str], str] = {}

    def _key(self, c: Candidate) -> str:
        # Key by commit AND command: command-only candidates share a commit, so
        # keying on commit alone would make them all reuse the first run's cache.
        import hashlib

        h = hashlib.md5(
            ((c.commit or "") + "|" + " ".join(c.command)).encode()
        ).hexdigest()[:8]
        label = (c.label or "cand").replace("/", "_").replace(" ", "_")[:24]
        return f"{label}_{h}"

    def _dump_folder(self, c: Candidate, mode: str) -> str:
        return os.path.join(self.run_dir, f"{self._key(c)}_{mode}_dump")

    def _run(self, c: Candidate, mode: str, extra: list[str]) -> str:
        key = (self._key(c), mode)
        if key in self._log_cache:
            return self._log_cache[key]
        log = os.path.join(self.run_dir, f"{key[0]}_{mode}.log")
        argv = [
            os.path.join(self.repo_root, "run_train.sh"),
            *self.base_command,
            *c.command,
            f"--metrics.log_freq={self.log_freq}",
            f"--dump_folder={self._dump_folder(c, mode)}",
            *extra,
        ]
        # Run inside repo_root (the experiment's worktree): cwd so relative paths
        # (tokenizer, dumps) resolve there, and PYTHONPATH so the worktree's
        # torchtitan (with this candidate's edits) is imported, not the primary.
        env = {
            **os.environ,
            "NGPU": str(self.ngpu),
            "MODULE": self.module,
            "CONFIG": self.config,
            "PYTHONPATH": self.repo_root
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        with open(log, "w") as f:
            subprocess.run(
                argv, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=self.repo_root
            )
        self._log_cache[key] = log
        return log

    def run_throughput(self, c, steps, window):
        text = open(self._run(c, "tps", [f"--training.steps={steps}"])).read()
        meas = M.measure(text, window)
        if not meas.steps or not meas.loss_finite:
            return ThroughputResult(ok=False, crash_text=text[-4000:])
        return ThroughputResult(
            ok=True,
            tps_mean=meas.tps_mean,
            tps_cv=meas.tps_cv,
            peak_mem_gb=meas.peak_memory_gb,
        )

    # --- faithfulness probe -------------------------------------------------
    def _short_steps(
        self, c: Candidate, mode: str, deterministic: bool
    ) -> list[M.Step]:
        extra = [f"--training.steps={self.verify_steps}", "--debug.seed=42"]
        if deterministic:
            extra.append("--debug.deterministic")
        text = open(self._run(c, mode, extra)).read()
        return M.parse_steps(text)

    def deterministic_steps(self, c: Candidate) -> list[M.Step]:
        """Per-step loss+grad_norm of a short seed-pinned DETERMINISTIC run."""
        return self._short_steps(c, "verify", deterministic=True)

    def jitter_steps(self, c: Candidate) -> list[M.Step]:
        """Same seed/data but NON-deterministic: the golden's own run-to-run
        rounding noise (float non-associativity), used to size the faithfulness
        bands. This is same-data noise, not seed-to-seed (different-data) variation
        which would be far too loose and let real changes hide."""
        return self._short_steps(c, "jitter", deterministic=False)

    @staticmethod
    def _rel_devs(values: list[float], golden: list[float]) -> list[float]:
        n = min(len(values), len(golden))
        return [(values[i] - golden[i]) / max(abs(golden[i]), 1e-9) for i in range(n)]

    def _check_axis(
        self, values: list[float], golden: list[float], band: float
    ) -> AxisCheck:
        """Faithful on this axis iff within the band AND non-trending.

        Magnitude: no per-step relative deviation exceeds the band (golden's own
        rounding noise x headroom). Trend: the SIGNED mean deviation stays within
        ``trend_factor`` x band -- unbiased rounding noise averages to ~0, while a
        real systematic change shows a consistent same-sign offset.
        """
        devs = self._rel_devs(values, golden)
        if not devs:
            return AxisCheck(False, float("inf"), float("inf"), band, "no steps")
        max_dev = max(abs(d) for d in devs)
        mean_dev = sum(devs) / len(devs)
        ok = (max_dev <= band) and (abs(mean_dev) <= band * self.trend_factor)
        return AxisCheck(
            ok,
            max_dev,
            mean_dev,
            band,
            f"max {max_dev:.2e} mean {mean_dev:+.2e} vs band {band:.2e}",
        )

    def run_verify(self, c):
        gl, gg = self.golden_det_losses, self.golden_det_grad_norms
        if not gl:
            return VerifyResult(False, "no golden trajectory; treat as affecting")
        steps = self.deterministic_steps(c)
        if len(steps) < len(gl):
            return VerifyResult(
                False, "deterministic verify run failed; treat as affecting"
            )
        la = self._check_axis([s.loss for s in steps], gl, self.loss_band)
        ga = self._check_axis([s.grad_norm for s in steps], gg, self.grad_band)
        return VerifyResult(
            faithful=(la.ok and ga.ok), detail=f"loss[{la.detail}] grad[{ga.detail}]"
        )


def classify_crash(text: str) -> cc.CrashVerdict:
    return cc.classify(text)
