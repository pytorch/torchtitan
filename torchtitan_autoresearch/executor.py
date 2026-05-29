"""The GPU-execution boundary, behind an injectable interface.

The harness orchestration (gate, quality, api) only depends on this Protocol, so
the whole control flow is testable on CPU with `FakeExecutor`. `SubprocessExecutor`
is the real implementation that shells out to the TorchTitan launcher and the
verify/eval torchrun entrypoints.

Throughput is the climbed axis; quality (eval loss, lower is better) is the
floored axis; verify decides whether a change is faithful enough to skip the eval.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Protocol

from torchtitan_autoresearch import crash_classify as cc
from torchtitan_autoresearch import measure as M
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
class EvalResult:
    ok: bool
    eval_loss: float = float("inf")
    crash_text: str = ""


class Executor(Protocol):
    def run_throughput(self, c: Candidate, steps: int, window: tuple[int, int]) -> ThroughputResult: ...
    def run_verify(self, c: Candidate) -> VerifyResult: ...
    def run_eval(self, c: Candidate) -> EvalResult: ...


# ---------------------------------------------------------------------------
# Fake executor — deterministic canned results, for CPU testing of the harness.
# ---------------------------------------------------------------------------
class FakeExecutor:
    """Drives any scenario from a per-candidate spec dict keyed by commit.

    spec fields: tps, cv, faithful, eval_loss, crash (text), eval_crash (text).
    Missing fields fall back to sensible defaults.
    """

    def __init__(self, specs: dict[str, dict]):
        self.specs = specs

    def _spec(self, c: Candidate) -> dict:
        # Key by commit, then label, then family — so tests can key by whatever is
        # stable (git shas are assigned at runtime under a real session).
        for key in (c.commit, c.label, c.family):
            if key and key in self.specs:
                return self.specs[key]
        return {}

    def run_throughput(self, c, steps, window):
        s = self._spec(c)
        if s.get("crash"):
            return ThroughputResult(ok=False, crash_text=s["crash"])
        return ThroughputResult(
            ok=True, tps_mean=float(s.get("tps", 0.0)),
            tps_cv=float(s.get("cv", 0.0)), peak_mem_gb=float(s.get("peak_mem", 0.0)),
        )

    def run_verify(self, c):
        s = self._spec(c)
        return VerifyResult(faithful=bool(s.get("faithful", True)), detail=s.get("verify_detail", ""))

    def run_eval(self, c):
        s = self._spec(c)
        if s.get("eval_crash"):
            return EvalResult(ok=False, crash_text=s["eval_crash"])
        return EvalResult(ok=True, eval_loss=float(s.get("eval_loss", float("inf"))))


# ---------------------------------------------------------------------------
# Subprocess executor — the real implementation (needs GPUs + TorchTitan).
# ---------------------------------------------------------------------------
class SubprocessExecutor:
    """Runs candidates through the real TorchTitan launcher on GPUs.

    Three run modes, each cached per (candidate, mode):
      - ``tps``    : fast timed run (no validation, non-deterministic) -> throughput.
      - ``verify`` : short deterministic run (seed-pinned) whose per-step loss
                     trajectory is compared to the golden's; a faithful match means
                     the change did not move the math, so quality is preserved and
                     the eval is skipped (the design's verify-routing).
      - ``eval``   : validator-enabled run -> held-out eval loss (quality signal),
                     only run when verify says the change is quality-affecting.

    Measurement is harness-pinned: ``--metrics.log_freq`` and step caps come from
    the constitution, not the candidate. ``base_command`` carries fixed regime
    flags prepended to each candidate command. ``golden_det_losses`` is the
    faithfulness anchor, set by the driver after calibration.
    """

    def __init__(
        self,
        repo_root: str,
        log_freq: int,
        ngpu: int,
        *,
        base_command: list[str] | None = None,
        val_dataset: str = "c4_test",
        val_steps: int = 8,
        verify_steps: int = 3,
        verify_tol: float = 5e-3,
        run_dir: str = "/tmp/ar_runs",
    ):
        self.repo_root = repo_root
        self.log_freq = log_freq
        self.ngpu = ngpu
        self.base_command = base_command or []
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.verify_steps = verify_steps
        self.verify_tol = verify_tol
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.golden_det_losses: list[float] = []  # set by the driver after calibration
        self._log_cache: dict[tuple[str, str], str] = {}

    def _key(self, c: Candidate) -> str:
        return (c.commit or c.label or "run")[:16].replace("/", "_")

    def _run(self, c: Candidate, mode: str, extra: list[str]) -> str:
        key = (self._key(c), mode)
        if key in self._log_cache:
            return self._log_cache[key]
        log = os.path.join(self.run_dir, f"{key[0]}_{mode}.log")
        argv = [
            os.path.join(self.repo_root, "run_train.sh"),
            *self.base_command, *c.command,
            f"--metrics.log_freq={self.log_freq}",
            f"--dump_folder={os.path.join(self.run_dir, key[0] + '_' + mode + '_dump')}",
            *extra,
        ]
        env = {**os.environ, "NGPU": str(self.ngpu), "MODULE": "qwen3", "CONFIG": "qwen3_14b"}
        with open(log, "w") as f:
            subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env)
        self._log_cache[key] = log
        return log

    def run_throughput(self, c, steps, window):
        text = open(self._run(c, "tps", [f"--training.steps={steps}"])).read()
        meas = M.measure(text, window)
        if not meas.steps or not meas.loss_finite:
            return ThroughputResult(ok=False, crash_text=text[-4000:])
        return ThroughputResult(ok=True, tps_mean=meas.tps_mean, tps_cv=meas.tps_cv,
                                peak_mem_gb=meas.peak_memory_gb)

    def deterministic_losses(self, c: Candidate) -> list[float]:
        """Per-step loss of a short seed-pinned deterministic run (the faithfulness probe)."""
        text = open(self._run(c, "verify", [
            f"--training.steps={self.verify_steps}",
            "--debug.seed=42", "--debug.deterministic",
        ])).read()
        return [s.loss for s in M.parse_steps(text)]

    def run_verify(self, c):
        golden = self.golden_det_losses
        if not golden:
            return VerifyResult(faithful=False, detail="no golden trajectory; treat as affecting")
        losses = self.deterministic_losses(c)
        if len(losses) < len(golden):
            # deterministic run failed (e.g. compile+deterministic conflict) -> eval to be safe
            return VerifyResult(faithful=False, detail="deterministic verify run failed; eval")
        worst = max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(losses, golden))
        faithful = worst <= self.verify_tol
        return VerifyResult(faithful=faithful,
                            detail=f"max loss rel-dev {worst:.2e} vs tol {self.verify_tol:.0e}")

    def run_eval(self, c, steps=10):
        text = open(self._run(c, "eval", [
            f"--training.steps={steps}",
            "--validator.enable",
            f"--validator.dataloader.dataset={self.val_dataset}",
            f"--validator.freq={steps}",
            f"--validator.steps={self.val_steps}",
        ])).read()
        loss = M.parse_validation_loss(text)
        if loss is None:
            return EvalResult(ok=False, crash_text=text[-4000:])
        return EvalResult(ok=True, eval_loss=loss)


def classify_crash(text: str) -> cc.CrashVerdict:
    return cc.classify(text)
