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

    Each candidate is one training run with TorchTitan's validator enabled, so a
    single run yields both throughput (per-step `tps`, parsed by ``measure``) and
    the held-out eval loss (`validate step: ... loss:`, the quality signal). The
    log is cached per candidate so ``run_throughput`` and ``run_eval`` share one
    GPU run.

    Measurement is harness-pinned: ``--metrics.log_freq`` and the step cap come
    from the constitution, not the candidate. ``base_command`` carries the fixed
    regime flags (e.g. the bf16 golden dtype) prepended to each candidate command.

    Verify (the faithful/affecting routing optimization) is not yet wired, so
    ``run_verify`` is conservative: every candidate is treated as quality-affecting
    and gets a real eval. This is safe (degradation-sensitive) at the cost of
    eval-ing changes that are actually quality-neutral.
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
        run_dir: str = "/tmp/ar_runs",
    ):
        self.repo_root = repo_root
        self.log_freq = log_freq
        self.ngpu = ngpu
        self.base_command = base_command or []
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._log_cache: dict[str, str] = {}  # candidate key -> log path

    def _key(self, c: Candidate) -> str:
        return (c.commit or c.label or "run")[:16].replace("/", "_")

    def _run_once(self, c: Candidate, steps: int) -> str:
        """Run one validation-enabled training run; return the log path (cached)."""
        key = self._key(c)
        if key in self._log_cache:
            return self._log_cache[key]
        log = os.path.join(self.run_dir, f"{key}.log")
        argv = [
            os.path.join(self.repo_root, "run_train.sh"),
            *self.base_command, *c.command,
            f"--training.steps={steps}",
            f"--metrics.log_freq={self.log_freq}",
            "--validator.enable",
            f"--validator.dataloader.dataset={self.val_dataset}",
            f"--validator.freq={steps}",          # validate once, at the last step
            f"--validator.steps={self.val_steps}",
            f"--dump_folder={os.path.join(self.run_dir, key + '_dump')}",
        ]
        env = {**os.environ, "NGPU": str(self.ngpu), "MODULE": "qwen3", "CONFIG": "qwen3_14b"}
        with open(log, "w") as f:
            subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env)
        self._log_cache[key] = log
        return log

    def run_throughput(self, c, steps, window):
        text = open(self._run_once(c, steps)).read()
        meas = M.measure(text, window)
        if not meas.steps or not meas.loss_finite:
            return ThroughputResult(ok=False, crash_text=text[-4000:])
        return ThroughputResult(ok=True, tps_mean=meas.tps_mean, tps_cv=meas.tps_cv,
                                peak_mem_gb=meas.peak_memory_gb)

    def run_verify(self, c):
        # Conservative: route every candidate to the real eval (verify-routing TODO).
        return VerifyResult(faithful=False, detail="verify-routing not wired; eval all")

    def run_eval(self, c, steps=10):
        text = open(self._run_once(c, steps)).read()
        loss = M.parse_validation_loss(text)
        if loss is None:
            return EvalResult(ok=False, crash_text=text[-4000:])
        return EvalResult(ok=True, eval_loss=loss)


def classify_crash(text: str) -> cc.CrashVerdict:
    return cc.classify(text)
