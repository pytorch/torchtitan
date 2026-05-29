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
    """Runs candidates through the TorchTitan launcher and torchrun entrypoints.

    Measurement is harness-pinned: ``--metrics.log_freq`` and the step cap come
    from the constitution, not the candidate. Throughput is parsed by ``measure``;
    verify/eval entrypoints print one parseable line each.
    """

    def __init__(self, repo_root: str, log_freq: int, ngpu: int = 8, run_dir: str = "/tmp"):
        self.repo_root = repo_root
        self.log_freq = log_freq
        self.ngpu = ngpu
        self.run_dir = run_dir

    def _run(self, argv: list[str], log_path: str) -> int:
        env = {**os.environ, "NGPU": str(self.ngpu)}
        with open(log_path, "w") as f:
            return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env).returncode

    def run_throughput(self, c, steps, window):
        log = os.path.join(self.run_dir, f"tps_{c.commit[:7]}.log")
        argv = [
            os.path.join(self.repo_root, "run_train.sh"), *c.command,
            f"--training.steps={steps}", f"--metrics.log_freq={self.log_freq}",
        ]
        self._run(argv, log)
        text = open(log).read() if os.path.exists(log) else ""
        meas = M.measure(text, window)
        if not meas.steps or not meas.loss_finite:
            return ThroughputResult(ok=False, crash_text=text[-4000:])
        return ThroughputResult(
            ok=True, tps_mean=meas.tps_mean, tps_cv=meas.tps_cv,
            peak_mem_gb=meas.peak_memory_gb,
        )

    def run_verify(self, c):
        import re
        log = os.path.join(self.run_dir, f"verify_{c.commit[:7]}.log")
        argv = [
            os.path.join(os.path.dirname(__file__), "run_verify.sh"), *c.command,
            "--debug.seed=42", "--debug.deterministic",
        ]
        self._run(argv, log)
        text = open(log).read() if os.path.exists(log) else ""
        m = re.search(r"VERIFY:\s*status=(\w+)", text)
        # The verify entrypoint reports faithfulness vs the golden; "pass" is
        # accepted as a synonym for "faithful" for older entrypoint builds.
        faithful = bool(m) and m.group(1) in ("faithful", "pass")
        return VerifyResult(faithful=faithful, detail=text[-2000:])

    def run_eval(self, c):
        import re
        log = os.path.join(self.run_dir, f"eval_{c.commit[:7]}.log")
        argv = [
            os.path.join(self.repo_root, "run_train.sh"), *c.command,
            f"--metrics.log_freq={self.log_freq}", "--eval.enable",
        ]
        self._run(argv, log)
        text = open(log).read() if os.path.exists(log) else ""
        m = re.search(r"EVAL:\s*eval_loss=([-\d.eE+]+)", text)
        if not m:
            return EvalResult(ok=False, crash_text=text[-4000:])
        return EvalResult(ok=True, eval_loss=float(m.group(1)))


def classify_crash(text: str) -> cc.CrashVerdict:
    return cc.classify(text)
