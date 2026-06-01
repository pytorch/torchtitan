# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The GPU-execution boundary, behind an injectable interface.

The harness orchestration (gate, quality, api) only depends on this Protocol, so
the whole control flow is testable on CPU with `FakeExecutor`. `SubprocessExecutor`
is the real implementation that shells out to the TorchTitan launcher and the
verify/eval torchrun entrypoints.

Throughput is the climbed axis; quality (eval loss, lower is better) is the
floored axis; verify decides whether a change is faithful enough to skip the eval.
"""

from __future__ import annotations

import math
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
    global_batch: int = (
        0  # effective global batch size; converts a token budget -> steps
    )
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
    def run_throughput(
        self, c: Candidate, steps: int, window: tuple[int, int]
    ) -> ThroughputResult:
        ...

    def run_verify(self, c: Candidate) -> VerifyResult:
        ...

    def run_eval(
        self, c: Candidate, *, global_batch: int = 0, run_tag: str = ""
    ) -> EvalResult:
        ...


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

    def run_eval(self, c, *, global_batch: int = 0, run_tag: str = ""):
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
      - ``warm``   : one-time golden pre-train (``warm_steps``) that saves a full
                     checkpoint; every eval warm-starts from it so the quality
                     signal is taken past warmup, not in from-scratch chaos.
      - ``eval``   : validator-enabled run -> held-out eval loss (quality signal),
                     warm-started from the golden checkpoint when available, and
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
        seq_len: int = 4096,
        eval_tokens: int = 0,
        eval_fallback_steps: int = 50,
        warm_steps: int = 0,
        lr_total_steps: int = 0,
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
        self.seq_len = seq_len
        self.eval_tokens = eval_tokens
        self.eval_fallback_steps = eval_fallback_steps
        self.warm_steps = warm_steps  # pre-train the golden this far past warmup
        self.lr_total_steps = lr_total_steps  # real LR-schedule horizon (decoupled)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.golden_det_losses: list[float] = []  # set by the driver after calibration
        self.warm_ckpt_path: str = (
            ""  # set by prepare_warm_checkpoint; evals warm-start here
        )
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
            "MODULE": "qwen3",
            "CONFIG": "qwen3_14b",
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
            global_batch=meas.global_batch,
        )

    def _short_losses(
        self, c: Candidate, mode: str, deterministic: bool
    ) -> list[float]:
        extra = [f"--training.steps={self.verify_steps}", "--debug.seed=42"]
        if deterministic:
            extra.append("--debug.deterministic")
        text = open(self._run(c, mode, extra)).read()
        return [s.loss for s in M.parse_steps(text)]

    def deterministic_losses(self, c: Candidate) -> list[float]:
        """Per-step loss of a short seed-pinned deterministic run (the faithfulness probe)."""
        return self._short_losses(c, "verify", deterministic=True)

    def jitter_losses(self, c: Candidate) -> list[float]:
        """Same seed/data but NON-deterministic -- exposes the run-to-run rounding
        jitter used to calibrate the faithfulness tolerance (the 'harmless noise'
        scale). This is same-data rounding noise, not seed-to-seed (different-data)
        variation, which would be far too loose and let precision changes hide."""
        return self._short_losses(c, "jitter", deterministic=False)

    def run_verify(self, c):
        golden = self.golden_det_losses
        if not golden:
            return VerifyResult(
                faithful=False, detail="no golden trajectory; treat as affecting"
            )
        losses = self.deterministic_losses(c)
        if len(losses) < len(golden):
            # deterministic run failed (e.g. compile+deterministic conflict) -> eval to be safe
            return VerifyResult(
                faithful=False, detail="deterministic verify run failed; eval"
            )
        worst = max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(losses, golden))
        faithful = worst <= self.verify_tol
        return VerifyResult(
            faithful=faithful,
            detail=f"max loss rel-dev {worst:.2e} vs tol {self.verify_tol:.0e}",
        )

    def eval_steps_for(self, global_batch: int) -> int:
        """Steps to train before the held-out eval, at a fixed *token* budget.

        Equal-compute, not equal-steps: a recipe with a bigger global batch runs
        proportionally fewer steps to consume the same tokens, so the quality
        comparison is per-compute (the objective) rather than penalizing/rewarding
        whoever happens to see more data at a fixed step count.
        """
        if self.eval_tokens and global_batch:
            return max(1, math.ceil(self.eval_tokens / (global_batch * self.seq_len)))
        return self.eval_fallback_steps

    def prepare_warm_checkpoint(self, golden: Candidate) -> str:
        """Pre-train the golden ``warm_steps`` (past warmup) and save a full
        checkpoint the evals warm-start from. Paid once per experiment.

        Solves substrate soft point #2: a from-scratch held-out eval is dominated
        by warmup chaos, so a short eval is pure noise. Continuing an already-good
        model a few *post-warmup* steps is a tight, cheap, honest quality signal --
        and it matches the operating assumption that the starting model is good and
        we are sensitive to degradations. ``lr_scheduler.total_steps`` keeps the LR
        curve on its real horizon so step W lands where it truly would in training.

        Returns the checkpoint folder path (also stored on ``self.warm_ckpt_path``),
        or "" if warm start is disabled (``warm_steps<=0``) or the run failed.
        """
        if self.warm_steps <= 0:
            return ""
        extra = [
            f"--training.steps={self.warm_steps}",
            "--checkpoint.enable",
            "--checkpoint.folder=ckpt",
            f"--checkpoint.interval={self.warm_steps}",  # one full checkpoint, at step W
            "--checkpoint.no_last_save_model_only",  # full state (model+opt+sched)
        ]
        if self.lr_total_steps:
            extra.append(f"--lr_scheduler.total_steps={self.lr_total_steps}")
        self._run(golden, "warm", extra)
        # DCP writes {dump}/ckpt/step-W/{.metadata,*.distcp}; initial_load_path must
        # point at the STEP dir (where .metadata lives), not the parent folder.
        step_dir = os.path.join(
            self._dump_folder(golden, "warm"), "ckpt", f"step-{self.warm_steps}"
        )
        if not os.path.isfile(os.path.join(step_dir, ".metadata")):
            return ""
        self.warm_ckpt_path = step_dir
        return step_dir

    def run_eval(self, c, *, global_batch: int = 0, run_tag: str = ""):
        eval_steps = self.eval_steps_for(global_batch)
        extra = [
            "--validator.enable",
            f"--validator.dataloader.dataset={self.val_dataset}",
            f"--validator.steps={self.val_steps}",
        ]
        if self.warm_ckpt_path:
            # Warm-start from the golden checkpoint: resume at step W (full state),
            # train eval_steps more *post-warmup* steps, then validate. The fresh
            # per-run dump keeps checkpoint.folder empty so initial_load_path is honored.
            total = self.warm_steps + eval_steps
            extra += [
                "--checkpoint.enable",
                f"--checkpoint.initial_load_path={self.warm_ckpt_path}",
                "--checkpoint.no_initial_load_model_only",  # resume opt+sched, not just weights
                "--checkpoint.load_only",  # load the warm state; never save (eval is throwaway)
                f"--training.steps={total}",
                f"--validator.freq={total}",
            ]
            if self.lr_total_steps:
                extra.append(f"--lr_scheduler.total_steps={self.lr_total_steps}")
        else:
            # No warm checkpoint: from-scratch eval (toy/offline mode).
            extra += [
                f"--training.steps={eval_steps}",
                f"--validator.freq={eval_steps}",
            ]
        # run_tag distinguishes repeated golden evals so noise calibration gets
        # independent (non-cached) runs rather than the same cached log.
        text = open(self._run(c, "eval" + run_tag, extra)).read()
        loss = M.parse_validation_loss(text)
        if loss is None:
            return EvalResult(ok=False, crash_text=text[-4000:])
        return EvalResult(ok=True, eval_loss=loss)


def classify_crash(text: str) -> cc.CrashVerdict:
    return cc.classify(text)
