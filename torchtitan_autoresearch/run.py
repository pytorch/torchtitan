# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The autoresearch creating loop -- the internal worker the observer launches.

This is NOT the human entry point. Autoresearch is started only through the
observer (``python -m torchtitan_autoresearch.observe start --tag <tag>``), which
spawns this loop as a separate background process. Running this module directly is
refused (it checks the observer marker env var) so there is exactly one way to
start an experiment.

The loop detects the GPU count (constitution ``ngpu: auto``), creates an isolated
experiment branch, calibrates the golden (quality bar, faithfulness anchor,
throughput noise), then runs the hill-climb with the quality gate live.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

from torchtitan_autoresearch.agent import KnobAgent
from torchtitan_autoresearch.api import Harness
from torchtitan_autoresearch.constitution import load_constitution
from torchtitan_autoresearch.executor import SubprocessExecutor
from torchtitan_autoresearch.loop import run_loop
from torchtitan_autoresearch.session import start_run
from torchtitan_autoresearch.state import HarnessState
from torchtitan_autoresearch.types import Candidate


def _repo_root() -> str:
    return os.getcwd()


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("AR_RUN_FROM_OBSERVER") != "1":
        print(
            "autoresearch is started only through the observer:\n"
            "  python -m torchtitan_autoresearch.observe start --tag <tag>",
            file=sys.stderr,
        )
        return 2
    ap = argparse.ArgumentParser(prog="torchtitan_autoresearch.run")
    ap.add_argument(
        "--tag", required=True, help="experiment tag -> branch autoresearch/<tag>"
    )
    ap.add_argument(
        "--dataset",
        default="c4",
        help="dataset for training AND its held-out eval split "
        "(c4 -> train c4 / eval c4_validation; c4_test -> both local/offline)",
    )
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--here", default="", help="repo root (default: cwd)")
    args = ap.parse_args(argv)

    # A single dataset choice decides both the training set and its held-out eval
    # split. c4's held-out split is c4_validation; toy c4_test has no separate split.
    eval_split = {"c4": "c4_validation", "c4_test": "c4_test"}.get(
        args.dataset, args.dataset
    )
    train_dataset, eval_dataset = args.dataset, eval_split

    repo = args.here or _repo_root()
    const = os.path.join(repo, "torchtitan_autoresearch/Constitution.md")
    ideas = os.path.join(repo, "torchtitan_autoresearch/Ideas.md")
    rules = load_constitution(const)
    ngpu = rules.resolve_ngpu()
    steps, window = rules.steps("screen"), rules.window("screen")
    run_dir = args.run_dir or f"/tmp/ar_{args.tag}"
    os.makedirs(run_dir, exist_ok=True)
    statefile = os.path.join(run_dir, "state.json")
    ledger = os.path.join(run_dir, "results.tsv")

    if ngpu <= 0:
        print("no GPUs detected; autoresearch needs at least one GPU", file=sys.stderr)
        return 2
    print(
        f"[setup] world size (auto) = {ngpu} | dataset = {args.dataset} | "
        f"(train {train_dataset} / eval {eval_dataset}) | run dir = {run_dir}"
    )

    # Create the experiment in its OWN git worktree so all its commits/resets/runs
    # are isolated from the primary checkout (which stays free for development).
    sess = start_run(repo, args.tag, rules, worktree_path=os.path.join(run_dir, "wt"))
    print(
        f"[setup] experiment branch: {sess.branch} off {sess.base_commit[:7]} "
        f"(isolated worktree: {sess.repo_root})"
    )

    # The executor runs in the worktree; the train dataset is a locked workload
    # field set by the human here (not a candidate knob).
    ex = SubprocessExecutor(
        sess.repo_root,
        log_freq=rules.log_freq,
        ngpu=ngpu,
        base_command=[f"--dataloader.dataset={train_dataset}"],
        val_dataset=eval_dataset,
        seq_len=rules.seq_len,
        eval_tokens=rules.eval_tokens,
        eval_fallback_steps=rules.eval_fallback_steps,
        val_steps=rules.eval_val_steps,
        warm_steps=rules.eval_warm_steps,
        lr_total_steps=rules.eval_lr_total_steps,
        run_dir=os.path.join(run_dir, "runs"),
    )
    try:
        # --- Calibrate the golden: throughput, faithfulness anchor, quality bar ---
        print(
            "[calibrate] running golden baseline (throughput, deterministic, eval)...",
            flush=True,
        )
        gold = Candidate(label="golden", command=[])
        tr = ex.run_throughput(gold, steps, window)
        if not tr.ok:
            print(
                "[calibrate] golden baseline failed to run; aborting:\n",
                tr.crash_text[-1500:],
            )
            return 1
        det = ex.deterministic_losses(gold)
        ex.golden_det_losses = det
        # Calibrate the faithfulness tolerance from the golden's own rounding
        # jitter (same seed/data, nondeterministic vs deterministic) x headroom,
        # so rounding-noise changes pass as faithful but real math changes don't.
        jit = ex.jitter_losses(gold)
        if jit and len(jit) == len(det):
            jitter = max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(jit, det))
            ex.verify_tol = max(jitter * 10.0, 5e-4)
            print(
                f"[calibrate] faithfulness tol = {ex.verify_tol:.2e} "
                f"(golden rounding jitter {jitter:.2e} x10)"
            )
        # Warm checkpoint (substrate soft point #2): pre-train the golden past
        # warmup once so every held-out eval continues an already-good model a few
        # post-warmup steps instead of measuring from-scratch chaos. Paid once.
        if rules.eval_warm_steps > 0:
            print(
                f"[calibrate] building golden warm checkpoint @ {rules.eval_warm_steps} "
                f"steps (LR horizon {rules.eval_lr_total_steps or 'run-length'})...",
                flush=True,
            )
            wp = ex.prepare_warm_checkpoint(gold)
            print(
                f"[calibrate] warm checkpoint: {wp or 'FAILED -> evals run from scratch'}",
                flush=True,
            )

        # Quality bar + eval-noise band: repeat the golden's held-out eval at a
        # fixed TOKEN budget (equal-compute, not equal-steps) and measure its
        # run-to-run spread. With a warm checkpoint these post-warmup evals are a
        # tight signal; the floor uses the measured noise so the gate never rejects
        # a candidate on a difference the eval itself cannot resolve.
        gb = tr.global_batch
        eval_steps = ex.eval_steps_for(gb)
        reps = rules.eval_calibration_repeats
        print(
            f"[calibrate] golden eval x{reps} at ~{eval_steps} post-warmup steps "
            f"({rules.eval_tokens:,} tokens / global batch {gb})...",
            flush=True,
        )
        eval_losses: list[float] = []
        last_crash = ""
        for i in range(reps):
            er = ex.run_eval(gold, global_batch=gb, run_tag=f"_cal{i}")
            if er.ok:
                eval_losses.append(er.eval_loss)
                print(
                    f"[calibrate]   golden eval[{i}] = {er.eval_loss:.4f}", flush=True
                )
            else:
                last_crash = er.crash_text or ""
                print(f"[calibrate]   golden eval[{i}] failed", flush=True)
        if not eval_losses:
            print(
                "[calibrate] golden eval failed on every repeat; aborting:\n",
                last_crash[-1500:],
            )
            return 1
        mean = sum(eval_losses) / len(eval_losses)
        std = statistics.stdev(eval_losses) if len(eval_losses) > 1 else 0.0
        band = max(rules.epsilon_rel * mean, rules.eval_z * std)
        print(
            f"[calibrate] golden: tps={tr.tps_mean:.0f} (cv {tr.tps_cv:.3f})  "
            f"eval_loss={mean:.4f} +/- {std:.4f}  "
            f"floor=golden+{band:.4f} (rejects degradation > {band / mean * 100:.1f}% of golden)  "
            f"det_losses={[round(x, 2) for x in det]}"
        )
        HarnessState(
            golden_commit=sess.base_commit[:7],
            golden_eval_loss=mean,
            golden_eval_losses=eval_losses,
            eval_noise_abs=std,
            eval_noise_rel=(std / mean if mean else 0.0),
            golden_det_losses=det,
            champion_commit=sess.base_commit[:7],
            champion_tps=[tr.tps_mean],
            tps_cv=max(tr.tps_cv, 0.01),
            tps_tail_pct=3.7,  # heavy-tail default; refine with repeated golden runs
        ).save(statefile)

        H = Harness(
            constitution_path=const,
            ideas_path=ideas,
            ledger_path=ledger,
            statefile=statefile,
            executor=ex,
            session=sess,
            report_path=os.path.join(run_dir, "report.json"),
        )

        print(
            f"[loop] starting hill-climb (max {args.max_iters} candidates). "
            f"Follow: python -m torchtitan_autoresearch.observe watch --run-dir {run_dir}",
            flush=True,
        )
        summary = run_loop(H, KnobAgent(), max_iters=args.max_iters, report_every=1)
        print("[done]", summary)
        print(
            "ledger:\n" + (open(ledger).read() if os.path.exists(ledger) else "(empty)")
        )
    finally:
        sess.remove()  # tear down the worktree; the branch + its commits are kept
    return 0


if __name__ == "__main__":
    sys.exit(main())
