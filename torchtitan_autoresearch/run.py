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

    # v1 trains on this dataset; there is no held-out eval split (faithfulness-only).
    train_dataset = args.dataset

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
        f"[setup] world size (auto) = {ngpu} | train dataset = {train_dataset} | "
        f"run dir = {run_dir}"
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
        run_dir=os.path.join(run_dir, "runs"),
    )
    try:
        # --- Calibrate the golden: throughput + faithfulness anchors/bands ---
        # v1 is faithfulness-only: no held-out eval. We anchor on the golden's
        # short deterministic loss+grad_norm trajectory and size the per-axis
        # faithfulness bands from its OWN run-to-run rounding jitter.
        print(
            "[calibrate] running golden baseline (throughput + faithfulness anchor)...",
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
        det = ex.deterministic_steps(gold)
        if not det:
            print("[calibrate] golden deterministic run produced no steps; aborting")
            return 1
        det_losses = [s.loss for s in det]
        det_grads = [s.grad_norm for s in det]
        ex.golden_det_losses = det_losses
        ex.golden_det_grad_norms = det_grads

        # Bands = the golden's own rounding noise (deterministic vs same-seed
        # non-deterministic) x headroom, floored. A candidate is faithful only if
        # it stays within this band AND is non-trending (see executor._check_axis).
        jit = ex.jitter_steps(gold)

        def _band(jit_vals: list[float], det_vals: list[float]) -> float:
            n = min(len(jit_vals), len(det_vals))
            if n == 0:
                return 5e-4
            worst = max(
                abs(jit_vals[i] - det_vals[i]) / max(abs(det_vals[i]), 1e-9)
                for i in range(n)
            )
            return max(worst * ex.band_headroom, 5e-4)

        ex.loss_band = _band([s.loss for s in jit], det_losses)
        ex.grad_band = _band([s.grad_norm for s in jit], det_grads)
        print(
            f"[calibrate] golden: tps={tr.tps_mean:.0f} (cv {tr.tps_cv:.3f})  "
            f"loss_band={ex.loss_band:.2e}  grad_band={ex.grad_band:.2e}  "
            f"(jitter x{ex.band_headroom:g}, non-trending @ {ex.trend_factor:g}x band)  "
            f"det_losses={[round(x, 2) for x in det_losses]}",
            flush=True,
        )
        HarnessState(
            golden_commit=sess.base_commit[:7],
            golden_det_losses=det_losses,
            golden_det_grad_norms=det_grads,
            loss_band=ex.loss_band,
            grad_band=ex.grad_band,
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
