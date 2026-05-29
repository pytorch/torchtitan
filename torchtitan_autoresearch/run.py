"""Single-command autoresearch entrypoint.

Starts a fresh experiment on whatever machine you run it on: detects the GPU
count, creates an isolated experiment branch, calibrates the golden (quality bar,
faithfulness anchor, throughput noise), then runs the hill-climbing loop with the
quality gate live. Hardware is not assumed -- the constitution's ``ngpu: auto``
is resolved here.

    python -m torchtitan_autoresearch.run --tag may30-qwen3 [--eval-dataset c4_validation]

Run from the repo root (so the package and run_train.sh are found). For real c4
eval set ``--eval-dataset c4_validation`` and pass your proxy env if needed.
Follow along read-only from another shell with
``python -m torchtitan_autoresearch.observe --run-dir <dir>``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
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
    ap = argparse.ArgumentParser(prog="torchtitan_autoresearch.run")
    ap.add_argument("--tag", required=True, help="experiment tag -> branch autoresearch/<tag>")
    ap.add_argument("--eval-dataset", default="c4_test",
                    help="held-out eval dataset (c4_test offline, c4_validation real)")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--here", default="",
                    help="repo root (default: cwd)")
    args = ap.parse_args(argv)

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
    print(f"[setup] world size (auto) = {ngpu} | eval dataset = {args.eval_dataset} | run dir = {run_dir}")

    ex = SubprocessExecutor(repo, log_freq=rules.log_freq, ngpu=ngpu,
                            val_dataset=args.eval_dataset, run_dir=os.path.join(run_dir, "runs"))

    sess = start_run(repo, args.tag, rules)
    print(f"[setup] experiment branch: {sess.branch} off {sess.base_commit[:7]}")

    # --- Calibrate the golden: throughput, faithfulness anchor, quality bar ---
    print("[calibrate] running golden baseline (throughput, deterministic, eval)...", flush=True)
    gold = Candidate(label="golden", command=[])
    tr = ex.run_throughput(gold, steps, window)
    if not tr.ok:
        print("[calibrate] golden baseline failed to run; aborting:\n", tr.crash_text[-1500:])
        return 1
    det = ex.deterministic_losses(gold)
    er = ex.run_eval(gold, steps=steps)
    ex.golden_det_losses = det
    print(f"[calibrate] golden: tps={tr.tps_mean:.0f} (cv {tr.tps_cv:.3f})  "
          f"eval_loss={er.eval_loss if er.ok else 'n/a'}  det_losses={[round(x,2) for x in det]}")
    HarnessState(
        golden_commit=sess.base_commit[:7],
        golden_eval_loss=(er.eval_loss if er.ok else None),
        golden_det_losses=det,
        champion_commit=sess.base_commit[:7],
        champion_tps=[tr.tps_mean],
        tps_cv=max(tr.tps_cv, 0.01),
        tps_tail_pct=3.7,  # heavy-tail default; refine with repeated golden runs
    ).save(statefile)

    H = Harness(constitution_path=const, ideas_path=ideas, ledger_path=ledger,
                statefile=statefile, executor=ex, session=sess,
                report_path=os.path.join(run_dir, "report.json"))

    print(f"[loop] starting hill-climb (max {args.max_iters} candidates). "
          f"Follow along: python -m torchtitan_autoresearch.observe --run-dir {run_dir}", flush=True)
    summary = run_loop(H, KnobAgent(), max_iters=args.max_iters, report_every=1)
    print("[done]", summary)
    print("ledger:\n" + (open(ledger).read() if os.path.exists(ledger) else "(empty)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
